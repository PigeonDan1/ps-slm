# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import time
import yaml
from contextlib import nullcontext
from pathlib import Path
from pkg_resources import packaging
import datetime

import functools
import hydra
import torch
# import torch.npu.nccl as nccl
import torch.distributed as dist
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import LlamaTokenizer
from typing import Any, Callable, List, Optional
from textwrap import dedent
from hydra import version
from hydra.main import _UNSPECIFIED_, _get_rerun_conf
from hydra._internal.deprecation_warning import deprecation_warning
from hydra._internal.utils import _run_hydra, get_args_parser
from hydra.types import TaskFunction
from hydra.core.utils import _flush_loggers, configure_log

# from dataset.speech_dataset_large import MultiTaskDynamicBatchDataset,MultiTaskDataset
from utils.checkpoint_handler import save_model_checkpoint_deepspeed
from utils.memory_utils import MemoryTrace

import wandb
import logging

logger = logging.getLogger(__name__)

# For deepspeed --local_rank argument
def deepspeed_main_wrapper(
    config_path: Optional[str] = _UNSPECIFIED_,
    config_name: Optional[str] = None,
    version_base: Optional[str] = _UNSPECIFIED_,
) -> Callable[[TaskFunction], Any]:
    """
    :param config_path: The config path, a directory where Hydra will search for
                        config files. This path is added to Hydra's searchpath.
                        Relative paths are interpreted relative to the declaring python
                        file. Alternatively, you can use the prefix `pkg://` to specify
                        a python package to add to the searchpath.
                        If config_path is None no directory is added to the Config search path.
    :param config_name: The name of the config (usually the file name without the .yaml extension)
    """

    version.setbase(version_base)

    if config_path is _UNSPECIFIED_:
        if version.base_at_least("1.2"):
            config_path = None
        elif version_base is _UNSPECIFIED_:
            url = "https://hydra.cc/docs/1.2/upgrades/1.0_to_1.1/changes_to_hydra_main_config_path"
            deprecation_warning(
                message=dedent(
                    f"""
                config_path is not specified in @hydra.main().
                See {url} for more information."""
                ),
                stacklevel=2,
            )
            config_path = "."
        else:
            config_path = "."

    def main_decorator(task_function: TaskFunction) -> Callable[[], None]:
        @functools.wraps(task_function)
        def decorated_main(cfg_passthrough: Optional[DictConfig] = None) -> Any:
            if cfg_passthrough is not None:
                return task_function(cfg_passthrough)
            else:
                args_parser = get_args_parser()
                args_parser.add_argument("--local_rank", type=int, default=-1)
                args = args_parser.parse_args()
                if args.experimental_rerun is not None:
                    cfg = _get_rerun_conf(args.experimental_rerun, args.overrides)
                    task_function(cfg)
                    _flush_loggers()
                else:
                    # no return value from run_hydra() as it may sometime actually run the task_function
                    # multiple times (--multirun)
                    _run_hydra(
                        args=args,
                        args_parser=args_parser,
                        task_function=task_function,
                        config_path=config_path,
                        config_name=config_name,
                    )

        return decorated_main

    return main_decorator


def deepspeed_join(group_join):
    """
    Copy from wenet:https://github.com/wenet-e2e/wenet/blob/main/wenet/utils/executor.py#L64
    """
    try:
        # NOTE(xcsong): Why we need a new group?
        #   Because Deepspeed has its own group where all the relevant communication
        #   operations are executed. If we add a communication operation that is not
        #   managed by Deepspeed in this group, it's highly likely to cause
        #   communication chaos, resulting in hard-to-troubleshoot hangs.
        dist.monitored_barrier(group=group_join,
                               timeout=group_join.options._timeout)
    except RuntimeError as e:
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        logger.info("Detected uneven workload distribution."  +
                     "Break current worker to manually join all workers, " +
                     "world_size {}, current rank {}, current local_rank {}\n".
                     format(world_size, rank, local_rank))
        return True
    return False


def set_tokenizer_params(tokenizer: LlamaTokenizer):
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"


# Converting Bytes to Megabytes
def byte2mb(x):
    return int(x / 2**20)


def train(
    model,
    train_dataloader,
    eval_dataloader,
    tokenizer,
    gradient_accumulation_steps,
    train_config,
    log_config,
    local_rank=None,
    rank=None,
):
    """
    Trains the model on the given dataloader
    """
    # if train_config.use_fp16 and train_config.enable_fsdp:
    #     scaler = ShardedGradScaler()
    # elif train_config.use_fp16 and not train_config.enable_fsdp:
    #     scaler = torch.cuda.amp.GradScaler()
    if train_config.enable_ddp:
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        world_size = 1

    # --- NPU autocast -> CUDA autocast ---
    autocast_cm = torch.cuda.amp.autocast(dtype=torch.bfloat16) if train_config.use_fp16 else nullcontext()

    train_prep = []
    train_loss = []
    train_acc = []
    val_prep = []
    val_loss = []
    val_acc = []
    epoch_times = []
    checkpoint_times = []
    results = {}
    best_val_loss = float("inf")
    best_val_acc = 0.0
    total_step = 0
    data_cnt = 0
    for epoch in range(train_config.num_epochs):
        dist.barrier()
        group_join = dist.new_group(
            backend="gloo", timeout=datetime.timedelta(seconds=7))
        epoch_start_time = time.perf_counter()
        with MemoryTrace() as memtrace:  # track the memory usage
            model.train()
            total_loss = torch.tensor(0.0).to(f"cuda:{local_rank}")
            total_acc = 0
            if train_config.batching_strategy != "dynamic":
                total_length = len(train_dataloader)//gradient_accumulation_steps
                pbar = tqdm(colour="blue", desc=f"Training Epoch: {epoch+1}", total=total_length, dynamic_ncols=True)
            else:
                total_length = len(train_dataloader)//(gradient_accumulation_steps*world_size)
                pbar = tqdm(colour="blue", desc=f"Training Epoch: {epoch+1}", total=total_length, dynamic_ncols=True)
            for step, batch in enumerate(train_dataloader):
                if train_config.batching_strategy == "dynamic" and deepspeed_join(group_join):
                    break
                assert "input_ids" in batch.keys()
                for key in batch.keys():
                    batch[key] = (
                        batch[key].to(f"cuda:{local_rank}").half()
                        if isinstance(batch[key], torch.Tensor)
                        and batch[key].dtype == torch.float32
                        else (
                            batch[key].to(f"cuda:{local_rank}")
                            if isinstance(batch[key], torch.Tensor)
                            else batch[key]
                        )
                    )
                with autocast_cm:
                    outputs, *rest = model(**batch)
                acc = rest[0] if rest else -1
                loss = outputs.loss

                loss = loss / gradient_accumulation_steps
                acc = acc / gradient_accumulation_steps

                if log_config.use_wandb and step % log_config.log_interval == 0:
                    if train_config.enable_fsdp or train_config.enable_ddp:
                        if rank == 0:
                            wandb.log(
                                {
                                    "train_inner/train_inner_loss": loss,
                                    "train_inner/train_inner_accuracy": acc,
                                },
                                step=(epoch * total_length + step) if train_config.batching_strategy != "dynamic" else step + 1,
                            )
                    else:
                        wandb.log(
                            {
                                "train_inner/train_inner_loss": loss,
                                "train_inner/train_inner_accuracy": acc,
                            },
                            step=(epoch * total_length + step) if train_config.batching_strategy != "dynamic" else step + 1,
                        )
                total_loss += loss.detach().float()
                total_acc += acc

                # deepspeed should handle gradient accumulate
                model.backward(loss)
                model.step()
                if train_config.batching_strategy != 'dynamic':
                    if (step + 1) % gradient_accumulation_steps == 0 :
                        pbar.update(1)
                else:
                    pbar.update(batch["input_ids"].shape[0])

                data_cnt += batch["input_ids"].shape[0]
                pbar.set_description(
                    f"Training Epoch: {epoch+1}/{train_config.num_epochs}, {step if train_config.batching_strategy != 'dynamic' else data_cnt}/{len(train_dataloader)/world_size} completed (loss: {loss.detach().float()}, acc: {acc})"
                )

                if ( step + 1) % train_config.validation_interval == 0 and train_config.run_validation:
                    eval_ppl, eval_epoch_loss, *rest = evaluation(
                        model, train_config, eval_dataloader, local_rank, tokenizer
                    )
                    eval_epoch_acc = rest[0] if rest else -1
                    checkpoint_start_time = time.perf_counter()

                    
                    if train_config.save_model and (eval_epoch_loss < best_val_loss or eval_epoch_acc > best_val_acc):
                        checkpoint_name = f"{train_config.model_name}_epoch_{str(epoch+1)}_step_{step+1}"
                        save_model_checkpoint_deepspeed(
                            model, train_config, checkpoint_name
                        )

                    checkpoint_end_time = time.perf_counter() - checkpoint_start_time
                    checkpoint_times.append(checkpoint_end_time)
                    if eval_epoch_loss < best_val_loss:
                        best_val_loss = eval_epoch_loss
                        if rank == 0:
                            logger.info(
                                f"best eval loss on epoch {epoch+1} is {best_val_loss}"
                            )
                    val_loss.append(eval_epoch_loss)
                    val_prep.append(eval_ppl)
                    if rest:
                        if eval_epoch_acc > best_val_acc:
                            best_val_acc = eval_epoch_acc
                            if rank == 0:
                                logger.info(
                                    f"best eval acc on epoch {epoch+1} is {best_val_acc}"
                                )
                        val_acc.append(rest[0])
                    else:
                        val_acc.append(-1)

                    if log_config.use_wandb:
                        if rank == 0:
                            wandb.log(
                                {
                                    "valid/val_epoch_loss": eval_epoch_loss,
                                    "valid/val_perplexity": eval_ppl,
                                    "valid/best_val_loss": best_val_loss,
                                    "valid/val_accuracy": val_acc[-1],
                                    "valid/val_best_accuracy": best_val_acc,
                                }
                            )

                if train_config.run_test_during_validation:
                    if rank == 0:
                        logger.info("=====================================")
                        logger.info(
                            f"Test the file {train_config.run_test_during_validation_file} during validation:"
                        )
                        with autocast_cm:
                            logger.info(
                                model.inference(
                                    train_config.run_test_during_validation_file,
                                    train_config.run_test_during_validation_prompt,
                                )
                            )
                        logger.info("=====================================")
                    dist.barrier()
            pbar.close()
        dist.destroy_process_group(group_join)

        total_step += (step + 1)

        epoch_end_time = time.perf_counter() - epoch_start_time
        epoch_times.append(epoch_end_time)
        # Reducing total_loss across all devices if there's more than one cuda device
        if torch.cuda.device_count() > 1 and (
            train_config.enable_fsdp or train_config.enable_ddp
        ):
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_acc, op=dist.ReduceOp.SUM)
        train_epoch_loss = total_loss / (step + 1)
        train_epoch_acc = total_acc / (step + 1)
        if train_config.enable_fsdp or train_config.enable_ddp:
            train_epoch_loss = train_epoch_loss / world_size
            train_epoch_acc = train_epoch_acc / world_size
        train_perplexity = torch.exp(train_epoch_loss)

        train_prep.append(train_perplexity)
        train_loss.append(train_epoch_loss)
        train_acc.append(train_epoch_acc)

        if log_config.use_wandb:
            if train_config.enable_fsdp or train_config.enable_ddp:
                if rank == 0:
                    wandb.log(
                        {
                            "train/train_perplexity": train_perplexity,
                            "train/train_epoch_loss": train_epoch_loss,
                            "train/train_epoch_acc": train_epoch_acc,
                        }
                    )
            else:
                wandb.log(
                    {
                        "train/train_perplexity": train_perplexity,
                        "train/train_epoch_loss": train_epoch_loss,
                        "train/train_epoch_acc": train_epoch_acc,
                    }
                )

        if rank == 0:
            logger.info(
                f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s"
            )

        if rank == 0:
            logger.info(f"Max cuda memory allocated was {memtrace.peak} GB")
            logger.info(f"Max cuda memory reserved was {memtrace.max_reserved} GB")
            logger.info(f"Peak active cuda memory was {memtrace.peak_active_gb} GB")
            logger.info(f"cuda Malloc retires : {memtrace.npu_malloc_retires}")
            logger.info(
                f"CPU Total Peak Memory consumed during the train (max): {memtrace.cpu_peaked + memtrace.cpu_begin} GB"
            )

    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    avg_checkpoint_time = (
        sum(checkpoint_times) / len(checkpoint_times)
        if len(checkpoint_times) > 0
        else 0
    )
    avg_train_prep = sum(train_prep) / len(train_prep)
    avg_train_loss = sum(train_loss) / len(train_loss)
    avg_train_acc = sum(train_acc) / len(train_acc)
    if train_config.run_validation:
        avg_eval_prep = sum(val_prep) / len(val_prep)
        avg_eval_loss = sum(val_loss) / len(val_loss)
        avg_eval_acc = sum(val_acc) / len(val_acc)

    results["avg_train_prep"] = avg_train_prep
    results["avg_train_loss"] = avg_train_loss
    results["avg_train_acc"] = avg_train_acc
    if train_config.run_validation:
        results["avg_eval_prep"] = avg_eval_prep
        results["avg_eval_loss"] = avg_eval_loss
        results["avg_eval_acc"] = avg_eval_acc
    results["avg_epoch_time"] = avg_epoch_time
    results["avg_checkpoint_time"] = avg_checkpoint_time

    return results


def evaluation(model, train_config, eval_dataloader, local_rank, tokenizer):
    """
    Evaluates the model on the given dataloader
    """
    import math
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{local_rank}")

    model.eval()
    eval_preds = []

    eval_loss = torch.zeros(1, device=device)
    eval_acc = torch.zeros(1, device=device)

    autocast_cm = (
        torch.cuda.amp.autocast(dtype=torch.bfloat16)
        if train_config.use_fp16
        else nullcontext()
    )

    tot_step = 0
    with MemoryTrace() as memtrace:
        if train_config.batching_strategy != "dynamic":
            total_length = len(eval_dataloader)
            pbar = tqdm(
                colour="green",
                desc="Evaluating Epoch",
                total=total_length,
                dynamic_ncols=True,
            )
        else:
            pbar = tqdm(
                colour="green",
                desc="Evaluating Epoch",
                dynamic_ncols=True,
            )

        for step, batch in enumerate(eval_dataloader):
            for key in batch.keys():
                if isinstance(batch[key], torch.Tensor):
                    if batch[key].dtype == torch.float32:
                        batch[key] = batch[key].to(device).half()
                    else:
                        batch[key] = batch[key].to(device)

            with torch.no_grad():
                with autocast_cm:
                    outputs, *rest = model(**batch)

                loss = outputs.loss

                if rest:
                    acc_val = rest[0]
                    if not isinstance(acc_val, torch.Tensor):
                        acc_val = torch.tensor(acc_val, device=device, dtype=torch.float32)
                    else:
                        acc_val = acc_val.to(device=device, dtype=torch.float32)
                else:
                    acc_val = torch.tensor(-1.0, device=device, dtype=torch.float32)

                eval_loss += loss.detach().float()
                eval_acc += acc_val

            preds = torch.argmax(outputs.logits, -1)
            eval_preds.extend(
                tokenizer.batch_decode(
                    preds.detach().cpu().numpy(),
                    skip_special_tokens=True,
                )
            )

            cur_loss = (eval_loss / (step + 1)).item()
            cur_acc = (eval_acc / (step + 1)).item()

            pbar.update(1)
            pbar.set_description(
                f"step: {step+1}/{total_length if train_config.batching_strategy != 'dynamic' else '' }, "
                f"eval_loss: {cur_loss:.4f}, eval_acc: {cur_acc:.4f}"
            )
            tot_step = step + 1

    if torch.cuda.device_count() > 1:
        dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(eval_acc, op=dist.ReduceOp.SUM)

    num_steps = (len(eval_dataloader)
                 if train_config.batching_strategy != "dynamic"
                 else (tot_step + 1))


    eval_epoch_loss = (eval_loss / num_steps) / world_size  
    eval_epoch_acc = (eval_acc / num_steps) / world_size

    eval_ppl = torch.exp(eval_epoch_loss).item()
    eval_epoch_loss_val = eval_epoch_loss.item()
    eval_epoch_acc_val = eval_epoch_acc.item()

    if local_rank == 0:
        logger.info(
            f" eval_ppl={eval_ppl} eval_epoch_loss={eval_epoch_loss_val} eval_epoch_acc={eval_epoch_acc_val}"
        )

    model.train()

    return eval_ppl, eval_epoch_loss_val, eval_epoch_acc_val


def freeze_transformer_layers(model, num_layer):
    for i, layer in enumerate(model.model.layers):
        if i < num_layer:
            for param in layer.parameters():
                param.requires_grad = False


def check_frozen_layers_peft_model(model):
    for i, layer in enumerate(model.base_model.model.model.layers):
        for name, param in layer.named_parameters():
            logger.info(
                f"Layer {i}, parameter {name}: requires_grad = {param.requires_grad}"
            )


def setup():
    """Initialize the process group for distributed training"""
    # --- HCCL -> NCCL ---
    dist.init_process_group("nccl")


def setup_environ_flags(rank):
    """Set environment flags for debugging purposes"""
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    # --- HCCL -> NCCL ---
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)
    # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    # This flag will help with cuda memory fragmentations that can lead into OOM in some cases.
    if rank == 0:
        logger.info(f"--> Running with torch dist debug set to detail")


def cleanup():
    """Clean up the process group after training"""
    dist.destroy_process_group()


def clear_gpu_cache(rank=None):
    """Clear the GPU cache for all ranks"""
    if rank == 0:
        logger.info(f"Clearing GPU cache for all ranks")
    # --- torch.npu.empty_cache() -> cuda ---
    torch.cuda.empty_cache()


def get_parameter_dtypes(model):
    """Get the data types of model parameters"""
    parameter_dtypes = {}
    for name, parameter in model.named_parameters():
        parameter_dtypes[name] = parameter.dtype
    return parameter_dtypes


def print_model_size(model, config, rank: int = 0) -> None:
    """
    log model name, the number of trainable parameters and initialization time.
    """
    if rank == 0:
        logger.info(f"--> Model {config.model_name}")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(
            f"--> {config.model_name} has {total_params / 1e6} Million params\n"
        )


def print_module_size(module, module_name, rank: int = 0) -> None:
    """
    Print module name, the number of trainable parameters and initialization time.
    """
    if rank == 0:
        logger.info(f"--> Module {module_name}")
        total_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        logger.info(f"--> {module_name} has {total_params / 1e6} Million params\n")


def save_train_params(train_config, fsdp_config, rank):
    """
    This function saves the train_config and FSDP config into a train_params.yaml.
    """
    train_config_dict = {
        k: str(v) for k, v in vars(train_config).items() if not k.startswith("__")
    }
    fsdp_config_dict = {
        k: str(v) for k, v in vars(fsdp_config).items() if not k.startswith("__")
    }
    train_params_dict = {**train_config_dict, **fsdp_config_dict}
    folder_name = (
        train_config.dist_checkpoint_root_folder
        + "/"
        + train_config.dist_checkpoint_folder
        + "-"
        + train_config.model_name
    )

    save_dir = Path.cwd() / folder_name
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    config_yaml = yaml.dump(train_params_dict, indent=4)
    file_name = os.path.join(save_dir, "train_params.yaml")

    if os.path.isdir(file_name):
        logger.info(f"Error: {file_name} is a directory, not a file.")
    else:
        with open(file_name, "w") as f:
            f.write(config_yaml)
        if rank == 0:
            logger.info(f"training params are saved in {file_name}")
