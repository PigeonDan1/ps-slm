import sys
import os
import logging
import random
import torch
import torch.distributed as dist
import deepspeed
import torch_npu
import editdistance
from torch_npu.contrib import transfer_to_npu
from tqdm import tqdm

from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict

from dataclasses import dataclass, field
from omegaconf import DictConfig, OmegaConf
import hydra

from simulator.factory import create_training_model
from simulator.dataset import SimulatorDataset, SimulatorCollate
from utils.deepspeed_utils import clear_gpu_cache, setup_environ_flags

# --- 参数清洗逻辑 ---
cleaned_argv = []
i = 0
while i < len(sys.argv):
    arg = sys.argv[i]
    if arg.startswith("--local_rank"):
        if "=" in arg:
            if len(arg.split("=")) > 1:
                val = arg.split("=")[1]
                os.environ["LOCAL_RANK"] = val
            i += 1 
        else:
            if i + 1 < len(sys.argv):
                val = sys.argv[i+1]
                os.environ["LOCAL_RANK"] = val
                i += 2 
            else:
                i += 1
    else:
        cleaned_argv.append(arg)
        i += 1
sys.argv = cleaned_argv

torch_npu.npu.set_compile_mode(jit_compile=False)
torch_npu.npu.config.allow_internal_format = False

logger = logging.getLogger(__name__)


@dataclass
class RunConfig:
    model_config: dict = field(default_factory=dict)
    dataset_config: dict = field(default_factory=dict)
    train_config: dict = field(default_factory=dict)
    log_config: dict = field(default_factory=dict)
    deepspeed_config: str = ""
    # [新增] 预训练权重路径
    ckpt_path: str = "" 

@hydra.main(config_path="conf", config_name="simulator_config", version_base=None)
def main_hydra(cfg: DictConfig):
    run_config = RunConfig()
    cfg = OmegaConf.merge(run_config, cfg)
    main(cfg)

def main(kwargs: DictConfig):
    train_config = kwargs.train_config
    model_config = kwargs.model_config
    dataset_config = kwargs.dataset_config
    log_config = kwargs.log_config
    ds_config_path = kwargs.deepspeed_config
    ckpt_path = kwargs.ckpt_path # [新增]

    # --- 环境设置 ---
    setup_logging(log_config, train_config.output_dir)
    
    step_logger = None
    rank = int(os.environ.get("RANK", -1))
    if rank == 0:
        step_logger = logging.getLogger("step_logger")
        step_logger.propagate = False 
        step_logger.setLevel(logging.INFO)
        log_path = os.path.join(train_config.output_dir, "train.log")
        fh = logging.FileHandler(log_path, mode='a')
        fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
        step_logger.addHandler(fh)

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank != -1:
        torch.npu.set_device(local_rank)
        clear_gpu_cache(local_rank)
        setup_environ_flags(rank)
        if not dist.is_initialized():
            deepspeed.init_distributed(dist_backend="hccl")

    set_seed(train_config.seed)

    if rank == 0:
        logger.info(f"Training Config: {OmegaConf.to_yaml(train_config)}")
        
    # --- 数据集构建 ---
    logger.info(f"Loading datasets... (Rank {rank})")
    train_dataset = SimulatorDataset(dataset_config.train_path, max_len=model_config.max_len)
    
    dev_dataset = None
    if train_config.get("run_validation", False):
        dev_dataset = SimulatorDataset(dataset_config.dev_path, max_len=model_config.max_len)
    
    collate_fn = SimulatorCollate(max_len=model_config.max_len)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, shuffle=True, drop_last=True
    )
    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_config.batch_size_per_gpu,
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True
    )

    dev_dataloader = None
    if dev_dataset:
        dev_sampler = torch.utils.data.distributed.DistributedSampler(
            dev_dataset, shuffle=False, drop_last=False
        )
        dev_dataloader = torch.utils.data.DataLoader(
            dev_dataset,
            batch_size=train_config.batch_size_per_gpu, 
            sampler=dev_sampler,
            collate_fn=collate_fn,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True
        )

    logger.info(f"Creating model... (Rank {rank})")
    model_wrapper, tokenizer = create_training_model(model_config, dataset_config)
    
    # 加载ckpt
    if ckpt_path and os.path.exists(ckpt_path):
        if rank == 0:
            logger.info(f"--> Loading pretrained checkpoint from: {ckpt_path}")
        
        state_dict = torch.load(ckpt_path, map_location="cpu")
        
        # 处理可能的键名不匹配 (如果旧ckpt里有 module. 前缀)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
            elif k.startswith("simulator."): 
                # 旧 wrapper 可能直接存了 simulator.xxx，如果你现在结构没变，直接用
                new_state_dict[k] = v
            else:
                new_state_dict[k] = v
        
        # [核心] strict=False
        # 允许加载 Encoder/Decoder 权重，同时忽略新加的 condition_projector (随机初始化)
        missing_keys, unexpected_keys = model_wrapper.load_state_dict(new_state_dict, strict=False)
        
        if rank == 0:
            logger.info(f"--> Missing keys (Initialized randomly): {missing_keys}")
            logger.info(f"--> Unexpected keys (Ignored): {unexpected_keys}")
    else:
        if rank == 0 and ckpt_path:
            logger.warning(f"--> Checkpoint path provided but not found: {ckpt_path}")
    # ========================================================

    if rank == 0:
        total_params = sum(p.numel() for p in model_wrapper.parameters() if p.requires_grad)
        logger.info(f"--> Model Parameters: {total_params / 1e6:.2f} Million")


    logger.info(f"Initializing DeepSpeed... (Rank {rank})")
    parameters = filter(lambda p: p.requires_grad, model_wrapper.parameters())
    
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model_wrapper,
        model_parameters=parameters,
        config=ds_config_path
    )


    global_step = 0
    best_loss = float('inf')
    num_epochs = train_config.num_epochs

    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        model_engine.train()
        
        pbar = None
        if rank == 0:
            logger.info(f"=== Starting Epoch {epoch+1}/{num_epochs} ===")
            pbar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}", dynamic_ncols=True)


        for step, batch in enumerate(train_dataloader):
            batch = to_device(batch, model_engine.device)
            
            # 纯CE loss SFT
            outputs, acc = model_engine(
                text_onehot=batch["text_onehot"],
                text_mask=batch["text_mask"],
                target_psd=batch["target_psd"],
                target_lengths=batch["target_lengths"],
                bucket_id=batch["bucket_id"]
            )
            loss = outputs.loss

            model_engine.backward(loss)
            model_engine.step()
            
            global_step += 1

            if rank == 0:
                cur_lr = model_engine.get_lr()[0]
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}", 
                    acc=f"{acc.item():.4f}", 
                    lr=f"{cur_lr:.2e}")
                pbar.update(1)
                
                if global_step % log_config.log_interval == 0:
                    step_logger.info(
                        f"Epoch: {epoch+1}, Step: {step}, Global: {global_step}, "
                        f"Loss: {loss.item():.4f}, Acc: {acc.item():.4f}, LR: {cur_lr:.2e}" 
                    )

            if train_config.run_validation and global_step % train_config.validation_interval == 0:
                if rank == 0 and pbar: pbar.close()
                
                val_loss = evaluate(model_engine, dev_dataloader, rank, model_config)
                model_engine.train()
                
                if rank == 0:
                    remaining = len(train_dataloader) - (step + 1)
                    if remaining > 0:
                        pbar = tqdm(total=remaining, desc=f"Epoch {epoch+1} (Cont)", dynamic_ncols=True)

                if train_config.save_model:
                    save_path = os.path.join(train_config.output_dir, "checkpoints")
                    tag = f"step_{global_step}"
                    
                    if rank == 0:
                        logger.info(f"Saving DeepSpeed shards to {save_path} tag {tag}")
                    
                    model_engine.save_checkpoint(save_path, tag)
                    dist.barrier()

                    if rank == 0:
                        checkpoint_dir = os.path.join(save_path, tag)
                        output_file = os.path.join(checkpoint_dir, "pytorch_model.bin")
                        try:
                            convert_zero_checkpoint_to_fp32_state_dict(save_path, output_file)
                            logger.info("Conversion successful!")
                        except Exception as e:
                            logger.error(f"Failed to convert checkpoint: {e}")

                    if val_loss < best_loss:
                        best_loss = val_loss
                        if rank == 0:
                            logger.info(f"New best validation loss: {best_loss:.4f}")

        if rank == 0 and pbar:
            pbar.close()

    if rank == 0:
        logger.info("Training Finished.")

def evaluate(model_engine, dataloader, rank, model_config):
    if dataloader is None:
        return float('inf')
        
    model_engine.eval()
    
    # 局部累加器（初始化为 Python 标量）
    local_sum_ce = 0.0
    local_sum_acc = 0.0
    local_steps = 0
    
    pbar = None
    if rank == 0:
        pbar = tqdm(total=len(dataloader), desc="Validating (MBR Soft-Inference)", 
                    colour="green", dynamic_ncols=True, leave=False)

    with torch.no_grad():
        for batch in dataloader:
            batch = to_device(batch, model_engine.device)
            
            # 1. 计算基础 Teacher Forcing 指标
            outputs, acc = model_engine(
                text_onehot=batch["text_onehot"],
                text_mask=batch["text_mask"],
                target_psd=batch["target_psd"],
                target_lengths=batch["target_lengths"],
                bucket_id=batch["bucket_id"]
            )
            loss_ce = outputs.loss

            # 3. 本地累加指标
            local_sum_ce += loss_ce.item()
            local_sum_acc += acc.item() if isinstance(acc, torch.Tensor) else acc
            local_steps += 1
            
            if rank == 0:
                pbar.update(1)
    
    if rank == 0 and pbar:
        pbar.close()

    # --- 关键修改：分布式汇总 ---
    if dist.is_initialized():
        # 封装到张量，统一转移到 NPU，强制 float32 避免类型报错
        metrics = torch.tensor([
            local_sum_ce, local_sum_acc, float(local_steps)
        ], device=model_engine.device, dtype=torch.float32)
        
        dist.barrier() 
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
        
        # 解包全局汇总值
        sum_ce, sum_acc, global_steps = metrics.tolist()
    else:
        sum_ce, sum_acc, global_steps = local_sum_ce, local_sum_acc, float(local_steps)

    avg_loss = sum_ce / (global_steps + 1e-6)
    avg_acc = sum_acc / (global_steps + 1e-6)
    
    if rank == 0:
        logger.info(f"--- Validation Result ---")
        logger.info(f"CE Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f}")
        
    return avg_loss

def to_device(batch, device):
    new_batch = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            new_batch[k] = v.to(device)
        else:
            new_batch[k] = v
    return new_batch

def setup_logging(log_config, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(output_dir, "train.log"), mode='a')
        ]
    )

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch_npu.npu.manual_seed_all(seed)
    except:
        pass

if __name__ == "__main__":
    main_hydra()