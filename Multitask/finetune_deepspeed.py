from typing import Optional
import os
import logging
import random 
from dataclasses import dataclass, field
from omegaconf import DictConfig, OmegaConf
import wandb
import deepspeed
import torch
from functools import partial
from dataset.speech_dataset_large import MultiTaskDynamicBatchDataset,MultiTaskDataset,window_class
from aispeech_asr_config import ModelConfig, TrainConfig, DataConfig, LogConfig, FSDPConfig
from utils.deepspeed_utils import deepspeed_main_wrapper, clear_gpu_cache, setup_environ_flags, train
from utils.model_utils import get_custom_model_factory
from utils.dataset_utils import get_preprocessed_dataset
from utils.config_utils import get_dataloader_kwargs


@dataclass
class RunConfig:
    dataset_config: DataConfig = field(default_factory=DataConfig)
    model_config: ModelConfig = field(default_factory=ModelConfig)
    train_config: TrainConfig = field(default_factory=TrainConfig)
    log_config: LogConfig = field(default_factory=LogConfig)
    fsdp_config: FSDPConfig = field(default_factory=FSDPConfig)
    debug: bool = field(default=False, metadata={"help": "Use pdb when true"})
    metric: str = field(default="acc", metadata={"help": "The metric for evaluation"})
    ckpt_path: Optional[str] = field(
        default=None, metadata={"help": "The path to projector checkpoint"}
    )
    deepspeed_config : str =""
    deepspeed_ckpt_path: Optional[str] = field(
        default=None, metadata={"help": "The path to projector checkpoint"}
    )
    deepspeed_ckpt_id: Optional[str] = field(
        default=None, metadata={"help": "The id to projector checkpoint"}
    )


@deepspeed_main_wrapper(config_name=None, version_base=None)
def main_hydra(cfg: DictConfig):
    run_config = RunConfig()
    cfg = OmegaConf.merge(run_config, cfg)

    kwargs = cfg
    log_level = getattr(logging, kwargs.get("log_level", "INFO").upper())
    
    logging.basicConfig(level=log_level)
    
    if kwargs.get("debug", False):
        import pdb;
        pdb.set_trace()
        
    main(kwargs)


def main(kwargs: DictConfig):
    # Update the configuration for the training and sharding process
    # train_config, fsdp_config, model_config, log_config = TRAIN_CONFIG(), FSDP_CONFIG(), MODEL_CONFIG(), LOG_CONFIG()
    # update_config((train_config, fsdp_config, model_config, log_config), **kwargs)

    train_config, model_config, log_config, dataset_config, deepspeed_config = kwargs.train_config, \
                                                                          kwargs.model_config, \
                                                                          kwargs.log_config, \
                                                                          kwargs.dataset_config, \
                                                                          kwargs.deepspeed_config
    del kwargs.train_config
    del kwargs.model_config
    del kwargs.log_config
    del kwargs.dataset_config
    
    # Set log
    if not os.path.exists(os.path.dirname(log_config.log_file)):
        os.makedirs(os.path.dirname(log_config.log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filemode='w'
    )

    logger = logging.getLogger()  
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(filename=log_config.log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)

    logger.handlers[0].setLevel(logging.INFO)
    console_formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logger.handlers[0].setFormatter(console_formatter) 

    logger.addHandler(file_handler)


    # Set the seeds for reproducibility
    torch.manual_seed(train_config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_config.seed)
    random.seed(train_config.seed)

    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    logger.info(f"local_rank: {local_rank}, rank: {rank}, world_size: {world_size}")

    # --- NPU -> GPU ---
    torch.cuda.set_device(local_rank)
    clear_gpu_cache(local_rank)
    setup_environ_flags(rank)

    if rank == 0:
        logger.info("train_config: {}".format(train_config))
        logger.info("model_config: {}".format(model_config))
        logger.info("log_config: {}".format(log_config))

    # Set wandb
    if rank == 0:
        if log_config.use_wandb:
            if not os.path.exists(log_config.wandb_dir):
                os.makedirs(log_config.wandb_dir, exist_ok=True)
            wandb_config={"train_config": train_config, "model_config": model_config, "log_config": log_config}
            wandb.init(dir=log_config.wandb_dir, entity=log_config.wandb_entity_name, project=log_config.wandb_project_name,name=log_config.wandb_exp_name ,config=wandb_config)


    model_factory = get_custom_model_factory(model_config, logger) # pj: we can implace our own method here for creating other recipes
    model, tokenizer = model_factory(train_config, model_config, **kwargs)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # If you are facing problem from limited memory(<=256GB), you can try to replace the above code with the following code
    # for i in range(rank):
    #     while not os.path.isfile(f".{i}.done"):
    #         pass
    # assert not os.path.isfile(f".{rank}.done"), f".{rank}.done already exists!"
    # model_factory = get_custom_model_factory(model_config, logger)
    # model, tokenizer = model_factory(train_config, model_config, **kwargs)
    # parameters = filter(lambda p: p.requires_grad, model.parameters())
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model.half()
    # with open(f".{rank}.done", "w"):
    #     pass


    # Initialize the optimizer and learning rate scheduler
    model_engine, _, _, _ = deepspeed.initialize(
        model=model, model_parameters=parameters, config=deepspeed_config
    )

    
    # Convert the model to bfloat16 if fsdp and pure_bf16 is enabled
    # if (train_config.enable_fsdp or train_config.enable_ddp) and fsdp_config.pure_bf16:
    #     model.to(torch.bfloat16)

    #setting up FSDP if enable_fsdp is enabled
    # if train_config.enable_ddp:
    #     model = model.cuda(local_rank)
    #     model = DDP(model, device_ids=[local_rank],
    #                 find_unused_parameters=kwargs.get("train_conf", {}).get("find_unused_parameters", False))
    # elif not train_config.quantization:
    #     model.to(device)

    # dataset_config = generate_dataset_config(train_config, kwargs)
    logger.info("dataset_config: {}".format(dataset_config))
    if rank == 0:
        if log_config.use_wandb:
            wandb.config.update({"dataset_config": dataset_config})
    
    # # Load and preprocess the dataset for training and validation
    # dataset_train = get_preprocessed_dataset(
    #     tokenizer,
    #     dataset_config,
    #     split="train",
    # )
    # if (not (train_config.enable_fsdp or train_config.enable_ddp) or rank == 0) and train_config.batching_strategy != "dynamic":
    #     logger.info(f"--> Training Set Length = {len(dataset_train)}")
    # dataset_val = get_preprocessed_dataset(
    #     tokenizer,
    #     dataset_config,
    #     split="val",
    # )
    # if not (train_config.enable_fsdp or train_config.enable_ddp) or rank == 0 and train_config.batching_strategy != "dynamic":
    #     logger.info(f"--> Validation Set Length = {len(dataset_val)}")
    dataset = MultiTaskDataset(dataset_config, tokenizer, "train")
    dataset_train = MultiTaskDynamicBatchDataset(dataset,partial(window_class,max_frame_length = dataset_config.train_max_frame_length,ds_rate = dataset_config.ds_rate))
    dataset = MultiTaskDataset(dataset_config, tokenizer, "val")
    dataset_val = MultiTaskDynamicBatchDataset(dataset,partial(window_class,max_frame_length = dataset_config.eval_max_frame_length,ds_rate = dataset_config.ds_rate))
    train_dl_kwargs = get_dataloader_kwargs(train_config, dataset_train, tokenizer, "train")

    # Create DataLoaders for the training and validation dataset
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        **train_dl_kwargs,
    )

    eval_dataloader = None
    if train_config.run_validation:
        val_dl_kwargs = get_dataloader_kwargs(train_config, dataset_val, tokenizer, "val")

        eval_dataloader = torch.utils.data.DataLoader(
            dataset_val,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            **val_dl_kwargs,
        )


    # Start the training process
    results = train(
        model_engine,
        train_dataloader,
        eval_dataloader,
        tokenizer,
        train_config.gradient_accumulation_steps,
        train_config,
        log_config,
        local_rank,
        rank,
    )
    if rank==0:
        [logger.info(f'Key: {k}, Value: {v}') for k, v in results.items()]

    if rank == 0:
        if log_config.use_wandb:
            wandb.finish()

if __name__ == "__main__":
    main_hydra()
