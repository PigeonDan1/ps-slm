import os
import random

import logging

import hydra
import logging
from dataclasses import dataclass, field

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from tqdm import tqdm

from typing import Optional
from aispeech_asr_config import ModelConfig, TrainConfig, DataConfig, LogConfig, FSDPConfig
from utils.model_utils import get_custom_model_factory
from utils.dataset_utils import get_preprocessed_dataset


@dataclass
class RunConfig:
    dataset_config: DataConfig = field(default_factory=DataConfig)
    model_config: ModelConfig = field(default_factory=ModelConfig)
    train_config: TrainConfig = field(default_factory=TrainConfig)
    log_config: LogConfig = field(default_factory=LogConfig)
    fsdp_config: FSDPConfig = field(default_factory=FSDPConfig)
    debug: bool = field(default=False, metadata={"help": "Use pdb when true"})
    metric: str = field(default="acc", metadata={"help": "The metric for evaluation"})
    decode_log: str = field(
        default="output/decode_log",
        metadata={"help": "The prefix for the decode output"},
    )
    ckpt_path: str = field(
        default="output/model.pt", metadata={"help": "The path to projector checkpoint"}
    )
    peft_ckpt: Optional[str] = field(
        default=None,
        metadata={
            "help": "The path to peft checkpoint, should be a directory including adapter_config.json"
        },
    )


@hydra.main(config_name=None, version_base=None)
def main_hydra(cfg: DictConfig):
    run_config = RunConfig()
    cfg = OmegaConf.merge(run_config, cfg)
    # kwargs = to_plain_list(cfg)
    log_level = getattr(logging, cfg.get("log_level", "INFO").upper())

    logging.basicConfig(level=log_level)

    if cfg.get("debug", False):
        import pdb

        pdb.set_trace()

    main(cfg)


def main(kwargs: DictConfig):

	# Update the configuration for the training and sharding process
	# train_config, fsdp_config, model_config, log_config = TRAIN_CONFIG(), FSDP_CONFIG(), MODEL_CONFIG(), LOG_CONFIG()
	# update_config((train_config, fsdp_config, model_config, log_config), **kwargs)
	train_config, fsdp_config, model_config, log_config, dataset_config = kwargs.train_config, \
	                                                                      kwargs.fsdp_config, \
	                                                                      kwargs.model_config, \
	                                                                      kwargs.log_config, \
	                                                                      kwargs.dataset_config

	OmegaConf.set_struct(kwargs,False)
	del kwargs["train_config"]
	del kwargs["fsdp_config"]
	del kwargs["model_config"]
	del kwargs["log_config"]
	del kwargs["dataset_config"]
	OmegaConf.set_struct(kwargs,True)

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
    
	logger.info("train_config: {}".format(train_config))
	logger.info("fsdp_config: {}".format(fsdp_config))
	logger.info("model_config: {}".format(model_config))

	
	# Set the seeds for reproducibility
	torch.npu.manual_seed(train_config.seed)
	torch.manual_seed(train_config.seed)
	random.seed(train_config.seed)
	
	model_factory = get_custom_model_factory(model_config, logger)
	model, tokenizer = model_factory(train_config, model_config, **kwargs)
	device = torch.device(f"npu:{train_config.device}" if torch.npu.is_available() else "cpu") # FIX(MZY): put the whole model to device.
	model.to(device)
	model.eval()

	# dataset_config = generate_dataset_config(train_config, kwargs)
	logger.info("dataset_config: {}".format(dataset_config))
	dataset_test = get_preprocessed_dataset(
        tokenizer,
        dataset_config,
        split="test",
    )

	test_dataloader = torch.utils.data.DataLoader(
            dataset_test,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
			shuffle=False,
            batch_size=train_config.val_batch_size,
			drop_last=False,
			collate_fn=dataset_test.collator
        )
	

	logger.info("=====================================")
	pred_path = kwargs.get('decode_log') + "_pred"
	gt_path = kwargs.get('decode_log') + "_gt"
	with open(pred_path, "w") as pred, open(gt_path, "w") as gt:
		for step, batch in tqdm(enumerate(test_dataloader), total=len(test_dataloader) if train_config.batching_strategy != "dynamic" else ""):
			for key in batch.keys():
				batch[key] = batch[key].to(device) if isinstance(batch[key], torch.Tensor) else batch[key]
			model_outputs = model.generate(**batch)
			output_text = model.tokenizer.batch_decode(model_outputs, add_special_tokens=False, skip_special_tokens=True)
			print(output_text)
			# input()
			for key, text, target in zip(batch["keys"], output_text, batch["targets"]):
				pred.write(key + "\t" + text.replace("\n", " ") + "\n")
				gt.write(key + "\t" + target + "\n")


if __name__ == "__main__":
    main_hydra()
