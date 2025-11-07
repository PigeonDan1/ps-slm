from dataclasses import dataclass, field
from typing import Optional, List
from torch.distributed.fsdp import ShardingStrategy


@dataclass
class ConformerConfig:
    input_size : int = 80
    activation_type: str = "swish"
    attention_dropout_rate: float = 0.0
    attention_heads: int = 12
    cnn_module_kernel: int = 33
    cnn_module_norm: str = "layer_norm"
    dropout_rate: float = 0.1
    input_layer: str = "conv2d"
    linear_units: int = 3584
    normalize_before: bool = True
    num_blocks: int = 16
    output_size: int = 768
    pos_enc_layer_type: str = "rel_pos"
    positional_dropout_rate: float = 0.1
    selfattention_layer_type: str = "rel_selfattn"
    use_cnn_module: bool = True   


@dataclass
class ModelConfig:
    file: str = "model/aispeech_asr.py:model_factory" # pj: you can also use your own model_factory here
    llm_name: str = "Qwen2.5-7B-Instruct"
    llm_path: str = "PATH/to/LLAMA/7B"
    llm_type: str = "decoder_only"
    llm_dim: int = 4096
    encoder_name: str = 'whisper'
    encoder_config: ConformerConfig = field(default_factory=ConformerConfig)  # or None, use wenet_model_path instead
    encoder_path: Optional[str] = "/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/model/whisper-large-v3"
    encoder_dim: int = 768
    encoder_projector: str = "linear"
    encoder_projector_ds_rate: int = 2
    ctc_linear: Optional[str] = None

@dataclass
class PeftConfig:
    peft_method: str = "lora" # None , llama_adapter, prefix
    r: int = 64
    lora_alpha: int = 16
    target_modules: List = field(default_factory=lambda: [ "q_proj","k_proj", "v_proj", "o_proj", "up_proj","gate_proj","down_proj"])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    lora_dropout: float = 0.05
    inference_mode: bool = False


@dataclass
class FbankConfig:
    num_mel_bins: int = 80  # 梅尔频率滤波器组的滤波器数量为80
    frame_length: int = 25  # 音频帧的长度为25毫秒
    frame_shift: int = 10  # 帧移为10毫秒
    dither: float = 0.001  # 抖动系数为0.001
    window_type: str = "hamming"  # 使用hamming窗口类型
    use_energy: bool = False  # 不使用能量特征
    low_freq:int = 0  # 低频截止频率为0Hz
    high_freq: int = 8000  # 高频截止频率为8000Hz
    htk_compat: bool = True  # 尝试使其与HTK兼容


@dataclass
class TrainConfig:
    model_name:str = "asr_model"
    enable_ddp:bool = False
    enable_deepspeed:bool = False
    enable_fsdp:bool = False
    low_cpu_fsdp:bool = False
    run_validation:bool = True
    batch_size_training: Optional[int] = None
    batching_strategy:str = field(default="packing", metadata={
        "help":"alternative: padding"
    }) #
    context_length:int = 4096
    gradient_accumulation_steps:int = 1
    num_epochs:int = 3
    num_workers_dataloader:int = 1
    warmup_steps:int = 1000
    total_steps:int = 100000
    validation_interval:int = 1000
    lr:float = 1e-4
    weight_decay:float = 0.0
    gamma:float = 0.85
    seed:int = 42
    use_fp16:bool = False
    mixed_precision:bool = True
    val_batch_size:Optional[int] = None
    do_psd:bool = False
    ctc_posterior:Optional[bool] = False
    voca_trans:Optional[bool] = False
    use_peft:bool = False
    use_emb:bool = False
    gt_emb:bool = False
    gt_emb_noise:bool = False
    top1_emb:bool = False
    cross_attn:bool = False
    peft_config:PeftConfig = field(default_factory=PeftConfig)
    output_dir:str = "PATH/to/save/PEFT/model"
    freeze_layers:bool = False
    freeze_projector:bool = False
    num_freeze_layers:int = 1
    quantization:bool = False
    one_gpu:bool = False
    save_model:bool = True
    dist_checkpoint_root_folder:str = "PATH/to/save/FSDP/model" # will be used if using FSDP
    dist_checkpoint_folder:str = "fine-tuned" # will be used if using FSDP
    save_optimizer:bool = False # will be used if using FSDP
    use_fast_kernels:bool = False # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    run_test_during_validation:bool = False
    run_test_during_validation_file:str = "test.wav"
    run_test_during_validation_prompt:str = "<|ASR|>"
    freeze_llm:bool = field(default=False, metadata={
        "help": "whether to freeze llm when finetuning, should be true when use peft finetuning"
    })
    freeze_encoder:bool = False
    device: Optional[int] = 0 
    gaussian_sim: bool = False


@dataclass
class DataConfig:
    file: Optional[str] = "dataset/speech_dataset_large.py:get_speech_dataset"
    dataset: str = "multitask_dataset"
    encoder: str = "whisper"
    encoder_path: Optional[str] = None
    max_audio_length : int = 30
    train_max_frame_length: int = 1500
    ds_rate: int = 8
    eval_max_frame_length: int = 2000
    multitask_prompt_path: str = "conf/multiprompt.jsonl"
    prompt_style: str = "<|im_start|>user\n{}<speech><|im_end|>\n<|im_start|>assistant\n"
    append_info_tasks : List = field(default_factory=lambda: ["hotword"])
    # file: str = "dataset/speech_dataset_large.py:get_speech_dataset"
    train_scp_file_path: str = ""
    dev_scp_file_path: str = ""
    test_scp_file_path: str = ""
    train_split: str = "train"
    dev_split: str = "dev"
    test_split:str = "test"
    pad_or_trim: bool = True
    prompt: Optional[str] = None
    inference_mode: bool = False
    lower: bool = False
    fix_length_audio: int = -1
    fbankConfig: FbankConfig = field(default_factory=FbankConfig)
    inference_mode:bool = False
    input_type: str = field(default="raw", metadata={
                                "help":"Use raw when input is wav, mel when for whisper"
                            })
    mel_size: int = field(default=80, metadata={
        "help": "80 for whisper large v1 and v2, 128 for v3"
    })
    normalize: Optional[bool] = field(default=False, metadata={
        "help": "whether input is normalized, used for models such as wavlm"
    })


@dataclass
class FSDPConfig:
    mixed_precision: bool = True
    use_fp16: bool = False
    # sharding_strategy = "FULL_SHARD" #ShardingStrategy = ShardingStrategy.FULL_SHARD
    sharding_strategy: ShardingStrategy = "SHARD_GRAD_OP" #ShardingStrategy.NO_SHARD #MZY: set NO_SHARD when use DDP
    checkpoint_type: str = "SHARDED_STATE_DICT"  # alternatively can use SHARDED_STATE_DICT save one file per rank, and can resize the world-size.
    fsdp_activation_checkpointing: bool = True
    fsdp_cpu_offload: bool = False
    pure_bf16: bool = False
    optimizer: str = "AdamW"


@dataclass
class LogConfig:
    use_wandb: bool = False
    wandb_dir: str = "tmp/test_wandb"
    wandb_entity_name: str = "project_name"
    wandb_project_name: str = "project_name"
    wandb_exp_name: str = "exp_name"
    log_file: str = "tmp/test.log"
    log_interval: int = 5
