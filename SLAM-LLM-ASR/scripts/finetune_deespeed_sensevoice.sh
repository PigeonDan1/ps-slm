#!/bin/bash
# export PYTHONPATH=/root/fairseq:$PYTHONPATH
# export ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false
export HCCL_CONNECT_TIMEOUT=7200
# export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=2
export ASCEND_LAUNCH_BLOCKING=0
export CPU_AFFINITY_CONF=2  # 细粒度绑核
run_dir=/aistor/aispeech/hpc_stor01/home/pengjing00sx/Github/ps-slm/SLAM-LLM-ASR
cd $run_dir
code_dir=.
dataset=librispeech
task=asr
train_scp_file_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/data/${dataset}/${task}/train/
dev_scp_file_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/data/${dataset}/${task}/dev/
train_max_frame_length=2000
eval_max_frame_length=3000
multitask_prompt_path=conf/multiprompt.jsonl
# prompt_style="\{\}\\<speech\\>" # "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n" | "USER: {}\n ASSISTANT:"
projector=linear

use_peft=true # For llm
use_fp16=true
freeze_encoder=true
do_psd=false # whether use psd to ds

# use absolute path
deepspeed_config=conf/ds_config.json

# Choose Encoder
encoder_name=sensevoice
speech_encoder_path=/aistor/aispeech/hpc_stor01/group/asr/model/SenseVoiceSmall
encoder_dim=512
encoder_projector_ds_rate=5 # downsampling rate
# Choose LLM
llm_name=Qwen2.5-1.5B-Instruct
llm_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/model/Qwen2.5-1.5B-Instruct
llm_dim=1536
model_factory=model/ps-slm.py:model_factory # create your own model_factory
output_dir=${code_dir}/exp/$(date +"%Y%m%d-%H%M")-$dataset-lora${use_peft}_${task}_instruct

hydra_args="
hydra.run.dir=$output_dir \
++model_config.file=$model_factory \
++model_config.llm_path=$llm_path \
++model_config.llm_dim=$llm_dim \
++model_config.encoder_name=$encoder_name \
++model_config.encoder_path=$speech_encoder_path \
++model_config.encoder_dim=$encoder_dim \
++model_config.encoder_projector=$projector \
++model_config.encoder_projector_ds_rate=$encoder_projector_ds_rate \
++dataset_config.encoder=$encoder_name \
++dataset_config.encoder_path=$speech_encoder_path \
++dataset_config.train_max_frame_length=$train_max_frame_length \
++dataset_config.eval_max_frame_length=$eval_max_frame_length \
++dataset_config.multitask_prompt_path=$multitask_prompt_path \
++dataset_config.train_scp_file_path=$train_scp_file_path \
++dataset_config.dev_scp_file_path=$dev_scp_file_path \
++train_config.model_name=ps-slm \
++train_config.num_epochs=5 \
++train_config.freeze_encoder=$freeze_encoder \
++train_config.do_psd=$do_psd \
++train_config.freeze_llm=true \
++train_config.use_peft=$use_peft \
++train_config.batching_strategy=dynamic \
++train_config.validation_interval=2000 \
++train_config.num_workers_dataloader=4 \
++train_config.output_dir=$output_dir \
++metric=acc \
"

# deepspeed \
#     --num_nodes 1 \
#     --num_gpus 2 \
#     $code_dir/finetune_deepspeed.py \
#     ++train_config.enable_fsdp=false \
#     ++train_config.enable_ddp=true \
#     ++train_config.use_fp16=$use_fp16 \
#     ++deepspeed_config=$deepspeed_config \
#     ${hydra_args}


HOST_FILE="/tmp/"${JobID}                        #生成的hostfile的完整文件名，$JobID调度系统会自动生成
 
echo "${VC_MASTER_HOSTS} slots=${GPU_PER_TASK}" > ${HOST_FILE}
echo "${VC_WORKER_HOSTS}" | awk -F ',' -v gpu_num=$GPU_PER_TASK '{for (i=1; i<=NF; i++) print $i" slots="gpu_num}' >> ${HOST_FILE}

deepspeed \
    --node_rank=$RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --hostfile $HOST_FILE \
    --no_ssh \
    $code_dir/finetune_deepspeed.py \
    ++train_config.enable_fsdp=false \
    ++train_config.enable_ddp=true \
    ++train_config.use_fp16=$use_fp16 \
    ++deepspeed_config=$deepspeed_config \
    ${hydra_args}
