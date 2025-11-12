#!/bin/bash
# export PYTHONPATH=/root/fairseq:$PYTHONPATH
export TOKENIZERS_PARALLELISM=false
export HCCL_CONNECT_TIMEOUT=7200
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=2 
export MASTER_ADDR=127.0.0.1

run_dir=/hpc_stor03/sjtu_home/jing.peng/workspace/ps-slm/Multitask # change this to your local dir
cd $run_dir
code_dir=.
dataset=librispeech
task=asr

train_scp_file_path=/hpc_stor03/sjtu_home/bohan.li/projects/ps-slm/data/librispeech_asr/train # set your train scp path
dev_scp_file_path=/hpc_stor03/sjtu_home/bohan.li/projects/ps-slm/data/librispeech_asr/dev # set your dev scp path

train_max_frame_length=3000
eval_max_frame_length=3000
multitask_prompt_path=conf/multiprompt.jsonl

projector=linear-silu # simple linear for ctc head, linear is normal type, cross-attention 
# ctc_linear=/aistor/aispeech/hpc_stor01/home/pengjing00sx/Github/ps-slm/ps-ctc/exp_sensevoice_librispeech_qwen_frozen/epoch_5.pt

use_peft=false # For llm
use_emb=false # For llm input_embs
gt_emb=true # whether use gt's emb as input
gt_emb_noise=true # whether use noise(CPS)
top1_emb=false # whether use top1's emb as input
do_psd=true # whether use psd to ds
ctc_posterior=true # whether use ctc posterior
voca_trans=false # whether use vocabulary transfer

use_fp16=false
deepspeed_config=conf/ds_config.json

# Choose Encoder
encoder_name=sensevoice
speech_encoder_path=/hpc_stor03/sjtu_home/bohan.li/projects/ps-slm/pretrained_models/SenseVoiceSmall
encoder_dim=25055 #25055 for sensevoice vocabulary size #512 for sensevoice raw dimension
freeze_encoder=true
encoder_projector_ds_rate=1 # downsampling rate
freeze_projector=false

# Choose LLM
llm_name=Qwen2.5-1.5B-Instruct
llm_path=/hpc_stor03/sjtu_home/jing.peng/nfs/model/qwen/Qwen/Qwen2___5-1___5B-Instruct
llm_dim=1536 #151936 for llm voc # 1536 for raw dim

model_factory=model/ps-slm.py:model_factory # you can also create your own model_factory

# prompt_style='<|im_start|>user\\n<speech>{}<|im_end|>\\n<|im_start|>assistant\\n' # audio first
output_dir=${code_dir}/exp/$(date +"%Y%m%d-%H%M")-$dataset-lora${use_peft}_${task}_instruct_do_psd_${do_psd}_ds_${encoder_projector_ds_rate}_ctc_posterior_${ctc_posterior}_voca_trans_${voca_trans}

hydra_args="
hydra.run.dir=$output_dir \
++model_config.file=$model_factory \
++model_config.llm_path=$llm_path \
++model_config.llm_dim=$llm_dim \
++model_config.llm_name=$llm_name \
++model_config.encoder_name=$encoder_name \
++model_config.encoder_path=$speech_encoder_path \
++model_config.encoder_dim=$encoder_dim \
++model_config.encoder_projector=$projector \
++model_config.encoder_projector_ds_rate=$encoder_projector_ds_rate \
++model_config.ctc_linear=$ctc_linear \
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
++train_config.freeze_projector=$freeze_projector \
++train_config.do_psd=$do_psd \
++train_config.ctc_posterior=$ctc_posterior \
++train_config.voca_trans=$voca_trans \
++train_config.freeze_llm=true \
++train_config.use_peft=$use_peft \
++train_config.use_emb=$use_emb \
++train_config.gt_emb=$gt_emb \
++train_config.gt_emb_noise=$gt_emb_noise \
++train_config.top1_emb=$top1_emb \
++train_config.batching_strategy=dynamic \
++train_config.validation_interval=10000 \
++train_config.num_workers_dataloader=4 \
++train_config.output_dir=$output_dir \
++metric=acc \
"

# if you want to run locally for test, uncomment the following lines:
# deepspeed \
#     --num_nodes 1 \
#     --num_gpus 2 \
#     $code_dir/finetune_deepspeed.py \
#     ++train_config.enable_fsdp=false \
#     ++train_config.enable_ddp=true \
#     ++train_config.use_fp16=$use_fp16 \
#     ++deepspeed_config=$deepspeed_config \
#     ${hydra_args}

# if you want to run on cluster, use the following lines:
HOST_FILE="/tmp/"${JobID}                        
 
echo "${VC_MASTER_HOSTS} slots=${GPU_PER_TASK}" > ${HOST_FILE}
echo "${VC_WORKER_HOSTS}" | awk -F ',' -v gpu_num=$GPU_PER_TASK '{for (i=1; i<=NF; i++) print $i" slots="gpu_num}' >> ${HOST_FILE}


deepspeed \
    $code_dir/finetune_deepspeed.py \
    ++train_config.enable_fsdp=false \
    ++train_config.enable_ddp=true \
    ++train_config.use_fp16=$use_fp16 \
    ++deepspeed_config=$deepspeed_config \
    ${hydra_args}
