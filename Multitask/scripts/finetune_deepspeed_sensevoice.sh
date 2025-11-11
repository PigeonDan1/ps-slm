#!/bin/bash
# export PYTHONPATH=/root/fairseq:$PYTHONPATH
# export ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=false
export HCCL_CONNECT_TIMEOUT=7200
# export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=2
export ASCEND_LAUNCH_BLOCKING=1
export CPU_AFFINITY_CONF=2  
run_dir=/aistor/sjtu/hpc_stor01/home/pengjing/workingspace/ps-slm/Multitask # change this to your local dir
cd $run_dir
code_dir=.
dataset=multitask_large
task=asr-st-slu
if [ "$dataset" = "asr" ] || [ "$dataset" = "multitask_large" ]; then
    train_scp_file_path=/aistor/sjtu/hpc_stor01/home/yangyi/data/${dataset}/train
    dev_scp_file_path=/aistor/sjtu/hpc_stor01/home/yangyi/data/${dataset}/dev
else
    train_scp_file_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/data/${dataset}/${task}/train/
    dev_scp_file_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/data/${dataset}/${task}/dev/
fi

train_max_frame_length=1500 # you can change this accroding to your process memory
eval_max_frame_length=2500
multitask_prompt_path=conf/multiprompt.jsonl

projector=linear-silu # simple linear for ctc head, linear is normal type, cross-attention 
# ctc_linear=/aistor/aispeech/hpc_stor01/home/pengjing00sx/Github/ps-slm/ps-ctc/exp_sensevoice_librispeech_qwen_frozen/epoch_5.pt

use_peft=false # For llm
use_emb=false # For llm input_embs
gt_emb=true # whether use gt's emb as input
gt_emb_noise=false # whether use noise
gaussian_sim=true # whether use gaussian sim as input
top1_emb=false # whether use top1's emb as input
use_fp16=true
freeze_encoder=true
freeze_projector=false
do_psd=true # whether use psd to ds
ctc_posterior=true # whether use ctc posterior
voca_trans=false # whether use vocabulary transfer
# use absolute path
deepspeed_config=conf/ds_config.json

# Choose Encoder
encoder_name=sensevoice
speech_encoder_path=/aistor/sjtu/hpc_stor01/home/yangyi/model/SenseVoiceSmall
encoder_dim=25055 #25055 #512
encoder_projector_ds_rate=1 # downsampling rate

# Choose LLM
llm_name=Qwen2.5-1.5B-Instruct
llm_path=/aistor/sjtu/hpc_stor01/home/yangyi/model/Qwen2.5-1.5B-Instruct
llm_dim=1536 #151936 # 1536 3584
model_factory=model/ps-slm.py:model_factory # create your own model_factory

# prompt_style='<|im_start|>user\\n<speech>{}<|im_end|>\\n<|im_start|>assistant\\n' # audio first
prompt_fig=instruction_first

output_dir=${code_dir}/exp/$(date +"%Y%m%d-%H%M")-$dataset-lora${use_peft}_${task}_instruct_do_psd_${do_psd}_ds_${encoder_projector_ds_rate}_ctc_posterior_${ctc_posterior}_voca_trans_${voca_trans}_${prompt_fig}

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
++train_config.validation_interval=1000 \
++train_config.num_workers_dataloader=4 \
++train_config.output_dir=$output_dir \
++metric=acc \
"

# if you want to run on local, then:
deepspeed \
    --num_nodes 1 \
    --num_gpus 8 \
    $code_dir/finetune_deepspeed.py \
    ++train_config.enable_fsdp=false \
    ++train_config.enable_ddp=true \
    ++train_config.use_fp16=$use_fp16 \
    ++deepspeed_config=$deepspeed_config \
    ${hydra_args}

# if you run on vc slurms, then:
# HOST_FILE="/tmp/"${JobID}                        
 
# echo "${VC_MASTER_HOSTS} slots=${GPU_PER_TASK}" > ${HOST_FILE}
# echo "${VC_WORKER_HOSTS}" | awk -F ',' -v gpu_num=$GPU_PER_TASK '{for (i=1; i<=NF; i++) print $i" slots="gpu_num}' >> ${HOST_FILE}

# deepspeed \
#     --node_rank=$RANK \
#     --master_addr $MASTER_ADDR \
#     --master_port $MASTER_PORT \
#     --hostfile $HOST_FILE \
#     --no_ssh \
#     $code_dir/finetune_deepspeed.py \
#     ++train_config.enable_fsdp=false \
#     ++train_config.enable_ddp=true \
#     ++train_config.use_fp16=$use_fp16 \
#     ++deepspeed_config=$deepspeed_config \
#     ${hydra_args}
