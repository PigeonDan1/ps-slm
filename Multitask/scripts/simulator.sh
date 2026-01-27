#!/bin/bash

# ================= 环境变量设置 =================
export PYTHONUNBUFFERED=1         
export TOKENIZERS_PARALLELISM=false
export HCCL_CONNECT_TIMEOUT=7200
export HYDRA_FULL_ERROR=1
export OMP_NUM_THREADS=1
export CPU_AFFINITY_CONF=1

export PATH=/usr/local/python3.10.15/bin:$PATH

# ================= 路径配置 =================
run_dir=$(cd $(dirname $0)/..; pwd)
cd $run_dir || exit 1
code_dir=.

DATA_ROOT="/aistor/aispeech/hpc_stor01/home/wangchenghao00sx/workingspace/TASU-simulator/Multitask/data"
train_data_path="${DATA_ROOT}/train/multitask_augmented.jsonl"
dev_data_path="${DATA_ROOT}/dev/multitask_augmented.jsonl"
tokenizer_path="/aistor/aispeech/hpc_stor01/home/wangchenghao00sx/.cache/modelscope/hub/models/iic/SenseVoiceSmall"

# 训练权重路径
ckpt_path=""
model_name="ctc_simulator_ar_control_augmentedData" 
num_epochs=50
batch_size_per_gpu=32 
val_batch_size=32
grad_accum=1
lr=1e-4
val_interval=1099

exp_tag="ar_control_augmentation" 
output_dir="${code_dir}/exp/simulator_${exp_tag}_$(date +"%Y%m%d-%H%M")"
mkdir -p $output_dir

deepspeed_config="${code_dir}/conf/simulator_config.json"

hydra_args="
hydra.run.dir=$output_dir \
++dataset_config.train_path=$train_data_path \
++dataset_config.dev_path=$dev_data_path \
++dataset_config.tokenizer_path=$tokenizer_path \
++model_config.max_len=160 \
++train_config.model_name=$model_name \
++train_config.num_epochs=$num_epochs \
++train_config.batch_size_per_gpu=$batch_size_per_gpu \
++train_config.val_batch_size=$val_batch_size \
++train_config.gradient_accumulation_steps=$grad_accum \
++train_config.lr=$lr \
++train_config.total_steps=16500 \
++train_config.warmup_steps=125 \
++train_config.output_dir=$output_dir \
++train_config.enable_deepspeed=true \
++train_config.use_fp16=true \
++train_config.num_workers_dataloader=4 \
++train_config.validation_interval=$val_interval \
++train_config.run_validation=true \
++deepspeed_config=$deepspeed_config \
++ckpt_path=$ckpt_path \
"


# 如果环境变量设置了 ASCEND_VISIBLE_DEVICES，则优先使用（但这里为了避开2号卡，建议不要设置环境变量或者确认环境变量里没有2）
# 为了保险，这里我们可以强制覆盖，或者只在未设置时使用
if [ -z "$ASCEND_VISIBLE_DEVICES" ]; then
    echo "--> No ASCEND_VISIBLE_DEVICES set, using hardcoded list: $TARGET_GPU_IDS"
else
    echo "--> Warning: ASCEND_VISIBLE_DEVICES is set to $ASCEND_VISIBLE_DEVICES."
    echo "--> Overriding with target list: $TARGET_GPU_IDS to avoid GPU 2."
fi

if [ -z "$VC_MASTER_HOSTS" ] && [ -z "$RANK" ]; then
    echo "--> Detected Local Environment."
    # 强制清理环境变量，防止 deepspeed 混淆
    unset ASCEND_VISIBLE_DEVICES #本地避免卡
    TARGET_GPU_IDS="2,3"
    NUM_GPUS=2
    RANDOM_PORT=$((29500 + $RANDOM % 100 + 50))
    
    echo "--> Using NPUs: $TARGET_GPU_IDS (Count: $NUM_GPUS)"
    
    # 生成临时的 hostfile
    HOST_FILE="my_local_hostfile_$$"
    echo "localhost slots=$NUM_GPUS" > $HOST_FILE
    
    # [核心修改] 使用 --include localhost:1,3,4,5,6,7
    # 这样 DeepSpeed 会准确地只拉起这几张卡
    eval deepspeed \
        --hostfile $HOST_FILE \
        --include localhost:$TARGET_GPU_IDS \
        --master_port $RANDOM_PORT \
        $code_dir/train_simulator.py ${hydra_args}
        
    rm $HOST_FILE
else
    # 集群环境逻辑 (通常不需要改，因为集群调度器分配好了)
    echo "--> Detected Cluster Environment."
    HOST_FILE="/tmp/${JobID:-hostfile}"
    SSH_PORT=6666
    if [ "${RANK}" = "0" ]; then
        /usr/sbin/sshd -p ${SSH_PORT}
        echo "${VC_MASTER_HOSTS} slots=${GPU_PER_TASK}" > "${HOST_FILE}"
        if [ -n "${VC_WORKER_HOSTS}" ]; then
            echo "${VC_WORKER_HOSTS}" | awk -F ',' -v gpu_num="$GPU_PER_TASK" '{for (i=1; i<=NF; i++) print $i" slots="gpu_num}' >> "${HOST_FILE}"
        fi
        eval deepspeed --hostfile "$HOST_FILE" --ssh_port "$SSH_PORT" --master_port "$MASTER_PORT" $code_dir/train_simulator.py ${hydra_args}
    else
        /usr/sbin/sshd -D -p ${SSH_PORT}
    fi
fi