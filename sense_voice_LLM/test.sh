#!/bin/bash

# 设置 VC 提交参数
PROJECT="pdgpu-sjtu-ai"
IMAGE="hub.szaic.com/hpc/ai_asr-jingpeng-ps-slm:v2.0"
CPU=80
MEMORY="480G"
NODES=1
GPUS=8
JOB_NUM=$NODES # 👈 你要运行的 Job 数，例如 1~10
SCRIPT_PATH="/aistor/sjtu/hpc_stor01/home/yangyi/Legoslm/sense_voice_LLM/scripts/finetune_deespeed_sensevoice.sh"
LOG_PATH="/aistor/sjtu/hpc_stor01/home/yangyi/Legoslm/sense_voice_LLM/logs"

# 创建日志目录（如不存在）
mkdir -p "$LOG_PATH"

# 执行 VC 提交（添加 JOB 数组部分）
vc submit -p "$PROJECT"  -i "$IMAGE" -c "$CPU" -m "$MEMORY" -n "$NODES"  -g "$GPUS" "JOB=1:$JOB_NUM" "$LOG_PATH/output_sleeper_JOB.log" --cmd "/usr/sbin/sshd -D -p 6666"

# 提示提交完成
echo "VC 任务数组（1~$JOB_NUM）已提交，日志保存在: $LOG_PATH/output_sleeper.log"
