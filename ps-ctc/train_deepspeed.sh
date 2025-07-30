#!/bin/bash

# 自动选择可用端口（29500-29999）
get_free_port() {
  while :
  do
    PORT=$(shuf -i 29500-29999 -n 1)
    ss -lpn | grep ":$PORT " > /dev/null
    if [ $? -ne 0 ]; then
      echo $PORT
      return
    fi
  done
}

PORT=$(get_free_port)
echo "[INFO] Using MASTER_PORT=$PORT"

# ✅ 传参进去 --master_port
deepspeed --num_gpus 8 \
  --master_port $PORT \
  train_deepspeed.py \
  --jsonl /aistor/aispeech/hpc_stor01/group/asr/english/librispeech/asr/train/multitask.jsonl \
  --valid_jsonl /aistor/aispeech/hpc_stor01/group/asr/english/librispeech/asr/dev/multitask.jsonl \
  --epochs 10 \
  --batch_size 4 \
  --lr 2e-5 \
  --save_path /aistor/aispeech/hpc_stor01/home/pengjing00sx/Github/Legoslm/whisper-ctc/exp/whisper_ctc.pt \
  --deepspeed /aistor/aispeech/hpc_stor01/home/pengjing00sx/Github/Legoslm/whisper-ctc/config/ds_config.json
