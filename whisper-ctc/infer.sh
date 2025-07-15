#!/bin/bash

# ========================
# Inference Shell Script
# ========================

# 用户可修改的参数
JSONL_PATH="/aistor/aispeech/hpc_stor01/group/asr/english/librispeech/asr/test-clean/multitask.jsonl"
CHECKPOINT_PATH="/aistor/aispeech/hpc_stor01/home/pengjing00sx/Github/Legoslm/whisper-ctc/ckpt/whisper_ctc_ddp_epoch0.pt"
BATCH_SIZE=4
DEVICE=0

# 运行
echo "Starting Inference..."
python infer.py \
    --jsonl "$JSONL_PATH" \
    --checkpoint "$CHECKPOINT_PATH" \
    --batch_size $BATCH_SIZE \
    --device $DEVICE 

echo "Inference Done!"
