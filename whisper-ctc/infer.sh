#!/bin/bash

# ========================
# Inference Shell Script
# ========================

# 用户可修改的参数
JSONL_PATH="/aistor/aispeech/hpc_stor01/group/asr/english/librispeech/asr/test-other/multitask.jsonl"
CHECKPOINT_PATH="/aistor/aispeech/hpc_stor01/home/pengjing00sx/Github/Legoslm/whisper-ctc/exp_librispeech_qwen/whisper_ctc_ddp_epoch2.pt"
BATCH_SIZE=16
DEVICE=0
MODE="beam"
# VOCAB="/aistor/aispeech/hpc_stor01/home/pengjing00sx/Github/Legoslm/Whisper_CTC_Greedy/dataset/vocabulary.txt"

# 运行
echo "Starting Inference..."
python infer.py \
    --jsonl "$JSONL_PATH" \
    --checkpoint "$CHECKPOINT_PATH" \
    --batch_size $BATCH_SIZE \
    --device $DEVICE \
    --mode $MODE 
    # --vocab_file $VOCAB

echo "Inference Done!"
