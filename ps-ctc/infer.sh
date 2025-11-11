#!/bin/bash

# ========================
# Inference Shell Script
# ========================

# 用户可修改的参数
JSONL_PATH="/hpc_stor03/sjtu_home/bohan.li/projects/ps-slm/data/librispeech_test_clean.jsonl"
# "/aistor/aispeech/hpc_stor01/group/asr/mandarin/aishell-1/asr/test/multitask.jsonl"
# "/aistor/aispeech/hpc_stor01/group/asr/english/librispeech/asr/test-other/multitask.jsonl"
CHECKPOINT_PATH="/hpc_stor03/sjtu_home/bohan.li/projects/ps-slm/ps-ctc/exp/exp_sensevoice_librispeech_qwen_frozen/epoch_10.pt"
BATCH_SIZE=24
DEVICE=0
MODE="beam"
SPLIT="test-clean"
TOKENIZER="/hpc_stor03/sjtu_home/bohan.li/projects/ps-slm/pretrained_models/Qwen2-1.5B-Instruct"
# VOCAB="/aistor/aispeech/hpc_stor01/home/pengjing00sx/Github/Legoslm/Whisper_CTC_Greedy/dataset/vocabulary.txt"

# 运行
echo "Starting Inference..."
python infer.py \
    --jsonl "$JSONL_PATH" \
    --split $SPLIT \
    --checkpoint "$CHECKPOINT_PATH" \
    --tokenizer_path $TOKENIZER \
    --batch_size $BATCH_SIZE \
    --device $DEVICE \
    --mode $MODE \
    --encoder_name sensevoice \
    --encoder_path /hpc_stor03/sjtu_home/bohan.li/projects/ps-slm/pretrained_models/SenseVoiceSmall
    # --tokenizer_path $TOKENIZER
    # --vocab_file $VOCAB

echo "Inference Done!"
