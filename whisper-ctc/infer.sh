#!/bin/bash

# ========================
# Inference Shell Script
# ========================

# 用户可修改的参数
JSONL_PATH="/aistor/aispeech/hpc_stor01/group/asr/english/librispeech/asr/test-other/multitask.jsonl"
# "/aistor/aispeech/hpc_stor01/group/asr/mandarin/aishell-1/asr/test/multitask.jsonl"
# "/aistor/aispeech/hpc_stor01/group/asr/english/librispeech/asr/test-other/multitask.jsonl"
CHECKPOINT_PATH="/aistor/aispeech/hpc_stor01/home/pengjing00sx/Github/Legoslm/whisper-ctc/exp_librispeech_qwen_frozen/whisper_ctc_ddp_epoch2.pt"
BATCH_SIZE=24
DEVICE=0
MODE="beam"
# TOKENIZER="/aistor/aispeech/hpc_stor01/home/pengjing00sx/nfs/model/AI-ModelScope/gemma-2b"
# VOCAB="/aistor/aispeech/hpc_stor01/home/pengjing00sx/Github/Legoslm/Whisper_CTC_Greedy/dataset/vocabulary.txt"

# 运行
echo "Starting Inference..."
python infer.py \
    --jsonl "$JSONL_PATH" \
    --checkpoint "$CHECKPOINT_PATH" \
    --batch_size $BATCH_SIZE \
    --device $DEVICE \
    --mode $MODE 
    # --tokenizer_path $TOKENIZER
    # --vocab_file $VOCAB

echo "Inference Done!"
