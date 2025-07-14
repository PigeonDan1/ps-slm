# 8 卡 NPU
torchrun --nproc_per_node=8 train_ddp.py \
    --jsonl /aistor/aispeech/hpc_stor01/group/asr/english/librispeech/export_llm/multitask.jsonl \
    --batch_size 2 \
    --epochs 3 \
    --lr 1e-5 \
    --save_path ./whisper_ctc_ddp.pt