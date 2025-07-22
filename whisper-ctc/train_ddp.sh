# 8 卡 NPU for librispeech, whisper is hot, tokenizer from qwen
torchrun --nproc_per_node=8 --master_port=29601 train_ddp.py \
    --jsonl /aistor/aispeech/hpc_stor01/group/asr/english/librispeech/asr/train/multitask.jsonl \
    --batch_size 24 \
    --epochs 10 \
    --lr 3e-4 \
    --save_path ./exp_librispeech_qwen_frozen/whisper_ctc_ddp.pt \
    --valid_jsonl /aistor/aispeech/hpc_stor01/group/asr/english/librispeech/asr/dev/multitask.jsonl 
    # --vocab_file /aistor/aispeech/hpc_stor01/home/pengjing00sx/Github/Legoslm/whisper-ctc/vocab/lang_bpe_500.model
# 8 卡 NPU for aishell1, whisper is frozen, voca is ours
# torchrun --nproc_per_node=8 --master_port=29601 train_ddp.py \
#     --jsonl /aistor/aispeech/hpc_stor01/group/asr/mandarin/aishell-1/asr/train/multitask.jsonl \
#     --batch_size 24 \
#     --epochs 10 \
#     --lr 3e-4 \
#     --save_path ./exp_aishell1_qwen/whisper_ctc_ddp.pt \
#     --valid_jsonl /aistor/aispeech/hpc_stor01/group/asr/mandarin/aishell-1/asr/dev/multitask.jsonl 
    # --vocab_file /aistor/aispeech/hpc_stor01/home/pengjing00sx/Github/Legoslm/whisper-ctc/vocab/vocabulary.txt
