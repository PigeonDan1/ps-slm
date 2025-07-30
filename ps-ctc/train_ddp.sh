# 杀干净残留进程
pkill -9 python && sleep 10
export MASTER_ADDR=127.0.0.1
export WORLD_SIZE=8
# 8 卡 NPU for librispeech, whisper is hot, tokenizer from qwen
torchrun --nproc_per_node=8 --master_port=29602 train_ddp.py \
    --jsonl /aistor/aispeech/hpc_stor01/group/asr/english/librispeech/asr/train/multitask.jsonl \
    --tokenizer_path  /aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/model/Qwen2.5-1.5B-Instruct \
    --batch_size 16 \
    --epochs 10 \
    --encoder_name sensevoice \
    --encoder_path /aistor/aispeech/hpc_stor01/group/asr/model/SenseVoiceSmall \
    --lr 3e-4 \
    --save_path ./exp_sensevoice_librispeech_qwen_frozen \
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
