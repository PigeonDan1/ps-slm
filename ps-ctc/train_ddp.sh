# 杀干净残留进程
# pkill -9 python && sleep 10
# export MASTER_ADDR=127.0.0.1
export WORLD_SIZE=2
# 8 卡 NPU for librispeech, whisper is hot, tokenizer from qwen
CUDA_VISIBLE_DEVICES=5,6 torchrun --nproc_per_node=2 --master_port=29602 train_ddp.py \
    --jsonl ../data/librispeech_train.jsonl \
    --tokenizer_path  /hpc_stor03/sjtu_home/bohan.li/projects/ps-slm/pretrained_models/Qwen2-1.5B-Instruct \
    --batch_size 16 \
    --epochs 20 \
    --encoder_name sensevoice \
    --encoder_path /hpc_stor03/sjtu_home/bohan.li/projects/ps-slm/pretrained_models/SenseVoiceSmall \
    --lr 3e-4 \
    --save_path ./exp/exp_sensevoice_librispeech_qwen_frozen \
    --valid_jsonl /hpc_stor03/sjtu_home/bohan.li/projects/ps-slm/data/toy_valid.jsonl
    # --vocab_file /aistor/aispeech/hpc_stor01/home/pengjing00sx/Github/Legoslm/whisper-ctc/vocab/lang_bpe_500.model
