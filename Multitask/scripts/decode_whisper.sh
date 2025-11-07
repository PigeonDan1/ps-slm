#!/bin/bash
run_dir=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/project/SLAM-LLM-ASR-Whisper
cd  $run_dir
code_dir=.

projector=linear
encoder_name=whisper
use_peft=true
use_fp16=false
eval_max_frame_length=4000
ckpt_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/project/SLAM-LLM-ASR-WhisperSLAM-LLM-ASR/exp/20250509-1623-aishell-1-loratrue_hotword_instruct/aispeech_asr_epoch_1_step_10
dataset=aishell-1
test_scp_file_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/project/SLAM-LLM-ASR-WhisperSLAM-LLM-ASR/multitask.jsonl


# Choose Encoder
encoder_name=conformer
speech_encoder_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/model/conformer
encoder_dim=768
encoder_projector_ds_rate=2

llm_name="Qwen2.5-7B-Instruct"
llm_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/model/Qwen2.5-7B-Instruct
llm_dim=3584 


decode_log=$ckpt_path/decode_${dataset}_${task}_${target}
python \
    $code_dir/inference_batch.py \
    hydra.run.dir=$ckpt_path \
    ++model_config.encoder_projector_ds_rate=$encoder_projector_ds_rate \
    ++model_config.llm_path=$llm_path \
    ++model_config.llm_dim=$llm_dim \
    ++model_config.encoder_name=$encoder_name \
    ++model_config.encoder_path=$speech_encoder_path \
    ++model_config.encoder_dim=$encoder_dim \
    ++model_config.encoder_projector=$projector \
    ++dataset_config.dataset=$dataset \
    ++dataset_config.test_scp_file_path=$test_scp_file_path \
    ++dataset_config.inference_mode=true \
    ++train_config.model_name=aispeech_asr \
    ++train_config.use_peft=$use_peft \
    ++train_config.batching_strategy=dynamic \
    ++train_config.num_epochs=1 \
    ++train_config.num_workers_dataloader=0 \
    ++train_config.output_dir=$output_dir \
    ++decode_log=$decode_log \
    ++ckpt_path=$ckpt_path/pytorch_model.bin

python utils/wenet_compute_cer.py --char=1 -v=1 ${decode_log}_gt ${decode_log}_pred > ${decode_log}_cer
python utils/wenet_compute_cer.py --char=1 -v=1 ${decode_log}_gt ${decode_log}_pred > ${decode_log}_cer