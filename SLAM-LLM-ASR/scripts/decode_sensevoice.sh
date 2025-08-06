#!/bin/bash
run_dir=/aistor/aispeech/hpc_stor01/home/pengjing00sx/Github/ps-slm/SLAM-LLM-ASR
cd  $run_dir
code_dir=.

projector=linear #simple_linear
# ctc_linear=/aistor/aispeech/hpc_stor01/home/pengjing00sx/Github/ps-slm/ps-ctc/exp_sensevoice_librispeech_qwen_frozen/epoch_5.pt # need to load pretrained ctc head if ctc head is frozen

use_peft=true
use_fp16=false
gt_emb=false # whether use gt's emb as input, actually here refers to gt one-hot
eval_max_frame_length=3000
ckpt_path=/aistor/aispeech/hpc_stor01/home/pengjing00sx/Github/ps-slm/SLAM-LLM-ASR/exp/20250725-2053-librispeech-loratrue_asr_instruct_do_psd_false_ds_5/ps-slm_epoch_5_step_1800
dataset=librispeech
task=asr
split=test-other

if [ "$dataset" = "librispeech" ]; then
    test_scp_file_path="/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/data/${dataset}/${task}/${split}/"
elif [ "$dataset" = "tts_en_rare_words" ]; then
    test_scp_file_path="/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/data/test/${dataset}"
fi

# test_scp_file_path=/aistor/aispeech/hpc_stor01/home/pengjing00sx/nfs/data/test/librispeech_st/
# Choose Encoder
encoder_name=sensevoice
speech_encoder_path=/aistor/aispeech/hpc_stor01/group/asr/model/SenseVoiceSmall
encoder_dim=512 #25055 #512
encoder_projector_ds_rate=5

do_psd=false # whether use psd to ds
ctc_posterior=false # whether use ctc posterior
voca_trans=false # whether use vocabulary transfer
top1_emb=false
llm_name="Qwen2.5-1.5B-Instruct"
llm_path=/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/model/Qwen2.5-1.5B-Instruct
llm_dim=1536 #151936 #1536

model_factory=model/ps-slm.py:model_factory # create your own model_factory
run_decode_device=0 # run decode on certain device
decode_log=$ckpt_path/decode_${dataset}_${task}_${split}
python \
    $code_dir/inference_batch.py \
    hydra.run.dir=$ckpt_path \
    ++model_config.file=$model_factory \
    ++model_config.encoder_projector_ds_rate=$encoder_projector_ds_rate \
    ++model_config.llm_path=$llm_path \
    ++model_config.llm_dim=$llm_dim \
    ++model_config.encoder_name=$encoder_name \
    ++model_config.encoder_path=$speech_encoder_path \
    ++model_config.encoder_dim=$encoder_dim \
    ++model_config.encoder_projector=$projector \
    ++model_config.ctc_linear=$ctc_linear \
    ++dataset_config.dataset=$dataset \
    ++dataset_config.encoder=$encoder_name \
    ++dataset_config.encoder_path=$speech_encoder_path \
    ++dataset_config.test_scp_file_path=$test_scp_file_path \
    ++dataset_config.inference_mode=true \
    ++train_config.model_name=ps-slm \
    ++train_config.device=$run_decode_device \
    ++train_config.use_peft=$use_peft \
    ++train_config.batching_strategy=dynamic \
    ++train_config.gt_emb=$gt_emb \
    ++train_config.top1_emb=$top1_emb \
    ++train_config.num_epochs=1 \
    ++train_config.do_psd=$do_psd \
    ++train_config.ctc_posterior=$ctc_posterior \
    ++train_config.voca_trans=$voca_trans \
    ++train_config.num_workers_dataloader=0 \
    ++train_config.output_dir=$output_dir \
    ++decode_log=$decode_log \
    ++ckpt_path=$ckpt_path/pytorch_model.bin

python utils/wenet_compute_cer.py --char=1 -v=1 ${decode_log}_gt ${decode_log}_pred > ${decode_log}_wer
python utils/wenet_compute_cer.py --char=1 -v=1 ${decode_log}_gt ${decode_log}_pred > ${decode_log}_wer