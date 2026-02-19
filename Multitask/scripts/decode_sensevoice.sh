#!/bin/bash
run_dir=/aistor/sjtu/hpc_stor01/home/wangchenghao/workingspace/TASU-simulator/Multitask # change this to your local dir
cd  $run_dir
code_dir=.

projector=linear-silu #simple_linear
# ctc_linear=/aistor/aispeech/hpc_stor01/home/pengjing00sx/Github/ps-slm/ps-ctc/exp_sensevoice_librispeech_qwen_frozen/epoch_5.pt # need to load pretrained ctc head if ctc head is frozen

use_peft=false
use_fp16=false
gt_emb=false # whether use gt's emb as input, actually here refers to gt one-hot
eval_max_frame_length=1500
ckpt_path=/aistor/sjtu/hpc_stor01/home/wangchenghao/workingspace/TASU-simulator/Multitask/exp/company/exp1_simulator/company_exp1_sim_phase1

task=asr
split=test-other
dataset=medical

# TBD: u should change paths to your own paths
if [ "$task" = "asr" ]; then
    if [ "$dataset" = "librispeech" ]; then
        test_scp_file_path="/aistor/sjtu/hpc_stor01/home/wangchenghao/workingspace/ps-slm/Multitask/data/${split}"
    elif [ "$dataset" = "commonvoice" ]; then
        test_scp_file_path="/aistor/sjtu/hpc_stor01/home/yangyi/data/common_voice/${split}/"
    elif [ "$dataset" = "slidespeech" ]; then
        test_scp_file_path="/aistor/aispeech/hpc_stor01/home/wangchenghao00sx/workingspace/dataset_reproduce/slidespeech_test/"
    elif [ "$dataset" = "tts_en_rare_words" ]; then
        test_scp_file_path="/aistor/aispeech/hpc_stor01/home/fangyangui/workingspace/data/test/${dataset}"
    elif [ "$dataset" = "MLS_en" ]; then
        test_scp_file_path="/aistor/sjtu/hpc_stor01/home/yangyi/data/MultiLingualSpeechRecognition_MLS-en/"
    elif [ "$dataset" = "TED" ]; then
        test_scp_file_path="/aistor/sjtu/hpc_stor01/home/yangyi/data/tedlium3/"
    elif [ "$dataset" = "medical" ]; then
        test_scp_file_path="/aistor/sjtu/hpc_stor01/home/wangchenghao/workingspace/TASU-simulator/Multitask/medical/test"
    elif [ "$dataset" = "gigaspeech" ]; then
        test_scp_file_path="/aistor/aispeech/hpc_stor01/home/wangchenghao00sx/workingspace/dataset_reproduce/gigaspeech_test/"
    fi
elif [ "$task" = "st" ]; then
    test_scp_file_path="/aistor/sjtu/hpc_stor01/home/yangyi/data/multitask_small/${split}"
elif [ "$task" = "qa" ]; then
    test_scp_file_path="/aistor/sjtu/hpc_stor01/home/yangyi/data/${dataset}"
elif [ "$task" = "SLU" ]; then
    test_scp_file_path="/aistor/sjtu/hpc_stor01/home/yangyi/data/slurp/test"
elif [ "$task" = "sentiment" ]; then
    test_scp_file_path="/aistor/sjtu/hpc_stor01/home/yangyi/data/GLUE/sst2"
fi

# Choose Encoder
encoder_name=sensevoice
speech_encoder_path=/aistor/sjtu/hpc_stor01/home/yangyi/model/SenseVoiceSmall
encoder_dim=25055 #25055 #512
encoder_projector_ds_rate=1

do_psd=true # whether use psd to ds
ctc_posterior=true # whether use ctc posterior
voca_trans=false # whether use vocabulary transfer
top1_emb=false
llm_name="Qwen2.5-1.5B-Instruct"
llm_path=/aistor/sjtu/hpc_stor01/home/yangyi/model/Qwen2.5-1.5B-Instruct
llm_dim=1536 #151936 #1536 3584
model_factory=model/ps-slm.py:model_factory # u can also create your own model_factory
run_decode_device=0  # run decode on certain device
# decode_log=$ckpt_path/decode_${dataset}_${task}_${split}
output_dir="${run_dir}/ckpt_text"
mkdir -p "$output_dir"
decode_log="$output_dir/decode_${dataset}_${task}_${split}"

# 1. 定义文件路径
original_jsonl="${test_scp_file_path}/multitask.jsonl"
split_base_dir="${run_dir}/temp_splits_$(date +"%m%d_%H%M")"
mkdir -p "$split_base_dir"

# 2. 将原始 jsonl 均匀切分为 8 份
# 生成 part_00.jsonl, part_01.jsonl ...
split -n l/8 -d --additional-suffix=.jsonl "$original_jsonl" "${split_base_dir}/part_"

export ASCEND_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# 3. 为每个 NPU 创建环境并并行启动
for i in {0..7}; do
    export PROC_RANK=$i
    export ASCEND_VISIBLE_DEVICES=$i
    
    echo "Starting Rank $PROC_RANK on NPU $ASCEND_VISIBLE_DEVICES"

    rank_dir="${split_base_dir}/rank_$i"
    mkdir -p "$rank_dir"
    
    # 将切分好的数据移动到 rank 目录，并改回代码硬编码要求的名称
    mv "${split_base_dir}/part_0${i}.jsonl" "${rank_dir}/multitask.jsonl"
    
    export ASCEND_VISIBLE_DEVICES=$i
    export PROC_RANK=$i
    
    echo "Starting Rank $i on NPU $i, using data in $rank_dir"
    
    python $code_dir/inference_batch.py \
        hydra.run.dir=$output_dir \
        ++dataset_config.test_scp_file_path="$rank_dir" \
        ++decode_log=${decode_log}_${i} \
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
        ++dataset_config.inference_mode=true \
        ++train_config.model_name=ps-slm \
        ++train_config.device=$i \
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
        ++ckpt_path=$ckpt_path/pytorch_model.bin \
        & # 后台并行
    
    sleep 2
done

# 4. 等待所有进程结束
wait
echo "All parallel processes finished."

# 5. 合并 8 个分片的结果
> "${decode_log}_pred"
> "${decode_log}_gt"

for i in {0..7}; do
    cat "${decode_log}_${i}_pred" >> "${decode_log}_pred"
    cat "${decode_log}_${i}_gt" >> "${decode_log}_gt"
    # 删除临时输出分片
    rm "${decode_log}_${i}_pred" "${decode_log}_${i}_gt"
done

# 6. 删除临时数据文件夹
rm -rf "$split_base_dir"

python clean_marks.py ${decode_log}_gt
python clean_marks.py ${decode_log}_pred

python utils/wenet_compute_cer.py --char=1 -v=1 ${decode_log}_gt ${decode_log}_pred > ${decode_log}_wer
python utils/wenet_compute_cer.py --char=1 -v=1 ${decode_log}_gt ${decode_log}_pred > ${decode_log}_wer
