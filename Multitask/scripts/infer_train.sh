#!/bin/bash

# ================= 信号捕获与清理 =================
cleanup() {
    echo ""
    echo "!!! Caught Ctrl+C/Signal. Killing all background NPU processes... !!!"
    pkill -P $$ 
    pkill -9 -f inference_simulator.py
    exit 1
}
trap cleanup SIGINT SIGTERM

# ================= 路径与环境配置 =================
NPU_IDS=(0 1 2 3 4 5 6 7)
WORLD_SIZE=${#NPU_IDS[@]}

export PYTHONUNBUFFERED=1
run_dir=$(cd $(dirname $0)/..; pwd)
cd $run_dir || exit 1

INPUT_JSONL="${run_dir}/data/train/multitask.jsonl"
OUT_DIR="${run_dir}/data/tempResult/reconstruction_train"
# MBR路径
CHECKPOINT_PATH="/aistor/sjtu/hpc_stor01/home/wangchenghao/workingspace/ps-slm/Multitask/exp/simulator_ar_control_feedback_fromAR_20260112-1815/checkpoints/step_14287/pytorch_model.bin/pytorch_model.bin"
# 无反馈路径
# CHECKPOINT_PATH="/aistor/sjtu/hpc_stor01/home/wangchenghao/workingspace/ps-slm/Multitask/exp/simulator_ar_control/checkpoints/pytorch_model.bin"
MAX_SAMPLES=2000

echo "=========================================================="
echo "--> Starting Reconstruction Test (Fitting Check)"
echo "--> Input: $INPUT_JSONL"
echo "--> Output Dir: $OUT_DIR"
echo "=========================================================="

mkdir -p "$OUT_DIR"
rm -f "$OUT_DIR"/temp_npu*.jsonl

# ================= 并行推理 =================
pids=()
for rank in "${!NPU_IDS[@]}"; do
    NPU=${NPU_IDS[$rank]}
    
    # [修改] 不再传递 --wer, --s_rate 等固定参数
    python infer_train.py \
        --rank $rank \
        --world_size $WORLD_SIZE \
        --device_id $NPU \
        --checkpoint_path "$CHECKPOINT_PATH" \
        --max_samples "$MAX_SAMPLES" \
        --input_jsonl "$INPUT_JSONL" \
        --output_dir "$OUT_DIR" \
        --use_reconstruction_mode & 
    
    pids+=($!)
done

# 等待所有进程完成
fail_flag=0
for pid in "${pids[@]}"; do
    wait $pid 
    status=$?
    if [ $status -ne 0 ] && [ $status -ne 143 ] && [ $status -ne 130 ]; then
        echo "--> [Error] Process $pid failed with status $status."
        fail_flag=1
    fi
done

if [ $fail_flag -ne 0 ]; then
    echo "--> [Error] Inference failed. Aborting."
    exit 1
fi

# ================= 合并结果 =================
FINAL_JSONL="$OUT_DIR/reconstruction_train.jsonl"
if ls "$OUT_DIR"/temp_npu*.jsonl >/dev/null 2>&1; then
    cat "$OUT_DIR"/temp_npu*.jsonl > "$FINAL_JSONL"
    rm "$OUT_DIR"/temp_npu*.jsonl
    echo "--> [Done] Inference finished. Saved to $FINAL_JSONL"
else
    echo "--> [Error] No output files found!"
    exit 1
fi

# ================= 后置分桶统计 (触发) =================
echo "--> Starting bucketed statistical analysis..."
# 调用统计脚本（稍后讨论其实现）
python eval_reconstruction.py \
    --input_jsonl "$FINAL_JSONL" \
    --tokenizer_path "/aistor/sjtu/hpc_stor01/home/yangyi/model/SenseVoiceSmall"

echo "--> All tasks finished successfully."