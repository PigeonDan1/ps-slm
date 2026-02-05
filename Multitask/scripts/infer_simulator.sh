#!/bin/bash

cleanup() {
    echo ""
    echo "!!! Caught Ctrl+C/Signal. Killing all background NPU processes... !!!"
    pkill -P $$ 
    pkill -9 -f inference_simulator.py
    exit 1
}

trap cleanup SIGINT SIGTERM

# 修改为3个bucket
BUCKETS=(1 3)

NPU_IDS=(0 1 2 3 4 5 6 7)
WORLD_SIZE=${#NPU_IDS[@]}

export PYTHONUNBUFFERED=1
run_dir=$(cd $(dirname $0)/..; pwd)
cd $run_dir || exit 1

# 请确保输入文件路径正确
INPUT_JSONL="${run_dir}/data/test-clean/multitask.jsonl"

# ================= 循环执行 =================
for BUCKET in "${BUCKETS[@]}"; do
    # 修改输出目录名称，以反映 Bucket 档位
    OUT_DIR="${run_dir}/data/test-clean/test_bucket_control/simulator_B${BUCKET}"
    
    echo "=========================================================="
    echo "--> [Task] Processing Bucket: $BUCKET "
    mkdir -p "$OUT_DIR"

    # 清理旧数据
    rm -f "$OUT_DIR"/temp_npu*.jsonl

    pids=()
    for rank in "${!NPU_IDS[@]}"; do
        NPU=${NPU_IDS[$rank]}
        
        # 修改调用接口：移除 --s_rate, --d_rate, --i_rate, --wer
        # 统一使用 --bucket_id
        python inference_simulator.py \
            --rank $rank \
            --world_size $WORLD_SIZE \
            --device_id $NPU \
            --input_jsonl "$INPUT_JSONL" \
            --output_dir "$OUT_DIR" \
            --bucket_id $BUCKET & 
        
        pids+=($!)
    done

    # 错误监测与等待
    fail_flag=0
    for pid in "${pids[@]}"; do
        wait $pid 
        status=$?
        if [ $status -ne 0 ]; then
             if [ $status -ne 143 ] && [ $status -ne 130 ]; then
                echo "--> [Error] Process $pid failed with status $status."
                fail_flag=1
             fi
        fi
    done

    if [  $fail_flag -ne 0 ]; then
        echo "--> [Error] Task failed. Aborting merge."
        exit 1
    fi

    # 合并 JSONL 结果
    FINAL_JSONL="$OUT_DIR/multitask.jsonl"
    if ls "$OUT_DIR"/temp_npu*.jsonl >/dev/null 2>&1; then
        cat "$OUT_DIR"/temp_npu*.jsonl > "$FINAL_JSONL"
        rm "$OUT_DIR"/temp_npu*.jsonl
        echo "--> [Done] Saved to $FINAL_JSONL"
    else
        echo "--> [Error] No output files found!"
        exit 1
    fi
done

echo "--> All bucket inference tasks finished successfully."