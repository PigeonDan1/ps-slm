#!/bin/bash

# ================= 信号捕获与清理 (新增部分) =================
# 定义清理函数：杀掉所有子进程
cleanup() {
    echo ""
    echo "!!! Caught Ctrl+C/Signal. Killing all background NPU processes... !!!"
    # pkill -P $$ 表示杀掉当前脚本进程($$)产生的所有子进程
    pkill -P $$ 
    # 为了保险，再次尝试通过名字清理（可选）
    pkill -9 -f inference_simulator.py
    exit 1
}

# 捕获 SIGINT (Ctrl+C) 和 SIGTERM 信号，触发 cleanup 函数
trap cleanup SIGINT SIGTERM

# ================= 控制参数 =================
# WERS=("10" "12" "14" "16" "18" "20" "30" "40" "50" "80")
# S_RATES=("0.052" "0.062" "0.073" "0.083" "0.094" "0.10" "0.16" "0.21" "0.26" "0.416")
# D_RATES=("0.019" "0.022" "0.026" "0.030" "0.033" "0.037" "0.056" "0.074" "0.093" "0.144")
# I_RATES=("0.030" "0.036" "0.042" "0.048" "0.054" "0.060" "0.090" "0.12" "0.15" "0.232")

WERS=("0" "1" "2" "3" "4" "5" "6" "7" "8" "9" )
S_RATES=("0.00" "0.0052" "0.010" "0.016" "0.021" "0.026" "0.031" "0.036" "0.042" "0.047")
D_RATES=("0.00" "0.0018" "0.0037" "0.0055" "0.0074" "0.0092" "0.011" "0.013" "0.015" "0.017" )
I_RATES=("0.00" "0.0029" "0.0059" "0.0088" "0.012" "0.015" "0.018" "0.021" "0.024" "0.027")


NPU_IDS=(0 1 2 3 4 5 6 7)
WORLD_SIZE=${#NPU_IDS[@]}

export PYTHONUNBUFFERED=1
run_dir=$(cd $(dirname $0)/..; pwd)
cd $run_dir || exit 1
INPUT_JSONL="${run_dir}/data/test-clean/multitask.jsonl"

# ================= 循环执行 =================
for i in "${!WERS[@]}"; do
    WER_VAL=${WERS[$i]}
    S_RATE=${S_RATES[$i]}
    D_RATE=${D_RATES[$i]}
    I_RATE=${I_RATES[$i]}
    
    OUT_DIR="${run_dir}/data/test-clean/test_without_feedback/simulator_WER${WER_VAL}"
    echo "=========================================================="
    echo "--> [Task $i] WER: ${WER_VAL}%, S: $S_RATE, D: $D_RATE, I: $I_RATE"
    mkdir -p "$OUT_DIR"

    # 清理旧数据
    rm -f "$OUT_DIR"/temp_npu*.jsonl

    pids=()
    for rank in "${!NPU_IDS[@]}"; do
        NPU=${NPU_IDS[$rank]}
        
        python inference_simulator.py \
            --rank $rank \
            --world_size $WORLD_SIZE \
            --device_id $NPU \
            --input_jsonl "$INPUT_JSONL" \
            --output_dir "$OUT_DIR" \
            --s_rate $S_RATE \
            --d_rate $D_RATE \
            --i_rate $I_RATE \
            --wer 0.${WER_VAL} & 
        
        pids+=($!)
    done

    # 错误监测与等待
    fail_flag=0
    for pid in "${pids[@]}"; do
        # wait如果不加处理，收到Ctrl+C会退出，但我们需要trap接管
        wait $pid 
        status=$?
        # 如果返回值非0，且不是因为被杀掉（130/143通常是信号中断），标记失败
        if [ $status -ne 0 ]; then
             # 如果是我们自己按了Ctrl+C，trap会处理，这里只需检测非人为的错误
             if [ $status -ne 143 ] && [ $status -ne 130 ]; then
                echo "--> [Error] Process $pid failed with status $status."
                fail_flag=1
             fi
        fi
    done

    if [ $fail_flag -ne 0 ]; then
        echo "--> [Error] Task failed. Aborting merge."
        exit 1
    fi

    # 合并
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

echo "--> All tasks finished successfully."