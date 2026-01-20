#NAR模型生成数据
#!/bin/bash

export PYTHONUNBUFFERED=1         
export OMP_NUM_THREADS=1

# 确保 Python 路径
export PATH=/usr/local/python3.10.15/bin:$PATH

run_dir=$(cd $(dirname $0)/..; pwd)
cd $run_dir || exit 1

echo "--> Starting 8 independent processes..."

# 循环启动 8 个后台任务
for rank in {0..7}; do
    echo "Launching Rank $rank..."
    python inference_oracle.py --rank $rank --world_size 8 &
done

# 等待所有后台任务结束
wait

echo "--> All processes finished. Merging JSONL files..."

# 手动合并文件
DATA_DIR="/aistor/sjtu/hpc_stor01/home/wangchenghao/workingspace/ps-slm/Multitask/data"

for split in "train" "dev"; do
    output_file="${DATA_DIR}/${split}/multitask_sim_oracle.jsonl"
    rm -f $output_file # 清理旧的
    
    echo "Merging ${split}..."
    cat ${DATA_DIR}/${split}/multitask_sim_oracle_rank*.jsonl > $output_file
    
    rm ${DATA_DIR}/${split}/multitask_sim_oracle_rank*.jsonl
    
    echo "Created: $output_file"
done

echo "--> Done!"