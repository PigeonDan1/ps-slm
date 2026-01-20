#!/bin/bash

# ================= 配置 =================
# 指定使用 NPU 1-7 (共7张卡)
# 注意：DeepSpeed 的 include 语法中，localhost 后面跟具体的卡号
TARGET_DEVICES="1,2,3,4,5,6,7"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export PATH=/usr/local/python3.10.15/bin:$PATH

# 定位工作目录
run_dir=$(cd $(dirname $0)/..; pwd)
cd $run_dir || exit 1

echo "--> Working Directory: $run_dir"
echo "--> Using NPUs: $TARGET_DEVICES"

# ================= 启动 =================
# 直接使用 DeepSpeed Launcher
# 它会自动拉起7个进程，并自动分配 Rank
deepspeed \
    --include localhost:$TARGET_DEVICES \
    preprocess_data.py