#!/bin/bash

# 指定需要占用的 NPU ID
NPU_IDS=(1 2 3 4 5 6 7)

# 捕获 Ctrl+C (SIGINT) 和 终止信号 (SIGTERM)
trap "echo -e '\n[!] 正在停止任务并清理 NPU 进程...'; kill 0; exit" SIGINT SIGTERM

echo "[+] 正在 NPU ${NPU_IDS[*]} 上启动保活任务..."
echo "[+] 按下 Ctrl+C 可同时停止所有任务并清理环境。"

for id in "${NPU_IDS[@]}"; do
    # 启动 Python 循环计算：每隔 2 秒进行一次简单的矩阵乘法
    python3 -c "
import torch
import torch_npu
import time

device = torch.device('npu:$id')
# 稍微调大一点矩阵，约占用 800MB-1GB 显存
a = torch.randn(8000, 8000).to(device)
b = torch.randn(8000, 8000).to(device)

print(f'--> NPU $id 重载任务已启动', flush=True)

while True:
    # 连续进行 100 次矩阵乘法，强迫 AICore 持续工作
    for _ in range(100):
        c = torch.matmul(a, b)
    
    # 极短的休眠，防止脚本把 CPU 线程吃死
    time.sleep(0.1)
" &
done

# 等待所有后台进程
wait