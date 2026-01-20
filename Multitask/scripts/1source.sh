# 定义你的基础路径
BASE_DIR="/aistor/sjtu/hpc_stor01/home/wangchenghao/workingspace/ps-slm/Multitask"
LOG_PATH="$BASE_DIR/logs/debug_interactive.log"

# 创建日志目录
mkdir -p "$BASE_DIR/logs"

# 提交任务 (申请最后 1 个 NPU)
vc submit -p "pdgpu-sjtu-ai" \
  -i "hub.szaic.com/sjtu/sjtu_yukai-chenghao.wang-ps-slm:v1.0" \
  -c 8 -m "120G" -g 1 \
  "JOB=1:1" "$LOG_PATH" \
  --cmd "sleep infinity"