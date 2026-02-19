BASE_DIR="/aistor/sjtu/hpc_stor01/home/wangchenghao/workingspace/TASU-simulator/Multitask"
LOG_PATH="$BASE_DIR/logs/debug_interactive.log"

mkdir -p "$BASE_DIR/logs"

# 提交任务
vc submit -p "pdgpu-sjtu-ai" \
  -i "hub.szaic.com/sjtu/sjtu_yukai-chenghao.wang-ps-slm:v1.0" \
  -c 120 -m "256G" -g 8 \
  "JOB=1:1" "$LOG_PATH" \
  --cmd "sleep infinity"