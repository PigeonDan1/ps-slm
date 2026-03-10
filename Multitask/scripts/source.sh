BASE_DIR="/aistor/sjtu/hpc_stor01/home/wangchenghao/workingspace/TASU-simulator/Multitask"
LOG_PATH="$BASE_DIR/logs/debug_interactive.log"

mkdir -p "$BASE_DIR/logs"

# 提交任务
vc submit -p "pdgpu-sjtu-ai" \
  -v /aistor/aispeech:/aistor/aispeech:rw \
  -v /hpc_stor08:/hpc_stor08:rw \
  -v /srvstor:/srvstor:rw \
  -i "hub.szaic.com/sjtu/sjtu_yukai-chenghao.wang-ps-slm:v1.0" \
  -c 120 -m "256G" -g 8 \
  "JOB=1:1" "$LOG_PATH" \
  --cmd "sleep infinity"