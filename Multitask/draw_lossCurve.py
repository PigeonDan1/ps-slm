import re
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1. 读取日志（改成你的日志文件）
# =========================
log_file = "exp/simulator_ar_control_feedback_fromAR_PureMBR_20260121-1854/train.log"  # 或直接把日志内容写成字符串

with open(log_file, "r") as f:
    lines = f.readlines()

# =========================
# 2. 正则解析 Global / Loss / Penalty
# =========================
pattern = re.compile(
    r"Global:\s*(\d+).*?Loss:\s*([0-9.]+).*?Penalty:\s*([0-9.]+)"
)

steps = []
losses = []
penalties = []

for line in lines:
    m = pattern.search(line)
    if m:
        steps.append(int(m.group(1)))
        losses.append(float(m.group(2)))
        penalties.append(float(m.group(3)))

steps = np.array(steps)
losses = np.array(losses)
penalties = np.array(penalties)

# =========================
# 3. 下采样（趋势线）
# =========================
def moving_average(x, window=20):
    return np.convolve(x, np.ones(window) / window, mode="valid")

window = 20
loss_ma = moving_average(losses, window)
penalty_ma = moving_average(penalties, window)
steps_ma = steps[window - 1:]


# =========================
# 4. 画图
# =========================
plt.figure(figsize=(10, 6))

# ---- Loss ----
plt.subplot(2, 1, 1)
plt.plot(steps, losses, alpha=0.25, linewidth=1, label="Loss (raw)")
plt.plot(steps_ma, loss_ma, linewidth=2.5, label="Loss (moving avg)")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.3)

# ---- Penalty ----
plt.subplot(2, 1, 2)
plt.plot(steps, penalties, alpha=0.25, linewidth=1, label="Penalty (raw)")
plt.plot(steps_ma, penalty_ma, linewidth=2.5, label=f"Penalty (moving average)")
plt.xlabel("Global Step")
plt.ylabel("Penalty")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.3)

plt.tight_layout()

output_path = "earliest_loss_penalty_curve.png"
plt.savefig(output_path, dpi=200, bbox_inches="tight")
plt.close()

print(f"Saved figure to: {output_path}")
