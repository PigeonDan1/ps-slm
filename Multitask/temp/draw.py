import matplotlib.pyplot as plt
import re
import os

def plot_training_log(log_path, save_name="loss_curve.png"):
    steps, losses = [], []

    # 匹配 Global step 和 Loss
    pattern = re.compile(r"Global: (\d+), Loss: ([\d.]+)")

    with open(log_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                steps.append(int(match.group(1)))
                losses.append(float(match.group(2)))

    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses)
    plt.xlabel("Global Step")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)

    # 保存在当前目录
    plt.savefig(save_name)
    plt.close()

if __name__ == "__main__":
    path = "/aistor/aispeech/hpc_stor01/home/wangchenghao00sx/workingspace/TASU-simulator/Multitask/exp/simulator_ar_control_augmentation_20260203-1921/train.log"
    plot_training_log(path)
