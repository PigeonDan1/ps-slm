import matplotlib.pyplot as plt
import re

def plot_training_log(log_path):
    steps, losses, accs = [], [], []
    
    # 匹配 Global step, Loss 和 Acc
    pattern = re.compile(r"Global: (\d+), Loss: ([\d.]+), Acc: ([\d.]+)")

    with open(log_path, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                steps.append(int(match.group(1)))
                losses.append(float(match.group(2)))
                accs.append(float(match.group(3)))

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 绘制 Loss
    color_loss = 'tab:red'
    ax1.set_xlabel('Global Step')
    ax1.set_ylabel('Loss', color=color_loss)
    ax1.plot(steps, losses, color=color_loss, label='Loss')
    ax1.tick_params(axis='y', labelcolor=color_loss)

    # 绘制 Accuracy (共享 X 轴)
    ax2 = ax1.twinx()
    color_acc = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color_acc)
    ax2.plot(steps, accs, color=color_acc, label='Accuracy')
    ax2.tick_params(axis='y', labelcolor=color_acc)

    plt.title('Training Loss and Accuracy')
    fig.tight_layout()
    plt.savefig('training_metrics.png')
    plt.show()

if __name__ == "__main__":
    path = "/aistor/aispeech/hpc_stor01/home/wangchenghao00sx/workingspace/TASU-simulator/Multitask/exp/simulator_ar_control_augmentation_20260130-0137/train.log"
    plot_training_log(path)