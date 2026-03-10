import random
import os

def sample_jsonl(input_path, output_path, ratio=0.5):
    """按比例随机抽取 jsonl 行并保存。"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            if random.random() < ratio:
                f_out.write(line)

def process_tasks():
    tasks = [
        (
            "/aistor/sjtu/hpc_stor01/home/yangyi/data/multitask_large/train/multitask.jsonl",
            "/aistor/sjtu/hpc_stor01/home/wangchenghao/workingspace/TASU-simulator/Multitask/data/half_multitask_large/train/multitask.jsonl"
        ),
        (
            "/aistor/sjtu/hpc_stor01/home/yangyi/data/multitask_large/dev/multitask.jsonl",
            "/aistor/sjtu/hpc_stor01/home/wangchenghao/workingspace/TASU-simulator/Multitask/data/half_multitask_large/dev/multitask.jsonl"
        )
    ]
    
    for src, dst in tasks:
        print(f"Processing: {src} -> {dst}")
        sample_jsonl(src, dst)
    print("Done.")

if __name__ == "__main__":
    process_tasks()