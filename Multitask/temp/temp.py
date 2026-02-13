import os
import random

def merge_and_shuffle_jsonl():
    base_path = "/aistor/aispeech/hpc_stor01/home/wangchenghao00sx/workingspace/TASU-simulator/Multitask/data"
    
    sources = [
        os.path.join(base_path, "temp_jsonl/libri_origin_50k.jsonl"),
        os.path.join(base_path, "train/train_bucket_control/simulator_B2/multitask.jsonl"),
    ]
    
    target_dir = os.path.join(base_path, "libri_sim_real/simulator_B2")
    target_file = os.path.join(target_dir, "multitask.jsonl")
    
    all_data = []

    # 读取所有数据到内存
    for fname in sources:
        if os.path.exists(fname):
            with open(fname, 'r', encoding='utf-8') as f:
                all_data.extend(f.readlines())
        else:
            print(f"Warning: {fname} not found.")

    # 核心：打乱顺序
    random.shuffle(all_data)

    # 写入新文件
    os.makedirs(target_dir, exist_ok=True)
    with open(target_file, 'w', encoding='utf-8') as f:
        f.writelines(all_data)

if __name__ == "__main__":
    merge_and_shuffle_jsonl()