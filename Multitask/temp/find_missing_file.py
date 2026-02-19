import json
import os
from tqdm import tqdm

def check_psd_files(jsonl_path):
    # 确保输入路径为绝对路径
    abs_jsonl_path = os.path.abspath(jsonl_path)
    missing_count = 0
    
    if not os.path.exists(abs_jsonl_path):
        print(f"Error: JSONL file not found at {abs_jsonl_path}")
        return

    # 预读行数用于进度条展示
    with open(abs_jsonl_path, 'r', encoding='utf-8') as f:
        total = sum(1 for _ in f)

    with open(abs_jsonl_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=total, desc="Checking PSD paths"):
            try:
                data = json.loads(line)
                psd_path = data.get("psd_path")
                
                # 直接检查 JSON 中的绝对路径是否存在
                if psd_path and not os.path.exists(psd_path):
                    missing_count += 1
            except json.JSONDecodeError:
                continue

    print(f"\n--- Check Completed ---")
    if missing_count > 0:
        print(f"Total missing files: {missing_count}")
    else:
        print("Success: All files exist.")

if __name__ == "__main__":
    # 使用你当前环境的绝对路径
    BASE_DIR = "/aistor/aispeech/hpc_stor01/home/wangchenghao00sx/workingspace/TASU-simulator/Multitask"
    DATA_PATH = os.path.join(BASE_DIR, "data/train/multitask_filtered_augmented_3bucket.jsonl")
    
    check_psd_files(DATA_PATH)