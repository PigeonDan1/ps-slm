import json
import os
from tqdm import tqdm

def clean_jsonl(input_path, output_path):
    input_abs = os.path.abspath(input_path)
    output_abs = os.path.abspath(output_path)
    
    kept_count = 0
    total_count = 0

    # 预读行数用于 tqdm
    with open(input_abs, 'r', encoding='utf-8') as f:
        total = sum(1 for _ in f)

    with open(input_abs, 'r', encoding='utf-8') as f_in, \
         open(output_abs, 'w', encoding='utf-8') as f_out:
        
        for line in tqdm(f_in, total=total, desc="Cleaning JSONL"):
            total_count += 1
            try:
                data = json.loads(line.strip())
                psd_path = data.get("psd_path")
                
                # 核心逻辑：只有路径存在才写入新文件
                if psd_path and os.path.exists(psd_path):
                    f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
                    kept_count += 1
            except json.JSONDecodeError:
                continue

    print(f"\n--- Cleaning Completed ---")
    print(f"Original records: {total_count}")
    print(f"Kept records:     {kept_count}")
    print(f"Removed records:  {total_count - kept_count}")
    print(f"Cleaned file saved to: {output_abs}")

if __name__ == "__main__":
    BASE_DIR = "/aistor/aispeech/hpc_stor01/home/wangchenghao00sx/workingspace/TASU-simulator/Multitask"
    IN_FILE = os.path.join(BASE_DIR, "data/train/multitask_filtered_augmented.jsonl")
    OUT_FILE = os.path.join(BASE_DIR, "data/train/multitask_filtered_augmented_exist.jsonl")
    
    clean_jsonl(IN_FILE, OUT_FILE)