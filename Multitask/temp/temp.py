import json
import numpy as np
import os

def stat_psd_lengths(jsonl_paths):
    all_lens = []
    for path in jsonl_paths:
        if not os.path.exists(path):
            print(f"Skipping {path}: File not found")
            continue
        
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    # 直接从预处理好的 psd_len 字段读取
                    if 'psd_len' in item:
                        all_lens.append(int(item['psd_len']))
                except:
                    continue
    
    if not all_lens:
        print("No psd_len data found.")
        return

    all_lens = np.array(all_lens)
    print(f"--- Statistics for {len(all_lens)} samples ---")
    print(f"Max Length:    {np.max(all_lens)}")
    print(f"Mean Length:   {np.mean(all_lens):.2f}")
    print(f"95th Percentile: {np.percentile(all_lens, 95)}")
    print(f"99th Percentile: {np.percentile(all_lens, 99)}")
    print(f"Samples > 160:  {np.sum(all_lens > 160)} ({np.sum(all_lens > 160)/len(all_lens)*100:.2f}%)")

if __name__ == "__main__":
    # 请根据实际路径修改
    data_dir = "/aistor/aispeech/hpc_stor01/home/wangchenghao00sx/workingspace/TASU-simulator/Multitask/data"
    files_to_stat = [
        os.path.join(data_dir, "train/multitask_augmented.jsonl"),
        os.path.join(data_dir, "dev/multitask_augmented.jsonl")
    ]
    stat_psd_lengths(files_to_stat)