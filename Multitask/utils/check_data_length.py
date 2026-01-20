import os
import json
import numpy as np
import glob

# ================= 配置 =================
BASE_WORK_DIR = "/aistor/sjtu/hpc_stor01/home/wangchenghao/workingspace/ps-slm/Multitask"
DATA_DIR = os.path.join(BASE_WORK_DIR, "data")

def scan_jsonl(split):
    jsonl_path = os.path.join(DATA_DIR, split, "multitask.jsonl")
    if not os.path.exists(jsonl_path):
        print(f"[Warn] {jsonl_path} not found.")
        return [], []
    
    psd_lens = []
    text_lens = []
    
    print(f"--> Scanning {split} set: {jsonl_path} ...")
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                item = json.loads(line)
                # 读取 metadata 中的长度
                # 注意：这是 raw psd length，不含 EOS
                p_len = item.get('psd_len', 0)
                t_len = item.get('text_len', 0)
                
                # 实际上模型需要的长度是 p_len + 1 (EOS)
                psd_lens.append(p_len + 1)
                text_lens.append(t_len)
            except:
                continue
    return psd_lens, text_lens

def print_stats(name, data):
    if not data:
        print(f"  {name}: No data.")
        return
    
    data = np.array(data)
    print(f"  === {name} Statistics ===")
    print(f"    Count: {len(data)}")
    print(f"    Min:   {np.min(data)}")
    print(f"    Max:   {np.max(data)}  <-- 重点关注")
    print(f"    Mean:  {np.mean(data):.2f}")
    print(f"    P90:   {np.percentile(data, 90):.0f}")
    print(f"    P95:   {np.percentile(data, 95):.0f}")
    print(f"    P99:   {np.percentile(data, 99):.0f}")
    print(f"    P99.9: {np.percentile(data, 99.9):.0f}")

def main():
    all_psd_lens = []
    all_text_lens = []
    
    for split in ["train", "dev"]:
        p_lens, t_lens = scan_jsonl(split)
        all_psd_lens.extend(p_lens)
        all_text_lens.extend(t_lens)
        
    print("\n" + "="*40)
    print("GLOBAL STATISTICS (Train + Dev)")
    print("="*40)
    
    # 你的模型配置 max_len 需要覆盖这两个的最大值
    print_stats("PSD-CTC Length (with EOS)", all_psd_lens)
    print("-" * 20)
    print_stats("Text One-hot Length", all_text_lens)
    
    # 建议值
    max_p = np.max(all_psd_lens) if all_psd_lens else 0
    max_t = np.max(all_text_lens) if all_text_lens else 0
    safe_max = max(max_p, max_t)
    
    print("\n" + "="*40)
    print(f"SUGGESTION: Set 'max_len' >= {safe_max}")
    # 考虑到 padding 通常设为 8 或 16 的倍数对硬件更友好
    suggested_cfg = ((safe_max + 15) // 16) * 16
    print(f"Recommended Config: max_len = {suggested_cfg}")
    print("="*40)

if __name__ == "__main__":
    main()