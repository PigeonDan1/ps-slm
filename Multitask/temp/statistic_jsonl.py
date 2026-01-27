#统计jsonl数据增强前后WER分布
import json
import os
from collections import Counter
from tqdm import tqdm

# ORIGINAL_JSONL = "data/dev/multitask.jsonl"
# AUGMENTED_JSONL = "data/dev/multitask_augmented.jsonl"

ORIGINAL_JSONL = "data/train/multitask.jsonl"
AUGMENTED_JSONL = "data/train/multitask_augmented.jsonl"
def get_bucket_id(wer):
    """根据项目定义的 WER 区间进行分桶"""
    if wer < 0.05: return 1
    elif wer < 0.10: return 2
    elif wer < 0.20: return 3
    else: return 4

def analyze_file(file_path):
    """分析单个文件的行数和分桶分布"""
    if not os.path.exists(file_path):
        return None
    
    counts = Counter()
    total_lines = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"Analyzing {os.path.basename(file_path)}"):
            line = line.strip()
            if not line: continue
            try:
                data = json.loads(line)
                # 优先读取已有的 bucket_id，否则根据 bpe_error 计算
                if "bucket_id" in data:
                    bid = data["bucket_id"]
                else:
                    wer = data.get("bpe_error", {}).get("WER", 0.0)
                    bid = get_bucket_id(wer)
                
                counts[bid] += 1
                total_lines += 1
            except Exception:
                continue
                
    return {"total": total_lines, "buckets": dict(sorted(counts.items()))}

def main():
    print("--> 正在开始统计分析...\n")
    orig_res = analyze_file(ORIGINAL_JSONL)
    aug_res = analyze_file(AUGMENTED_JSONL)

    if not orig_res:
        print(f"[错误] 无法找到原始文件: {ORIGINAL_JSONL}")
        return

    print("\n" + "="*70)
    print(f"{'区间 (Bucket)':<20} | {'原始样本数':<15} | {'增强后样本数':<15} | {'增长率':<10}")
    print("-" * 70)

    for bid in range(1, 5):
        orig_count = orig_res["buckets"].get(bid, 0)
        aug_count = aug_res["buckets"].get(bid, 0) if aug_res else 0
        ratio = aug_count / orig_count if orig_count > 0 else 0
        
        print(f"{f'Bucket {bid}':<20} | {orig_count:<15,} | {aug_count:<15,} | {ratio:.2f}x")

    print("-" * 70)
    orig_total = orig_res["total"]
    aug_total = aug_res["total"] if aug_res else 0
    total_ratio = aug_total / orig_total if orig_total > 0 else 0
    
    print(f"{'总计 (Total)':<20} | {orig_total:<15,} | {aug_total:<15,} | {total_ratio:.2f}x")
    print("="*70)

    if aug_res:
        print("\n--> 增强后分布占比检查:")
        for bid in range(1, 5):
            count = aug_res["buckets"].get(bid, 0)
            percent = (count / aug_total) * 100 if aug_total > 0 else 0
            print(f"    Bucket {bid}: {percent:>6.2f}%")

if __name__ == "__main__":
    main()