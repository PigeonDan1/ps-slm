#绘制增强后数据的直方图并观察分组合理性
import json
import os
from collections import defaultdict, Counter
from tqdm import tqdm

def analyze_full_dataset(input_path):
    # 定义区间
    intervals = {
        1: (0.0, 0.06),
        2: (0.1, 0.4),
        3: (0.45, 1.5)
    }
    
    # 统计变量
    text_groups = defaultdict(list)
    overall_wer_counts = Counter()  # 全量直方图统计
    
    if not os.path.exists(input_path):
        print(f"Error: 找不到文件 {input_path}")
        return

    # 1. 第一遍扫描：收集所有数据并统计全局分布
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="扫描全量数据"):
            item = json.loads(line)
            base_key = item['key'].split('_aug_')[0]
            wer = item.get('bpe_error', {}).get('WER', 0)
            
            # 记录到分组
            text_groups[base_key].append(wer)
            
            # 记录到全局直方图 (取 0-1 之间，步长 0.01)
            bin_idx = int(wer * 100)
            if bin_idx < 100:
                overall_wer_counts[bin_idx] += 1
            else:
                overall_wer_counts[100] += 1 # 100% 以上的统一归类

    # 2. 逻辑验证：检查“三桶各一”的文本数量
    valid_groups_count = 0
    bucket_availability = {1: 0, 2: 0, 3: 0} # 统计每个桶分别有多少文本支持
    
    for base_key, wers in text_groups.items():
        has_b1 = any(intervals[1][0] <= v < intervals[1][1] for v in wers)
        has_b2 = any(intervals[2][0] <= v < intervals[2][1] for v in wers)
        has_b3 = any(intervals[3][0] <= v < intervals[3][1] for v in wers)
        
        if has_b1: bucket_availability[1] += 1
        if has_b2: bucket_availability[2] += 1
        if has_b3: bucket_availability[3] += 1
        
        if has_b1 and has_b2 and has_b3:
            valid_groups_count += 1

    # 3. 打印报告
    print("\n" + "="*60)
    print("【70万全量数据】分布与约束分析报告")
    print("-" * 60)
    print(f"总样本数: {sum(overall_wer_counts.values())}")
    print(f"独立文本数 (Base Keys): {len(text_groups)}")
    print("-" * 60)
    print("1. 设定区间与文本覆盖情况 (只要文本有一个变体落在区间即计入):")
    for b_id, (low, high) in intervals.items():
        count = bucket_availability[b_id]
        print(f"   Bucket {b_id} [{low:.2f} - {high:.2f}): {count} 个文本支持 ({count/len(text_groups):.1%})")
    
    print("-" * 60)
    print(f"2. 满足『一文本三桶各一』强约束的可提取组数: {valid_groups_count}")
    
    target = 30000
    if valid_groups_count >= target:
        print(f"   结论: ✅ 数据充足！可以提取 {target} 组 (占可用组的 {target/valid_groups_count:.1%})")
    else:
        print(f"   结论: ❌ 数据不足！缺口 {target - valid_groups_count} 组")

    print("-" * 60)
    print("3. 全量数据 WER 粗略分布 (10% 步长):")
    for i in range(0, 100, 10):
        bin_sum = sum(overall_wer_counts[j] for j in range(i, i+10))
        bar = "#" * int(bin_sum / sum(overall_wer_counts.values()) * 50)
        print(f"   {i:2d}% - {i+10:2d}% : {bin_sum:7d} {bar}")
    print(f"   > 100%     : {overall_wer_counts[100]:7d}")
    print("="*60 + "\n")

if __name__ == "__main__":
    PATH = "/aistor/aispeech/hpc_stor01/home/wangchenghao00sx/workingspace/TASU-simulator/Multitask/data/train/multitask_filtered_augmented_exist.jsonl"
    analyze_full_dataset(PATH)