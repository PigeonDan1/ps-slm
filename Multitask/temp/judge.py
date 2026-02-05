import json
import os
from collections import defaultdict, Counter
from tqdm import tqdm

def verify_constraints(file_path, expected_count=None):
    print(f"\n正在检查文件: {os.path.basename(file_path)}")
    
    stats = {
        "total_lines": 0,
        "bucket_counts": Counter(),
        "groups": defaultdict(list),
        "adjacency_violations": 0
    }
    
    prev_base_key = None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Checking"):
            item = json.loads(line)
            stats["total_lines"] += 1
            
            # 1. 提取 base_key
            base_key = item['key'].split('_aug_')[0]
            bucket_id = item.get('bucket_id')
            
            # 2. 统计 Bucket
            stats["bucket_counts"][bucket_id] += 1
            
            # 3. 记录组信息
            stats["groups"][base_key].append(bucket_id)
            
            # 4. 检查物理相邻 (约束：相同 base_key 不能挨着)
            if base_key == prev_base_key:
                stats["adjacency_violations"] += 1
            prev_base_key = base_key

    # 逻辑判断
    is_balanced = len(set(stats["bucket_counts"].values())) <= 1
    
    # 检查每组是否都是完整的 {1, 2, 3}
    group_integrity = True
    for b_key, b_ids in stats["groups"].items():
        if sorted(b_ids) != [1, 2, 3]:
            group_integrity = False
            break

    # 打印结果
    print("-" * 30)
    print(f"总行数: {stats['total_lines']}")
    if expected_count:
        print(f"数量验证: {'通过' if stats['total_lines'] == expected_count else '失败 (预期 ' + str(expected_count) + ')'}")
    
    print(f"Bucket 分布: {dict(stats['bucket_counts'])} -> {'平衡' if is_balanced else '不平衡'}")
    print(f"组完整性 (1文本3变体): {'通过' if group_integrity else '失败'}")
    print(f"物理相邻违规数: {stats['adjacency_violations']} -> {'通过 (已打散)' if stats['adjacency_violations'] == 0 else '失败 (存在相邻)'}")
    
    return set(stats["groups"].keys()), group_integrity

def run_full_check(base_dir):
    val_file = os.path.join(base_dir, "val_augmented_3bucket.jsonl")
    train_file = os.path.join(base_dir, "train_augmented_3bucket.jsonl")
    
    # 检查单个文件约束
    val_keys, val_ok = verify_constraints(val_file, expected_count=1998)
    train_keys, train_ok = verify_constraints(train_file)
    
    # 检查集合交集 (泄漏检查)
    overlap = val_keys.intersection(train_keys)
    print("\n" + "="*40)
    print("全局约束检查:")
    print(f"训练集与验证集重叠文本数: {len(overlap)}")
    if len(overlap) == 0:
        print("结果: 验证集完全独立，无数据泄漏。")
    else:
        print("警告: 发现数据泄漏！")
    print("="*40)

if __name__ == "__main__":
    BASE_DIR = "/aistor/aispeech/hpc_stor01/home/wangchenghao00sx/workingspace/TASU-simulator/Multitask/data/train"
    run_full_check(BASE_DIR)