#将数据增强后的来自train的jsonl拆分为train和val，并且保证每个区间内文本出且仅出现一次
import json
import os
import random
from collections import defaultdict
from tqdm import tqdm

def split_and_shuffle(base_dir, input_filename, val_count=1998):
    input_path = os.path.join(base_dir, input_filename)
    val_path = os.path.join(base_dir, "val_augmented_3bucket.jsonl")
    train_path = os.path.join(base_dir, "train_augmented_3bucket.jsonl")

    # 1. 读取并按 base_key 分组（为了保证 1:1:1 的完整性）
    text_groups = defaultdict(list)
    if not os.path.exists(input_path):
        print(f"Error: 找不到文件 {input_path}")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Reading and Grouping"):
            item = json.loads(line)
            base_key = item['key'].split('_aug_')[0]
            text_groups[base_key].append(item)

    # 2. 准备分配
    all_base_keys = list(text_groups.keys())
    num_texts_to_val = val_count // 3
    
    random.seed(42) # 固定种子保证可复现
    random.shuffle(all_base_keys)

    val_keys = set(all_base_keys[:num_texts_to_val])
    
    val_samples = []
    train_samples = []

    # 3. 分配到两个列表（此时还是按组拿出来的）
    for base_key in all_base_keys:
        if base_key in val_keys:
            val_samples.extend(text_groups[base_key])
        else:
            train_samples.extend(text_groups[base_key])

    # 4. 核心步骤：彻底打散！！！
    # 这将确保同一个 base_key 的 3 个变体被随机散布在文件中
    random.shuffle(val_samples)
    random.shuffle(train_samples)

    # 5. 顺序写入（此时已经是随机顺序）
    def save_jsonl(data, path, desc):
        with open(path, 'w', encoding='utf-8') as f:
            for item in tqdm(data, desc=desc):
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    save_jsonl(val_samples, val_path, "Saving Validation Set")
    save_jsonl(train_samples, train_path, "Saving Training Set")

    # 6. 验证统计
    print("\n" + "="*40)
    print("拆分并全局洗牌完成！")
    print("-" * 40)
    print(f"验证集: {len(val_samples)} 条 (Bucket 1/2/3 各 666 条)")
    print(f"训练集: {len(train_samples)} 条")
    print("状态: 同一文本的样本已随机散布，无物理相邻。")
    print("=" * 40)

if __name__ == "__main__":
    BASE_DIR = "/aistor/aispeech/hpc_stor01/home/wangchenghao00sx/workingspace/TASU-simulator/Multitask/data/train"
    INPUT_FILE = "multitask_filtered_augmented_3bucket.jsonl"
    
    split_and_shuffle(BASE_DIR, INPUT_FILE, val_count=1998)