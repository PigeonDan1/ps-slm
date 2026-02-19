import json
import os
import random
from collections import defaultdict
from tqdm import tqdm

def clean_three_buckets(input_path, output_path):
    text_groups = defaultdict(list)
    if not os.path.exists(input_path):
        print(f"Error: 找不到文件 {input_path}")
        return

    # 1. 读入并按基础 ID 分组
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="读取数据"):
            item = json.loads(line)
            # 假设 key 格式为 "ID_aug_H"，提取 ID
            base_key = item['key'].split('_aug_')[0]
            text_groups[base_key].append(item)

    # 遵循最小改动，保留 4 变体过滤
    valid_groups = {k: sorted(v, key=lambda x: x['bpe_error']['WER']) 
                    for k, v in text_groups.items() if len(v) == 4}

    # 2. 提取并分配 (强制指定区间)
    # B1: [0, 0.06) | B2: [0.1, 0.4) | B3: [0.45, 1.5)
    final_data = []
    
    for base_key, items in tqdm(valid_groups.items(), desc="逻辑过滤"):
        wers = [x['bpe_error']['WER'] for x in items]
        
        # 根据强制区间寻找满足条件的索引
        idx_b1 = [i for i, v in enumerate(wers) if 0.0 <= v < 0.06]
        idx_b2 = [i for i, v in enumerate(wers) if 0.1 <= v < 0.4]
        idx_b3 = [i for i, v in enumerate(wers) if 0.45 <= v < 1.5]

        # 必须三个区间都至少有一个变体落入
        if idx_b1 and idx_b2 and idx_b3:
            # 每个文本在每个 Bucket 中只取一个（默认取在该区间内 WER 最低的）
            b1_item = items[idx_b1[0]]
            b2_item = items[idx_b2[0]]
            b3_item = items[idx_b3[0]]
            
            b1_item['bucket_id'] = 1
            b2_item['bucket_id'] = 2
            b3_item['bucket_id'] = 3
            
            final_data.extend([b1_item, b2_item, b3_item])

    # 3. 全局洗牌 (防止同一文本变体相邻)
    random.seed(42)
    random.shuffle(final_data)

    # 4. 写入
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in tqdm(final_data, desc="写入数据"):
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    total_groups = len(final_data) // 3
    print("\n" + "="*40)
    print(f"清洗完成！")
    print(f"最终每个 Bucket 数量均为: {total_groups}")
    print(f"区间依据: B1:[0, 0.06) | B2:[0.1, 0.4) | B3:[0.45, 1.5)")
    print("="*40)

if __name__ == "__main__":
    BASE_DIR = "/aistor/aispeech/hpc_stor01/home/wangchenghao00sx/workingspace/TASU-simulator/Multitask/data/train"
    IN = os.path.join(BASE_DIR, "multitask_filtered_augmented_exist.jsonl")
    OUT = os.path.join(BASE_DIR, "multitask_filtered_augmented_3bucket.jsonl")
    clean_three_buckets(IN, OUT)