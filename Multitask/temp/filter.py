#筛选出WER低于0.05的jsonl行
import json
import os
from tqdm import tqdm

def filter_multitask_jsonl(input_path):
    output_path = os.path.join(os.path.dirname(input_path), "multitask_filtered.jsonl")
    
    # 获取总行数以初始化进度条
    with open(input_path, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)

    original_count = 0
    filtered_count = 0

    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        for line in tqdm(f_in, total=total_lines, desc="Filtering JSONL"):
            original_count += 1
            data = json.loads(line)
            
            # 核心逻辑：筛选 $WER \leq 0.05$ 并增加 bucket_id
            if data.get("bpe_error", {}).get("WER", 1.0) <= 0.05:
                data["bucket_id"] = 1
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                filtered_count += 1
    
    print(f"\n--- 处理统计 ---")
    print(f"原本总样本数: {original_count}")
    print(f"已复制样本数 ($WER \leq 0.05$): {filtered_count}")
    print(f"新文件路径: {output_path}")

if __name__ == "__main__":
    filter_multitask_jsonl("data/dev/multitask.jsonl")