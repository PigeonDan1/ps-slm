#删除jsonl中所有的增强后.pt文件
import json
import os
from tqdm import tqdm

def cleanup_augmented_files(jsonl_path):
    if not os.path.exists(jsonl_path):
        print(f"错误: 找不到文件 {jsonl_path}")
        return

    # 统计信息
    total_found = 0
    deleted_count = 0
    failed_count = 0

    # 首先读取所有行以确定进度条总长度
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"--- 开始扫描并删除文件 ---")
    
    for line in tqdm(lines, desc="清理进度", unit="行"):
        try:
            item = json.loads(line.strip())
            key = item.get('key', '')
            sim_psd_path = item.get('sim_psd_path', '')

            # 判断条件：Key 包含 _aug_ 字符串，且 sim_psd_path 存在
            if sim_psd_path:
                total_found += 1
                if os.path.exists(sim_psd_path):
                    try:
                        os.remove(sim_psd_path)
                        deleted_count += 1
                    except Exception as e:
                        print(f"\n[错误] 无法删除 {sim_psd_path}: {e}")
                        failed_count += 1
        except json.JSONDecodeError:
            continue

    print(f"\n{'='*40}")
    print(f"清理完成!")
    print(f"发现样本: {total_found}")
    print(f"成功删除文件: {deleted_count}")
    print(f"失败/已不存在: {failed_count}")
    print(f"{'='*40}")

if __name__ == "__main__":
    # 请确认该路径为您需要清理的训练集 jsonl
    target_jsonl = "/aistor/aispeech/hpc_stor01/home/wangchenghao00sx/workingspace/TASU-simulator/Multitask/data/medical/simulator_B2/to_delete.jsonl"
    cleanup_augmented_files(target_jsonl)