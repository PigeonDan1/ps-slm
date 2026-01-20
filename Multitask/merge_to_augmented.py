# 合并已经生成的5个温度数据的jsonl
import json
import os
import glob
from tqdm import tqdm

# ================= 配置 =================
# 这里对应 shell 脚本里定义的后缀
SUFFIXES = ["_temp_0.1", "_temp_0.5", "_temp_1.0", "_temp_1.5", "_temp_2.0"]
DATA_ROOT = "/aistor/sjtu/hpc_stor01/home/wangchenghao/workingspace/ps-slm/Multitask/data"
SPLITS = ["train", "dev"]

def get_real_dir_path(base_dir, suffix):
    """
    尝试匹配两种可能的文件夹命名格式，找到真实存在的那个
    """
    # 可能性 1: sim_ar__temp_0.1 (生成脚本实际产生的)
    path1 = f"{base_dir}_{suffix}"
    if os.path.exists(path1):
        return path1
        
    # 可能性 2: sim_ar_temp_0.1 (预期的)
    path2 = f"{base_dir}{suffix}"
    if os.path.exists(path2):
        return path2
        
    return None

def fix_filename_in_dir(dir_path):
    """
    在目录下查找 multitask_npu*.jsonl 并重命名为 multitask.jsonl
    """
    target_file = os.path.join(dir_path, "multitask.jsonl")
    
    # 如果已经修正过了，直接返回
    if os.path.exists(target_file):
        return target_file
        
    # 查找带 NPU 后缀的文件
    pattern = os.path.join(dir_path, "multitask_npu*.jsonl")
    found_files = glob.glob(pattern)
    
    if len(found_files) > 0:
        src_file = found_files[0]
        print(f"      [Renaming] {os.path.basename(src_file)} -> multitask.jsonl")
        os.rename(src_file, target_file)
        return target_file
    else:
        return None

def merge_split(split):
    print(f"\n--> Processing {split} set...")
    merged_data = {}
    
    base_sim_dir = os.path.join(DATA_ROOT, split, "sim_ar")
    
    for i, suffix in enumerate(SUFFIXES):
        # 1. 找到正确的文件夹
        dir_path = get_real_dir_path(base_sim_dir, suffix)
        
        if not dir_path:
            print(f"    [Error] Directory not found for suffix {suffix} (Epoch {i})")
            continue
            
        # 2. 修正该文件夹内的文件名
        jsonl_path = fix_filename_in_dir(dir_path)
        
        if not jsonl_path:
            print(f"    [Error] No jsonl file found in {dir_path}")
            continue
            
        # 3. 读取数据
        print(f"    Reading Epoch {i} from: {os.path.basename(dir_path)}")
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    key = item['key']
                    path = item['sim_psd_path']
                    
                    if key not in merged_data:
                        # 第一次遇到这个 key，复制元数据
                        new_item = item.copy()
                        del new_item['sim_psd_path']
                        # 初始化 5 个位置的列表
                        new_item['sim_psd_paths'] = [None] * len(SUFFIXES)
                        merged_data[key] = new_item
                    
                    # 将路径填入对应 Epoch 的位置
                    merged_data[key]['sim_psd_paths'][i] = path
                except Exception as e:
                    pass

    # 4. 保存为 multitask.jsonl
    # [关键修改] 文件名改为 multitask.jsonl，覆盖原始文件或作为新输入
    output_path = os.path.join(DATA_ROOT, split, "multitask.jsonl")
    
    print(f"    Writing merged file to: {output_path}")
    
    valid_count = 0
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for key, item in tqdm(merged_data.items()):
            # 只要有路径就写入
            if any(p is not None for p in item['sim_psd_paths']):
                f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
                valid_count += 1
                
    print(f"--> Done {split}! Total merged: {valid_count}")

if __name__ == "__main__":
    for split in SPLITS:
        merge_split(split)