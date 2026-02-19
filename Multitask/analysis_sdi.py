import json
import os
import torch
import numpy as np
from tqdm import tqdm

# ================= 配置 =================
# 您的数据根目录
DATA_ROOT = "/aistor/sjtu/hpc_stor01/home/wangchenghao/workingspace/ps-slm/Multitask/data"

DATASETS = {
    "test-clean": {
        "input": os.path.join(DATA_ROOT, "test-clean", "multitask.jsonl"),
        "output": os.path.join(DATA_ROOT, "test-clean", "multitask_sdi.jsonl")
    },
    "test-other": {
        "input": os.path.join(DATA_ROOT, "test-other", "multitask.jsonl"),
        "output": os.path.join(DATA_ROOT, "test-other", "multitask_sdi.jsonl")
    }
}

VOCAB_SIZE_REAL = 25055 

def restore_sequence_clean(pt_path):
    """
    从 .pt 还原 BPE 序列，并严格过滤 Blank (0)
    """
    try:
        data = torch.load(pt_path, map_location='cpu')
        
        # 1. 获取 GT IDs
        if 'text_ids' in data:
            gt_ids = data['text_ids'].tolist() if isinstance(data['text_ids'], np.ndarray) else data['text_ids']
        else:
            return None, None

        # 2. 获取 Pred IDs
        if 'psd_indices' not in data:
            return None, None

        indices = data['psd_indices'].astype(np.int64)
        values = data['psd_values'].astype(np.float32)
        T = indices.shape[0]
        
        dense = torch.zeros(T, VOCAB_SIZE_REAL, dtype=torch.float32)
        # 防止越界
        indices = np.clip(indices, 0, VOCAB_SIZE_REAL - 1)
        dense.scatter_(1, torch.from_numpy(indices), torch.from_numpy(values))
        
        raw_pred_ids = dense.argmax(dim=-1).tolist()
        
        # [核心] 过滤 Blank Token (0)
        # 这是计算正确 SDI 的关键前提
        pred_ids_clean = [x for x in raw_pred_ids if x != 0]
        
        return gt_ids, pred_ids_clean
    except Exception as e:
        # print(f"Error loading {pt_path}: {e}")
        return None, None

def compute_sdi_dp(ref, hyp):
    """
    动态规划计算 BPE 级别的 S, D, I
    """
    len_r, len_h = len(ref), len(hyp)
    
    # Cost Matrix: Rows=Ref, Cols=Hyp (int32 节省内存)
    dp = np.zeros((len_r + 1, len_h + 1), dtype=np.int32)
    
    # 初始化
    for i in range(len_r + 1): dp[i][0] = i
    for j in range(len_h + 1): dp[0][j] = j
        
    # 填充
    for i in range(1, len_r + 1):
        for j in range(1, len_h + 1):
            cost = 0 if ref[i-1] == hyp[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,     # Deletion
                dp[i][j-1] + 1,     # Insertion
                dp[i-1][j-1] + cost # Match/Substitution
            )
            
    # 回溯 (Backtrace)
    s, d, ins = 0, 0, 0
    i, j = len_r, len_h
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i-1] == hyp[j-1]:
            i -= 1; j -= 1 # Match
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            s += 1; i -= 1; j -= 1 # Substitution
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            d += 1; i -= 1 # Deletion
        elif j > 0 and dp[i][j] == dp[i][j-1] + 1:
            ins += 1; j -= 1 # Insertion
        else:
            i -= 1; j -= 1 # 兜底
            
    return s, d, ins

def process_split(split_name, config):
    input_path = config["input"]
    output_path = config["output"]
    
    print(f"\n{'='*30} Processing {split_name} Set {'='*30}")
    print(f"Input : {input_path}")
    print(f"Output: {output_path}")
    
    if not os.path.exists(input_path):
        print(f"Error: Input file not found!")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    print(f"Total lines: {len(lines)}")
    valid_count = 0
    
    # [新增] 整体统计变量
    global_stats = {
        "S": 0, "D": 0, "I": 0, "RefLen": 0
    }
    
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for line in tqdm(lines):
            try:
                item = json.loads(line)
                pt_path = item.get('psd_path')
                
                sdi_result = None
                
                # 只有当 psd_path 存在时才计算
                if pt_path and os.path.exists(pt_path):
                    # 1. 还原 BPE 序列
                    gt_bpe, pred_bpe = restore_sequence_clean(pt_path)
                    
                    if gt_bpe is not None and pred_bpe is not None:
                        # 2. 计算 SDI
                        s, d, i = compute_sdi_dp(gt_bpe, pred_bpe)
                        ref_len = len(gt_bpe)
                        
                        # 单条 WER
                        if ref_len > 0:
                            wer = (s + d + i) / ref_len
                        else:
                            wer = 0.0 if len(pred_bpe) == 0 else 1.0
                            
                        sdi_result = {
                            "S": s,
                            "D": d,
                            "I": i,
                            "WER": round(wer, 4), # 保留4位小数
                            "RefLen": ref_len
                        }
                        
                        # [新增] 更新整体统计
                        global_stats["S"] += s
                        global_stats["D"] += d
                        global_stats["I"] += i
                        global_stats["RefLen"] += ref_len
                        
                        valid_count += 1
                
                # 3. 写入新字段
                item['bpe_error'] = sdi_result
                
                # 4. 写入新文件
                f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
                
            except Exception as e:
                # 遇到解析错误，原样写入（防止丢数据）
                f_out.write(line)
                
    print(f"--> Done! Successfully computed SDI for {valid_count} samples.")
    
    # [新增] 打印整体统计报告
    if global_stats["RefLen"] > 0:
        total_err = global_stats["S"] + global_stats["D"] + global_stats["I"]
        ref_len = global_stats["RefLen"]
        
        print("\n" + "-"*40)
        print(f"Overall BPE WER Report ({split_name})")
        print("-"*40)
        print(f"Total Ref Tokens : {ref_len}")
        print(f"Substitution     : {global_stats['S']} ({global_stats['S']/ref_len:.2%})")
        print(f"Deletion         : {global_stats['D']} ({global_stats['D']/ref_len:.2%})")
        print(f"Insertion        : {global_stats['I']} ({global_stats['I']/ref_len:.2%})")
        print(f"Overall WER      : {total_err/ref_len:.2%}")
        print("-"*40 + "\n")
    else:
        print("\n[Warn] No valid reference tokens found for overall statistics.\n")

if __name__ == "__main__":
    for split, cfg in DATASETS.items():
        process_split(split, cfg)