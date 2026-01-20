#使用DTW算法，计算CE，KL，Acc，Prob diff
#专用于测试模拟数据质量（对比real PSD_CTC）
import json
import os
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from scipy.spatial.distance import cdist

INPUT_JSONL = "/aistor/sjtu/hpc_stor01/home/wangchenghao/workingspace/ps-slm/Multitask/data/dev/multitask.jsonl"

TEMP_INDEX = 2 # 对应temperature为1
VOCAB_SIZE_REAL = 25055 
EPS = 1e-8 

def restore_dense(pt_path):
    """从 .pt 还原为稠密概率矩阵 [T, V]"""
    try:
        data = torch.load(pt_path, map_location='cpu')
        if 'psd_indices' not in data or 'psd_values' not in data:
            return None
            
        indices = data['psd_indices'].astype(np.int64)
        values = data['psd_values'].astype(np.float32)
        T = indices.shape[0]
        
        dense = torch.zeros(T, VOCAB_SIZE_REAL, dtype=torch.float32)
        indices = np.clip(indices, 0, VOCAB_SIZE_REAL - 1)
        dense.scatter_(1, torch.from_numpy(indices), torch.from_numpy(values))
        
        # 归一化
        row_sums = dense.sum(dim=1, keepdim=True) + EPS
        dense = dense / row_sums
        
        return dense
    except Exception as e:
        return None

def compute_dtw_path(seq_a, seq_b):
    """计算 DTW 路径 (欧氏距离)"""
    dist_mat = cdist(seq_a, seq_b, metric='euclidean')
    n, m = dist_mat.shape

    acc_cost = np.zeros((n + 1, m + 1))
    acc_cost[0, 0] = 0
    acc_cost[1:, 0] = np.inf
    acc_cost[0, 1:] = np.inf
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = dist_mat[i-1, j-1]
            acc_cost[i, j] = cost + min(
                acc_cost[i-1, j],   # Insertion
                acc_cost[i, j-1],   # Deletion
                acc_cost[i-1, j-1]  # Match
            )

    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i-1, j-1))
        curr_cost = acc_cost[i, j]
        if i > 0 and j > 0 and np.isclose(curr_cost, acc_cost[i-1, j-1] + dist_mat[i-1, j-1]):
            i -= 1; j -= 1
        elif i > 0 and np.isclose(curr_cost, acc_cost[i-1, j] + dist_mat[i-1, j-1]):
            i -= 1
        else:
            j -= 1
    
    path.reverse()
    return path

def calculate_metrics_aligned(real_tensor, sim_tensor):
    """计算指标"""
    real_np = real_tensor.numpy()
    sim_np = sim_tensor.numpy()

    path = compute_dtw_path(real_np, sim_np)
    
    real_indices = [p[0] for p in path]
    sim_indices = [p[1] for p in path]
    
    aligned_real = real_tensor[real_indices] # [K, V]
    aligned_sim = sim_tensor[sim_indices]    # [K, V]
    
    aligned_sim_safe = aligned_sim + EPS
    aligned_real_safe = aligned_real + EPS
    
    ce = -(aligned_real * torch.log(aligned_sim_safe)).sum(dim=-1).mean().item()
    kl = (aligned_real * (torch.log(aligned_real_safe) - torch.log(aligned_sim_safe))).sum(dim=-1).mean().item()

    real_ids = aligned_real.argmax(dim=-1)
    sim_ids = aligned_sim.argmax(dim=-1)
    acc = (real_ids == sim_ids).float().mean().item()

    max_real_probs = aligned_real.max(dim=-1).values # [K]
    max_sim_probs = aligned_sim.max(dim=-1).values   # [K]
    
    prob_diff = torch.abs(max_real_probs - max_sim_probs).mean().item()
    
    return {
        "CE": ce,
        "KL": kl,
        "ACC": acc,
        "ProbDiff": prob_diff
    }

def main():
    if not os.path.exists(INPUT_JSONL):
        print("Input file not found.")
        return

    print(f"Reading {INPUT_JSONL}...")
    with open(INPUT_JSONL, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    metrics_sum = {"CE": 0.0, "KL": 0.0, "ACC": 0.0, "ProbDiff": 0.0}
    count = 0
    
    print(f"Processing {len(lines)} samples (Target Temp Index: {TEMP_INDEX})...")
    
    for line in tqdm(lines):
        item = json.loads(line)
        
        real_path = item.get('psd_path')
        sim_paths = item.get('sim_psd_paths')
        
        if not real_path or not sim_paths or not isinstance(sim_paths, list):
            continue
            
        sim_path = sim_paths[TEMP_INDEX]
        
        if not os.path.exists(real_path) or not os.path.exists(sim_path):
            continue
            
        real_dense = restore_dense(real_path)
        sim_dense = restore_dense(sim_path)
        
        if real_dense is None or sim_dense is None:
            continue
            
        m = calculate_metrics_aligned(real_dense, sim_dense)
        
        for k in metrics_sum:
            metrics_sum[k] += m[k]
        count += 1
            
    if count > 0:
        print(f"Evaluation Results (Temp Index {TEMP_INDEX})")
        print(f"Sample Count: {count}")
        print(f"CE       : {metrics_sum['CE']/count:.4f}")
        print(f"KL       : {metrics_sum['KL']/count:.4f}")
        print(f"ACC      : {metrics_sum['ACC']/count:.4f}")
        print(f"Prob Diff: {metrics_sum['ProbDiff']/count:.4f}")
    else:
        print("No valid samples processed.")

if __name__ == "__main__":
    main()