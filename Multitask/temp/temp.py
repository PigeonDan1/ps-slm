import torch
import numpy as np
import os

# 配置与 dataset.py 保持一致
VOCAB_SIZE_BASE = 25055 
VOCAB_SIZE_FULL = 25056
EOS_INDEX = 25055

def verify_restore(pt_path):
    print(f"--- 正在检查文件: {os.path.basename(pt_path)} ---")
    
    # 1. 加载数据
    data = torch.load(pt_path, map_location='cpu')
    
    indices = data['psd_indices']  # numpy array (flattened)
    values = data['psd_values']    # numpy array (flattened)
    original_shape = data['shape'] # [T, 25055]
    
    print(f"平铺后的索引长度: {indices.shape[0]}")
    print(f"保存的原始形状 (T, V): {original_shape}")
    
    # 2. 核心恢复逻辑：必须利用 shape 中的 T 来 reshape
    T = original_shape[0]
    V = original_shape[1] # 应该是 25055
    
    try:
        # 将 1D 数组还原回 [T, 10]
        # 因为我们存的是 Top-10，所以每一行对应 10 个元素
        t_indices = torch.from_numpy(indices.astype(np.int64)).reshape(T, -1)
        t_values = torch.from_numpy(values.astype(np.float32)).reshape(T, -1)
        
        print(f"成功恢复索引形状: {t_indices.shape}") # 应该是 [T, 10]
        
        # 3. 构建稠密矩阵
        dense_base = torch.zeros((T, VOCAB_SIZE_BASE), dtype=torch.float32)
        
        # 使用 scatter_ 填充。此时 t_indices 是 2D，dense_base 也是 2D，维度匹配
        dense_base.scatter_(1, t_indices, t_values)
        
        print(f"稠密矩阵构建成功，形状: {dense_base.shape}")
        
        # 4. 验证数据正确性
        # 每一行的概率和应该接近 top-k 的能量总和
        row_sum_max = dense_base.sum(dim=1).max().item()
        print(f"单行最大概率和: {row_sum_max:.4f} (Top-10 能量和)")
        
        # 5. 模拟 dataset.py 的后续操作（加 EOS）
        padding_col = torch.zeros((T, 1), dtype=torch.float32)
        dense_extended = torch.cat([dense_base, padding_col], dim=1)
        eos_frame = torch.zeros((1, VOCAB_SIZE_FULL), dtype=torch.float32)
        eos_frame[0, EOS_INDEX] = 1.0
        final_matrix = torch.cat([dense_extended, eos_frame], dim=0)
        
        print(f"最终输出矩阵形状 (含 EOS): {final_matrix.shape}") # 应该是 [T+1, 25056]
        print("恢复逻辑验证通过！\n")
        
    except Exception as e:
        print(f"恢复失败！错误原因: {e}")

if __name__ == "__main__":
    # 请替换为你报错的具体路径
    sample_path = "/aistor/aispeech/hpc_stor01/home/wangchenghao00sx/workingspace/TASU-simulator/Multitask/data/train/augmented_psd_files/LibriSpeech-other-naven-6549-71114-0024-F-180513_aug_L.pt"
    
    if os.path.exists(sample_path):
        verify_restore(sample_path)
    else:
        print(f"未找到文件: {sample_path}")