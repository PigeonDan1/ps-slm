import torch
from torch.utils.data import Dataset
from typing import Dict, Any, List
import json
import os
import numpy as np

# SenseVoice 基础词表大小 (不含 EOS)
VOCAB_SIZE_BASE = 25055 
# 扩展后的词表大小 (含 EOS)
VOCAB_SIZE_FULL = 25056
# EOS 在扩展词表中的索引
EOS_INDEX = 25055

class SimulatorDataset(Dataset):
    """
    加载预处理后的 .pt 数据，还原稀疏矩阵，添加 EOS，并执行固定长度 Padding。
    """
    def __init__(self, jsonl_path: str, max_len: int = 100):
        super().__init__()
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"JSONL file not found at: {jsonl_path}")

        self.max_len = max_len
        self.data = []
        
        # 读取 JSONL 元数据
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    # 必须包含预处理后的路径
                    if 'psd_path' in item:
                        self.data.append(item)
                except json.JSONDecodeError:
                    continue
        
        print(f"[Dataset] Loaded {len(self.data)} samples from {jsonl_path}")

    def __len__(self) -> int:
        return len(self.data)

    def _restore_psd_matrix(self, indices: np.ndarray, values: np.ndarray, original_len: int):
        """
        还原步骤：
        1. 稀疏 (Top10) -> 稠密 [T, 25055]
        2. 扩展维度 -> [T, 25056]
        3. 添加 EOS -> [T+1, 25056]
        """
        # 1. 还原基础稠密矩阵 [T, 25055]
        # indices, values shape: [T, 10]
        T = indices.shape[0]
        dense_base = torch.zeros((T, VOCAB_SIZE_BASE), dtype=torch.float32)
        
        # 使用 scatter 填充
        t_indices = torch.from_numpy(indices.astype(np.int64))
        t_values = torch.from_numpy(values.astype(np.float32))
        dense_base.scatter_(1, t_indices, t_values)
        
        # 2. 扩展维度到 25056 (最后一列为 EOS位，当前全是 0)
        # [T, 25055] -> [T, 25056]
        padding_col = torch.zeros((T, 1), dtype=torch.float32)
        dense_extended = torch.cat([dense_base, padding_col], dim=1)
        
        # 3. 构造 EOS 帧 [1, 25056] (前 25055 为 0, 最后一位为 1)
        eos_frame = torch.zeros((1, VOCAB_SIZE_FULL), dtype=torch.float32)
        eos_frame[0, EOS_INDEX] = 1.0
        
        # 4. 拼接
        # Result: [T+1, 25056]
        final_matrix = torch.cat([dense_extended, eos_frame], dim=0)
        
        return final_matrix

    def _restore_text_onehot(self, ids: np.ndarray):
        """
        还原步骤：
        1. ID 序列 -> One-hot [L, 25055] (文本不需要 EOS)
        """
        L = ids.shape[0]
        dense_text = torch.zeros((L, VOCAB_SIZE_BASE), dtype=torch.float32)
        
        # scatter 填充
        t_ids = torch.from_numpy(ids.astype(np.int64)).unsqueeze(1)
        dense_text.scatter_(1, t_ids, 1.0)
        
        return dense_text

    def _pad_or_truncate(self, tensor: torch.Tensor, target_len: int):
        """
        固定长度处理：截断或补 0
        Input: [L, D]
        Output: [target_len, D], valid_len
        """
        curr_len = tensor.size(0)
        dim = tensor.size(1)
        
        if curr_len >= target_len:
            # 截断
            return tensor[:target_len, :], target_len
        else:
            # Padding (补全 0)
            pad_len = target_len - curr_len
            padding = torch.zeros((pad_len, dim), dtype=torch.float32)
            padded_tensor = torch.cat([tensor, padding], dim=0)
            return padded_tensor, curr_len

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        pt_path = item['psd_path']
        
        try:
            # 1. 加载 .pt
            # map_location='cpu' 确保不占用 NPU 显存，数据在 Dataset 层面是 CPU 的
            data = torch.load(pt_path, map_location='cpu')
            
            # 2. 还原 PSD-CTC (Target)
            # 输出形状: [T_real + 1, 25056]
            psd_tensor = self._restore_psd_matrix(
                data['psd_indices'], 
                data['psd_values'], 
                item['psd_len']
            )
            
            # 3. 还原 Text One-hot (Input)
            # 输出形状: [L_text, 25055]
            text_tensor = self._restore_text_onehot(
                data['text_ids']
            )
            
            # 4. 固定长度处理 (Padding / Truncation)
            # 对齐到 self.max_len (100)
            
            # 处理 Audio Target
            padded_psd, valid_psd_len = self._pad_or_truncate(psd_tensor, self.max_len)
            
            # 处理 Text Input
            padded_text, valid_text_len = self._pad_or_truncate(text_tensor, self.max_len)
            
            return {
                "text_onehot": padded_text,      # [100, 25055]
                "text_len": valid_text_len,      # int
                "target_psd": padded_psd,        # [100, 25056]
                "target_len": valid_psd_len,     # int (含EOS)
                "key": item.get("key", str(idx))
            }
            
        except Exception as e:
            print(f"[Dataset Error] Failed to load {pt_path}: {e}")
            # 返回全 0 的 dummy 数据防止 Dataloader 崩溃
            # 注意：实际训练中最好过滤掉这些数据
            dummy_text = torch.zeros((self.max_len, VOCAB_SIZE_BASE), dtype=torch.float32)
            dummy_psd = torch.zeros((self.max_len, VOCAB_SIZE_FULL), dtype=torch.float32)
            return {
                "text_onehot": dummy_text,
                "text_len": 0,
                "target_psd": dummy_psd,
                "target_len": 0,
                "key": "error"
            }

class SimulatorCollate:
    """
    简单的 Collate，因为数据已经在 Dataset 里 Pad 到了固定长度。
    这里只需要 stack 并生成 mask。
    """
    def __init__(self, max_len: int = 100):
        self.max_len = max_len

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # 过滤加载失败的数据
        batch = [item for item in batch if item["key"] != "error"]
        if not batch:
            return {}

        # 1. Stack Tensors
        # text_onehot: [B, 100, 25055]
        text_onehot = torch.stack([item["text_onehot"] for item in batch])
        # target_psd:  [B, 100, 25056]
        target_psd = torch.stack([item["target_psd"] for item in batch])
        
        # Lengths
        text_lens = torch.tensor([item["text_len"] for item in batch], dtype=torch.long)
        target_lens = torch.tensor([item["target_len"] for item in batch], dtype=torch.long)
        
        # 2. 生成 Mask (1 for Valid, 0 for Padding)
        # Shape: [B, 100]
        # range: [0, 1, ..., 99]
        seq_range = torch.arange(self.max_len).unsqueeze(0).expand(len(batch), -1)
        
        text_mask = (seq_range < text_lens.unsqueeze(1)).float()
        target_mask = (seq_range < target_lens.unsqueeze(1)).float()

        return {
            "text_onehot": text_onehot,
            "text_mask": text_mask,
            "target_psd": target_psd,
            "target_lengths": target_lens, # 给 LengthRegulator 用
            "loss_mask": target_mask,      # 给 Loss 计算用
            "keys": [item["key"] for item in batch]
        }