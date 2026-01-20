import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
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
    加载预处理后的 .pt 数据，还原稀疏矩阵，添加 EOS。
    """
    def __init__(self, jsonl_path: str, max_len: int = 160):
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
                    if 'psd_path' in item:
                        self.data.append(item)
                except json.JSONDecodeError:
                    continue
        
        print(f"[Dataset] Loaded {len(self.data)} samples from {jsonl_path}")

    def __len__(self) -> int:
        return len(self.data)

    def _restore_psd_matrix(self, indices: np.ndarray, values: np.ndarray):
        T = indices.shape[0]
        dense_base = torch.zeros((T, VOCAB_SIZE_BASE), dtype=torch.float32)
        t_indices = torch.from_numpy(indices.astype(np.int64))
        t_values = torch.from_numpy(values.astype(np.float32))
        dense_base.scatter_(1, t_indices, t_values)
        padding_col = torch.zeros((T, 1), dtype=torch.float32)
        dense_extended = torch.cat([dense_base, padding_col], dim=1)
        eos_frame = torch.zeros((1, VOCAB_SIZE_FULL), dtype=torch.float32)
        eos_frame[0, EOS_INDEX] = 1.0
        final_matrix = torch.cat([dense_extended, eos_frame], dim=0)
        return final_matrix

    def _restore_text_onehot(self, ids: np.ndarray):
        L = ids.shape[0]
        dense_text = torch.zeros((L, VOCAB_SIZE_BASE), dtype=torch.float32)
        t_ids = torch.from_numpy(ids.astype(np.int64)).unsqueeze(1)
        dense_text.scatter_(1, t_ids, 1.0)
        return dense_text

    def _normalize_sdi_stats(self, bpe_error: Dict) -> torch.Tensor:
        if not bpe_error:
            return torch.zeros(4, dtype=torch.float32)
            
        ref_len = bpe_error.get("RefLen", 0)
        if ref_len <= 0: ref_len = 1.0 
            
        s = bpe_error.get("S", 0)
        d = bpe_error.get("D", 0)
        i = bpe_error.get("I", 0)
        wer = bpe_error.get("WER", 0.0)
        
        s_rate = s / ref_len
        d_rate = d / ref_len
        i_rate = i / ref_len
        
        return torch.tensor([s_rate, d_rate, i_rate, wer], dtype=torch.float32)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        pt_path = item['psd_path']
        
        try:
            data = torch.load(pt_path, map_location='cpu')
            psd_tensor = self._restore_psd_matrix(data['psd_indices'], data['psd_values'])
            
            # [功能保持] 使用 data['text_ids'] 生成 onehot
            text_tensor = self._restore_text_onehot(data['text_ids'])
            
            # [核心新增] 获取原始 text_ids 并进行长度截断，确保与 onehot 对齐
            raw_text_ids = data['text_ids']
            if raw_text_ids.shape[0] > self.max_len:
                raw_text_ids = raw_text_ids[:self.max_len]
            # 转为 LongTensor 以供后续编辑距离计算
            text_ids_tensor = torch.from_numpy(raw_text_ids.astype(np.int64))
            
            error_stats = self._normalize_sdi_stats(item.get("bpe_error", None))
            
            if psd_tensor.size(0) > self.max_len:
                psd_tensor = psd_tensor[:self.max_len, :]
            
            if text_tensor.size(0) > self.max_len:
                text_tensor = text_tensor[:self.max_len, :]
            
            return {
                "text_onehot": text_tensor,      
                "text_ids": text_ids_tensor,     # [新增] 返回原始 ID 序列
                "target_psd": psd_tensor,        
                "error_stats": error_stats,      
                "key": item.get("key", str(idx))
            }
            
        except Exception as e:
            print(f"[Dataset Error] Failed to load {pt_path}: {e}")
            return {
                "text_onehot": torch.zeros((1, VOCAB_SIZE_BASE), dtype=torch.float32),
                "text_ids": torch.zeros(1, dtype=torch.long), # [新增] Error case 占位
                "target_psd": torch.zeros((1, VOCAB_SIZE_FULL), dtype=torch.float32),
                "error_stats": torch.zeros(4, dtype=torch.float32), 
                "key": "error"
            }

class SimulatorCollate:
    def __init__(self, max_len: int = 160):
        self.max_len = max_len 

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = [item for item in batch if item["key"] != "error"]
        if not batch: return {}

        text_list = [item["text_onehot"] for item in batch]
        # [核心新增] 提取 text_ids 列表
        text_ids_list = [item["text_ids"] for item in batch]
        
        target_list = [item["target_psd"] for item in batch]
        error_stats_list = [item["error_stats"] for item in batch]
        
        text_lens = torch.tensor([t.size(0) for t in text_list], dtype=torch.long)
        target_lens = torch.tensor([t.size(0) for t in target_list], dtype=torch.long)
        
        text_padded = pad_sequence(text_list, batch_first=True, padding_value=0.0)
        
        # [核心新增] 对 text_ids 进行 Padding
        # 使用 25055 (EOS) 作为填充值，这在 pt.py 的比对逻辑中会被自动过滤，不影响计算
        text_ids_padded = pad_sequence(text_ids_list, batch_first=True, padding_value=25055)
        
        target_padded = pad_sequence(target_list, batch_first=True, padding_value=0.0)
        error_stats_batch = torch.stack(error_stats_list)
        
        B_text, L_text, _ = text_padded.size()
        text_seq_range = torch.arange(L_text).unsqueeze(0).expand(B_text, -1)
        text_mask = (text_seq_range < text_lens.unsqueeze(1)).float()
        
        B_tgt, L_tgt, _ = target_padded.size()
        tgt_seq_range = torch.arange(L_tgt).unsqueeze(0).expand(B_tgt, -1)
        loss_mask = (tgt_seq_range < target_lens.unsqueeze(1)).float()

        return {
            "text_onehot": text_padded,
            "text_ids": text_ids_padded, # [新增] 将 Padding 后的 ID 序列存入 batch
            "text_mask": text_mask,
            "target_psd": target_padded,
            "target_lengths": target_lens,
            "loss_mask": loss_mask,
            "error_stats": error_stats_batch,
            "keys": [item["key"] for item in batch]
        }