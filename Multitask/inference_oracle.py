# NAR模型生成模拟数据（溺爱模式）
import os
import sys
import json
import torch
import torch_npu
import numpy as np
from tqdm import tqdm
import argparse
from torch.utils.data import Dataset, DataLoader

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from simulator.model import CTCTransformerSimulator
from simulator.config import SimulatorConfig


CHECKPOINT_PATH = "/aistor/sjtu/hpc_stor01/home/wangchenghao/workingspace/ps-slm/Multitask/exp/simulator_v2_nar_bf16_20251130-0950/exp/simulator_v2_nar_bf16_20251130-0950/ctc_simulator_nar_epoch_35_step_500/pytorch_model.bin"
BASE_WORK_DIR = "/aistor/sjtu/hpc_stor01/home/wangchenghao/workingspace/ps-slm/Multitask"
DATA_DIR = os.path.join(BASE_WORK_DIR, "data")

SPLITS = ["train", "dev"]
VOCAB_SIZE_BASE = 25055
MAX_LEN = 160
TOP_K = 10

class StandaloneDataset(Dataset):
    def __init__(self, jsonl_path, rank, world_size, max_len=160):
        self.data = []
        self.max_len = max_len
        
        if not os.path.exists(jsonl_path):
            return

        print(f"[Rank {rank}] Loading and slicing data from {jsonl_path}...")
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        # [核心] 手动切分数据：每 world_size 行取一行
        # 例如 8 卡：Rank 0 取 0, 8, 16...
        my_lines = lines[rank::world_size]
        
        for line in my_lines:
            try:
                item = json.loads(line.strip())
                if 'psd_path' in item:
                    self.data.append(item)
            except:
                continue
        print(f"[Rank {rank}] Assigned {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def _restore_text_onehot(self, ids: np.ndarray):
        L = ids.shape[0]
        dense_text = torch.zeros((L, VOCAB_SIZE_BASE), dtype=torch.float32)
        t_ids = torch.from_numpy(ids.astype(np.int64)).unsqueeze(1)
        dense_text.scatter_(1, t_ids, 1.0)
        return dense_text

    def __getitem__(self, idx):
        item = self.data[idx]
        pt_path = item['psd_path']
        
        try:
            data = torch.load(pt_path, map_location='cpu')
            text_ids = data['text_ids']
            real_psd_len = data['psd_values'].shape[0] # Real Len
            
            text_tensor = self._restore_text_onehot(text_ids) 
            
            L = text_tensor.size(0)
            if L < self.max_len:
                padding = torch.zeros((self.max_len - L, VOCAB_SIZE_BASE), dtype=torch.float32)
                text_padded = torch.cat([text_tensor, padding], dim=0)
                text_mask = torch.cat([torch.ones(L), torch.zeros(self.max_len - L)], dim=0)
            else:
                text_padded = text_tensor[:self.max_len]
                text_mask = torch.ones(self.max_len)

            return {
                "text_onehot": text_padded,
                "text_mask": text_mask,
                "real_len": real_psd_len,
                "raw_item": json.dumps(item, ensure_ascii=False)
            }
        except Exception as e:
            print(f"Error loading {pt_path}: {e}")
            return None

def oracle_collate(batch):
    batch = [x for x in batch if x is not None]
    if not batch: return {}
    return {
        "text_onehot": torch.stack([x['text_onehot'] for x in batch]),
        "text_mask": torch.stack([x['text_mask'] for x in batch]),
        "real_lens": torch.tensor([x['real_len'] for x in batch], dtype=torch.long),
        "raw_items": [x['raw_item'] for x in batch]
    }

# ================= 主逻辑 =================

def load_model(device):
    # Config 需要补全 vocab_size
    config = SimulatorConfig(
        vocab_size=25055, ctc_vocab_size=25055, d_model=512, n_head=8,
        num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048,
        dropout=0.0, max_len=MAX_LEN
    )
    model = CTCTransformerSimulator(config)
    
    # 静默加载，不打印太多日志
    state_dict = torch.load(CHECKPOINT_PATH, map_location='cpu')
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'): k = k[7:]
        if k.startswith('simulator.'): k = k[10:]
        new_state_dict[k] = v
        
    model.load_state_dict(new_state_dict, strict=True)
    model.to(device)
    model.eval()
    model.bfloat16() 
    return model

def save_compressed_pt(probs, save_path):
    topk_values, topk_indices = torch.topk(probs, TOP_K, dim=-1)
    vals_np = topk_values.cpu().to(torch.float16).numpy()
    idxs_np = topk_indices.cpu().to(torch.int16).numpy()
    torch.save({"psd_values": vals_np, "psd_indices": idxs_np}, save_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    args = parser.parse_args()
    
    rank = args.rank
    world_size = args.world_size
    
    # 1. 绑定设备
    device = torch.device(f"npu:{rank}")
    torch.npu.set_device(device)
    
    print(f"--> [Rank {rank}] Started on NPU {rank}")

    model = load_model(device)

    for split in SPLITS:
        jsonl_path = os.path.join(DATA_DIR, split, "multitask.jsonl")
        sim_dir = os.path.join(DATA_DIR, split, "sim")
        
        # 仅 Rank 0 创建目录，其他稍微等等
        if rank == 0:
            os.makedirs(sim_dir, exist_ok=True)
        # 简单的 sleep 代替 barrier，防止 io 冲突
        import time; time.sleep(2) 
        
        # 加载分片数据
        dataset = StandaloneDataset(jsonl_path, rank, world_size, max_len=MAX_LEN)
        
        dataloader = DataLoader(
            dataset, batch_size=64, shuffle=False, 
            collate_fn=oracle_collate, num_workers=0, pin_memory=True
        )
        
        temp_jsonl = os.path.join(DATA_DIR, split, f"multitask_sim_oracle_rank{rank}.jsonl")
        f_out = open(temp_jsonl, 'w', encoding='utf-8')
        
        # 只有 Rank 0 显示进度条，避免刷屏
        iterator = tqdm(dataloader, desc=f"Rank {rank} {split}", disable=(rank!=0))
        
        for batch in iterator:
            text_onehot = batch['text_onehot'].to(device).bfloat16()
            text_mask = batch['text_mask'].to(device).bfloat16()
            real_lens = batch['real_lens'].to(device)
            raw_items = batch['raw_items']
            
            # [溺爱模式] Oracle Length
            target_lengths = real_lens + 1
            
            with torch.no_grad():
                logits = model(text_onehot, text_mask, target_lengths=target_lengths)
                probs = torch.softmax(logits, dim=-1)
            
            for i, prob in enumerate(probs):
                valid_len = target_lengths[i].item()
                valid_prob = prob[:valid_len, :] 
                
                item = json.loads(raw_items[i])
                key = item['key']
                
                sim_filename = f"{key}_sim_oracle.pt"
                sim_path = os.path.join(sim_dir, sim_filename)
                
                save_compressed_pt(valid_prob, sim_path)
                
                item['psd_sim_path'] = sim_path
                item['psd_sim_len'] = valid_len
                item['mode'] = 'oracle'
                
                f_out.write(json.dumps(item, ensure_ascii=False) + "\n")
        
        f_out.close()
        print(f"--> [Rank {rank}] Finished {split}")

if __name__ == "__main__":
    main()