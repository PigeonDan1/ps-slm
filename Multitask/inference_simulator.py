import os
import json
import torch
import torch_npu
from tqdm import tqdm
import numpy as np
import argparse
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from simulator.model import CTCTransformerSimulator
from simulator.config import SimulatorConfig
from model.tokenizer import SenseVoiceTokenizer

# ================= 配置 =================
CHECKPOINT_PATH = "/aistor/sjtu/hpc_stor01/home/wangchenghao/workingspace/TASU-simulator/Multitask/exp/simulator_ckpt/pytorch_model.bin"
TOKENIZER_PATH = "/aistor/sjtu/hpc_stor01/home/yangyi/model/SenseVoiceSmall"
BATCH_SIZE = 16 
TOP_K_SPARSE = 10 
VOCAB_SIZE_BASE = 25055 

CONFIG = SimulatorConfig(
    vocab_size=0,
    ctc_vocab_size=25055,
    d_model=512,
    n_head=8,
    num_encoder_layers=6,
    num_decoder_layers=6,
    dim_feedforward=2048,
    dropout=0.0,
    max_len=160
)

class InferenceDataset(Dataset):
    def __init__(self, lines, tokenizer, max_len=160):
        self.lines = lines
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self): return len(self.lines)

    def _restore_text_onehot(self, ids: np.ndarray):
        L = ids.shape[0]
        dense_text = torch.zeros((L, VOCAB_SIZE_BASE), dtype=torch.float32)
        if L > 0:
            t_ids = torch.from_numpy(ids.astype(np.int64)).unsqueeze(1)
            dense_text.scatter_(1, t_ids, 1.0)
        return dense_text

    def __getitem__(self, idx):
        try:
            item = json.loads(self.lines[idx])
            # [核心修复] 直接从 jsonl 获取文本，不再尝试加载不存在的 .pt 文件
            text = item.get('target', item.get('GT', ''))
            processed_text = text.lower().replace("'", "")
            
            # 获取 ID 序列
            raw_tokens = self.tokenizer.encode(processed_text)
            
            # [严格一致性] 过滤 0 和 25055，确保输入特征与训练时完全一致
            token_ids = [x for x in raw_tokens if x != 0 and x != 25055]
            text_ids = np.array(token_ids, dtype=np.int32)[:self.max_len]
            
            return {
                "key": item['key'], 
                "text_onehot": self._restore_text_onehot(text_ids), 
                "text_ids": text_ids,
                "item": item
            }
        except Exception as e:
            return None

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch: return None
    text_list = [b['text_onehot'] for b in batch]
    text_padded = pad_sequence(text_list, batch_first=True, padding_value=0.0)
    # 计算 mask
    lens = torch.tensor([t.size(0) for t in text_list])
    mask = torch.arange(text_padded.size(1)).expand(len(batch), -1) < lens.unsqueeze(1)
    return {
        "keys": [b['key'] for b in batch], 
        "text_onehot": text_padded, 
        "text_mask": mask.float(), 
        "text_ids": [b['text_ids'] for b in batch],
        "items": [b['item'] for b in batch]
    }

def save_sparse_tensor(tensor, path, text_ids):
    # [核心逻辑] 保持 Top-10 保存，确保 pt.py 可以正确恢复
    val, idx = torch.topk(tensor, TOP_K_SPARSE, dim=-1)
    torch.save({
        'psd_indices': idx.cpu().numpy().astype(np.int32),
        'psd_values': val.cpu().numpy().astype(np.float16),
        'shape': list(tensor.shape),
        'text_ids': text_ids.astype(np.int32) 
    }, path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--device_id", type=int, required=True)
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--bucket_id", type=int, default=1)
    args = parser.parse_args()

    device = torch.device(f"npu:{args.device_id}")
    torch.npu.set_device(device)

    tokenizer = SenseVoiceTokenizer(TOKENIZER_PATH)
    CONFIG.vocab_size = tokenizer.vocab_size
    model = CTCTransformerSimulator(CONFIG)
    
    # 加载权重
    sd = torch.load(CHECKPOINT_PATH, map_location='cpu')
    model.load_state_dict({k.replace("simulator.", "").replace("module.", ""): v for k, v in sd.items()})
    model.to(device).eval()

    with open(args.input_jsonl, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()
    lines = all_lines[args.rank::args.world_size]
    
    # 传入 tokenizer 修复加载逻辑
    dataset = InferenceDataset(lines, tokenizer, max_len=160)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=4)
    
    temp_jsonl = os.path.join(args.output_dir, f"temp_npu{args.device_id}.jsonl")
    os.makedirs(args.output_dir, exist_ok=True)

    with open(temp_jsonl, 'w', encoding='utf-8') as f_out:
        for batch in tqdm(loader, desc=f"NPU {args.device_id}"):
            if batch is None: continue
            
            with torch.no_grad():
                b_size = len(batch['keys'])
                bucket_id_batch = torch.full((b_size,), args.bucket_id, device=device, dtype=torch.long)
                
                probs_b = model.inference(
                    batch['text_onehot'].to(device), 
                    batch['text_mask'].to(device), 
                    bucket_id=bucket_id_batch, 
                    max_len=160, 
                    temperature=0.5
                )
            
            for i, key in enumerate(batch['keys']):
                probs = probs_b[i]
                
                # EOS 截断处理
                pred_ids = probs.argmax(dim=-1)
                eos_indices = (pred_ids == 25055).nonzero(as_tuple=True)[0]
                if len(eos_indices) > 0:
                    valid_len = eos_indices[0].item()
                    probs = probs[:valid_len, :]
                
                # 保存为 .pt，包含 text_ids 和 Top-10 PSD
                save_path = os.path.join(args.output_dir, f"{key}.pt")
                save_sparse_tensor(probs, save_path, batch['text_ids'][i])
                
                # 更新 jsonl 信息
                item = batch['items'][i].copy()
                item['sim_psd_path'] = save_path
                item['bucket_id'] = args.bucket_id
                f_out.write(json.dumps(item, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    main()