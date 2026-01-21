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
CHECKPOINT_PATH = "/aistor/sjtu/hpc_stor01/home/wangchenghao/workingspace/ps-slm/Multitask/exp/simulator_ar_control_feedback_fromAR_20260112-1815/checkpoints/step_14287/pytorch_model.bin/pytorch_model.bin"
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
    def __init__(self, lines):
        self.lines = lines
    
    def __len__(self): return len(self.lines)

    def _restore_text_onehot(self, ids: np.ndarray):
        L = ids.shape[0]
        dense_text = torch.zeros((L, VOCAB_SIZE_BASE), dtype=torch.float32)
        t_ids = torch.from_numpy(ids.astype(np.int64)).unsqueeze(1)
        dense_text.scatter_(1, t_ids, 1.0)
        return dense_text

    def __getitem__(self, idx):
        try:
            item = json.loads(self.lines[idx])
            data = torch.load(item['psd_path'], map_location='cpu')
            text_ids = data['text_ids'][:160]
            return {
                "key": item['key'], 
                "text_onehot": self._restore_text_onehot(text_ids), 
                "text_ids": text_ids, # [新增] 保持原始 ID
                "item": item
            }
        except: return None

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch: return None
    text_list = [b['text_onehot'] for b in batch]
    text_padded = pad_sequence(text_list, batch_first=True, padding_value=0.0)
    mask = torch.arange(text_padded.size(1)).expand(len(batch), -1) < torch.tensor([t.size(0) for t in text_list]).unsqueeze(1)
    return {
        "keys": [b['key'] for b in batch], 
        "text_onehot": text_padded, 
        "text_mask": mask.float(), 
        "text_ids": [b['text_ids'] for b in batch], # [新增]
        "items": [b['item'] for b in batch]
    }

def save_sparse_tensor(tensor, path, text_ids):
    val, idx = torch.topk(tensor, TOP_K_SPARSE, dim=-1)
    torch.save({
        'psd_indices': idx.cpu().numpy().astype(np.int32),
        'psd_values': val.cpu().numpy().astype(np.float16),
        'shape': list(tensor.shape),
        'text_ids': text_ids.astype(np.int32) # [关键修复] 将文本 ID 存入新文件
    }, path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--device_id", type=int, required=True)
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--s_rate", type=float, default=0.0)
    parser.add_argument("--d_rate", type=float, default=0.0)
    parser.add_argument("--i_rate", type=float, default=0.0)
    parser.add_argument("--wer", type=float, default=0.0)
    args = parser.parse_args()

    device = torch.device(f"npu:{args.device_id}")
    torch.npu.set_device(device)

    tokenizer = SenseVoiceTokenizer(TOKENIZER_PATH)
    CONFIG.vocab_size = tokenizer.vocab_size
    model = CTCTransformerSimulator(CONFIG)
    
    sd = torch.load(CHECKPOINT_PATH, map_location='cpu')
    model.load_state_dict({k.replace("simulator.", "").replace("module.", ""): v for k, v in sd.items()})
    model.to(device).eval()

    cond = torch.tensor([args.s_rate, args.d_rate, args.i_rate, args.wer], device=device).unsqueeze(0)

    with open(args.input_jsonl, 'r', encoding='utf-8') as f:
        lines = f.readlines()[args.rank::args.world_size]
    
    loader = DataLoader(InferenceDataset(lines), batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=4)
    temp_jsonl = os.path.join(args.output_dir, f"temp_npu{args.device_id}.jsonl")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(temp_jsonl, 'w', encoding='utf-8') as f_out:
        for batch in tqdm(loader, desc=f"NPU {args.device_id}"):
            if batch is None: continue
            with torch.no_grad():
                probs_b = model.inference(batch['text_onehot'].to(device), batch['text_mask'].to(device), 
                                         error_stats=cond.expand(len(batch['keys']), -1), 
                                         max_len=160, temperature=0.5)
            
            for i, key in enumerate(batch['keys']):
                probs = probs_b[i]
                # [EOS 处理] 找到第一个 EOS 位置并截断（不保留 EOS 帧）
                pred_ids = probs.argmax(dim=-1)
                eos_indices = (pred_ids == 25055).nonzero(as_tuple=True)[0]
                if len(eos_indices) > 0:
                    valid_len = eos_indices[0].item()
                    probs = probs[:valid_len, :]
                
                save_path = os.path.join(args.output_dir, f"{key}.pt")
                # [修正调用] 传入 text_ids
                save_sparse_tensor(probs, save_path, batch['text_ids'][i])
                
                item = batch['items'][i].copy()
                item['sim_psd_path'] = save_path
                f_out.write(json.dumps(item, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    main()