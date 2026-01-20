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
CHECKPOINT_PATH = "/aistor/sjtu/hpc_stor01/home/wangchenghao/workingspace/ps-slm/Multitask/exp/simulator_ar_control/checkpoints/pytorch_model.bin"
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
    def __init__(self, lines, use_reconstruction_mode=False):
        self.lines = lines
        self.use_reconstruction_mode = use_reconstruction_mode
    
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
            
            # [新增] 提取样本级 error_stats
            if self.use_reconstruction_mode:
                bpe = item.get('bpe_error', {})
                ref_len = bpe.get('RefLen', 1e-6)
                # 构造 [S_rate, D_rate, I_rate, WER]
                error_stats = torch.tensor([
                    bpe.get('S', 0) / ref_len,
                    bpe.get('D', 0) / ref_len,
                    bpe.get('I', 0) / ref_len,
                    bpe.get('WER', 0.0)
                ], dtype=torch.float32)
            else:
                error_stats = None

            return {
                "key": item['key'], 
                "text_onehot": self._restore_text_onehot(text_ids), 
                "text_ids": text_ids,
                "error_stats": error_stats,
                "item": item
            }
        except Exception as e:
            return None

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch: return None
    
    text_list = [b['text_onehot'] for b in batch]
    text_padded = pad_sequence(text_list, batch_first=True, padding_value=0.0)
    
    mask = torch.arange(text_padded.size(1)).expand(len(batch), -1) < torch.tensor([t.size(0) for t in text_list]).unsqueeze(1)
    
    # [新增] 处理 error_stats 的 batch 化
    error_stats_list = [b['error_stats'] for b in batch if b['error_stats'] is not None]
    batch_error_stats = torch.stack(error_stats_list) if error_stats_list else None

    return {
        "keys": [b['key'] for b in batch], 
        "text_onehot": text_padded, 
        "text_mask": mask.float(), 
        "text_ids": [b['text_ids'] for b in batch],
        "error_stats": batch_error_stats,
        "items": [b['item'] for b in batch]
    }

def save_sparse_tensor(tensor, path, text_ids):
    # 为节省空间，依然保存 top-k
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
    # 保持旧参数兼容，若非重构模式则使用它们
    parser.add_argument("--s_rate", type=float, default=0.0)
    parser.add_argument("--d_rate", type=float, default=0.0)
    parser.add_argument("--i_rate", type=float, default=0.0)
    parser.add_argument("--wer", type=float, default=0.0)
    # [新增] 模式切换
    parser.add_argument("--use_reconstruction_mode", action="store_true")
    args = parser.parse_args()

    device = torch.device(f"npu:{args.device_id}")
    torch.npu.set_device(device)

    tokenizer = SenseVoiceTokenizer(TOKENIZER_PATH)
    CONFIG.vocab_size = tokenizer.vocab_size
    model = CTCTransformerSimulator(CONFIG)
    
    # 加载模型
    sd = torch.load(CHECKPOINT_PATH, map_location='cpu')
    model.load_state_dict({k.replace("simulator.", "").replace("module.", ""): v for k, v in sd.items()})
    model.to(device).eval()

    # 默认全局控制张量
    global_cond = torch.tensor([args.s_rate, args.d_rate, args.i_rate, args.wer], device=device).unsqueeze(0)

    with open(args.input_jsonl, 'r', encoding='utf-8') as f:
        lines = f.readlines()[args.rank::args.world_size]
    
    dataset = InferenceDataset(lines, use_reconstruction_mode=args.use_reconstruction_mode)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=4)
    temp_jsonl = os.path.join(args.output_dir, f"temp_npu{args.device_id}.jsonl")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(temp_jsonl, 'w', encoding='utf-8') as f_out:
        for batch in tqdm(loader, desc=f"NPU {args.device_id}"):
            if batch is None: continue
            
            # [关键修改] 选择控制信号源
            if args.use_reconstruction_mode and batch['error_stats'] is not None:
                current_cond = batch['error_stats'].to(device)
            else:
                current_cond = global_cond.expand(len(batch['keys']), -1)

            with torch.no_grad():
                # 注意：自回归推理使用采样温度，拟合测试建议设低一点（如 0.1 或 0.5）看回归性
                probs_b = model.inference(
                    batch['text_onehot'].to(device), 
                    batch['text_mask'].to(device), 
                    error_stats=current_cond, 
                    max_len=160, 
                    temperature=0.5
                )
            
            for i, key in enumerate(batch['keys']):
                probs = probs_b[i]
                
                # EOS 截断逻辑
                pred_ids = probs.argmax(dim=-1)
                eos_indices = (pred_ids == 25055).nonzero(as_tuple=True)[0]
                if len(eos_indices) > 0:
                    valid_len = eos_indices[0].item()
                    probs = probs[:valid_len, :]
                
                # 即使 valid_len 为 0，也至少保留一帧或跳过
                if probs.size(0) == 0:
                    probs = probs_b[i][:1, :] 

                save_path = os.path.join(args.output_dir, f"{key}.pt")
                save_sparse_tensor(probs, save_path, batch['text_ids'][i])
                
                item = batch['items'][i].copy()
                item['sim_psd_path'] = save_path
                f_out.write(json.dumps(item, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    main()