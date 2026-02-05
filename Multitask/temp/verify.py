#看看train是否遵循bucket_id
import os
import json
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import argparse
import torch_npu

from simulator.model import CTCTransformerSimulator
from simulator.config import SimulatorConfig
from model.tokenizer import SenseVoiceTokenizer

# ================= 配置区 =================
# 请根据实际情况修改 CKPT 路径
CHECKPOINT_PATH = "/aistor/aispeech/hpc_stor01/home/wangchenghao00sx/workingspace/TASU-simulator/Multitask/exp/simulator_ar_control_augmentation_20260203-1921/checkpoints/step_13440/pytorch_model.bin/pytorch_model.bin"
TOKENIZER_PATH = "/aistor/aispeech/hpc_stor01/home/wangchenghao00sx/.cache/modelscope/hub/models/iic/SenseVoiceSmall"
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

# ================= 编辑距离计算逻辑 (复用 pt.py) =================
class Calculator:
    def __init__(self):
        self.space = []
        self.cost = {'cor': 0, 'sub': 1, 'del': 1, 'ins': 1}

    def calculate(self, lab, rec):
        lab = [str(x) for x in lab]
        rec = [str(x) for x in rec]
        lab.insert(0, ''); rec.insert(0, '')
        while len(self.space) < len(lab): self.space.append([])
        for row in self.space:
            while len(row) < len(rec): row.append({'dist': 0, 'error': 'non'})
        for i in range(len(lab)): self.space[i][0]['dist'] = i; self.space[i][0]['error'] = 'del'
        for j in range(len(rec)): self.space[0][j]['dist'] = j; self.space[0][j]['error'] = 'ins'
        self.space[0][0]['error'] = 'non'
        for i in range(1, len(lab)):
            for j in range(1, len(rec)):
                min_dist = 1e9
                d = self.space[i-1][j]['dist'] + self.cost['del']
                if d < min_dist: min_dist, err = d, 'del'
                d = self.space[i][j-1]['dist'] + self.cost['ins']
                if d < min_dist: min_dist, err = d, 'ins'
                c = self.cost['cor'] if lab[i] == rec[j] else self.cost['sub']
                d = self.space[i-1][j-1]['dist'] + c
                if d < min_dist: min_dist, err = d, ('cor' if lab[i] == rec[j] else 'sub')
                self.space[i][j]['dist'] = min_dist; self.space[i][j]['error'] = err
        res = {'all': 0, 'cor': 0, 'sub': 0, 'ins': 0, 'del': 0}
        i, j = len(lab)-1, len(rec)-1
        while True:
            op = self.space[i][j]['error']
            if op == 'cor': res['all'] += 1; res['cor'] += 1; i -= 1; j -= 1
            elif op == 'sub': res['all'] += 1; res['sub'] += 1; i -= 1; j -= 1
            elif op == 'del': res['all'] += 1; res['del'] += 1; i -= 1
            elif op == 'ins': res['ins'] += 1; j -= 1
            else: break
        return res

def get_bucket_from_wer(wer):
    if wer < 0.04: return 1
    if wer < 0.12: return 2
    return 3

class VerifyDataset(Dataset):
    def __init__(self, lines, tokenizer, max_len=160):
        self.lines = lines
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self): return len(self.lines)

    def __getitem__(self, idx):
        item = json.loads(self.lines[idx])
        text = item.get('target', '')
        processed_text = text.lower().replace("'", "")
        raw_tokens = self.tokenizer.encode(processed_text)
        # 严格对齐训练过滤逻辑
        token_ids = [x for x in raw_tokens if x != 0 and x != 25055]
        text_ids = np.array(token_ids, dtype=np.int32)[:self.max_len]
        
        dense_text = torch.zeros((len(text_ids), VOCAB_SIZE_BASE), dtype=torch.float32)
        if len(text_ids) > 0:
            dense_text.scatter_(1, torch.from_numpy(text_ids.astype(np.int64)).unsqueeze(1), 1.0)

        return {
            "text_onehot": dense_text,
            "text_ids": text_ids,
            "target_bucket": int(item['bucket_id']),
            "key": item['key']
        }

def collate_fn(batch):
    text_list = [b['text_onehot'] for b in batch]
    text_padded = pad_sequence(text_list, batch_first=True, padding_value=0.0)
    lens = torch.tensor([t.size(0) for t in text_list])
    mask = torch.arange(text_padded.size(1)).expand(len(batch), -1) < lens.unsqueeze(1)
    return {
        "text_onehot": text_padded,
        "text_mask": mask.float(),
        "text_ids": [b['text_ids'] for b in batch],
        "target_buckets": torch.tensor([b['target_bucket'] for b in batch], dtype=torch.long)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=CHECKPOINT_PATH)
    parser.add_argument("--jsonl", type=str, default="data/train/train_augmented_3bucket.jsonl")
    args = parser.parse_args()

    device = torch.device("npu:0" if torch.npu.is_available() else "cpu")
    tokenizer = SenseVoiceTokenizer(TOKENIZER_PATH)
    CONFIG.vocab_size = tokenizer.vocab_size
    model = CTCTransformerSimulator(CONFIG)
    
    sd = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict({k.replace("simulator.", "").replace("module.", ""): v for k, v in sd.items()})
    model.to(device).eval()

    with open(args.jsonl, 'r', encoding='utf-8') as f:
        lines = [f.readline() for _ in range(2000)]
    
    loader = DataLoader(VerifyDataset(lines, tokenizer), batch_size=16, collate_fn=collate_fn)
    calc = Calculator()
    
    # 统计矩阵: [target_bucket][pred_bucket]
    results = np.zeros((5, 5), dtype=int)

    for batch in tqdm(loader, desc="Verifying Buckets"):
        with torch.no_grad():
            probs_b = model.inference(
                batch['text_onehot'].to(device), 
                batch['text_mask'].to(device), 
                bucket_id=batch['target_buckets'].to(device), 
                max_len=160, temperature=0.5
            )
        
        for i, probs in enumerate(probs_b):
            target_bid = batch['target_buckets'][i].item()
            ref_ids = batch['text_ids'][i]
            
            # 截断与过滤
            pred_ids = probs.argmax(dim=-1)
            eos_indices = (pred_ids == 25055).nonzero(as_tuple=True)[0]
            if len(eos_indices) > 0:
                pred_ids = pred_ids[:eos_indices[0].item()]
            hyp_ids = [x.item() for x in pred_ids if x != 0 and x != 25055]
            
            # 计算 WER
            res = calc.calculate(ref_ids, hyp_ids)
            n = res['all']
            wer = (res['sub'] + res['del'] + res['ins']) / n if n > 0 else 0
            pred_bid = get_bucket_from_wer(wer)
            
            results[target_bid][pred_bid] += 1

    # 打印报表
    print("\n" + "="*60)
    print(f"{'Target':<10} | {'Hit Rate':<12} | {'Distribution (B1|B2|B3)'}")
    print("-" * 60)
    for b in range(1, 5):
        total = results[b].sum()
        hit = results[b][b]
        acc = (hit / total * 100) if total > 0 else 0
        dist = "|".join([f"{results[b][i]:>3}" for i in range(1, 5)])
        print(f"Bucket {b:<3} | {acc:>7.2f}% ({hit:>3}/{total:<3}) | {dist}")
    print("="*60)

if __name__ == "__main__":
    main()