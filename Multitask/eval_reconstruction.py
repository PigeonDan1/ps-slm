# 测试train的拟合准确性——计算指标
import json
import os
import torch
import numpy as np
import re
import sys
import unicodedata
import argparse
from tqdm import tqdm
from model.tokenizer import SenseVoiceTokenizer

# ================= Configuration =================
# 关键修复：VOCAB_SIZE 必须包含 EOS (25055)
VOCAB_SIZE_REAL = 25056 

# 定义分桶区间 (Target WER)
BUCKETS = [
    (0.0, 0.02, "WER < 2%"),
    (0.02, 0.05, "2% <= WER < 5%"),
    (0.05, 0.10, "5% <= WER < 10%"),
    (0.10, 0.20, "10% <= WER < 20%"),
    (0.20, 1.00, "WER >= 20%")
]

# --- Normalization & Calculator 保持不变 (严谨复用) ---
puncts = ['!', ',', '?', '、', '。', '！', '，', '；', '？', '：', '「', '」', '︰', '『', '』', '《', '》']
spacelist = [' ', '\t', '\r', '\n']

def characterize(string):
    res = []
    i = 0
    while i < len(string):
        char = string[i]
        if char in puncts: i += 1; continue
        cat1 = unicodedata.category(char)
        if cat1 == 'Zs' or cat1 == 'Cn' or char in spacelist: i += 1; continue
        if cat1 == 'Lo':
            res.append(char); i += 1
        else:
            sep = ' '
            if char == '<': sep = '>'
            j = i + 1
            while j < len(string):
                c = string[j]
                if ord(c) >= 128 or (c in spacelist) or (c == sep): break
                j += 1
            if j < len(string) and string[j] == '>': j += 1
            res.append(string[i:j]); i = j
    return res

def normalize_standard(sentence_list):
    new_sentence = []
    for token in sentence_list:
        x = token.upper().replace("'", "")
        x = re.sub(r"[^A-Z0-9]+", "", x)
        if x: new_sentence.append(x)
    return new_sentence

class Calculator:
    def __init__(self):
        self.space = []
        self.cost = {'cor': 0, 'sub': 1, 'del': 1, 'ins': 1}
    def calculate(self, lab_raw, rec_raw):
        lab = [str(x) for x in lab_raw]
        rec = [str(x) for x in rec_raw]
        lab.insert(0, ''); rec.insert(0, '')
        while len(self.space) < len(lab): self.space.append([])
        for row in self.space:
            while len(row) < len(rec): row.append({'dist': 0, 'error': 'non'})
        for i in range(len(lab)): self.space[i][0]['dist'] = i; self.space[i][0]['error'] = 'del'
        for j in range(len(rec)): self.space[0][j]['dist'] = j; self.space[0][j]['error'] = 'ins'
        self.space[0][0]['error'] = 'non'
        for i in range(1, len(lab)):
            for j in range(1, len(rec)):
                min_dist = sys.maxsize
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

def restore_ids(pt_path):
    try:
        data = torch.load(pt_path, map_location='cpu')
        if 'text_ids' not in data: return None, None

        if 'shape' in data: T = data['shape'][0]
        else: T = data['psd_indices'].shape[0] // 10 if data['psd_indices'].ndim == 1 else data['psd_indices'].shape[0]

        idx = data['psd_indices'].reshape(T, 10).astype(np.int64)
        val = data['psd_values'].reshape(T, 10).astype(np.float32)
        
        dense = torch.zeros(T, VOCAB_SIZE_REAL)
        indices_clamped = np.clip(idx, 0, VOCAB_SIZE_REAL - 1)
        dense.scatter_(1, torch.from_numpy(indices_clamped), torch.from_numpy(val))
        
        pred = dense.argmax(dim=-1).tolist()
        hyp = [x for x in pred if x != 0 and x != 25055]
        ref = [x for x in data['text_ids'].tolist() if x != 0 and x != 25055]
        return hyp, ref
    except:
        return None, None

def get_bucket_idx(wer):
    for i, (low, high, _) in enumerate(BUCKETS):
        if low <= wer < high: return i
    return len(BUCKETS) - 1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--tokenizer_path", type=str, required=True)
    args = parser.parse_args()

    tokenizer = SenseVoiceTokenizer(args.tokenizer_path)
    calc = Calculator()
    
    # 每个桶初始化统计量
    # 格式: {bucket_idx: {"st_b": stats, "st_w": stats, "target_wer_sum": 0.0, "count": 0}}
    stats_buckets = []
    for _ in BUCKETS:
        stats_buckets.append({
            "st_b": {'all': 0, 'cor': 0, 'sub': 0, 'ins': 0, 'del': 0},
            "st_w": {'all': 0, 'cor': 0, 'sub': 0, 'ins': 0, 'del': 0},
            "target_wer_sum": 0.0,
            "count": 0
        })

    with open(args.input_jsonl, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Analyzing Reconstruction"):
        item = json.loads(line)
        target_bpe = item.get('bpe_error', {})
        target_wer = target_bpe.get('WER', 0.0)
        
        b_idx = get_bucket_idx(target_wer)
        path_key = 'sim_psd_path' if 'sim_psd_path' in item else 'psd_path'
        
        hyp_b, ref_b = restore_ids(item[path_key])
        if hyp_b is None or ref_b is None: continue

        bucket = stats_buckets[b_idx]
        bucket["count"] += 1
        bucket["target_wer_sum"] += target_wer

        # 1. BPE 级别计算 (复用逻辑)
        rb = calc.calculate(ref_b, hyp_b)
        for k in bucket["st_b"]: bucket["st_b"][k] += rb[k]

        # 2. Word 级别计算 (复用逻辑)
        gt_text = item.get('GT', item.get('target', ''))
        hyp_text = tokenizer.decode(hyp_b)
        lab_w = normalize_standard(characterize(gt_text))
        rec_w = normalize_standard(characterize(hyp_text))
        rw = calc.calculate(lab_w, rec_w)
        for k in bucket["st_w"]: bucket["st_w"][k] += rw[k]

    # 打印结果
    print("\n" + "="*80)
    print(f"{'Bucket Range':<20} | {'Count':<6} | {'Tgt WER':<8} | {'BPE WER':<8} | {'Word WER':<8} | {'Bias (BPE)':<8}")
    print("-" * 80)

    for i, (low, high, label) in enumerate(BUCKETS):
        b = stats_buckets[i]
        if b["count"] == 0:
            print(f"{label:<20} | 0      | N/A      | N/A      | N/A      | N/A")
            continue

        avg_target = (b["target_wer_sum"] / b["count"]) * 100
        
        # 计算 BPE 
        nb = b["st_b"]['all']
        bpe_wer = (nb_sum := (nb_err := nb - b["st_b"]['cor'] + b["st_b"]['ins'])) / (nb + 1e-6) * 100 if nb > 0 else 0
        # 严格对应 SDI 计算: (sub+del+ins)/all
        bpe_wer_clean = (b["st_b"]['sub'] + b["st_b"]['del'] + b["st_b"]['ins']) / (nb + 1e-6) * 100

        # 计算 Word
        nw = b["st_w"]['all']
        word_wer = (b["st_w"]['sub'] + b["st_w"]['del'] + b["st_w"]['ins']) / (nw + 1e-6) * 100

        bias = bpe_wer_clean - avg_target
        print(f"{label:<20} | {b['count']:<6} | {avg_target:>7.2f}% | {bpe_wer_clean:>7.2f}% | {word_wer:>7.2f}% | {bias:>+7.2f}%")

    print("="*80)

if __name__ == "__main__":
    main()