# 计算结果指标
import json
import os
import torch
import numpy as np
import re
import sys
import unicodedata
import math
from tqdm import tqdm
from model.tokenizer import SenseVoiceTokenizer

# 新增绘图依赖
import matplotlib.pyplot as plt
from scipy.stats import norm

BASE_DIR = "/aistor/aispeech/hpc_stor01/home/wangchenghao00sx/workingspace/TASU-simulator/Multitask/data/test-clean/test_bucket_control"
TOKENIZER_PATH = "/aistor/aispeech/hpc_stor01/home/wangchenghao00sx/.cache/modelscope/hub/models/iic/SenseVoiceSmall"

VOCAB_SIZE_REAL = 25056 

DATASETS = {
    # "1": os.path.join(BASE_DIR, "simulator_B1", "multitask.jsonl"),
    # "2": os.path.join(BASE_DIR, "simulator_B2", "multitask.jsonl"),
    "3": os.path.join(BASE_DIR, "simulator_B3", "multitask.jsonl")
}

# --- 核心辅助函数：判定 WER 属于哪个 Bucket ---
def get_bucket_id(wer_percent):
    if wer_percent < 6.0:
        return 1
    elif 10.0 <= wer_percent < 40.0:
        return 2
    elif 45.0 <= wer_percent < 150.0:
        return 3
    return 4

# --- Normalization & Calculator 保持绝对不动 ---
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
        if 'shape' in data:
            T = data['shape'][0]
        else:
            T = data['psd_indices'].shape[0] // 10 if data['psd_indices'].ndim == 1 else data['psd_indices'].shape[0]

        idx = data['psd_indices'].reshape(T, 10).astype(np.int64)
        val = data['psd_values'].reshape(T, 10).astype(np.float32)
        dense = torch.zeros(T, VOCAB_SIZE_REAL)
        indices_clamped = np.clip(idx, 0, VOCAB_SIZE_REAL - 1)
        dense.scatter_(1, torch.from_numpy(indices_clamped), torch.from_numpy(val))
        pred = dense.argmax(dim=-1).tolist()
        hyp = [x for x in pred if x != 0 and x != 25055]
        ref = data['text_ids'].tolist()
        ref = [x for x in ref if x != 0 and x != 25055]
        return hyp, ref
    except Exception:
        return None, None

def plot_wer_distribution(wer_list, target_bucket, output_dir):
    """
    绘制 WER 分布直方图及正态分布曲线
    """
    # 过滤 0-200% 范围内的样本
    plot_data = [w for w in wer_list if 0 <= w <= 200]
    if not plot_data:
        return

    plt.figure(figsize=(10, 6))
    
    # 绘制直方图，bins 间隔 5%
    bins = np.arange(0, 201, 5)
    # 使用 density=True 使 y 轴为概率密度，方便叠加正态曲线
    n, bins_edges, patches = plt.hist(plot_data, bins=bins, density=True, alpha=0.6, color='skyblue', edgecolor='black', label='WER Distribution')
    
    # 拟合正态分布（钟型曲线）
    mu, std = norm.fit(plot_data)
    xmin, xmax = 0, 200
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    
    plt.plot(x, p, 'r', linewidth=2, label=f'Normal Fit (μ={mu:.1f}, σ={std:.1f})')
    
    plt.title(f'WER Distribution - Target Bucket {target_bucket}')
    plt.xlabel('Word Error Rate (%)')
    plt.ylabel('Probability Density')
    plt.xlim(0, 200)
    plt.grid(axis='y', alpha=0.3)
    plt.legend()
    
    # 保存图片
    save_name = f"distribution_B{target_bucket}.png"
    save_path = os.path.join(output_dir, save_name)
    plt.savefig(save_path)
    plt.close()
    print(f"[*] Histogram saved to: {save_path}")

def main():
    tokenizer = SenseVoiceTokenizer(TOKENIZER_PATH)
    calc = Calculator()
    
    # 确保输出目录存在
    os.makedirs(BASE_DIR, exist_ok=True)

    for target_bucket, jsonl in DATASETS.items():
        print(f"\n--- Evaluating Target Bucket {target_bucket} ---")
        st_b, st_w = [{'all':0,'cor':0,'sub':0,'ins':0,'del':0} for _ in range(2)]
        
        correct_bucket_count = 0
        total_valid_samples = 0
        bucket_distribution = {1: 0, 2: 0, 3: 0, 4: 0}
        wer_list = [] # 用于存储当前 Bucket 的所有样本 WER
        
        if not os.path.exists(jsonl):
            print(f"File not found: {jsonl}")
            continue

        with open(jsonl, 'r') as f:
            lines = f.readlines()
            
        for line in tqdm(lines):
            item = json.loads(line)
            path_key = 'sim_psd_path' if 'sim_psd_path' in item else 'psd_path'
            hyp_b, ref_b = restore_ids(item[path_key])
            if hyp_b is None or ref_b is None: continue

            # 1. 计算样本 BPE 级别 WER
            rb = calc.calculate(ref_b, hyp_b)
            n_b = rb['all']
            if n_b > 0:
                sample_wer = (rb['sub'] + rb['del'] + rb['ins']) / n_b * 100
                wer_list.append(sample_wer) # 收集数据点用于画图
                
                # 判定实际落入的 bucket
                actual_bucket = get_bucket_id(sample_wer)
                bucket_distribution[actual_bucket] += 1
                if actual_bucket == int(target_bucket):
                    correct_bucket_count += 1
                total_valid_samples += 1

            # 累加统计用于打印平均指标
            for k in st_b: st_b[k] += rb[k]

            # 2. 计算 Word 级别指标 (保持原样)
            gt_text = item.get('GT', item.get('target', ''))
            hyp_text = tokenizer.decode(hyp_b)
            lab_w = normalize_standard(characterize(gt_text))
            rec_w = normalize_standard(characterize(hyp_text))
            rw = calc.calculate(lab_w, rec_w)
            for k in st_w: st_w[k] += rw[k]

        # 打印统计结果
        if total_valid_samples > 0:
            accuracy = correct_bucket_count / total_valid_samples * 100
            print(f">>> Bucket Accuracy (Hit Rate): {accuracy:.2f}% ({correct_bucket_count}/{total_valid_samples})")
            print(f"    Distribution: B1:{bucket_distribution[1]} | B2:{bucket_distribution[2]} | B3:{bucket_distribution[3]} | B4:{bucket_distribution[4]}")
            
            # 绘制分布图
            plot_wer_distribution(wer_list, target_bucket, BASE_DIR)

        for mode, s in [("BPE", st_b), ("WORD", st_w)]:
            if s['all'] > 0:
                n = s['all']
                wer = (s['sub'] + s['del'] + s['ins']) / n * 100
                print(f"[{mode}] Mean WER: {wer:.2f}% | S: {s['sub']/n*100:.2f}% | D: {s['del']/n*100:.2f}% | I: {s['ins']/n*100:.2f}% | N: {n}")

if __name__ == "__main__":
    main()