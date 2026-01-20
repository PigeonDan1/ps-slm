# 计算结果指标
import json
import os
import torch
import numpy as np
import re
import sys
import unicodedata
from tqdm import tqdm
from model.tokenizer import SenseVoiceTokenizer

# ================= Configuration =================
BASE_DIR = "/aistor/sjtu/hpc_stor01/home/wangchenghao/workingspace/ps-slm/Multitask/data/test-clean" #/without_feedback"
TOKENIZER_PATH = "/aistor/sjtu/hpc_stor01/home/yangyi/model/SenseVoiceSmall"

# [关键修复] Simulator 词表包含 EOS (25055)，所以维度必须是 25056
VOCAB_SIZE_REAL = 25056 

DATASETS = {
    "0": os.path.join(BASE_DIR, "simulator_WER0", "multitask.jsonl"),
    "1": os.path.join(BASE_DIR, "simulator_WER1", "multitask.jsonl"),
    "2": os.path.join(BASE_DIR, "simulator_WER2", "multitask.jsonl"),
    "3": os.path.join(BASE_DIR, "simulator_WER3", "multitask.jsonl"),
    "4": os.path.join(BASE_DIR, "simulator_WER4", "multitask.jsonl"),
    "5": os.path.join(BASE_DIR, "simulator_WER5", "multitask.jsonl"),
    "6": os.path.join(BASE_DIR, "simulator_WER6", "multitask.jsonl"),
    "7": os.path.join(BASE_DIR, "simulator_WER7", "multitask.jsonl"),
    "8": os.path.join(BASE_DIR, "simulator_WER8", "multitask.jsonl"),
    "9": os.path.join(BASE_DIR, "simulator_WER9", "multitask.jsonl"),
    # "10": os.path.join(BASE_DIR, "simulator_WER10", "multitask.jsonl"),
    # "12": os.path.join(BASE_DIR, "simulator_WER12", "multitask.jsonl"),
    # "14": os.path.join(BASE_DIR, "simulator_WER14", "multitask.jsonl"),
    # "16": os.path.join(BASE_DIR, "simulator_WER16", "multitask.jsonl"),
    # "18": os.path.join(BASE_DIR, "simulator_WER18", "multitask.jsonl"),
    # "20": os.path.join(BASE_DIR, "simulator_WER20", "multitask.jsonl"),
    # "30": os.path.join(BASE_DIR, "simulator_WER30", "multitask.jsonl"),
    # "40": os.path.join(BASE_DIR, "simulator_WER40", "multitask.jsonl"),
    # "50": os.path.join(BASE_DIR, "simulator_WER50", "multitask.jsonl"),
    # "80": os.path.join(BASE_DIR, "simulator_WER80", "multitask.jsonl"),
}

# --- Normalization & Calculator 保持不变 ---
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

# ================= 核心逻辑修复 =================
def restore_ids(pt_path):
    try:
        data = torch.load(pt_path, map_location='cpu')
        
        # 增加对 text_ids 存在的显式检查
        if 'text_ids' not in data:
            # print(f"Error: 'text_ids' missing in {pt_path}")
            return None, None

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
        
        # [关键修复] 同时移除 Blank(0) 和 EOS(25055)
        # 这样即使推理时由于各种原因没截断干净，也不会影响评测
        hyp = [x for x in pred if x != 0 and x != 25055]
        
        ref = data['text_ids'].tolist()
        # 对 Reference 也进行一次特殊 ID 过滤，确保万无一失
        ref = [x for x in ref if x != 0 and x != 25055]
        
        return hyp, ref
    except Exception as e:
        # print(f"Restore Error: {e}")
        return None, None

def main():
    tokenizer = SenseVoiceTokenizer(TOKENIZER_PATH)
    calc = Calculator()
    
    for name, jsonl in DATASETS.items():
        print(f"\n--- Evaluating {name} ---")
        st_b, st_w = [{'all':0,'cor':0,'sub':0,'ins':0,'del':0} for _ in range(2)]
        
        if not os.path.exists(jsonl):
            print(f"File not found: {jsonl}")
            continue

        with open(jsonl, 'r') as f:
            lines = f.readlines()
            
        for i, line in enumerate(tqdm(lines)):
            item = json.loads(line)
            
            # 这里确保路径正确
            path_key = 'sim_psd_path' if 'sim_psd_path' in item else 'psd_path'
            
            hyp_b, ref_b = restore_ids(item[path_key])
            
            # 如果 restore_ids 还是返回 None，说明 .pt 文件里真的没存 text_ids
            if hyp_b is None or ref_b is None:
                continue

            # BPE 级别
            rb = calc.calculate(ref_b, hyp_b)
            for k in st_b: st_b[k] += rb[k]

            # Word 级别 (这里 decode 之前 hyp_b 已经没有 EOS 了)
            gt_text = item.get('GT', item.get('target', ''))
            hyp_text = tokenizer.decode(hyp_b)
            lab_w = normalize_standard(characterize(gt_text))
            rec_w = normalize_standard(characterize(hyp_text))
            rw = calc.calculate(lab_w, rec_w)
            for k in st_w: st_w[k] += rw[k]

        # [修改] 打印逻辑：按照标准 SDI 公式计算百分比
        for mode, s in [("BPE", st_b), ("WORD", st_w)]:
            if s['all'] > 0:
                n = s['all']
                wer = (s['sub'] + s['del'] + s['ins']) / n * 100
                s_rate = s['sub'] / n * 100
                d_rate = s['del'] / n * 100
                i_rate = s['ins'] / n * 100
                print(f"[{mode}] WER: {wer:.2f}% | S: {s_rate:.2f}% | D: {d_rate:.2f}% | I: {i_rate:.2f}% | N: {n}")
            else:
                print(f"[{mode}] No valid samples found. 请确认 .pt 是否包含 text_ids 字段。")

if __name__ == "__main__":
    main()