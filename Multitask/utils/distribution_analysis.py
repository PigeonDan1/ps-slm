#!/usr/bin/env python3
"""
run_js_parallel_plus_metrics.py

并行计算以下三对分布的多指标：
(ctc, clean), (ctc, noise), (noise, clean)

指标包含：
- JS 距离（对称，帧均值）
- 对称交叉熵 SCE = 0.5*(CE(p||q) + CE(q||p))（帧均值）
- 帧级 argmax 一致率
- CTC 合并后的序列编辑距离 & 归一化编辑距离
- blank 帧占比（按 argmax==blank）
- 熵（帧均值）
并输出：
1) 按样本的明细 CSV
2) 若干可视化散点图（JS vs Δ、JS vs top1_acc、JS vs norm edit dist）
"""

import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import csv

# ---------------- 设备选择 ----------------
device = torch.device(
    "npu" if (torch.npu.is_available() if hasattr(torch, "npu") else False) else
    ("cuda" if torch.cuda.is_available() else "cpu")
)

# ---------------- 路径配置 ----------------
h5_path = Path("/aistor/aispeech/hpc_stor01/home/pengjing00sx/Github/ps-slm/SLAM-LLM-ASR/distribution/cpu_cache.h5")
out_dir = Path("/aistor/aispeech/hpc_stor01/home/pengjing00sx/Github/ps-slm/SLAM-LLM-ASR/distribution/plot")
out_dir.mkdir(exist_ok=True)

BLANK_ID = 0
EPS = 1e-12

# ---------------- 工具函数 ----------------
def interp_logits_then_softmax(x_np: np.ndarray, tgt_len: int) -> torch.Tensor:
    """
    在 log 概率域做线性插值，再 softmax 回概率。
    目的：避免直接在概率域插值带来的峰值形状扭曲。
    输入 x_np: [T, V] 概率；输出: [tgt_len, V] 概率 tensor
    """
    x = torch.from_numpy(x_np.astype(np.float32)).to(device).clamp_min(EPS)
    x = x.log()                                # 近似 logits（严格说是 log prob）
    x = x.unsqueeze(0).permute(0, 2, 1)        # [1,V,T]
    x = torch.nn.functional.interpolate(x, size=tgt_len, mode='linear', align_corners=False)
    x = x.permute(0, 2, 1).squeeze(0)          # [tgt_len,V]
    x = torch.softmax(x, dim=-1).clamp_min(EPS)
    return x

def js_frame_mean(p: torch.Tensor, q: torch.Tensor) -> float:
    """输入 [T,V] 概率，返回整条音频的 JS 均值（开方后的 JSD）。"""
    p, q = p.clamp_min(EPS), q.clamp_min(EPS)
    m = 0.5 * (p + q).clamp_min(EPS)
    kl_pm = (p * (p / m).log()).sum(dim=1)
    kl_qm = (q * (q / m).log()).sum(dim=1)
    js = torch.sqrt(0.5 * (kl_pm + kl_qm))
    return js.mean().item()

def symm_ce_frame_mean(p: torch.Tensor, q: torch.Tensor) -> float:
    """对称交叉熵 SCE = 0.5*(CE(p||q) + CE(q||p))，帧均值。"""
    p, q = p.clamp_min(EPS), q.clamp_min(EPS)
    ce_pq = (-p * q.log()).sum(dim=1)
    ce_qp = (-q * p.log()).sum(dim=1)
    sce = 0.5 * (ce_pq + ce_qp)
    return sce.mean().item()

def entropy_frame_mean(p: torch.Tensor) -> float:
    p = p.clamp_min(EPS)
    H = (-(p * p.log()).sum(dim=1)).mean().item()
    return H

def top1_acc_frame(p: torch.Tensor, q: torch.Tensor) -> float:
    """帧级 argmax 一致率。"""
    t1 = p.argmax(dim=-1)
    t2 = q.argmax(dim=-1)
    return (t1 == t2).float().mean().item()

def collapse_ctc(ids: np.ndarray, blank_id: int = BLANK_ID) -> list:
    """移除重复和 blank 的 CTC 合并。输入为帧级 argmax 序列。"""
    out = []
    prev = None
    for x in ids:
        if x == blank_id:
            prev = x
            continue
        if (prev is None) or (x != prev):
            out.append(int(x))
        prev = x
    return out

def edit_distance(a: list, b: list) -> int:
    """标准 Levenshtein 距离。"""
    la, lb = len(a), len(b)
    dp = [[0]*(lb+1) for _ in range(la+1)]
    for i in range(la+1):
        dp[i][0] = i
    for j in range(lb+1):
        dp[0][j] = j
    for i in range(1, la+1):
        for j in range(1, lb+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,      # 删除
                dp[i][j-1] + 1,      # 插入
                dp[i-1][j-1] + cost  # 替换/匹配
            )
    return dp[la][lb]

def blank_fraction_by_argmax(p: torch.Tensor, blank_id: int = BLANK_ID) -> float:
    """按 argmax==blank 的帧占比。"""
    return (p.argmax(dim=-1) == blank_id).float().mean().item()

def prepare_pair_arrays(f: h5py.File, g1: str, g2: str, k: str) -> tuple[torch.Tensor, torch.Tensor]:
    """读取两组分布，同步到相同长度并返回概率张量 [T,V]。"""
    x1 = f[g1][k][:]
    x2 = f[g2][k][:]
    T = max(x1.shape[0], x2.shape[0])
    p = interp_logits_then_softmax(x1, T)
    q = interp_logits_then_softmax(x2, T)
    return p, q

# ---------------- 子进程工作函数 ----------------
def _worker(key_and_triplet):
    """
    对单个样本 key，计算三个配对的多指标：
    pairs = [('ctc','clean'), ('ctc','noise'), ('noise','clean')]
    返回 dict：包含每一对的度量。
    """
    k, pairs = key_and_triplet
    out = {"key": k}

    with h5py.File(h5_path, 'r') as f:
        # 预先为 Δ 计算准备 ctc-clean / ctc-noise
        p_cc, q_cc = prepare_pair_arrays(f, 'ctc', 'clean', k)
        p_cn, q_cn = prepare_pair_arrays(f, 'ctc', 'noise', k)
        p_nc, q_nc = prepare_pair_arrays(f, 'noise', 'clean', k)

        stats = {}
        for (a, b), (p, q) in {
            ('ctc', 'clean'): (p_cc, q_cc),
            ('ctc', 'noise'): (p_cn, q_cn),
            ('noise', 'clean'): (p_nc, q_nc),
        }.items():
            js = js_frame_mean(p, q)
            sce = symm_ce_frame_mean(p, q)
            acc = top1_acc_frame(p, q)
            H_p, H_q = entropy_frame_mean(p), entropy_frame_mean(q)
            blank_p, blank_q = blank_fraction_by_argmax(p), blank_fraction_by_argmax(q)

            # CTC 合并后的编辑距
            ids_p = p.argmax(dim=-1).detach().cpu().numpy()
            ids_q = q.argmax(dim=-1).detach().cpu().numpy()
            seq_p = collapse_ctc(ids_p, BLANK_ID)
            seq_q = collapse_ctc(ids_q, BLANK_ID)
            ed = edit_distance(seq_p, seq_q)
            norm = ed / max(1, max(len(seq_p), len(seq_q)))

            prefix = f"{a}_{b}"
            stats[f"{prefix}_js"] = js
            stats[f"{prefix}_sce"] = sce
            stats[f"{prefix}_top1_acc"] = acc
            stats[f"{prefix}_entropy_{a}"] = H_p
            stats[f"{prefix}_entropy_{b}"] = H_q
            stats[f"{prefix}_blank_frac_{a}"] = blank_p
            stats[f"{prefix}_blank_frac_{b}"] = blank_q
            stats[f"{prefix}_edit_dist"] = ed
            stats[f"{prefix}_edit_norm"] = norm
            stats[f"{prefix}_len_{a}"] = len(seq_p)
            stats[f"{prefix}_len_{b}"] = len(seq_q)

        # Δ = JS(ctc, noise) - JS(ctc, clean)
        delta = stats["ctc_noise_js"] - stats["ctc_clean_js"]
        out.update(stats)
        out["delta"] = delta

    return out

# ---------------- 主流程 ----------------
def main() -> None:
    with h5py.File(h5_path, 'r') as f:
        keys = sorted(f['ctc'].keys())

    pairs = [('ctc', 'clean'), ('ctc', 'noise'), ('noise', 'clean')]

    # 并行计算
    n_jobs = min(cpu_count(), len(keys))
    with Pool(n_jobs) as pool:
        rows = list(tqdm(pool.imap(_worker, [(k, pairs) for k in keys]),
                         total=len(keys), desc="Parallel metrics"))

    # ---------- 汇总与保存 ----------
    # 1) 控制台摘要（沿用你的 Δ 统计）
    deltas = np.array([r["delta"] for r in rows], dtype=np.float32)
    print(f"Δ 均值: {deltas.mean():.3f}")
    print(f"Δ < 0 占比: {(deltas < 0).mean() * 100:.1f}%")

    # 2) 保存明细 CSV
    csv_path = out_dir / "pair_metrics_per_utt.csv"
    fieldnames = ["key"] + [k for k in rows[0].keys() if k != "key"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"明细指标已保存到 {csv_path}")

    # ---------- 可视化 ----------
    # a) 你原本的 Δ 散点图：x=JS(ctc,clean) vs y=Δ
    x = np.array([r["ctc_clean_js"] for r in rows], dtype=np.float32)
    y = deltas
    plt.figure(figsize=(5, 3.5))
    plt.scatter(x, y, s=10, alpha=0.6)
    plt.axhline(0, lw=1, ls='--')
    plt.xlabel('JS(CTC, Clean)')
    plt.ylabel('Δ = JS(CTC, Noise) − JS(CTC, Clean)')
    plt.title('Noise closer to CTC?  (Δ < 0 → Yes)')
    plt.tight_layout()
    p1 = out_dir / "delta_ctc_noise_clean.png"
    plt.savefig(p1, dpi=150)
    plt.close()
    print(f"结果图已保存到 {p1}")

    # b) JS vs 帧级 argmax 一致率（以 ctc-clean 为例）
    x = np.array([r["ctc_clean_js"] for r in rows], dtype=np.float32)
    y = np.array([r["ctc_clean_top1_acc"] for r in rows], dtype=np.float32)
    plt.figure(figsize=(5, 3.5))
    plt.scatter(x, y, s=10, alpha=0.6)
    plt.xlabel('JS(CTC, Clean)')
    plt.ylabel('Top-1 frame acc (CTC vs Clean)')
    plt.title('形状差异 vs 决策一致性')
    plt.tight_layout()
    p2 = out_dir / "js_vs_top1acc_ctc_clean.png"
    plt.savefig(p2, dpi=150)
    plt.close()
    print(f"结果图已保存到 {p2}")

    # c) JS vs 归一化编辑距（以 ctc-clean 为例）
    x = np.array([r["ctc_clean_js"] for r in rows], dtype=np.float32)
    y = np.array([r["ctc_clean_edit_norm"] for r in rows], dtype=np.float32)
    plt.figure(figsize=(5, 3.5))
    plt.scatter(x, y, s=10, alpha=0.6)
    plt.xlabel('JS(CTC, Clean)')
    plt.ylabel('Norm edit distance (CTC vs Clean)')
    plt.title('形状差异 vs 序列拓扑差异')
    plt.tight_layout()
    p3 = out_dir / "js_vs_editnorm_ctc_clean.png"
    plt.savefig(p3, dpi=150)
    plt.close()
    print(f"结果图已保存到 {p3}")

if __name__ == '__main__':
    main()
