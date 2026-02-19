import torch
import torch.nn.functional as F
import numpy as np
import kaldiio
import torchaudio
from model.SenseVoice import SenseVoiceSmall
from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank
from typing import Tuple

# 1. 还原函数 (保持一致)
def restore_dense_matrix(indices, values, target_vocab_size=25055):
    T = indices.shape[0]
    VOCAB_SIZE_FULL = 25056 
    EPS = 1e-8
    indices = torch.from_numpy(indices.astype(np.int64)).reshape(T, -1)
    values = torch.from_numpy(values.astype(np.float32)).reshape(T, -1)
    dense = torch.zeros(T, VOCAB_SIZE_FULL, dtype=torch.float32)
    dense.scatter_(1, indices, values)
    dense = dense[:, :target_vocab_size]
    row_sums = dense.sum(dim=1, keepdim=True) + EPS
    return dense / row_sums

# 2. 从你的模型代码中完整复制的 PSD 逻辑
def psd_logic(
        encoder_out: torch.Tensor,      # [B, T, D]
        encoder_out_lens: torch.Tensor, # [B]
        ctc_posterior: torch.Tensor,    # [B, T, V]  
        blank_id: int = 0,
        blank_threshold: float = 0.90
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, T, D = encoder_out.shape
    device  = encoder_out.device
    is_log_prob = ctc_posterior.max() <= 0
    ctc_probs = ctc_posterior.exp() if is_log_prob else ctc_posterior
    keep_frames, new_lens = [], []
    for b in range(B):
        L = encoder_out_lens[b].item()
        if L == 0:
            keep_frames.append(encoder_out.new_zeros(0, D))
            new_lens.append(0)
            continue
        ids = ctc_probs[b, :L].argmax(dim=-1)

        merged_feats, merged_blank_probs = [], []
        start = 0
        for end in range(1, L + 1):
            if end == L or ids[end] != ids[start]:
                seg_len = end - start
                char_id = ids[start].item()
                if char_id == blank_id:
                    for t in range(start, end):
                        merged_feats.append(encoder_out[b, t])
                        merged_blank_probs.append(ctc_probs[b, t, blank_id])
                else:
                    merged_feats.append(encoder_out[b, start:end].mean(dim=0))
                    avg_blank_prob = ctc_probs[b, start:end, blank_id].mean()
                    merged_blank_probs.append(avg_blank_prob)
                start = end

        merged_feats = torch.stack(merged_feats, dim=0)
        merged_blank_probs = torch.tensor(merged_blank_probs, device=device)
        mask = merged_blank_probs < blank_threshold
        keep = mask.nonzero(as_tuple=False).squeeze(-1)
        feats_after_blank = merged_feats[keep]
        keep_frames.append(feats_after_blank)
        new_lens.append(feats_after_blank.size(0))

    max_len = max(new_lens) if new_lens else 0
    padded = []
    for feat in keep_frames:
        pad_len = max_len - feat.size(0)
        if pad_len > 0:
            feat = F.pad(feat, (0, 0, 0, pad_len), value=0.)
        padded.append(feat)
    encoder_outs = torch.stack(padded, dim=0)
    new_lens = torch.tensor(new_lens, dtype=torch.long, device=device)
    return encoder_outs, new_lens

@torch.no_grad()
def diagnostic():
    ark_path = "/aistor/sjtu/hpc_stor01/home/yangyi/data/asr/train/data/data_wav.3.ark:6027407635"
    pt_path = "/aistor/sjtu/hpc_stor01/home/wangchenghao/workingspace/ps-slm/Multitask/data/train/LibriSpeech-clean-naven-100-121669-0000-F-180513.pt"
    encoder_path = "/aistor/sjtu/hpc_stor01/home/yangyi/model/SenseVoiceSmall"
    
    # 1. 加载模型
    model, kwargs = SenseVoiceSmall.from_pretrained(encoder_path)
    model.eval()
    frontend = kwargs.get("frontend")

    # 2. 实时路径
    audio_raw = kaldiio.load_mat(ark_path)[1].astype(np.float32) / 32768.0
    audio_list = load_audio_text_image_video([audio_raw], fs=frontend.fs, audio_fs=16000, data_type="sound")
    fbank, fbank_len = extract_fbank(audio_list, data_type="sound", frontend=frontend)
    
    B = fbank.size(0)
    l_q = model.embed(torch.tensor([[0]])).repeat(B, 1, 1)
    e_q = model.embed(torch.tensor([[1, 2]])).repeat(B, 1, 1)
    t_q = model.embed(torch.tensor([[2]])).repeat(B, 1, 1)
    speech = torch.cat([l_q, e_q, t_q, fbank], dim=1)
    speech_lens = fbank_len + 4
    enc_out, _ = model.encoder(speech, speech_lens)
    raw_logits = model.ctc.ctc_lo(enc_out[0] if isinstance(enc_out, tuple) else enc_out)
    
    # 获取后验概率
    ctc_posterior = torch.softmax(raw_logits, dim=-1)[:, 4:, :]
    encoder_out = enc_out[:, 4:, :]
    encoder_out_lens = torch.clamp(speech_lens - 4, min=0)

    # 【关键：应用 PSD】
    print(">>> 正在对实时特征应用 PSD...")
    # 注意：PSD 输出的是特征维度，我们需要的是概率维度
    # 根据你的 forward 逻辑：encoder_outs, len = self.psd(ctc_posterior, ...)
    # 你的 PSD 输入参数 encoder_out 其实传的是 ctc_posterior (概率矩阵)
    real_time_psd, _ = psd_logic(
        ctc_posterior, 
        encoder_out_lens, 
        ctc_posterior, 
        blank_id=model.blank_id, 
        blank_threshold=0.90
    )
    real_time_final = real_time_psd[0, :, :25055]

    # 3. 预提取路径
    pt_data = torch.load(pt_path, map_location='cpu', weights_only=False)
    restored_ctc = restore_dense_matrix(pt_data['psd_indices'], pt_data['psd_values'])

    print(f"\n实时 PSD 后 Shape: {real_time_final.shape}")
    print(f"预提取还原 Shape: {restored_ctc.shape}")

    # 对比...
    mse = torch.mean((real_time_final - restored_ctc[:real_time_final.size(0)])**2)
    print(f"MSE (PSD后): {mse.item():.8f}")

if __name__ == "__main__":
    diagnostic()