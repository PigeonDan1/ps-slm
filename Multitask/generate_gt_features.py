import os
import json
import torch
import torch_npu 
from torch_npu.contrib import transfer_to_npu 

import torchaudio
import torchaudio.compliance.kaldi as kaldi
import numpy as np
import argparse
import time
from tqdm import tqdm
from pathlib import Path

# [新增] 引入 kaldiio 读取 ark 数据
try:
    import kaldiio
except ImportError:
    print("Please install kaldiio: pip install kaldiio")

# 引入项目模块
from model.SenseVoice import SenseVoiceSmall
from model.tokenizer import SenseVoiceTokenizer

MODEL_PATH = "/aistor/sjtu/hpc_stor01/home/yangyi/model/SenseVoiceSmall"
TOP_K_SPARSE = 10 
LOG_INTERVAL = 1000

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    return parser.parse_args()

def load_audio_flexible(path):
    """
    兼容 .wav 路径和 .ark:offset 格式
    """
    if ":" in path:
        try:
            rate, array = kaldiio.load_mat(path)
            if array.dtype == np.int16:
                array = array.astype(np.float32) / 32768.0
            else:
                array = array.astype(np.float32)
            waveform = torch.from_numpy(array)
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            return waveform, rate
        except Exception as e:
            return None, None
    else:
        if not os.path.exists(path):
            return None, None
        try:
            waveform, sr = torchaudio.load(path)
            return waveform, sr
        except Exception as e:
            return None, None

def extract_fbank(waveform, sample_rate=16000):
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    # [注意] 您确认音频计算正确，保留 num_mel_bins=560 以匹配 embedding 维度
    fbank_features = kaldi.fbank(
        waveform,
        num_mel_bins=560, 
        frame_length=25,
        frame_shift=10,
        dither=0.0,
        window_type="hamming",
        use_energy=False,
    )
    return fbank_features

def psd_processing(ctc_posterior, blank_id=0, blank_threshold=0.90):
    T, V = ctc_posterior.shape
    ids = ctc_posterior.argmax(dim=-1)
    merged_probs = []
    
    start = 0
    for end in range(1, T + 1):
        if end == T or ids[end] != ids[start]:
            char_id = ids[start].item()
            if char_id == blank_id:
                for t in range(start, end):
                    merged_probs.append(ctc_posterior[t])
            else:
                avg_prob = ctc_posterior[start:end].mean(dim=0)
                merged_probs.append(avg_prob)
            start = end
            
    if not merged_probs: return None
    merged_probs = torch.stack(merged_probs, dim=0)
    
    blank_probs = merged_probs[:, blank_id]
    mask = blank_probs < blank_threshold
    keep = mask.nonzero(as_tuple=False).squeeze(-1)
    
    if keep.numel() > 0:
        final_probs = merged_probs[keep]
    else:
        final_probs = merged_probs[:1] 
    return final_probs

def save_sparse_tensor(psd_tensor, text_ids, path):
    """
    保存 PSD 和 Tokenized Text IDs
    """
    # 截断 PSD 维度
    if psd_tensor.size(-1) > 25055:
        psd_tensor = psd_tensor[:, :25055]
        
    values, indices = torch.topk(psd_tensor, TOP_K_SPARSE, dim=-1)
    values = values.cpu().numpy().flatten()
    indices = indices.cpu().numpy().flatten()
    
    # 转换 text_ids 为 numpy array 存储
    if isinstance(text_ids, list):
        text_ids = np.array(text_ids, dtype=np.int32)
    
    torch.save({
        'psd_indices': indices.astype(np.int32),
        'psd_values': values.astype(np.float16),
        'shape': psd_tensor.shape,
        'text_ids': text_ids.astype(np.int32) # [核心] 存储 Token ID
    }, path)

def main():
    args = get_args()
    
    if torch.npu.is_available():
        device = torch.device(f"npu:{args.device_id}")
        torch.npu.set_device(device)
        print(f"[Rank {args.rank}] Using NPU {args.device_id}", flush=True)
    else:
        device = torch.device("cpu")

    print(f"[Rank {args.rank}] Loading SenseVoice & Tokenizer...", flush=True)
    model, kwargs = SenseVoiceSmall.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()
    tokenizer = SenseVoiceTokenizer(MODEL_PATH)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    skip_audio_cnt = 0
    skip_text_cnt = 0
    
    with open(args.input_jsonl, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()
    
    my_lines = all_lines[args.rank::args.world_size]
    total_samples = len(my_lines)
    print(f"[Rank {args.rank}] Start processing {total_samples} samples...", flush=True)
    
    start_time = time.time()
    
    with open(args.output_jsonl, 'w', encoding='utf-8', buffering=1) as f_out:
        for i, line in enumerate(my_lines):
            try:
                if (i + 1) % LOG_INTERVAL == 0:
                    elapsed = time.time() - start_time
                    speed = (i + 1) / elapsed
                    remaining = (total_samples - (i + 1)) / speed
                    print(f"[Rank {args.rank}] Progress: {i + 1}/{total_samples} | Skips(Aud/Txt): {skip_audio_cnt}/{skip_text_cnt} | Speed: {speed:.1f} it/s", flush=True)
                
                item = json.loads(line)
                key = item['key']
                audio_path = item['path']
                
                # 兼容 target 字段作为 GT
                text = item.get('GT', item.get('text', item.get('target', '')))
                
                if not text: 
                    skip_text_cnt += 1
                    continue
                
                # [核心] 1. Tokenize 文本
                text_ids = tokenizer.encode(text) 
                
                # 读取音频
                waveform, sr = load_audio_flexible(audio_path)
                if waveform is None:
                    if skip_audio_cnt < 3: 
                        print(f"[Rank {args.rank}] [WARN] Audio Load Failed: {audio_path}", flush=True)
                    skip_audio_cnt += 1
                    continue
                    
                input_features = extract_fbank(waveform, sr).to(device)
                input_len = torch.tensor([input_features.size(0)], device=device)
                
                speech = input_features.unsqueeze(0) 
                
                with torch.no_grad():
                    B = 1
                    language_query = model.embed(torch.tensor([[0]], device=device)).repeat(B, 1, 1)
                    textnorm_query = model.embed(torch.tensor([[2]], device=device)).repeat(B, 1, 1)
                    event_emo_query = model.embed(torch.tensor([[1, 2]], device=device)).repeat(B, 1, 1)
                    
                    speech_in = torch.cat([language_query, event_emo_query, textnorm_query, speech], dim=1)
                    speech_lengths = input_len + 4
                    
                    encoder_out, _ = model.encoder(speech_in, speech_lengths)
                    encoder_out = encoder_out[:, 4:, :] 
                    logits = model.ctc.ctc_lo(encoder_out)
                    probs = torch.softmax(logits, dim=-1).squeeze(0)
                
                psd_probs = psd_processing(probs, blank_id=0)
                if psd_probs is None: continue
                    
                # [核心] 2. 计算长度信息
                psd_len = psd_probs.size(0)
                text_len = len(text_ids)

                save_filename = f"{key}.pt"
                save_path = os.path.join(args.output_dir, save_filename)
                
                # 保存 PSD 和 Token ID 到 .pt
                save_sparse_tensor(psd_probs, text_ids, save_path)
                
                # [核心] 3. 写入 Metadata 到 JSONL
                new_item = item.copy()
                new_item['psd_path'] = save_path 
                new_item['psd_len'] = psd_len   # 记录音频特征长度
                new_item['text_len'] = text_len # 记录文本 Token 长度
                
                f_out.write(json.dumps(new_item, ensure_ascii=False) + '\n')
                
            except Exception as e:
                print(f"[Rank {args.rank}] Error on {key}: {e}", flush=True)
                continue

    print(f"[Rank {args.rank}] Finished. Total Skips: Audio={skip_audio_cnt}, Text={skip_text_cnt}", flush=True)

if __name__ == "__main__":
    main()