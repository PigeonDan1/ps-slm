import os
import sys
import json
import torch
import torch_npu
import numpy as np
import torchaudio
import argparse
import re
from tqdm import tqdm
import deepspeed
import torch.distributed as dist

# 必须安装 kaldiio: pip install kaldiio
try:
    import kaldiio
except ImportError:
    print("Please install kaldiio: pip install kaldiio")

from model.SenseVoice import SenseVoiceSmall
from model.tokenizer import SenseVoiceTokenizer
from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank

# ================= 配置 =================
BASE_WORK_DIR = "/aistor/sjtu/hpc_stor01/home/wangchenghao/workingspace/ps-slm/Multitask"
OUTPUT_DATA_DIR = os.path.join(BASE_WORK_DIR, "data")
SOURCE_FILES = {
    "test-clean": "/aistor/sjtu/hpc_stor01/home/yangyi/data/asr/test-clean/multitask.jsonl",
    "test-other": "/aistor/sjtu/hpc_stor01/home/yangyi/data/asr/test-other/multitask.jsonl"
}
MODEL_PATH = "/aistor/sjtu/hpc_stor01/home/yangyi/model/SenseVoiceSmall"
BLANK_THRESHOLD = 0.90
TOP_K = 10 

def setup_distributed():
    if not dist.is_initialized():
        deepspeed.init_distributed(dist_backend="hccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.npu.set_device(rank)
    return rank, world_size

def load_audio_flexible(path):
    """
    兼容 .wav 路径和 .ark:offset 格式
    """
    if ":" in path:
        # Kaldi ark 格式
        rate, array = kaldiio.load_mat(path)
        waveform = torch.from_numpy(array).float()
        if waveform.abs().max() > 1.0:
            waveform /= 32768.0
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        return waveform, rate
    else:
        # 普通音频格式
        return torchaudio.load(path)

def psd_processing(ctc_posterior, blank_id=0, blank_threshold=0.90):
    T, V = ctc_posterior.shape
    ids = ctc_posterior.argmax(dim=-1)
    merged_probs = []
    start = 0
    for end in range(1, T + 1):
        if end == T or ids[end] != ids[start]:
            if ids[start] == blank_id:
                for t in range(start, end): merged_probs.append(ctc_posterior[t])
            else:
                merged_probs.append(ctc_posterior[start:end].mean(dim=0))
            start = end
    if not merged_probs: return None, 0
    merged_probs = torch.stack(merged_probs, dim=0)
    mask = merged_probs[:, blank_id] < blank_threshold
    keep = mask.nonzero(as_tuple=False).squeeze(-1)
    final_probs = merged_probs[keep] if keep.numel() > 0 else merged_probs[:1]
    return final_probs, final_probs.size(0)

def main():
    rank, world_size = setup_distributed()
    device = f"npu:{rank}"
    
    if rank == 0:
        print(f"--> [Rank 0] Loading model from {MODEL_PATH}...", flush=True)

    model, model_kwargs = SenseVoiceSmall.from_pretrained(MODEL_PATH, device=device)
    model.eval()
    frontend = model_kwargs.get("frontend")
    tokenizer = SenseVoiceTokenizer(MODEL_PATH)
    
    dist.barrier()

    for split, jsonl_path in SOURCE_FILES.items():
        split_dir = os.path.join(OUTPUT_DATA_DIR, split)
        if rank == 0:
            os.makedirs(split_dir, exist_ok=True)
            print(f"--> [Rank 0] Processing split: {split}", flush=True)
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
        
        my_lines = all_lines[rank::world_size]
        temp_path = os.path.join(split_dir, f"temp_rank{rank}.jsonl")
        
        with open(temp_path, 'w', encoding='utf-8') as out_f:
            pbar = tqdm(my_lines, desc=f"Rank {rank}", disable=(rank != 0))
            for line in pbar:
                try:
                    item = json.loads(line.strip())
                    audio_path = item['path']
                    
                    # 使用灵活加载函数
                    waveform, sr = load_audio_flexible(audio_path)
                    if sr != 16000:
                        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
                    
                    audio_list = load_audio_text_image_video([waveform.squeeze()], fs=16000, audio_fs=16000, data_type="sound")
                    input_f, input_l = extract_fbank(audio_list, data_type="sound", frontend=frontend)
                    
                    with torch.no_grad():
                        # ASR Prompt
                        l_q = model.embed(torch.tensor([[0]], device=device))
                        e_q = model.embed(torch.tensor([[1, 2]], device=device))
                        t_q = model.embed(torch.tensor([[1]], device=device)) # TextNorm=1
                        
                        speech_in = torch.cat([l_q, e_q, t_q, input_f.to(device)], dim=1)
                        encoder_out, _ = model.encoder(speech_in, input_l.to(device) + 4)
                        logits = model.ctc.ctc_lo(encoder_out[:, 4:, :])
                        
                        probs = torch.softmax(logits, dim=-1).squeeze(0)
                        psd_out, psd_len = psd_processing(probs, model.blank_id, BLANK_THRESHOLD)

                    if psd_out is None: continue

                    topk_v, topk_i = torch.topk(psd_out, TOP_K, dim=-1)
                    gt_text = item.get('GT', item.get('target', ''))
                    processed_text = gt_text.lower().replace("'", "")
                    token_ids = tokenizer.encode(processed_text)
                    
                    save_path = os.path.join(split_dir, f"{item['key']}.pt")
                    torch.save({
                        "psd_indices": topk_i.cpu().to(torch.int32).numpy().flatten(),
                        "psd_values": topk_v.cpu().to(torch.float16).numpy().flatten(),
                        "text_ids": np.array(token_ids, dtype=np.int32),
                        "shape": list(psd_out.shape)
                    }, save_path)
                    
                    item.update({
                        'psd_path': save_path, 
                        'psd_len': int(psd_len), 
                        'text_len': len(token_ids)
                    })
                    out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
                except Exception as e:
                    # 打印报错，方便观察
                    # print(f"Error on {item.get('key')}: {e}")
                    continue

    dist.barrier()
    if rank == 0:
        for split in SOURCE_FILES:
            final_jsonl = os.path.join(OUTPUT_DATA_DIR, split, "multitask.jsonl")
            with open(final_jsonl, 'w') as f_out:
                for r in range(world_size):
                    tmp = os.path.join(OUTPUT_DATA_DIR, split, f"temp_rank{r}.jsonl")
                    if os.path.exists(tmp):
                        with open(tmp, 'r') as f_tmp: f_out.write(f_tmp.read())
                        os.remove(tmp)
        print("--> All Done.", flush=True)

if __name__ == "__main__":
    main()