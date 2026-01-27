import os
import json
import torch
import torch_npu
import numpy as np
import torchaudio
import random
from tqdm import tqdm
import torch.distributed as dist
from scipy.signal import butter, sosfilt

from model.SenseVoice import SenseVoiceSmall
from model.tokenizer import SenseVoiceTokenizer
from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank
from preprocess_data import load_audio_flexible, psd_processing, setup_distributed
import pt 

# ================= 配置 =================
BASE_DIR = "/aistor/aispeech/hpc_stor01/home/wangchenghao00sx/workingspace/TASU-simulator/Multitask"
MODEL_PATH = "/aistor/aispeech/hpc_stor01/home/wangchenghao00sx/.cache/modelscope/hub/models/iic/SenseVoiceSmall"
INPUT_JSONL = os.path.join(BASE_DIR, "data/train/multitask.jsonl") 
OUTPUT_PT_DIR = os.path.join(BASE_DIR, "data/train/augmented_psd_files")
INFERENCE_BATCH_SIZE = 24 # 大规模处理时，适当增加 Batch Size 提升吞吐

class NativeAugmentor:
    @staticmethod
    def shift(samples):
        fraction = np.random.uniform(-0.03, 0.03)
        return np.roll(samples, int(len(samples) * fraction))

    @staticmethod
    def add_gaussian_snr(samples, min_db, max_db):
        snr = np.random.uniform(min_db, max_db)
        clean_rms = np.sqrt(np.mean(samples**2) + 1e-9)
        noise_rms = clean_rms / (10**(snr / 20))
        return samples + np.random.normal(0, noise_rms, samples.shape).astype(np.float32)

    @staticmethod
    def freq_mask(samples, sr=16000):
        bandwidth = np.random.randint(0.02 * sr // 2, 0.06 * sr // 2)
        start = np.random.randint(16, sr // 2 - bandwidth - 1)
        sos = butter(6, [start, start + bandwidth], btype="bandstop", output="sos", fs=sr)
        return sosfilt(sos, samples).astype(np.float32)

    @staticmethod
    def time_mask(samples):
        t_len = np.random.randint(int(len(samples) * 0.03), int(len(samples) * 0.07))
        t_start = np.random.randint(0, len(samples) - t_len)
        new_s = samples.copy()
        new_s[t_start : t_start + t_len] = 0
        return new_s

    @staticmethod
    def distortion(samples):
        gain = np.random.uniform(1.2, 2.0)
        distorted = np.tanh(gain * samples)
        return (np.sqrt(np.mean(samples**2)+1e-9) / np.sqrt(np.mean(distorted**2)+1e-9)) * distorted

def apply_aug(samples, tid):
    aug = NativeAugmentor()
    if tid == "L": return aug.add_gaussian_snr(aug.shift(samples), 22, 32)
    if tid == "M": return aug.add_gaussian_snr(aug.time_mask(aug.freq_mask(samples)), 16, 26)
    if tid == "H": return aug.add_gaussian_snr(aug.distortion(samples), 8, 18)
    return samples

def process_batch(task_buffer, model, device, frontend, tokenizer, calc, out_f, output_dir):
    if not task_buffer: return
    waves = []
    for item, tid in task_buffer:
        try:
            w, sr = load_audio_flexible(item['path'])
            if sr != 16000: w = torchaudio.transforms.Resample(sr, 16000)(w)
            waves.append(torch.from_numpy(apply_aug(w.numpy().squeeze(), tid)))
        except: continue
    if not waves: return

    audio_list = load_audio_text_image_video(waves, fs=16000, audio_fs=16000, data_type="sound")
    input_f, input_l = extract_fbank(audio_list, data_type="sound", frontend=frontend)
    
    with torch.no_grad():
        b_sz = len(waves)
        l_q, e_q, t_q = model.embed(torch.zeros(b_sz, 1, dtype=torch.long, device=device)), \
                        model.embed(torch.full((b_sz, 2), 1, dtype=torch.long, device=device)), \
                        model.embed(torch.full((b_sz, 1), 1, dtype=torch.long, device=device))
        encoder_out, _ = model.encoder(torch.cat([l_q, e_q, t_q, input_f.to(device)], dim=1), input_l.to(device) + 4)
        all_probs = torch.softmax(model.ctc.ctc_lo(encoder_out[:, 4:, :]), dim=-1)

    for idx, (item, tid) in enumerate(task_buffer):
        if idx >= len(all_probs): break
        psd_out, psd_len = psd_processing(all_probs[idx], model.blank_id, 0.90)
        if psd_out is None: continue
        
        ref_ids = [x for x in tokenizer.encode(item['target'].lower().replace("'", "")) if x != 0 and x != 25055]
        hyp_ids = [x for x in psd_out.argmax(dim=-1).tolist() if x != 0 and x != 25055]
        res = calc.calculate(ref_ids, hyp_ids)
        new_wer = (res['sub'] + res['del'] + res['ins']) / res['all'] if res['all'] > 0 else 0.0
        
        new_bid = 1 if new_wer < 0.05 else 2 if new_wer < 0.1 else 3 if new_wer < 0.2 else 4
        
        # 核心逻辑：确保 Bucket 1 总量不变，只向 2, 3, 4 输送
        if new_bid == 1: continue

        new_key = f"{item['key']}_aug_{tid}"
        save_path = os.path.join(output_dir, f"{new_key}.pt")
        top_v, top_i = torch.topk(psd_out, 10, dim=-1)
        torch.save({"psd_indices": top_i.cpu().to(torch.int32).numpy().flatten(),
                    "psd_values": top_v.cpu().to(torch.float16).numpy().flatten(),
                    "text_ids": np.array(ref_ids, dtype=np.int32), "shape": list(psd_out.shape)}, save_path)
        
        aug_item = item.copy()
        aug_item.update({"key": new_key, "psd_path": save_path, "psd_len": int(psd_len), "bucket_id": new_bid,
                         "bpe_error": {"RefLen": res['all'], "S": res['sub'], "D": res['del'], "I": res['ins'], "WER": new_wer}})
        out_f.write(json.dumps(aug_item, ensure_ascii=False) + "\n")

def main():
    rank, world_size = setup_distributed()
    device = f"npu:{rank}"
    if rank == 0: os.makedirs(OUTPUT_PT_DIR, exist_ok=True)
    dist.barrier()
    
    model, model_kwargs = SenseVoiceSmall.from_pretrained(MODEL_PATH, device=device)
    model.eval()
    frontend, tokenizer, calc = model_kwargs.get("frontend"), SenseVoiceTokenizer(MODEL_PATH), pt.Calculator()
    
    # 策略概率：目标是补齐 Bucket 2, 3, 4 缺失的约 40w 条数据
    # 这里几乎对 Bucket 1 做全量触发
    prob_map = {
        1: {"L": 0.95, "M": 0.95, "H": 0.98}, 
        2: {"M": 0.85, "H": 0.85},
        3: {"H": 0.85}
    }
    
    with open(INPUT_JSONL, 'r', encoding='utf-8') as f:
        my_lines = f.readlines()[rank::world_size]
    
    temp_jsonl = os.path.join(OUTPUT_PT_DIR, f"temp_rank{rank}.jsonl")
    buffer = []
    with open(temp_jsonl, 'w', encoding='utf-8') as out_f:
        for line in tqdm(my_lines, desc=f"Rank {rank}", disable=(rank!=0)):
            item = json.loads(line.strip())
            wer = item.get("bpe_error", {}).get("WER", 0.0)
            bid = 1 if wer < 0.05 else 2 if wer < 0.1 else 3 if wer < 0.2 else 4
            item["bucket_id"] = bid
            out_f.write(json.dumps(item, ensure_ascii=False) + "\n") 
            
            if bid in prob_map:
                for tid, p in prob_map[bid].items():
                    if random.random() < p: buffer.append((item, tid))
            
            if len(buffer) >= INFERENCE_BATCH_SIZE:
                process_batch(buffer, model, device, frontend, tokenizer, calc, out_f, OUTPUT_PT_DIR)
                buffer = []
        if buffer: process_batch(buffer, model, device, frontend, tokenizer, calc, out_f, OUTPUT_PT_DIR)

    dist.barrier()
    if rank == 0:
        final_jsonl = INPUT_JSONL.replace(".jsonl", "_augmented.jsonl")
        with open(final_jsonl, 'w', encoding='utf-8') as f_out:
            for r in range(world_size):
                tmp = os.path.join(OUTPUT_PT_DIR, f"temp_rank{r}.jsonl")
                if os.path.exists(tmp):
                    with open(tmp, 'r') as ft: f_out.write(ft.read())
                    os.remove(tmp)
        print(f"--> DONE. Final total targeted at ~700k lines.")

if __name__ == "__main__": main()