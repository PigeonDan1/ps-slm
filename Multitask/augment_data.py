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

BASE_DIR = "/aistor/aispeech/hpc_stor01/home/wangchenghao00sx/workingspace/TASU-simulator/Multitask"
MODEL_PATH = "/aistor/aispeech/hpc_stor01/home/wangchenghao00sx/.cache/modelscope/hub/models/iic/SenseVoiceSmall"
INPUT_JSONL = os.path.join(BASE_DIR, "data/train/multitask_filtered.jsonl") 
OUTPUT_PT_DIR = os.path.join(BASE_DIR, "data/train/augmented_psd_files")
INFERENCE_BATCH_SIZE = 48 # NPU 显存较大，可增加 Batch Size 提升并行效率

class NoiseAugmentor:
    """阶梯式增强：通过低通滤波和强底噪精准控制 WER 区间"""
    @staticmethod
    def add_gaussian_snr(samples, min_db, max_db):
        snr = np.random.uniform(min_db, max_db)
        clean_rms = np.sqrt(np.mean(samples**2) + 1e-9)
        noise_rms = clean_rms / (10**(snr / 20))
        return samples + np.random.normal(0, noise_rms, samples.shape).astype(np.float32)

    @staticmethod
    def apply_lowpass(samples, cutoff, sr=16000):
        # 稳定的高频切除，3000Hz以下会显著提升替换错误
        sos = butter(6, cutoff, btype="lowpass", output="sos", fs=sr)
        return sosfilt(sos, samples).astype(np.float32)

    @staticmethod
    def freq_mask(samples, sr=16000, width_ratio=0.15):
        # 拓宽掩码范围，直接切断关键共振峰
        bandwidth = np.random.randint(int(0.08 * sr // 2), int(width_ratio * sr // 2))
        start = np.random.randint(16, sr // 2 - bandwidth - 1)
        sos = butter(6, [start, start + bandwidth], btype="bandstop", output="sos", fs=sr)
        return sosfilt(sos, samples).astype(np.float32)

    @staticmethod
    def time_mask(samples, max_ratio=0.06):
        # 采用短而多次的遮蔽，模拟信号闪变而不破坏整体节奏
        new_s = samples.copy()
        for _ in range(random.randint(1, 2)):
            t_len = np.random.randint(int(len(samples) * 0.03), int(len(samples) * max_ratio))
            t_start = np.random.randint(0, len(samples) - t_len)
            new_s[t_start : t_start + t_len] = 0
        return new_s
        
def apply_noise_aug(samples, tid):
    aug = NoiseAugmentor()
    
    # L: 目标 15-30%。
    # 必须引入 3800Hz 低通 + 强噪声(0-5dB)，否则在大数据集上依然会堆积在 10% 以下。
    if tid == "L": 
        samples = aug.apply_lowpass(samples, cutoff=3800)
        samples = aug.freq_mask(samples, width_ratio=0.12)
        return aug.add_gaussian_snr(samples, 0, 5)
    
    # M: 目标 35-55%。
    # 低通压到 2500Hz（接近窄带上限），配合 0-3dB 噪声，这是填补 B2 区间的核心。
    if tid == "M": 
        samples = aug.apply_lowpass(samples, cutoff=2500)
        samples = aug.time_mask(samples, max_ratio=0.05)
        return aug.add_gaussian_snr(samples, 0, 3)
    
    # H: 目标 60-120% (上限 150%)。
    # 截止频率下压至 1800Hz，这是为了在不产生“幻听”的前提下，最大化替换错误。
    if tid == "H": 
        samples = aug.apply_lowpass(samples, cutoff=1800)
        # 增加掩码次数但保持短时长，模拟极差的网络抖动
        for _ in range(2): 
            samples = aug.time_mask(samples, max_ratio=0.04)
        return aug.add_gaussian_snr(samples, 2, 6)
    
    return samples

def process_batch(task_buffer, model, device, frontend, tokenizer, calc, out_f, output_dir):
    if not task_buffer: return
    waves = []
    valid_tasks = []
    for item, tid in task_buffer:
        try:
            w, sr = load_audio_flexible(item['path'])
            if sr != 16000: w = torchaudio.transforms.Resample(sr, 16000)(w)
            waves.append(torch.from_numpy(apply_noise_aug(w.numpy().squeeze(), tid)))
            valid_tasks.append((item, tid))
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

    for idx, (item, tid) in enumerate(valid_tasks):
        if idx >= len(all_probs): break
        psd_out, psd_len = psd_processing(all_probs[idx], model.blank_id, 0.90)
        if psd_out is None: continue
        
        ref_ids = [x for x in tokenizer.encode(item['target'].lower().replace("'", "")) if x != 0 and x != 25055]
        hyp_ids = [x for x in psd_out.argmax(dim=-1).tolist() if x != 0 and x != 25055]
        res = calc.calculate(ref_ids, hyp_ids)
        new_wer = (res['sub'] + res['del'] + res['ins']) / res['all'] if res['all'] > 0 else 0.0
        
        if new_wer > 2.0:
            continue
        # 记录新的 Bucket ID
        new_bid = 1 if new_wer < 0.05 else 2 if new_wer < 0.1 else 3 if new_wer < 0.2 else 4

        new_key = f"{item['key']}_aug_{tid}"
        save_path = os.path.join(output_dir, f"{new_key}.pt")
        top_v, top_i = torch.topk(psd_out, 10, dim=-1)
        torch.save({"psd_indices": top_i.cpu().to(torch.int32).numpy().flatten(),
                    "psd_values": top_v.cpu().to(torch.float16).numpy().flatten(),
                    "text_ids": np.array(ref_ids, dtype=np.int32), "shape": list(psd_out.shape)}, save_path)
        
        aug_item = item.copy()
        aug_item.update({
            "key": new_key, 
            "psd_path": save_path, 
            "psd_len": int(psd_len), 
            "bucket_id": new_bid,
            "bpe_error": {"RefLen": res['all'], "S": res['sub'], "D": res['del'], "I": res['ins'], "WER": new_wer}
        })
        out_f.write(json.dumps(aug_item, ensure_ascii=False) + "\n")

def main():
    rank, world_size = setup_distributed()
    device = f"npu:{rank}"
    if rank == 0: os.makedirs(OUTPUT_PT_DIR, exist_ok=True)
    dist.barrier()
    
    model, model_kwargs = SenseVoiceSmall.from_pretrained(MODEL_PATH, device=device)
    model.eval()
    frontend, tokenizer, calc = model_kwargs.get("frontend"), SenseVoiceTokenizer(MODEL_PATH), pt.Calculator()
    
    with open(INPUT_JSONL, 'r', encoding='utf-8') as f:
        my_lines = f.readlines()[rank::world_size]
    
    temp_jsonl = os.path.join(OUTPUT_PT_DIR, f"temp_rank{rank}.jsonl")
    buffer = []
    
    with open(temp_jsonl, 'w', encoding='utf-8') as out_f:
        for line in tqdm(my_lines, desc=f"Rank {rank}", disable=(rank!=0)):
            item = json.loads(line.strip())
            
            # 1. 写入原始数据（带原始 bucket_id）
            wer = item.get("bpe_error", {}).get("WER", 0.0)
            bid = 1 if wer < 0.05 else 2 if wer < 0.1 else 3 if wer < 0.2 else 4
            item["bucket_id"] = bid
            out_f.write(json.dumps(item, ensure_ascii=False) + "\n") 
            
            # 2. 准备 L, M, H 三种增强任务，不设概率，100% 触发
            for tid in ["L", "M", "H"]:
                buffer.append((item, tid))
                
                # 当 Buffer 达到推理批次大小时进行处理
                if len(buffer) >= INFERENCE_BATCH_SIZE:
                    process_batch(buffer, model, device, frontend, tokenizer, calc, out_f, OUTPUT_PT_DIR)
                    buffer = []
                    
        # 处理剩余 buffer
        if buffer: 
            process_batch(buffer, model, device, frontend, tokenizer, calc, out_f, OUTPUT_PT_DIR)

    dist.barrier()
    if rank == 0:
        final_jsonl = INPUT_JSONL.replace(".jsonl", "_augmented.jsonl")
        with open(final_jsonl, 'w', encoding='utf-8') as f_out:
            for r in range(world_size):
                tmp = os.path.join(OUTPUT_PT_DIR, f"temp_rank{r}.jsonl")
                if os.path.exists(tmp):
                    with open(tmp, 'r') as ft: f_out.write(ft.read())
                    os.remove(tmp)
        print(f"--> DONE. Total data is now 4x original.")

if __name__ == "__main__": main()