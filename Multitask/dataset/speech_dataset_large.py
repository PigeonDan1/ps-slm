# Modify Author: Jing Peng, Yi Yang, X-Lance Lab, SJTU

import torch
from torch.utils.data import Dataset, IterableDataset
import kaldiio
from functools import partial
import torch.distributed as dist
import numpy as np
import os
import json
import random
import re
import copy

# ==========================================
# 1. 工具函数：矩阵还原
# ==========================================

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

class MultiTaskDataset(IterableDataset):
    def __init__(self, dataset_config, tokenizer=None, split='train'):
        super().__init__()
        self.multitask_prompt_list = {}
        self.append_info_tasks = dataset_config.append_info_tasks
        with open(dataset_config.multitask_prompt_path) as f_prompt:
            for line in f_prompt:
                item = json.loads(line.strip())
                self.multitask_prompt_list.setdefault(item["task"], []).append(item["prompt"])

        self.dataset_config = dataset_config
        self.tokenizer = tokenizer
        self.split = split
        self.current_epoch = 0
        self.use_real_ctc = dataset_config.get("use_real_ctc", False) # 默认关闭，使用仿真数据
        print(f"是否use_real_ctc? {self.use_real_ctc}")
        # self.wer_levels = [0, 3, 6, 7, 9] 

        # 路径管理
        if split == "train":
            self.data_path = dataset_config.train_scp_file_path
            self.extra_data_path = dataset_config.train_scp_extra_path if dataset_config.train_scp_extra_path else None
            print(f"成功加载困难数据集{self.extra_data_path}")
        elif split in ["val", "dev"]:
            self.data_path = dataset_config.dev_scp_file_path
            self.extra_data_path = dataset_config.train_scp_extra_path if dataset_config.train_scp_extra_path else None
            print(f"成功加载困难数据集{self.extra_data_path}")
        elif split == "test":
            self.data_path = dataset_config.test_scp_file_path
        
        self.prompt_template = dataset_config.get("prompt_style", "{}")
        self.inference_mode = dataset_config.get("inference_mode", False)

        # [恢复功能] 为推理模式加载 SenseVoice 前端参数
        if self.inference_mode or self.split == "test":
            self.encoder_path = dataset_config.encoder_path
            from model.SenseVoice import SenseVoiceSmall
            _, kwargs = SenseVoiceSmall.from_pretrained(self.encoder_path)
            self.kwargs = kwargs # 包含 frontend 信息

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def __iter__(self):
        base_jsonl = os.path.join(self.data_path, "multitask.jsonl")
        extra_jsonl = None
        if self.split in ["train", "val", "dev"]:
            extra_jsonl = os.path.join(self.extra_data_path, "multitask.jsonl") if self.extra_data_path else None
        worker_info = torch.utils.data.get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        total_workers = num_workers * world_size
        worker_rank = rank * num_workers + (worker_info.id if worker_info else 0)

        b2_ratio = 0.0 #课程学习的困难数据集的比例
        if self.split in ["train"]:
            b2_ratio = (self.current_epoch - 1) * 0.05

        if not os.path.exists(base_jsonl): 
            return

        with open(base_jsonl, 'r', encoding='utf-8') as f:
            f_extra = open(extra_jsonl, 'r', encoding='utf-8') if extra_jsonl else None
            for data_index, line in enumerate(f):
                line_extra = f_extra.readline() if f_extra else None #二者指针同步
                if (data_index % total_workers) != worker_rank: 
                    continue
                
                if line_extra and random.random() < b2_ratio and extra_jsonl and line_extra:
                    item = json.loads(line_extra.strip())
                else:
                    item = json.loads(line.strip())
                key, task = item["key"], item["task"]
                
                input_features, input_feature_length = None, None
                sim_ctc, sim_ctc_len = None, None

                # 情况 A：训练/验证/开发
                if self.split in ["train", "val", "dev"]:
                    path_key = "psd_path" if self.use_real_ctc else "sim_psd_path"
                    pt_path = item.get(path_key)

                    if pt_path and os.path.exists(pt_path):
                        data = torch.load(pt_path, map_location='cpu', weights_only=False)
                        # 2. 内部逻辑已包含：从 25056 切除至 25055 (indices 中的 25055 被丢弃)
                        sim_ctc = restore_dense_matrix(data['psd_indices'], data['psd_values'])
                        sim_ctc_len = torch.tensor(sim_ctc.size(0), dtype=torch.long)
                        input_features = torch.zeros(sim_ctc.size(0), 560) 
                        input_feature_length = sim_ctc_len
                    else: 
                        continue

                # 情况 B：测试 (走真实音频提取逻辑)
                else:
                    ark_path = item["path"]
                    try:
                        import torchaudio
                        from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank
                        frontend = self.kwargs.get("frontend", None)
                        
                        if task in ["QA", "EN2DE"] or ark_path.endswith(".wav"):
                            waveform, _ = torchaudio.load(ark_path)
                            audio_raw = waveform.mean(dim=0) if waveform.shape[0] > 1 else waveform.squeeze(0)
                        else:
                            audio_raw = kaldiio.load_mat(ark_path)[1].astype(np.float32) / 32768.0

                        audio_list = load_audio_text_image_video([audio_raw], fs=frontend.fs, audio_fs=16000, data_type="sound")
                        fbank, fbank_len = extract_fbank(audio_list, data_type="sound", frontend=frontend)
                        input_features, input_feature_length = fbank[0], fbank_len[0]
                    except Exception as e:
                        print(f"[SKIP] key={key} err={e}")
                        continue

                # Prompt 处理逻辑保持不变
                prompt = random.choice(self.multitask_prompt_list[task])
                if task == "QA": 
                    prompt += f"\n\nTask: {item['task']}\n{item['prompt']}"
                prompt = self.prompt_template.format(prompt)
                if task in self.append_info_tasks: 
                    prompt = prompt.format(item[task])

                prompt_ids = self.tokenizer.encode(prompt)
                prompt_ids_tensor = torch.tensor(prompt_ids)

                target_text = item["target"]
                if not self.inference_mode:
                    if task == "ASR": 
                        target_text = re.sub(r"[^A-Za-z\s.,!?']+", "", target_text).lower().strip()
                    target_ids = self.tokenizer.encode(target_text) + [self.tokenizer.eos_token_id]
                    input_ids = torch.cat([prompt_ids_tensor, torch.tensor(target_ids)])
                else:
                    input_ids = prompt_ids_tensor
                
                result = {
                    "input_ids": input_ids,
                    "attention_mask": input_ids.ge(-1),
                    "input_features": input_features,
                    "input_feature_length": input_feature_length,
                    "sim_ctc": sim_ctc,
                    "sim_ctc_len": sim_ctc_len,
                    "key": key,
                    "target": target_text,
                    "GT": item.get("GT", "").encode('utf-8').decode('unicode_escape'),
                }
                if not self.inference_mode:
                    labels = input_ids.clone()
                    labels[:len(prompt_ids)] = self.tokenizer.default_ignore_token
                    result["labels"] = labels
                
                yield result

    def pad(self, sequence, max_length, padding_idx=0, padding_style="right"):
        if sequence is None: return None
        if isinstance(sequence, torch.Tensor):
            if len(sequence) < max_length:
                fill_shape = [max_length - len(sequence)] + list(sequence.size())[1:]
                fill = torch.full(fill_shape, padding_idx, dtype=sequence.dtype, device=sequence.device)
                return torch.cat((sequence, fill) if padding_style == "right" else (fill, sequence))
            return sequence[:max_length]
        return sequence

    def collator(self, samples):
        if not samples: return None
        padding_style = "left" if self.inference_mode else "right"
        input_ids_max = max([s['input_ids'].shape[0] for s in samples])
        
        # 处理特征长度
        feat_lens = [s["input_feature_length"] for s in samples]
        max_feat_len = max(feat_lens)

        res = {
            "input_ids": torch.stack([self.pad(s['input_ids'], input_ids_max, self.tokenizer.pad_token_id, padding_style) for s in samples]),
            "attention_mask": torch.stack([self.pad(s['attention_mask'], input_ids_max, False, padding_style) for s in samples]),
            "input_features": torch.stack([torch.nn.functional.pad(s["input_features"], (0,0,0, max_feat_len - s["input_features"].size(0))) for s in samples]),
            "input_feature_length": torch.tensor(feat_lens, dtype=torch.long),
            "GT": [s["GT"] for s in samples]
        }
        
        # 只有在训练模式下(sim_ctc不为None)才进行 sim_ctc 的 stack
        if samples[0]["sim_ctc"] is not None:
            res["sim_ctc"] = torch.stack([torch.nn.functional.pad(s["sim_ctc"], (0,0,0, max_feat_len - s["sim_ctc"].size(0))) for s in samples])
            res["sim_ctc_len"] = res["input_feature_length"]

        if self.inference_mode:
            res["keys"], res["targets"] = [s['key'] for s in samples], [s['target'] for s in samples]
        else:
            res["labels"] = torch.stack([self.pad(s['labels'], input_ids_max, self.tokenizer.default_ignore_token, padding_style) for s in samples])
        return res

class MultiTaskDynamicBatchDataset(IterableDataset):
    def __init__(self, dataset, window_class):
        super().__init__()
        self.dp, self.window_class, self.collator = dataset, window_class, dataset.collator
        self._buffer = []
        if hasattr(self.dp, 'set_epoch'): self.set_epoch = self.dp.set_epoch
    def __iter__(self):
        for elem in self.dp:
            if not self.window_class(elem, self._buffer): self._buffer.append(elem)
            else:
                if self._buffer: yield self._buffer
                self._buffer = [elem]
        if self._buffer: yield self._buffer
        self._buffer = []

def window_class(elem, buffer, max_frame_length, ds_rate):
    if not buffer: return True
    max_frame = max(len(elem["input_ids"]) + (elem["input_feature_length"] // ds_rate) - 1, 
                   max([len(_["input_ids"]) + (_["input_feature_length"] // ds_rate) - 1 for _ in buffer]))
    return (len(buffer) + 1) * max_frame > max_frame_length

def get_speech_dataset(dataset_config, tokenizer, split):
    dataset = MultiTaskDataset(dataset_config, tokenizer, split)
    max_len = dataset_config.train_max_frame_length if split == "train" else dataset_config.eval_max_frame_length
    return MultiTaskDynamicBatchDataset(dataset, partial(window_class, max_frame_length=max_len, ds_rate=dataset_config.ds_rate))