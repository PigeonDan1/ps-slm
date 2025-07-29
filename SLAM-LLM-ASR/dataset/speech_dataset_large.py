import torch
from torch.utils.data import Dataset,IterableDataset
import whisper
import kaldiio
import types
from functools import partial
import torch.distributed as dist
import string
import copy
import numpy as np
import copy
from tqdm import tqdm
import os
import json
import random
import torchaudio
import random
import logging
import subprocess
import torchaudio
import torchaudio.compliance.kaldi as kaldi

class MultiTaskDataset(IterableDataset):
    def __init__(self, dataset_config, tokenizer=None, split='train'):
        super().__init__()
        self.multitask_prompt_list = {}
        self.append_info_tasks = dataset_config.append_info_tasks
        with open(dataset_config.multitask_prompt_path) as f_prompt:
            for line in f_prompt:
                item = json.loads(line.strip())
                if item["task"] in self.multitask_prompt_list:
                    self.multitask_prompt_list[item["task"]].append(item["prompt"])
                else:
                    self.multitask_prompt_list[item["task"]] = [item["prompt"]]
        # print(f"[Prompt] {self.multitask_prompt_list}")
        if split == "train":
            self.data_path = dataset_config.train_scp_file_path
        elif split == "val":
            self.data_path = dataset_config.dev_scp_file_path
        elif split == "test":
            self.data_path = dataset_config.test_scp_file_path
        else:
            raise ValueError("Split must be train val test")
        
        self.prompt_template = dataset_config.get("prompt_style", "{}")
        self.dataset_config = dataset_config
        self.tokenizer = tokenizer
        self.split = split
        self.max_audio_length = dataset_config.get("max_audio_length", 30)
        self.inference_mode = dataset_config.get("inference_mode", False)
        self.sample_rate = 16000
        self.encoder_path = dataset_config.encoder_path
        from model.SenseVoice import SenseVoiceSmall
        encoder, kwargs = SenseVoiceSmall.from_pretrained(self.encoder_path)
        self.kwargs = kwargs
        del encoder # without memory waste

    def __iter__(self):
        multitask_task_path = os.path.join(self.data_path,"multitask.jsonl")
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  
            num_workers = 1
            worker_id = 0
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id

        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
        else:
            world_size = 1
            rank = 0

        total_num_workers = num_workers * world_size
        worker_rank = rank * num_workers + worker_id 
        with open(multitask_task_path) as f_task:
            for data_index,line in enumerate(f_task):
                if (data_index % total_num_workers) == worker_rank:
                    # try:
                    item = json.loads(line.strip())
                    ark_path = item["path"]
                    key = item["key"]
                    target = item["target"]
                    task = item["task"]
                    # pj: Modify the dataset based on your own encoder
                    if self.dataset_config.encoder == "whisper":
                        numpy_array = kaldiio.load_mat(ark_path)
                        audio_raw = numpy_array[1].astype(np.float32) / 32768
                        if len(audio_raw) / self.sample_rate > self.max_audio_length or len(audio_raw) / self.sample_rate < 0.1: 
                            continue
                        audio_raw = torch.from_numpy(audio_raw).float().unsqueeze(0)
                        input_features = self.extract_fbank(audio_raw)
                        audio_raw = whisper.pad_or_trim(audio_raw)
                        input_features = whisper.log_mel_spectrogram(audio_raw, n_mels=128)
                        input_features = input_features.squeeze(0)
                        input_feature_length = input_features.shape[1]
                    else:
                        import torchaudio
                        numpy_array = kaldiio.load_mat(ark_path)        # int16
                        audio_raw = numpy_array[1].astype(np.float32) / 32768.0
                        data_in = [audio_raw]      # List[np.ndarray]
                        kwargs = self.kwargs
                        frontend = kwargs.get("frontend", None)
                        from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank
                        audio_list = load_audio_text_image_video(
                            data_in,
                            fs=frontend.fs,
                            audio_fs=16000,
                            data_type="sound",
                        )
                        input_features, input_feature_length = extract_fbank(
                            audio_list,
                            data_type="sound",
                            frontend=frontend
                        ) # [1, T, D=560]
                        input_features, input_feature_length = input_features[0], input_feature_length[0]
                        # print(f"input_feat: {input_features.shape}")
                        # print(f"input_feat_len: {input_feature_length}")
                        # input()
                    
                    prompt = random.choice(self.multitask_prompt_list[task])
                    prompt = self.prompt_template.format(prompt)
                    
                    if task in self.append_info_tasks:
                        prompt = prompt.format(item[task])
                    # print(f"Prompt is {prompt}")
                    prompt_ids = self.tokenizer.encode(prompt)
                    # print(prompt)
                    prompt_length = len(prompt_ids)
                    prompt_ids = torch.tensor(prompt_ids)

                    if  not self.inference_mode:
                        import re
                        if re.fullmatch(r"[A-Za-z\s.,!?']+", target):
                            target = target.lower()
                        target_ids = self.tokenizer.encode(target)
                        target_ids.append(self.tokenizer.eos_token_id)
                        target_ids = torch.tensor(target_ids)
                        input_ids = torch.cat([prompt_ids,target_ids])
                    else:
                        input_ids = prompt_ids
                    attention_mask = input_ids.ge(-1)  
                    result = {
                            "input_ids": input_ids,
                            "attention_mask": attention_mask ,
                            "input_features": input_features ,
                            "input_feature_length":input_feature_length,
                            'key': key,
                            'target': target,
                    }

                    if  not self.inference_mode:
                        labels = copy.deepcopy(input_ids)
                        labels[:prompt_length] = self.tokenizer.default_ignore_token
                        result["labels"] = labels
                    yield result
                    # except:
                    #     print(data_index,item)
            
    def pad(self, sequence, max_length, padding_idx=0,padding_style = "right"):
            if isinstance(sequence, (int, list, tuple)):
                if len(sequence) < max_length:
                    if padding_style == "right":
                        sequence = sequence + [padding_idx] * (max_length - len(sequence))
                    else:
                        sequence =  [padding_idx] * (max_length - len(sequence)) + sequence 
                else:
                    sequence = sequence[:max_length]
            elif isinstance(sequence, torch.Tensor):
                if len(sequence) < max_length:
                    if padding_style == "right":
                        sequence = torch.cat(
                            (sequence, torch.full(([max_length - len(sequence)] + list(sequence.size())[1:]), padding_idx)))
                    else:
                        sequence = torch.cat(
                            (torch.full(([max_length - len(sequence)] + list(sequence.size())[1:]), padding_idx),sequence))
                else:
                    sequence = sequence[:max_length]
            elif isinstance(sequence, np.ndarray):
                if len(sequence) < max_length:
                    if padding_style == "right":
                        sequence = np.concatenate(
                            (sequence, np.full((max_length - len(sequence),) + sequence.shape[1:], padding_idx)))
                    else:
                        sequence = np.concatenate(
                            (np.full((max_length - len(sequence),) + sequence.shape[1:], padding_idx),sequence))
                else:
                    sequence = sequence[:max_length]
            else:
                raise Exception("Type mismatch during padding!")
            return sequence
    def extract_fbank(self,waveform):
        fbank_features = kaldi.fbank(
            waveform,
            num_mel_bins=self.dataset_config["fbankConfig"]["num_mel_bins"],  # 梅尔频率滤波器组的滤波器数量
            frame_length=self.dataset_config["fbankConfig"]["frame_length"],  # 音频帧的长度（毫秒）
            frame_shift=self.dataset_config["fbankConfig"]["frame_shift"],  # 帧移（毫秒）
            dither=self.dataset_config["fbankConfig"]["dither"] if self.split == "train" else 0,  # 抖动系数
            window_type=self.dataset_config["fbankConfig"]["window_type"],  # 窗口类型
            use_energy=self.dataset_config["fbankConfig"]["use_energy"],  # 是否使用能量特征
            low_freq=self.dataset_config["fbankConfig"]["low_freq"],  # 低频截止频率（Hz）
            high_freq=self.dataset_config["fbankConfig"]["high_freq"],  # 高频截止频率（Hz）
            htk_compat=self.dataset_config["fbankConfig"]["htk_compat"]  # 是否与HTK兼容
        )
        return fbank_features
    
    # pj: Add sensevoice 
    def collator(self, samples):
        assert samples is not None
        if self.inference_mode:
            padding_style = "left"
        else:
            padding_style = "right"

        input_ids_max_length = max([s['input_ids'].shape[0] for s in samples])
        input_ids = torch.stack([
            self.pad(s['input_ids'], input_ids_max_length,
                    self.tokenizer.pad_token_id, padding_style=padding_style)
            for s in samples
        ])
        attention_mask = torch.stack([
            self.pad(s['attention_mask'], input_ids_max_length,
                    False, padding_style=padding_style)
            for s in samples
        ])

        if not self.inference_mode:
            labels = torch.stack([
                self.pad(s['labels'], input_ids_max_length,
                        self.tokenizer.default_ignore_token, padding_style=padding_style)
                for s in samples
            ])

        if self.dataset_config.encoder == "whisper":
            # Whisper: [B, D, T]
            input_feature_length = torch.stack([torch.tensor(s["input_feature_length"]) for s in samples])
            max_feat_len = max([s['input_features'].shape[0] for s in samples])
            input_features = torch.stack([
                self.pad(s['input_features'], max_feat_len, 0, padding_style="right")
                for s in samples
            ])
        else:  # SenseVoice: [T, 560] -> [B, T_max, 560]
            input_feature_length = torch.tensor(
                [s["input_feature_length"] for s in samples], dtype=torch.long
            )  # [B]

            max_len = max([s["input_features"].size(0) for s in samples])

            input_features = torch.stack([
                torch.nn.functional.pad(
                    s["input_features"],
                    (0, 0, 0, max_len - s["input_features"].size(0)),  # (left, right, top, bottom)
                    value=0.0
                )
                for s in samples
            ])  # [B, T_max, 80]

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "input_features": input_features,
            "input_feature_length": input_feature_length,
        }
        if self.inference_mode:
            result["keys"] = [s['key'] for s in samples]
            result["targets"] = [s['target'] for s in samples]
        else:
            result["labels"] = labels
        return result

class MultiTaskDynamicBatchDataset(IterableDataset):
    def __init__(self, dataset: IterableDataset, window_class) -> None:
        super().__init__()
        self.dp = dataset
        
        assert window_class is not None
        self.window_class = window_class
        self.collator = self.dp.collator
        self._buffer = []
    def __iter__(self):
        for elem in self.dp:
            if not self.window_class(elem, self._buffer):
                self._buffer.append(elem)
            else:
                if len(self._buffer) > 0:
                    yield self._buffer
                del self._buffer
                self._buffer = [elem]
        if len(self._buffer) > 0:
            yield self._buffer
        del self._buffer
        self._buffer = []
         
    
def window_class(elem,buffer,max_frame_length,ds_rate):
    # return True 
    if len(buffer) == 0:
        return True
    max_frame = max(len(elem["input_ids"]) + (elem["input_feature_length"] // ds_rate) - 1,max([ len(_["input_ids"]) + (_["input_feature_length"] // ds_rate ) -1 for _ in buffer]))
    return (len(buffer) + 1) * max_frame > max_frame_length

def get_speech_dataset(dataset_config, tokenizer, split):
    dataset = MultiTaskDataset(dataset_config, tokenizer, split)
    if split == "train":
        dataset = MultiTaskDynamicBatchDataset(dataset,partial(window_class,max_frame_length = dataset_config.train_max_frame_length,ds_rate = dataset_config.ds_rate))
    else:
        dataset = MultiTaskDynamicBatchDataset(dataset,partial(window_class,max_frame_length = dataset_config.eval_max_frame_length,ds_rate = dataset_config.ds_rate))
    return dataset



    
