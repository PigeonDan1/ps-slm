import json
import torch
import torchaudio
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import kaldiio
import numpy as np
import re
import unicodedata
import torchaudio.compliance.kaldi as kaldi
import torchaudio.transforms as T

# 1. 随机 SpecAugment（频率 + 时间掩码）
def spec_augment(x, freq_mask=27, time_mask=100, num_masks=2):
    """
    x: [80, T]  float32
    return: [80, T] 原地 mask
    """
    # 频率掩码
    if freq_mask > 0:
        fmask = T.FrequencyMasking(freq_mask_param=freq_mask)
        x = fmask(x)

    # 时间掩码（最多 num_masks 次）
    if time_mask > 0:
        tmask = T.TimeMasking(time_mask_param=time_mask)
        for _ in range(num_masks):
            x = tmask(x)
    return x

def normalize_librispeech(text: str) -> str:
    """
    与 Kaldi、WeNet、ESPnet、Whisper 完全对齐的 LibriSpeech 文本归一化。
    输出：小写、无标点、保留空格，适合字符级或子词级 CTC。
    """
    # 1. Unicode 正规化
    text = unicodedata.normalize("NFKC", text)

    # 2. 全大写→小写
    text = text.lower()

    # 3. 移除标点，仅保留 a-z 和空格
    text = re.sub(r"[^a-z ]", " ", text)

    # 4. 合并连续空格
    text = re.sub(r"\s+", " ", text).strip()

    return text

class JsonlCTCDataset(Dataset):
    def __init__(self, jsonl_file, tokenizer, feature_extractor, max_input_length=3000, mode="train", encoder="whisper"):
        self.samples = []
        with open(jsonl_file) as f:
            for line in f:
                self.samples.append(json.loads(line.strip()))

        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.max_input_length = max_input_length
        self.mode = mode
        self.use_custom_vocab = hasattr(tokenizer, "text_to_sequence")
        self.encoder = encoder

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        path = sample["path"]
        text = sample["target"]
        uid = sample["key"] 

        import soundfile as sf
        sr, wav = kaldiio.load_mat(path)
        if path.endswith(".wav"):
            wav, sr = torchaudio.load(path)
            wav = wav.squeeze(0).numpy()
        else:
            sr, wav = kaldiio.load_mat(path)
            wav = wav.astype(np.float32) / 32768.0   # damn!!!!!!!!!, 必须要归一化！
        if sr != 16000:
            print(f"{path}'s sampling rate is not 16khz.")

        # 兼容 HuggingFace 和 Vocabulary 的 tokenizer
        if self.use_custom_vocab:
            labels = self.tokenizer.text_to_sequence(text)
        else:
            with self.tokenizer.as_target_tokenizer():
                if "librispeech" in path.lower():
                    text = normalize_librispeech(text) # 英文文本正则化 for qwen-2.5

                labels = self.tokenizer(text, add_special_tokens=False).input_ids
                # print(labels)
                # input()

        if self.encoder == "whisper":
            output = self.feature_extractor(
                wav,
                sampling_rate=16000,
                return_attention_mask=True,
            )
            input_features = output.input_features[0]        # [80, T]
            attention_mask = output.attention_mask[0]        # [T]
            input_length = int(attention_mask.sum()) // 10   # 注意：×2 ×5 下采样

            # SpecAugment for balance dev/train loss in librispeech 
            # if self.mode == "train" and "librispeech" in path.lower():
            #     input_features = torch.tensor(input_features, dtype=torch.float)  # [80, T]
            #     input_features = spec_augment(input_features)

            return {
                "uid": uid, 
                "encoder": self.encoder,
                "input_features": torch.tensor(input_features, dtype=torch.float),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                "input_length": input_length,
                "labels": torch.tensor(labels, dtype=torch.long),
            }
        else:
            input_features, input_length = self.feature_extractor(
                path,
                sampling_rate=16000
            ) # [T,D], [1]
            return {
                "uid": uid, 
                "encoder": self.encoder,
                "input_features": torch.tensor(input_features, dtype=torch.float),
                "input_length": input_length,
                "labels": torch.tensor(labels, dtype=torch.long),
            }

def ctc_collate_fn(batch):
    encoder = batch[0]["encoder"]
    input_lengths = torch.tensor([item["input_length"] for item in batch])  # scalar
    labels = [item["labels"] for item in batch]
    label_lengths = torch.tensor([len(x) for x in labels])
    uids = [item["uid"] for item in batch]

    # Pad labels to [B, L_max]
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)

    if encoder == "whisper":
        input_features = [item["input_features"] for item in batch]         # [D, T]
        input_features_stacked = torch.stack(input_features) 
        attention_masks = [item["attention_mask"] for item in batch]        # [T]
        attention_masks_stacked = torch.stack(attention_masks)
        return {
            "uid": uids,
            "input_features": input_features_stacked,      # [B, D, T]
            "attention_mask": attention_masks_stacked,     # [B, T]
            "input_lengths": input_lengths,               # [B]
            "labels": labels_padded,                      # [B, L]
            "label_lengths": label_lengths,               # [B]
        }
    else:
        max_len = max([s["input_features"].size(0) for s in batch])

        input_features = torch.stack([
            torch.nn.functional.pad(
                s["input_features"],
                (0, 0, 0, max_len - s["input_features"].size(0)),  # (left, right, top, bottom)
                value=0.0
            )
            for s in batch
        ])  # [B, T_max, 512]
        
        return {
            "uid": uids,
            "input_features": input_features,      # [B, D, T]
            "input_lengths": input_lengths,               # [B]
            "labels": labels_padded,                      # [B, L]
            "label_lengths": label_lengths,               # [B]
        }

# support sensevoice
import numpy as np
import kaldiio
import torch
from typing import Tuple
from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank
from funasr import AutoFrontend


class KaldiFbankExtractor(torch.nn.Module):
    """
    ark → 归一化 PCM → 重采样 → fbank 特征
    用法:
        extractor = KaldiFbankExtractor(
            fs=16000,
            frontend_name="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
            n_mels=80
        )
        feat, feat_len = extractor("data/raw/train.ark:123")
    """

    def __init__(self, kwargs):
        super().__init__()
        # 1. 采样率
        self.fs = kwargs.get("fs", 16000)
        # 2. 前端模型（不加载权重，仅取配置）
        self.frontend = kwargs.get("frontend", None)
        if self.frontend is None:
            raise ValueError(
                "必须提供 frontend_name，例如 "
                "'iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch'"
            )

    @torch.no_grad()
    def forward(self, ark_path, sampling_rate=16000) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        ark_path : str
            形如 "xxx.ark:123" 的 kaldi ark key

        Returns
        -------
        feat : torch.Tensor
            shape [T, D]  float32
        feat_len : torch.Tensor
            scalar int64
        """     
        pcm_int16 = kaldiio.load_mat(ark_path)          # (ch, N)
        pcm_float = pcm_int16[1].astype(np.float32) / 32768.0
        # 2. 封装成 list
        audio_list = load_audio_text_image_video(
            [pcm_float],
            fs=self.fs,
            audio_fs=sampling_rate,
            data_type="sound",
        )

        # 3. 提取 fbank
        input_features, input_feature_length = extract_fbank(
            audio_list,
            data_type="sound",
            frontend=self.frontend
        )        # feats: [1, T, D], lens: [1]
        input_features, input_feature_length = input_features[0], input_feature_length[0]
        return input_features, input_feature_length

import os
import sentencepiece as spm

class Vocabulary:
    def __init__(self, vocab_file):
        self.word2idx = {}
        self.idx2word = {}
        self.use_bpe = False  # 是否启用 BPE 分词
        self.sp_model = None  # sentencepiece 分词器
        self.vocab_file = vocab_file

        if "bpe" in vocab_file:
            print("Loading bpe vocabulary...")
            self._load_sentencepiece_model(vocab_file)
            self.use_bpe = True
        else:
            self.build_vocab(vocab_file)

    def _load_sentencepiece_model(self, model_path):
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(model_path)
        vocab_size = self.sp_model.get_piece_size()

        for idx in range(vocab_size):
            token = self.sp_model.id_to_piece(idx)
            self.word2idx[token] = idx
            self.idx2word[idx] = token

    def build_vocab(self, vocab_file):
        with open(vocab_file, 'r', encoding='utf-8') as f:
            for line in f:
                char, index = line.strip().split()
                self.word2idx[char] = int(index)
                self.idx2word[int(index)] = char

    def __len__(self):
        return len(self.word2idx)

    def text_to_sequence(self, text):
        unk_index = self.word2idx.get('<unk>', 0)
        if self.use_bpe:
            tokens = self.sp_model.encode(text, out_type=str)
            return [self.word2idx.get(t, unk_index) for t in tokens]
        else:
            return [self.word2idx.get(char, unk_index) for char in text]

    def sequence_to_text(self, sequence):
        if self.use_bpe:
            tokens = [self.idx2word.get(idx, '<unk>') for idx in sequence]
            return self.sp_model.decode_pieces(tokens)
        else:
            return ''.join([self.idx2word.get(idx, '<unk>') for idx in sequence])

