import json
import torch
import torchaudio
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import kaldiio
import numpy as np
import re
import unicodedata

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
# class JsonlCTCDataset(Dataset):
#     def __init__(self, jsonl_file, tokenizer, feature_extractor, max_input_length=3000):
#         self.samples = []
#         with open(jsonl_file) as f:
#             for line in f:
#                 self.samples.append(json.loads(line.strip()))

#         self.tokenizer = tokenizer
#         self.feature_extractor = feature_extractor
#         self.max_input_length = max_input_length

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         sample = self.samples[idx]
#         path = sample["path"]
#         text = sample["target"]
#         uid = sample["key"] 

#         if path.endswith(".wav"):
#             wav, sr = torchaudio.load(path)
#             wav = wav.squeeze(0).numpy()
#         else:
#             sr, wav = kaldiio.load_mat(path)
#         if sr != 16000:
#             print(f"{path}'s sampling rate is not 16khz.")

#         # Whisper feature extractor
#         # input_features = self.feature_extractor(
#         #     wav,
#         #     sampling_rate=16000
#         # ).input_features[0] # (80, 3000) fully padding
#         output = self.feature_extractor(
#             wav,
#             sampling_rate=16000,
#             return_attention_mask=True,
#         )
#         input_features = output.input_features[0]        # shape: [80, 3000]
#         attention_mask = output.attention_mask[0]        # shape: [3000], 1 表示有效帧，0 表示 padding
#         input_length = int(attention_mask.sum()) // 10   # x2 downsampling and 5 subsampling
#         # Tokenize target
#         with self.tokenizer.as_target_tokenizer():
#             labels = self.tokenizer(text, add_special_tokens=False).input_ids

#         return {
#             "uid": uid, 
#             "input_features": torch.tensor(input_features, dtype=torch.float),
#             "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
#             "input_length": input_length,
#             "labels": torch.tensor(labels, dtype=torch.long),

#         }
class JsonlCTCDataset(Dataset):
    def __init__(self, jsonl_file, tokenizer, feature_extractor, max_input_length=3000, mode="train"):
        self.samples = []
        with open(jsonl_file) as f:
            for line in f:
                self.samples.append(json.loads(line.strip()))

        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.max_input_length = max_input_length
        self.mode = mode
        # 判断是否是自定义 Vocabulary（是否含有 text_to_sequence 方法）
        self.use_custom_vocab = hasattr(tokenizer, "text_to_sequence")

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

        return {
            "uid": uid, 
            "input_features": torch.tensor(input_features, dtype=torch.float),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "input_length": input_length,
            "labels": torch.tensor(labels, dtype=torch.long),
        }

def ctc_collate_fn(batch):
    input_features = [item["input_features"] for item in batch]         # [D, T]
    attention_masks = [item["attention_mask"] for item in batch]        # [T]
    input_lengths = torch.tensor([item["input_length"] for item in batch])  # scalar
    labels = [item["labels"] for item in batch]
    label_lengths = torch.tensor([len(x) for x in labels])
    uids = [item["uid"] for item in batch]

    input_features_stacked = torch.stack(input_features) 
    attention_masks_stacked = torch.stack(attention_masks)

    # Pad labels to [B, L_max]
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)

    return {
        "uid": uids,
        "input_features": input_features_stacked,      # [B, D, T]
        "attention_mask": attention_masks_stacked,     # [B, T]
        "input_lengths": input_lengths,               # [B]
        "labels": labels_padded,                      # [B, L]
        "label_lengths": label_lengths,               # [B]
    }

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

# class Vocabulary:
#     def __init__(self, vocab_file):
#         self.word2idx = {}
#         self.idx2word = {}
#         self.build_vocab(vocab_file)

#     def build_vocab(self, vocab_file):
#         with open(vocab_file, 'r') as f:
#             for line in f:
#                 char, index = line.strip().split()
#                 self.word2idx[char] = int(index)
#                 self.idx2word[int(index)] = char

#     def __len__(self):
#         return len(self.word2idx)

#     def text_to_sequence(self, text):
#         unk_index = self.word2idx.get('<unk>', 0)
#         return [self.word2idx.get(char, unk_index) for char in text]

#     def sequence_to_text(self, sequence):
#         return ''.join([self.idx2word.get(idx, '<unk>') for idx in sequence])
