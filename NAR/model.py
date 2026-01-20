import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from dataclasses import dataclass

from .config import SimulatorConfig

@dataclass
class SimulatorOutput:
    loss: torch.Tensor
    logits: torch.Tensor

class LayerNorm(nn.LayerNorm):
    """支持 fp16 的 LayerNorm"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input):
        output = F.layer_norm(
            input.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        )
        return output.type_as(input)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout_rate):
        super().__init__()
        self.w_1 = nn.Linear(d_model, dim_feedforward)
        self.w_2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))

class MultiHeadedAttention(nn.Module):
    def __init__(self, n_head, d_model, dropout_rate):
        super().__init__()
        assert d_model % n_head == 0
        self.d_k = d_model // n_head
        self.h = n_head
        
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, query, key, value, mask=None):
        b, t_q, d = query.size()
        t_k = key.size(1)

        q = self.linear_q(query).view(b, t_q, self.h, self.d_k)
        k = self.linear_k(key).view(b, t_k, self.h, self.d_k)
        v = self.linear_v(value).view(b, t_k, self.h, self.d_k)

        q = q.transpose(1, 2) 
        k = k.transpose(1, 2) 
        v = v.transpose(1, 2) 

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k) 

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(2) 
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = torch.softmax(scores, dim=-1)
        p_attn = self.dropout(attn)

        x = torch.matmul(p_attn, v)
        x = x.transpose(1, 2).contiguous().view(b, t_q, d)
        return self.linear_out(x)

class ManualEncoderLayer(nn.Module):
    def __init__(self, size, n_head, dim_feedforward, dropout_rate):
        super().__init__()
        self.self_attn = MultiHeadedAttention(n_head, size, dropout_rate)
        self.feed_forward = PositionwiseFeedForward(size, dim_feedforward, dropout_rate)
        self.norm1 = LayerNorm(size)
        self.norm2 = LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask):
        residual = x
        x = self.norm1(x)
        x = residual + self.dropout(self.self_attn(x, x, x, mask))
        
        residual = x
        x = self.norm2(x)
        x = residual + self.dropout(self.feed_forward(x))
        return x

class ManualDecoderLayer(nn.Module):
    def __init__(self, size, n_head, dim_feedforward, dropout_rate):
        super().__init__()
        self.self_attn = MultiHeadedAttention(n_head, size, dropout_rate)
        self.cross_attn = MultiHeadedAttention(n_head, size, dropout_rate)
        self.feed_forward = PositionwiseFeedForward(size, dim_feedforward, dropout_rate)
        
        self.norm1 = LayerNorm(size)
        self.norm2 = LayerNorm(size)
        self.norm3 = LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, memory, tgt_mask, memory_mask):
        residual = x
        x = self.norm1(x)
        x = residual + self.dropout(self.self_attn(x, x, x, tgt_mask))
        
        residual = x
        x = self.norm2(x)
        x = residual + self.dropout(self.cross_attn(x, memory, memory, memory_mask))
        
        residual = x
        x = self.norm3(x)
        x = residual + self.dropout(self.feed_forward(x))
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 显式指定 dtype=torch.float 以避免 NPU 上的 Double Warning
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x_len: int) -> torch.Tensor:
        if x_len > self.pe.size(1):
            return self.pe[:, :self.pe.size(1), :]
        return self.pe[:, :x_len, :]

class LengthRegulator(nn.Module):
    def __init__(self, d_model, max_len=160):
        super().__init__()
        self.max_len = max_len
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_len)

    def forward(self, target_lengths: Optional[torch.Tensor] = None, max_len: Optional[int] = None):
        if target_lengths is not None: # 表示训练阶段
            seq_len = self.max_len
        else:
            assert max_len is not None # 推理阶段，保持接口稳定，可能传入不一样的max_len
            seq_len = max_len
            
        pos_emb = self.pos_encoder(seq_len) 
        return pos_emb


class CTCTransformerSimulator(nn.Module):
    def __init__(self, config: SimulatorConfig):
        super().__init__()
        self.config = config
        
        self.text_input_proj = nn.Linear(config.ctc_vocab_size, config.d_model)
        self.text_pos_encoder = PositionalEncoding(config.d_model, config.dropout, config.max_len)
        
        self.encoder_layers = nn.ModuleList([
            ManualEncoderLayer(config.d_model, config.n_head, config.dim_feedforward, config.dropout)
            for _ in range(config.num_encoder_layers)
        ])
        self.encoder_norm = LayerNorm(config.d_model)

        self.length_regulator = LengthRegulator(config.d_model, config.max_len)
        
        self.decoder_layers = nn.ModuleList([
            ManualDecoderLayer(config.d_model, config.n_head, config.dim_feedforward, config.dropout)
            for _ in range(config.num_decoder_layers)
        ])
        self.decoder_norm = LayerNorm(config.d_model)

        self.output_head = nn.Linear(config.d_model, config.ctc_vocab_size + 1)

    def encode_text(self, text_onehot, text_mask):
        x = self.text_input_proj(text_onehot)
        pe = self.text_pos_encoder(x.size(1))
        x = x + pe
        x = F.dropout(x, p=self.config.dropout, training=self.training)

        for layer in self.encoder_layers:
            x = layer(x, text_mask)
        
        return self.encoder_norm(x)

    def decode(self, memory, memory_mask, target_lengths=None, infer_max_len=None):
        batch_size = memory.size(0)
        query = self.length_regulator(target_lengths, infer_max_len)
        
        x = query.expand(batch_size, -1, -1)
        x = F.dropout(x, p=self.config.dropout, training=self.training)

        if target_lengths is not None:
            max_len = x.size(1) # 100
            tgt_mask = torch.arange(max_len, device=x.device).unsqueeze(0) < target_lengths.unsqueeze(1)
            tgt_mask = tgt_mask.float()
        else:
            tgt_mask = torch.ones(batch_size, x.size(1), device=x.device).float()

        for layer in self.decoder_layers:
            x = layer(x, memory, tgt_mask, memory_mask)
        
        x = self.decoder_norm(x)
        logits = self.output_head(x) 
        return logits

    def forward(self, text_onehot, text_mask, target_lengths):
        memory = self.encode_text(text_onehot, text_mask)
        logits = self.decode(memory, text_mask, target_lengths=target_lengths)
        return logits

    @torch.no_grad()
    def inference(self, text_onehot, text_mask, max_len=100):
        memory = self.encode_text(text_onehot, text_mask)
        logits = self.decode(memory, text_mask, infer_max_len=max_len)
        probs = torch.softmax(logits, dim=-1)
        return probs

class SimulatorTrainingWrapper(nn.Module):
    def __init__(self, simulator: CTCTransformerSimulator):
        super().__init__()
        self.simulator = simulator
        self.eos_id = 25055  # 明确指定 EOS ID

    def soft_cross_entropy(self, logits, target, mask):
        log_probs = F.log_softmax(logits, dim=-1)
        loss_per_frame = -(target * log_probs).sum(dim=-1)
        loss_masked = loss_per_frame * mask # mask为1代表有效，故下面归一化也要除mask之和代表总有效数
        total_loss = loss_masked.sum() / (mask.sum() + 1e-6)
        return total_loss

    def compute_accuracy(self, logits, target, mask):
        with torch.no_grad():
            pred_ids = logits.argmax(dim=-1)
            target_ids = target.argmax(dim=-1)
            correct = (pred_ids == target_ids) & (mask == 1)
            acc = correct.sum() / (mask.sum() + 1e-6)
        return acc

    def _monitor_eos_performance(self, logits, target_lengths, current_loss):
        """
        静默检查
        """
        with torch.no_grad():
            # 1. 获取最后一个有效帧的预测 ID
            # logits: [B, T, V]
            # last_indices: [B] (每个样本最后一步的索引)
            B = logits.size(0)
            last_indices = torch.clamp(target_lengths - 1, min=0)
            
            # [B, 1, V]
            last_frame_logits = logits[torch.arange(B, device=logits.device), last_indices]
            last_frame_preds = last_frame_logits.argmax(dim=-1) # [B]

            eos_correct = (last_frame_preds == self.eos_id).float().mean().item()

            # 3. 触发警告的条件：Loss 已经降下来了 (< 3.0)，但 EOS 准确率还很烂 (< 0.5)
            if (current_loss < 2.0 and eos_correct < 0.5):
                print(f"\n[Monitor] Low EOS Accuracy Warning!")
                print(f"    - Current Loss: {current_loss:.4f}")
                print(f"    - EOS Acc at last frame: {eos_correct * 100:.2f}%")
                print(f"    - Sample Preds at end: {last_frame_preds[:10].tolist()}")
                print(f"    - Expected EOS ID: {self.eos_id}")

    def forward(self, 
                text_onehot, 
                text_mask, 
                target_psd, 
                target_lengths,
                **kwargs):
        
        logits = self.simulator(text_onehot, text_mask, target_lengths)
        
        B, T, _ = logits.size()
        loss_mask = torch.arange(T, device=logits.device).unsqueeze(0) < target_lengths.unsqueeze(1)
        loss_mask = loss_mask.float()

        loss = self.soft_cross_entropy(logits, target_psd, loss_mask)
        acc = self.compute_accuracy(logits, target_psd, loss_mask)
        
        # [插入静默检查]
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                self._monitor_eos_performance(logits, target_lengths, loss.item())
        else:
            self._monitor_eos_performance(logits, target_lengths, loss.item())
        
        return SimulatorOutput(loss=loss, logits=logits), acc