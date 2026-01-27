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
    per_sample_loss: torch.Tensor = None  # 用于存储 [B] 维度的样本损失

# 组件
class LayerNorm(nn.LayerNorm):
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

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        T = x.size(1)
        if T > self.pe.size(1):
            return x + self.pe[:, :self.pe.size(1), :]
        return self.dropout(x + self.pe[:, :T, :])


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

#主模型
class CTCTransformerSimulator(nn.Module):
    def __init__(self, config: SimulatorConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.vocab_size_out = config.ctc_vocab_size + 1 

        self.text_input_proj = nn.Linear(config.ctc_vocab_size, config.d_model)
        self.text_pos_encoder = PositionalEncoding(config.d_model, config.dropout, config.max_len)
        
        self.condition_projector = nn.Embedding(5, config.d_model) # 0用于padding，1-4是id，共5个
        
        self.encoder_layers = nn.ModuleList([
            ManualEncoderLayer(config.d_model, config.n_head, config.dim_feedforward, config.dropout)
            for _ in range(config.num_encoder_layers)
        ])
        self.encoder_norm = LayerNorm(config.d_model)

        self.audio_input_proj = nn.Linear(self.vocab_size_out, config.d_model)
        
        self.sos_emb = nn.Parameter(torch.zeros(1, 1, config.d_model))
        nn.init.normal_(self.sos_emb, mean=0, std=0.02)
        
        self.decoder_pos_encoder = PositionalEncoding(config.d_model, config.dropout, config.max_len)
        
        self.decoder_layers = nn.ModuleList([
            ManualDecoderLayer(config.d_model, config.n_head, config.dim_feedforward, config.dropout)
            for _ in range(config.num_decoder_layers)
        ])
        self.decoder_norm = LayerNorm(config.d_model)

        self.output_head = nn.Linear(config.d_model, self.vocab_size_out)

    def encode_text(self, text_onehot, text_mask, bucket_id):
        x = self.text_input_proj(text_onehot)
        cond_emb = self.condition_projector(bucket_id).unsqueeze(1)
        x = torch.cat([cond_emb, x], dim=1)
        
        B = x.size(0)
        cond_mask = torch.ones(B, 1, device=x.device)
        new_mask = torch.cat([cond_mask, text_mask], dim=1)
        
        x = self.text_pos_encoder(x)
        for layer in self.encoder_layers:
            x = layer(x, new_mask)
            
        return self.encoder_norm(x), new_mask 

    def decode_audio_step(self, x, memory, memory_mask, past_len=0):
        B, L, D = x.size()
        causal_mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1) == 0
        causal_mask = causal_mask.float().unsqueeze(0).unsqueeze(0) 
        
        for layer in self.decoder_layers:
            x = layer(x, memory, causal_mask, memory_mask)
        
        x = self.decoder_norm(x)
        logits = self.output_head(x)
        return logits

    def forward(self, text_onehot, text_mask, target_psd, target_lengths, bucket_id):
        memory, memory_mask = self.encode_text(text_onehot, text_mask, bucket_id)

        prev_frames_emb = self.audio_input_proj(target_psd[:, :-1, :]) 
        sos = self.sos_emb.expand(prev_frames_emb.size(0), -1, -1)
        decoder_input = torch.cat([sos, prev_frames_emb], dim=1)
        
        decoder_input = self.decoder_pos_encoder(decoder_input)

        logits = self.decode_audio_step(decoder_input, memory, memory_mask)
        
        return logits

    @torch.no_grad()
    def inference(self, text_onehot, text_mask, bucket_id, max_len=160, temperature=0.5): 
        B = text_onehot.size(0)
        memory, memory_mask = self.encode_text(text_onehot, text_mask, bucket_id)
        
        current_input = self.sos_emb.expand(B, -1, -1)
        generated_logits = []
        history_emb = current_input
        
        for t in range(max_len):
            dec_input = self.decoder_pos_encoder(history_emb)
            logits_seq = self.decode_audio_step(dec_input, memory, memory_mask)
            
            last_logits = logits_seq[:, -1:, :]
            generated_logits.append(last_logits)
            
            next_probs = torch.softmax(last_logits / temperature, dim=-1) 
            next_in_emb = self.audio_input_proj(next_probs)
            history_emb = torch.cat([history_emb, next_in_emb], dim=1)
            
        full_logits = torch.cat(generated_logits, dim=1)
        probs = torch.softmax(full_logits / temperature, dim=-1)
        
        return probs

#封装
class SimulatorTrainingWrapper(nn.Module):
    def __init__(self, simulator: CTCTransformerSimulator):
        super().__init__()
        self.simulator = simulator
        self.eos_id = 25055

    def soft_cross_entropy(self, logits, target, mask):
        log_probs = F.log_softmax(logits, dim=-1)
        loss_per_frame = -(target * log_probs).sum(dim=-1)
        loss_masked = loss_per_frame * mask
        
        # 修改点：在 dim=1 (时间轴) 上求和，除以该样本的有效长度，保留 [B] 维度
        per_sample_loss = loss_masked.sum(dim=1) / (mask.sum(dim=1) + 1e-6)
        return per_sample_loss

    def compute_accuracy(self, logits, target, mask):
        with torch.no_grad():
            pred_ids = logits.argmax(dim=-1)
            target_ids = target.argmax(dim=-1)
            correct = (pred_ids == target_ids) & (mask == 1)
            acc = correct.sum() / (mask.sum() + 1e-6)
        return acc

    def _monitor_eos_performance(self, logits, target_lengths, current_loss):
        with torch.no_grad():
            B = logits.size(0)
            last_indices = torch.clamp(target_lengths - 1, min=0)
            last_frame_logits = logits[torch.arange(B, device=logits.device), last_indices]
            last_frame_preds = last_frame_logits.argmax(dim=-1)
            eos_correct = (last_frame_preds == self.eos_id).float().mean().item()
            if (current_loss < 2.0 and eos_correct < 0.5):
                print(f"\n[Monitor] Low EOS Accuracy: {eos_correct:.2%} (Loss: {current_loss:.2f})")

    def forward(self, text_onehot, text_mask, target_psd, target_lengths, bucket_id, **kwargs):
        # 纯CE loss，无需任何MBR
        logits = self.simulator(text_onehot, text_mask, target_psd, target_lengths, bucket_id)
        B, T, _ = logits.size()
        seq_range = torch.arange(T, device=logits.device).unsqueeze(0)
        loss_mask = (seq_range < target_lengths.unsqueeze(1)).float()

        per_sample_ce = self.soft_cross_entropy(logits, target_psd, loss_mask)
        loss_sft = per_sample_ce.mean()
        
        #注：acc是BPE序列中ID正确数/总数
        acc = self.compute_accuracy(logits, target_psd, loss_mask)
        
        return SimulatorOutput(loss=loss_sft, logits=logits, per_sample_loss=per_sample_ce), acc