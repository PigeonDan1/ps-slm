import torch
import torch.nn as nn
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class EncoderProjectorLinear(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.k = config.encoder_projector_ds_rate
        self.encoder_dim = config.encoder_dim
        self.llm_vocab = config.llm_dim 
        self.map = nn.Linear(self.encoder_dim * self.k, self.llm_vocab, bias=True)

    def forward(self, x):
        B, T, D = x.size()
        discard = T % self.k
        if discard:
            x = x[:, :-discard, :]
        T = x.size(1)
        x = x.view(B, T // self.k, D * self.k)      # [B, T', D*k]
        logits = self.map(x)                        # [B, T', V2]
        return logits                                # [B, T', V2]


class EncoderProjectorConcat(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.k = config.encoder_projector_ds_rate
        self.encoder_dim = config.encoder_dim
        self.llm_dim = config.llm_dim
        self.linear1 = nn.Linear(self.encoder_dim * self.k, 2048)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(2048, config.llm_dim)

    def forward(self, x):
        batch_size, seq_len, dim = x.size()
        num_frames_to_discard = seq_len % self.k
        if num_frames_to_discard > 0:
            x = x[:, :-num_frames_to_discard, :]
        seq_len = x.size(1)
        x = x.contiguous()
        x = x.view(batch_size, seq_len // self.k, dim * self.k)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


class EncoderProjectorCov1d(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.k = config.encoder_projector_ds_rate
        self.encoder_dim = config.encoder_dim
        self.llm_dim = config.llm_dim
        self.conv1d = nn.Conv1d(in_channels=self.encoder_dim, out_channels=self.encoder_dim, kernel_size=self.k, stride=self.k, padding=0)
        self.linear1 = nn.Linear(self.encoder_dim, 2048)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(2048, self.llm_dim)
        self.relu2 = nn.ReLU()
    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1d(x)
        x = x.transpose(1, 2)
        x = self.relu1(x)
        x = self.linear1(x)
        x = self.relu2(x)
        x = self.linear2(x)
        return x


class EncoderProjectorQFormer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder_dim = config.encoder_dim
        self.llm_dim = config.llm_dim
        from transformers import Blip2QFormerConfig, Blip2QFormerModel
        configuration = Blip2QFormerConfig()
        configuration.encoder_hidden_size = self.encoder_dim
        configuration.num_hidden_layers = config.qformer_layers
        self.query_len = int(config.get("query_len", 64))
        self.query = nn.Parameter(torch.zeros(1, self.query_len, configuration.hidden_size))
        self.query.data.normal_(mean=0.0, std=1.0)
        self.qformer = Blip2QFormerModel(configuration)
        self.linear = nn.Linear(configuration.hidden_size, self.llm_dim)
        self.norm = nn.LayerNorm(self.llm_dim, eps=1e-5)

    def forward(self, x, atts):
        query = self.query.expand(x.shape[0], -1, -1)
        query_output = self.qformer(
            query_embeds=query,
            encoder_hidden_states=x,
            encoder_attention_mask=atts,
            return_dict=True,
        )
        query_proj = self.norm(self.linear(query_output.last_hidden_state))
        return query_proj


class EncoderProjectorCTCCA(nn.Module):
    def __init__(self, config, n_heads=8):
        super().__init__()
        v1 = config.encoder_dim
        D = config.llm_dim
        self.W_q = nn.Linear(v1, D, bias=False)
        self.n_heads = n_heads

    def forward(self, post, llm_embed):
        B, T, _ = post.shape
        Q = self.W_q(post)                       # (B, T, D)
        K = llm_embed                  # (V2, D)
        V = llm_embed                # (V2, D)
        h = self.n_heads
        d = Q.size(-1) // h
        q = Q.view(B, T, h, d)                   # (B, T, h, d)
        k = K.view(-1, h, d)                     # (V2, h, d)
        v = V.view(-1, h, d)                     # (V2, h, d)
        scores = torch.einsum('bthd,vhd->bthv', q, k) / d**0.5
        attn = scores.softmax(dim=-1)
        z = torch.einsum('bthv,vhd->bthd', attn, v)
        Z = z.contiguous().view(B, T, -1)        # (B, T, D)
        return Z


class EncoderProjectorLinearSiLU(nn.Module):
    def __init__(self, config, bottleneck=2048):
        """
        1. First perform LayerNorm to balance the mean and variance
        2. Replace ReLU with SiLU (with negative leakage)
        3. Bottleneck
        """
        super().__init__()
        in_dim = config.encoder_dim 
        out_dim = config.llm_dim
        self.norm = nn.LayerNorm(in_dim)
        self.ffn = nn.Sequential(
            nn.Linear(in_dim, bottleneck, bias=True),
            nn.SiLU(),                      
            nn.Linear(bottleneck, out_dim, bias=True),
        )
        nn.init.kaiming_uniform_(self.ffn[0].weight, a=math.sqrt(5))
        nn.init.zeros_(self.ffn[2].bias)
        self.k = 1

    def forward(self, x):                  # (B,T,in_dim)
        x = self.norm(x)
        return self.ffn(x)                 # (B,T,out_dim)