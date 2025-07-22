import torch
import torch.nn as nn
from transformers import WhisperModel

# class WhisperCTC(nn.Module):
#     def __init__(self, whisper_name, vocab_size, freeze_encoder=False):
#         super().__init__()
#         # Load Whisper encoder only
#         self.encoder = WhisperModel.from_pretrained(whisper_name).encoder

#         if freeze_encoder:
#             for param in self.encoder.parameters():
#                 param.requires_grad = False

#         hidden_size = self.encoder.config.d_model
#         self.ctc_head = nn.Linear(hidden_size, vocab_size)

#     def forward(self, input_features, attention_mask=None):
#         """
#         input_features: (B, T, feature_dim)
#         attention_mask: (B, T) -> 1 for valid frames, 0 for padding
#         """
#         encoder_outputs = self.encoder(
#             input_features=input_features,
#             attention_mask=attention_mask
#         )
#         hidden_states = encoder_outputs.last_hidden_state  # (B, T, D)
#         logits = self.ctc_head(hidden_states)              # (B, T, V)
        
#         return logits

import torch
import torch.nn as nn
from transformers import WhisperModel

class WhisperCTC(nn.Module):
    def __init__(self, whisper_name, vocab_size, freeze_encoder=False):
        super().__init__()
        # 1. 加载 Whisper encoder
        self.encoder = WhisperModel.from_pretrained(whisper_name).encoder

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        hidden_size = self.encoder.config.d_model

        # 2. 下采样模块（时间维度缩小5倍）
        self.post_downsample = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=5,
            stride=5,
            padding=0
        )

        # 3. CTC 头
        self.ctc_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_features, attention_mask):
        """
        input_features: (B, D, T)
        attention_mask: (B, T)
        """
        # attention_mask=None # for compare
        # 1. Whisper Encoder
        encoder_outputs = self.encoder(
            input_features=input_features,
            attention_mask=attention_mask
        )
        hidden_states = encoder_outputs.last_hidden_state  # (B, T' = T // 2, D)
        
        # 2. Downsample
        x = hidden_states.transpose(1, 2)  # (B, D, T')
        x = self.post_downsample(x)        # (B, D, T'')
        x = x.transpose(1, 2)              # (B, T'', D)
        
        # 3. CTC Head
        logits = self.ctc_head(x)          # (B, T'', V)
        return logits
