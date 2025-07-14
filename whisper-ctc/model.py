import torch
import torch.nn as nn
from transformers import WhisperModel

class WhisperCTC(nn.Module):
    def __init__(self, whisper_name, vocab_size):
        super().__init__()
        self.encoder = WhisperModel.from_pretrained(whisper_name).encoder
        hidden_size = self.encoder.config.d_model
        self.ctc_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_features, attention_mask=None):
        encoder_outputs = self.encoder(input_features, attention_mask=attention_mask)
        hidden_states = encoder_outputs.last_hidden_state  # (B, T, D)
        logits = self.ctc_head(hidden_states)              # (B, T, V)
        return logits
