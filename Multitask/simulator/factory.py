from omegaconf import DictConfig
from typing import Tuple, Any, List, Union
import torch
import numpy as np

from .model import SimulatorTrainingWrapper, CTCTransformerSimulator
from .config import SimulatorConfig
from model.tokenizer import SenseVoiceTokenizer

class HFTokenizerWrapper:
    def __init__(self, tokenizer: SenseVoiceTokenizer):
        self.tokenizer = tokenizer
        # 记录真实词表上限 (不含 EOS)
        self.valid_vocab_limit = tokenizer.vocab_size
    
    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    @property
    def pad_token_id(self):
        return getattr(self.tokenizer, "pad_id", 0)

    @property
    def eos_token_id(self):
        return getattr(self.tokenizer, "eos_id", 1)

    def batch_decode(self, sequences: Union[List[int], np.ndarray, torch.Tensor], skip_special_tokens: bool = True, **kwargs) -> List[str]:
        decoded_results = []
        for seq in sequences:
            if isinstance(seq, torch.Tensor):
                if seq.is_cuda or (hasattr(seq, 'is_npu') and seq.is_npu):
                    seq = seq.cpu()
                seq = seq.tolist()
            elif isinstance(seq, np.ndarray):
                seq = seq.tolist()
            
            # 递归处理 Batch
            if isinstance(seq, list) and len(seq) > 0 and isinstance(seq[0], list):
                return [self.batch_decode([s], skip_special_tokens=skip_special_tokens)[0] for s in seq]
            
            # [核心修复] 过滤掉 EOS (25055) 和其他非法 ID，防止 SentencePiece 崩溃
            # 只保留 0 ~ 25054 之间的 ID
            safe_ids = [x for x in seq if 0 <= x < self.valid_vocab_limit]
            
            try:
                text = self.tokenizer.decode(safe_ids)
            except Exception:
                text = ""
            decoded_results.append(text)
            
        return decoded_results

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text)

def create_training_model(model_config: DictConfig, dataset_config: DictConfig, **kwargs):
    raw_tokenizer = SenseVoiceTokenizer(dataset_config.tokenizer_path)
    tokenizer = HFTokenizerWrapper(raw_tokenizer)

    simulator_config = SimulatorConfig(
        vocab_size=tokenizer.vocab_size,
        ctc_vocab_size=model_config.ctc_vocab_size,
        d_model=model_config.d_model,
        n_head=model_config.n_head,
        num_encoder_layers=model_config.num_encoder_layers,
        num_decoder_layers=model_config.num_decoder_layers,
        dim_feedforward=model_config.dim_feedforward,
        dropout=model_config.dropout,
        max_len=model_config.max_len,
    )
    simulator = CTCTransformerSimulator(simulator_config)
    training_wrapper = SimulatorTrainingWrapper(simulator=simulator)
    
    return training_wrapper, tokenizer