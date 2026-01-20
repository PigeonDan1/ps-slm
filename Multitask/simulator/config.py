from dataclasses import dataclass

@dataclass
class SimulatorConfig:
    vocab_size: int # 指文本的tokenizer词表大小
    ctc_vocab_size: int = 25055
    d_model: int = 512
    n_head: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    dim_feedforward: int = 2048
    dropout: float = 0.1
    max_len: int = 160

    # 默认打开反馈回路
    use_feedback: bool = True 
    feedback_weight: float = 2