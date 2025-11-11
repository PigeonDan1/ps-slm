import json
import sentencepiece as spm
import torch

class SenseVoiceTokenizer:
    def __init__(self, model_dir):
        sp_model = f"{model_dir}/chn_jpn_yue_eng_ko_spectok.bpe.model"
        self.sp = spm.SentencePieceProcessor(model_file=sp_model)

        with open(f"{model_dir}/tokens.json", encoding="utf-8") as f:
            self.id2tok = json.load(f)          # {str_id: str_token}

        self.pad_id = self.sp.pad_id()
        self.eos_id = self.sp.eos_id()

    def encode(self, text: str) -> list[int]:
        return self.sp.encode(text, out_type=int)

    def decode(self, ids: list[int]) -> str:
        ids = [i for i in ids if i not in (self.pad_id, self.eos_id)]
        return self.sp.decode(ids)
    
    @property
    def vocab_size(self):
        return self.sp.vocab_size()