# TASU: Text-Only Alignment for Speech Understanding
# Authors: Jing Peng*, Yi Yang (X-LANCE Lab, Shanghai Jiao Tong University)
# Repository: https://github.com/PigeonDan1/ps-slm 
# Adapted from: https://github.com/X-LANCE/SLAM-LLM 

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import logging
import types
from typing import List, Optional, Tuple, Union
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import PeftModel, LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from utils.metric import compute_accuracy
from utils.config_utils import generate_peft_config
from utils.model_utils import print_model_size, print_module_size
# from utils.npu_flash_attn import patch_npu_flash_attn

logger = logging.getLogger(__name__)


def setup_tokenizer(train_config, model_config, **kwargs):
    tokenizer = AutoTokenizer.from_pretrained(model_config.llm_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def setup_encoder(train_config, model_config, **kwargs):
    encoder_name = model_config.encoder_name
    from model.SenseVoice import SenseVoiceSmall
    encoder, kwargs = SenseVoiceSmall.from_pretrained(model_config.encoder_path)
    if train_config.freeze_encoder:
        for name, param in encoder.named_parameters(): 
            param.requires_grad = False
        encoder.eval()
        print_module_size(encoder, encoder_name, int(os.environ["RANK"]) if train_config.enable_fsdp or train_config.enable_ddp else 0)
    return encoder


def setup_encoder_projector(train_config, model_config, **kwargs):
    if model_config.encoder_projector == "linear":
        from model.projector import EncoderProjectorConcat
        encoder_projector = EncoderProjectorConcat(model_config)
    elif model_config.encoder_projector == "linear-silu":
        from model.projector import EncoderProjectorLinearSiLU
        encoder_projector = EncoderProjectorLinearSiLU(model_config)
        if train_config.freeze_projector:
            for name, param in encoder_projector.named_parameters(): 
                param.requires_grad = False
            encoder_projector.eval()
            print("Projector is frozen ...")
    elif model_config.encoder_projector == "cov1d-linear":
        from model.projector import EncoderProjectorCov1d
        encoder_projector = EncoderProjectorCov1d(model_config)
    elif model_config.encoder_projector == "q-former":
        from model.projector import EncoderProjectorQFormer
        encoder_projector = EncoderProjectorQFormer(model_config)
    elif model_config.encoder_projector == "cross-attention":
        from model.projector import EncoderProjectorCTCCA
        encoder_projector = EncoderProjectorCTCCA(model_config)
    elif model_config.encoder_projector == "simple_linear":
        from model.projector import EncoderProjectorLinear
        encoder_projector = EncoderProjectorLinear(model_config)
        if model_config.ctc_linear:
            ckpt = torch.load(
            model_config.ctc_linear,
            map_location="cpu"
            )
            state = ckpt.get("model", ckpt)          
            proj_state = {
                "weight": state["ctc_head.weight"],  # (151644, 512)
                "bias":   state["ctc_head.bias"],    # (151644,)
            }
            missing, unexpected = encoder_projector.map.load_state_dict(
                proj_state, strict=True
            )
            print("Pretrained CTC Head is loaded ...")
            if train_config.freeze_encoder:
                for name, param in encoder_projector.named_parameters(): 
                    param.requires_grad = False
                encoder_projector.eval()
                print("CTC Head is Frozen ...")
    return encoder_projector


def setup_llm(train_config, model_config, **kwargs):
    use_cache = False if train_config.enable_fsdp or train_config.enable_ddp else None

    model = AutoModelForCausalLM.from_pretrained(
                model_config.llm_path,
                load_in_8bit=True if train_config.quantization else None,
                device_map="auto" if train_config.quantization else None,
                use_cache=use_cache,
            )

    print_module_size(model, model_config.llm_name, int(os.environ["RANK"]) if train_config.enable_fsdp or train_config.enable_ddp else 0)

    # Prepare the model for int8 training if quantization is enabled
    if train_config.quantization:
        model = prepare_model_for_kbit_training(model)

    if train_config.freeze_llm: 
        for name, param in model.named_parameters(): 
            param.requires_grad = False
        model.eval()
        
    if kwargs.get("peft_ckpt", None): 
        logger.info("loading peft_ckpt from: {}".format(kwargs.get("peft_ckpt")))
        model = PeftModel.from_pretrained(model=model, model_id=kwargs.get("peft_ckpt"), is_trainable=True)
        model.print_trainable_parameters()
    elif train_config.use_peft:
        logger.info("setup peft...")
        peft_config = generate_peft_config(train_config)
        model = get_peft_model(model, peft_config)
        
        if train_config.use_emb:
            logger.info("embs are hot...")
            for name, p in model.named_parameters():
                if "embed_tokens" in name:
                    p.requires_grad = True
        
        model.print_trainable_parameters()
    print_module_size(model, model_config.llm_name, int(os.environ["RANK"]) if train_config.enable_fsdp or train_config.enable_ddp else 0)
    return model


def model_factory(train_config, model_config, **kwargs):
    # return necessary components for training
    tokenizer = setup_tokenizer(train_config, model_config, **kwargs)
    DEFAULT_SPEECH_TOKEN = "<speech>"
    DEFAULT_IGNORE_TOKEN = -100
    special_tokens_dict = {"additional_special_tokens": [DEFAULT_SPEECH_TOKEN]}
    tokenizer.add_special_tokens(special_tokens_dict)
    tokenizer.default_ignore_token = DEFAULT_IGNORE_TOKEN
    tokenizer.default_speech_token = tokenizer.convert_tokens_to_ids(
            DEFAULT_SPEECH_TOKEN
        )
    
    # llm
    llm = setup_llm(train_config, model_config, **kwargs)

    # encoder
    encoder = setup_encoder(train_config, model_config, **kwargs)
    
    # projector
    encoder_projector = setup_encoder_projector(
        train_config, model_config, **kwargs
    )

    model = slam_model_asr(
        encoder,
        llm,
        encoder_projector,
        tokenizer,
        train_config,
        model_config,
        **kwargs,
    )
    # patch_npu_flash_attn()
    ckpt_path = kwargs.get( "ckpt_path", None)
    if ckpt_path is not None:
        print(f"loading other parts from: {ckpt_path}")
        ckpt_dict = torch.load(ckpt_path, map_location="cpu")
        print("Keys in the checkpoint:")
        for key in ckpt_dict.keys():
            print(key)
        model.load_state_dict(ckpt_dict, strict=False)

    print_model_size(
        model,
        train_config,
        (
            int(os.environ["RANK"])
            if train_config.enable_fsdp or train_config.enable_ddp
            else 0
        ),
    )
    return model, tokenizer

class slam_model_asr(torch.nn.Module):
    def __init__(
        self,
        encoder,
        llm,
        encoder_projector,
        tokenizer,
        train_config,
        model_config,
        **kwargs,
    ):
        super().__init__()
        # modality encoder 
        self.encoder = encoder

        # llm
        self.llm = llm

        # projector
        self.encoder_projector = encoder_projector

        # tokenizer
        self.tokenizer = tokenizer
        self.metric = kwargs.get("metric", "acc")
        self.ctc_posterior = train_config.ctc_posterior
        self.train_config = train_config
        self.do_psd = train_config.do_psd
        self.voca_trans = train_config.voca_trans
        self.gt_emb = train_config.gt_emb
        self.gt_emb_noise = train_config.gt_emb_noise
        self.gaussian_sim = train_config.get("gaussian_sim", False)
        if model_config.encoder_projector == "cross-attention":
            self.cross_attn = True
        else:
            self.cross_attn = False
        from model.tokenizer import SenseVoiceTokenizer
        self.encoder_tokenizer = SenseVoiceTokenizer(model_config.encoder_path)  
    
        self.top1_emb = train_config.top1_emb
        self.model_config = model_config
        if train_config.get("enable_deepspeed", False):
            def new_forward(self, input):
                output = F.layer_norm(
                    input.float(),
                    self.normalized_shape,
                    self.weight.float() if self.weight is not None else None,
                    self.bias.float() if self.bias is not None else None,
                    self.eps,
                )
                return output.type_as(input)
            for item in self.modules():
                if isinstance(item, nn.LayerNorm):
                    item.forward = types.MethodType(new_forward, item)

    def psd(
            self,
            encoder_out: torch.Tensor,      # [B, T, D]
            encoder_out_lens: torch.Tensor, # [B]
            ctc_posterior: torch.Tensor,    # [B, T, V]  
            blank_id: int = 0,
            blank_threshold: float = 0.90
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        1. Only merge adjacent and identical non-blank character frames (blank frames are not merged)
        2. If a certain character appears in more than 5 consecutive frames, print it.
        3. Use a blank probability threshold of 0.9 to uniformly delete blank frames
        4. 0-pad
        return:
            encoder_outs : [B, T_new, D]
            new_lens     : [B]   
        """
        B, T, D = encoder_out.shape
        device  = encoder_out.device
        is_log_prob = ctc_posterior.max() <= 0
        ctc_probs = ctc_posterior.exp() if is_log_prob else ctc_posterior
        keep_frames, new_lens = [], []
        for b in range(B):
            L = encoder_out_lens[b].item()
            if L == 0:
                keep_frames.append(encoder_out.new_zeros(0, D))
                new_lens.append(0)
                continue
            ids = ctc_probs[b, :L].argmax(dim=-1)  # [L]

            # ---- Merge adjacent identical non-blank character frames ----
            merged_feats, merged_blank_probs = [], []
            start = 0
            for end in range(1, L + 1):
                if end == L or ids[end] != ids[start]:
                    seg_len = end - start
                    char_id = ids[start].item()

                    if char_id == blank_id:
                        # blank frame：Keep each frame separately
                        for t in range(start, end):
                            merged_feats.append(encoder_out[b, t])
                            merged_blank_probs.append(ctc_probs[b, t, blank_id])
                    else:
                        # Not blank: merge the entire paragraph
                        if seg_len > 5:
                            print(f"[PSD] Warning: batch={b}, char={char_id}, "
                                f"continuous frames={seg_len} (>5)")
                        merged_feats.append(encoder_out[b, start:end].mean(dim=0))
                        avg_blank_prob = ctc_probs[b, start:end, blank_id].mean()
                        merged_blank_probs.append(avg_blank_prob)
                    start = end

            merged_feats = torch.stack(merged_feats, dim=0)           # [T_merged, D]
            merged_blank_probs = torch.tensor(merged_blank_probs,
                                            device=device)        # [T_merged]

            # ---- Filter blanks with a threshold of 0.9 ----
            mask = merged_blank_probs < blank_threshold
            keep = mask.nonzero(as_tuple=False).squeeze(-1)
            feats_after_blank = merged_feats[keep]                    # [M, D]

            keep_frames.append(feats_after_blank)
            new_lens.append(feats_after_blank.size(0))

        # pad to batch's maximum length
        max_len = max(new_lens) if new_lens else 0
        if max_len == 0:
            return encoder_out.new_zeros(B, 0, D), \
                encoder_out.new_zeros(B, dtype=torch.long, device=device)

        padded = []
        for feat in keep_frames:
            pad_len = max_len - feat.size(0)
            if pad_len > 0:
                feat = F.pad(feat, (0, 0, 0, pad_len), value=0.)
            padded.append(feat)
        encoder_outs = torch.stack(padded, dim=0)  # [B, T_new, D]
        new_lens = torch.tensor(new_lens, dtype=torch.long, device=device)

        return encoder_outs, new_lens
    
    def ids2text(self, ids: torch.LongTensor, llm):
        """
        ids: [B, T]  padding
        llm: transformers.PreTrainedModel / AutoModelForCausalLM
        return: list[str]  The text of each sample
        """
        # 1. Change -100 to pad_token_id, keep the rest unchanged
        pad_id = llm.config.pad_token_id if llm.config.pad_token_id is not None else llm.config.eos_token_id
        ids = torch.where(ids == -100, pad_id, ids)

        # 2. decode
        text_list = self.tokenizer.batch_decode(
            ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        return text_list

    def ctc_pseudo_posterior(self, texts):
        """
        texts: list[str]  —— Decoded text
        return: 
            posterior: [B, L_max, vocab_size]  one-hot 
            lens:      [B]   real token length
        """
        tok = self.encoder_tokenizer
        ids_list = [tok.encode(t) for t in texts]

        # real length
        lens = torch.tensor([len(ids) for ids in ids_list], dtype=torch.long)
        max_len = lens.max().item()
        vocab_size = tok.vocab_size

        # one-hot 
        B = len(ids_list)
        posterior = torch.zeros(B, max_len, vocab_size, dtype=torch.float32)
        for b, ids in enumerate(ids_list):
            posterior[b, torch.arange(len(ids)), ids] = 1.0

        return posterior, lens

    def ctc_pseudo_posterior_noise(self, texts):
        """
        texts: list[str]  —— Decoded text
        return:
            posterior: [B, L_max, vocab_size]  Pseudo-posterior (smoothed + random addition/deletion)
            lens:      [B]   The actual token length of each sample after processing
        """
        print("Add noise simulation ...")
        tok = self.encoder_tokenizer
        vocab_size = tok.vocab_size
        device = next(self.parameters()).device

        drop_prob   = getattr(self, 'drop_prob',   0.05)          # drop probability
        insert_prob = getattr(self, 'insert_prob', 0.0)          # Relative length insertion ratio
        smooth_low  = getattr(self, 'smooth_low',  0.0)           # α range
        smooth_high = getattr(self, 'smooth_high', 0.1)
        blank_id    = self.encoder.blank_id  # preserve blank
        ids_list = [tok.encode(t) for t in texts]
        processed = []

        for ids in ids_list:
            ids_t = torch.tensor(ids, dtype=torch.long)
            onehot = torch.nn.functional.one_hot(ids_t, vocab_size).float()

            alpha = torch.empty(()).uniform_(smooth_low, smooth_high).item()
            soft = (1 - alpha) * onehot + alpha / vocab_size

            keep_mask = torch.rand(len(soft)) > drop_prob
            soft = soft[keep_mask]

            n_insert = int(len(soft) * insert_prob)
            for _ in range(n_insert):
                pos = torch.randint(0, len(soft) + 1, (1,)).item()
                if torch.rand(1) < 0.5 and len(soft) > 0:
                    dup = soft[pos - 1] if pos > 0 else soft[0]
                    soft = torch.cat([soft[:pos], dup.unsqueeze(0), soft[pos:]])
                else:
                    blank_vec = torch.zeros(vocab_size)
                    blank_vec[blank_id] = 1.0
                    soft = torch.cat([soft[:pos], blank_vec.unsqueeze(0), soft[pos:]])

            processed.append(soft)

        lens = torch.tensor([t.size(0) for t in processed], dtype=torch.long, device=device)
        max_len = lens.max().item()
        posterior = torch.zeros(len(processed), max_len, vocab_size,
                                dtype=torch.float32, device=device)
        for b, t in enumerate(processed):
            posterior[b, :t.size(0)] = t.to(device)
        return posterior, lens
    
    def forward(self,
                input_ids: torch.LongTensor = None,
                input_features: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                input_feature_length : Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                GT: Optional[List[str]] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                ):
        # input features: [bs, audio_lengths，560] , input_length: [bs]
        speech = input_features
        B = speech.size(0)

        language_query = self.encoder.embed(
            torch.tensor([[0]], device=speech.device)
        ).repeat(B, 1, 1)  # [B, 1, d_model]

        textnorm_query = self.encoder.embed(
            torch.tensor([[2]], device=speech.device)
        ).repeat(B, 1, 1)  # [B, 1, d_model]

        event_emo_query = self.encoder.embed(
            torch.tensor([[1, 2]], device=speech.device)
        ).repeat(B, 1, 1)  # [B, 2, d_model]

        speech = torch.cat([language_query, event_emo_query, textnorm_query, speech], dim=1)
        speech_lengths = input_feature_length + 4  
        
        raw_encoder_out, raw_encoder_out_lens = self.encoder.encoder(speech, speech_lengths)
        if isinstance(raw_encoder_out, tuple):
            raw_encoder_out = raw_encoder_out[0]

        # delete formulate output: first 4 tokens
        raw_logits = self.encoder.ctc.ctc_lo(raw_encoder_out)
        raw_ctc_posterior = torch.softmax(raw_logits, dim=-1)
        ctc_posterior = raw_ctc_posterior[:, 4:, :]
        encoder_out      = raw_encoder_out[:, 4:, :]          # [B, T-4, D]
        encoder_out_lens = torch.clamp(raw_encoder_out_lens - 4, min=0)   # [B]

        if self.ctc_posterior:
            print("Use CTC Posterior ...")
            if self.voca_trans == False: 
                if self.gt_emb:
                    print("Use Groundtruth Embeddings...")
                    texts = GT
                    device = labels.device
                    if self.gt_emb_noise:
                        encoder_outs, encoder_feature_length = self.ctc_pseudo_posterior_noise(texts)   # [B, L_max, vocab_size]
                    else:
                        encoder_outs, encoder_feature_length = self.ctc_pseudo_posterior(texts)   # [B, L_max, vocab_size]
                    encoder_outs          = encoder_outs.to(device, non_blocking=True)
                    encoder_feature_length = encoder_feature_length.to(device, non_blocking=True)
                else:
                    if self.do_psd:
                        encoder_outs, encoder_feature_length = self.psd(ctc_posterior, encoder_out_lens, ctc_posterior,self.encoder.blank_id)
                    else:
                        encoder_outs, encoder_feature_length = ctc_posterior, encoder_out_lens
                
                if self.cross_attn:
                    with torch.no_grad():  
                        llm_embedding = self.llm.get_input_embeddings().weight
                    llm_embedding = llm_embedding.detach()
                    projector_outs = self.encoder_projector(encoder_outs, llm_embedding) 
                    projector_feature_length = encoder_feature_length
                else:
                    projector_outs = self.encoder_projector(encoder_outs) 
                    projector_feature_length = encoder_feature_length // self.encoder_projector.k     

            else:
                print("Vocabulary Transform is ready ...")
                if self.do_psd: # projector serves as a ctc head
                    projector_outs = self.encoder_projector(encoder_outs) 
                    projector_feature_length = encoder_feature_length // self.encoder_projector.k
                    ctc_posterior = torch.softmax(projector_outs,dim=-1)
                    projector_outs, projector_feature_length= self.psd(projector_outs, projector_feature_length, ctc_posterior, 151643)
                    llm_embedding = self.llm.get_input_embeddings()
                    embed_matrix = llm_embedding.weight  # [llm_vocab, hidden]
                    V_real = projector_outs.size(-1) - 1
                    logits_no_blank = projector_outs[..., :V_real]            
                    ctc_outs = torch.softmax(logits_no_blank,dim=-1)
                    projector_outs = torch.einsum("btv,vh->bth", ctc_outs, embed_matrix[:V_real])
                    if self.top1_emb:
                        # logits_no_blank: [B, T, V_real]   (drop blank)
                        print("Use Top1 pred_emb ....")
                        top1_ids = ctc_outs.argmax(dim=-1).to(torch.int32)                   # [B, T]
                        projector_outs = embed_matrix[top1_ids]                    # [B, T, H]     

                else:
                    projector_outs = self.encoder_projector(encoder_outs) 
                    projector_feature_length = encoder_feature_length // self.encoder_projector.k
                    llm_embedding = self.llm.get_input_embeddings()
                    embed_matrix = llm_embedding.weight          
                    ctc_outs = torch.softmax(projector_outs,dim=-1)
                    projector_outs = torch.einsum("btv,vh->bth", ctc_outs, embed_matrix[:projector_outs.size(-1)])
                    if self.top1_emb:
                        print("Use Top1 pred_emb ....")
                        top1_ids = ctc_outs.argmax(dim=-1).to(torch.int32)                   # [B, T]
                        projector_outs = embed_matrix[top1_ids]                    # [B, T, H]     
        else:
            print("Use raw feature ...")
            if self.do_psd:
                encoder_outs, encoder_feature_length = self.psd(encoder_out, encoder_out_lens, ctc_posterior,self.encoder.blank_id)
            else:
                encoder_outs, encoder_feature_length = encoder_out, encoder_out_lens

            projector_outs = self.encoder_projector(encoder_outs) 
            projector_feature_length = encoder_feature_length // self.encoder_projector.k    

        inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        inputs_embeds, attention_mask, labels, position_ids, _ = self._merge_input_ids_with_audio_features(
                projector_outs, projector_feature_length, inputs_embeds, input_ids, attention_mask, labels
            )
        
        model_outputs = self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels,position_ids=position_ids)
        acc = -1
        if self.metric:
            with torch.no_grad():
                preds = torch.argmax(model_outputs.logits, -1)
                acc = compute_accuracy(preds.detach()[:, :-1], labels.detach()[:, 1:], ignore_label=self.tokenizer.default_ignore_token)

        return model_outputs, acc
    
    @torch.no_grad()
    def generate(self,
                input_ids: torch.LongTensor = None,
                input_features: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                input_feature_length : Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                targets: Optional[str] = None,
                **kwargs
                ):
        
        # input features: [bs, audio_lengths，560] , input_length: [bs]
        speech = input_features
        B = speech.size(0)

        language_query = self.encoder.embed(
            torch.tensor([[0]], device=speech.device)
        ).repeat(B, 1, 1)  # [B, 1, d_model]

        textnorm_query = self.encoder.embed(
            torch.tensor([[2]], device=speech.device)
        ).repeat(B, 1, 1)  # [B, 1, d_model]

        event_emo_query = self.encoder.embed(
            torch.tensor([[1, 2]], device=speech.device)
        ).repeat(B, 1, 1)  # [B, 2, d_model]

        speech = torch.cat([language_query, event_emo_query, textnorm_query, speech], dim=1)
        speech_lengths = input_feature_length + 4  
        
        raw_encoder_out, raw_encoder_out_lens = self.encoder.encoder(speech, speech_lengths)
        if isinstance(raw_encoder_out, tuple):
            raw_encoder_out = raw_encoder_out[0]

        # delete formulate output: first 4 tokens
        raw_logits = self.encoder.ctc.ctc_lo(raw_encoder_out)
        raw_ctc_posterior = torch.softmax(raw_logits, dim=-1)
        ctc_posterior = raw_ctc_posterior[:, 4:, :]
        encoder_out      = raw_encoder_out[:, 4:, :]          # [B, T-4, D]
        encoder_out_lens = torch.clamp(raw_encoder_out_lens - 4, min=0)   # [B]
        
        if self.ctc_posterior:
            print("Use CTC Posterior ...")
            if self.voca_trans == False: 
                if self.gt_emb:
                    print("Use Groundtruth Embeddings...")
                    import re
                    texts = [re.sub(r"[^A-Za-z\s.,!?]+", "", t).lower().strip()   
                            for t in targets]
                    device = input_ids.device
                    encoder_outs, encoder_feature_length = self.ctc_pseudo_posterior(texts)   # [B, L_max, vocab_size]
                    encoder_outs          = encoder_outs.to(device, non_blocking=True)
                    encoder_feature_length = encoder_feature_length.to(device, non_blocking=True)
                else:
                    if self.do_psd:
                        print("Apply PSD on CTC Posterior ...")
                        encoder_outs, encoder_feature_length = self.psd(ctc_posterior, encoder_out_lens, ctc_posterior,self.encoder.blank_id)
                    else:
                        encoder_outs, encoder_feature_length = ctc_posterior, encoder_out_lens
                if self.cross_attn:
                    with torch.no_grad():  
                        llm_embedding = self.llm.get_input_embeddings().weight
                    llm_embedding = llm_embedding.detach()
                    projector_outs = self.encoder_projector(encoder_outs, llm_embedding) 
                    projector_feature_length = encoder_feature_length 
                else:
                    projector_outs = self.encoder_projector(encoder_outs) 
                    projector_feature_length = encoder_feature_length // self.encoder_projector.k     

            else:
                print("Vocabulary Transform is ready ...")
                if self.do_psd: # projector serves as a ctc head
                    projector_outs = self.encoder_projector(encoder_outs) 
                    projector_feature_length = encoder_feature_length // self.encoder_projector.k
                    ctc_posterior = torch.softmax(projector_outs,dim=-1)
                    projector_outs, projector_feature_length= self.psd(projector_outs, projector_feature_length, ctc_posterior, self.encoder.blank_id)
                    llm_embedding = self.llm.get_input_embeddings()
                    embed_matrix = llm_embedding.weight  # [llm_vocab, hidden]
                    V_real = projector_outs.size(-1) - 1
                    logits_no_blank = projector_outs[..., :V_real]            
                    ctc_outs = torch.softmax(logits_no_blank,dim=-1)
                    projector_outs = torch.einsum("btv,vh->bth", ctc_outs, embed_matrix[:V_real])
                    if self.top1_emb:
                        # logits_no_blank: [B, T, V_real]   (drop blank)
                        print("Use Top1 pred_emb ....")
                        top1_ids = ctc_outs.argmax(dim=-1).to(torch.int32)                   # [B, T]
                        projector_outs = embed_matrix[top1_ids]                    # [B, T, H]     

                else:
                    projector_outs = self.encoder_projector(encoder_outs) 
                    projector_feature_length = encoder_feature_length // self.encoder_projector.k
                    llm_embedding = self.llm.get_input_embeddings()
                    embed_matrix = llm_embedding.weight          
                    ctc_outs = torch.softmax(projector_outs,dim=-1)
                    projector_outs = torch.einsum("btv,vh->bth", ctc_outs, embed_matrix[:projector_outs.size(-1)])
                    if self.top1_emb:
                        print("Use Top1 pred_emb ....")
                        top1_ids = ctc_outs.argmax(dim=-1).to(torch.int32)                   # [B, T]
                        projector_outs = embed_matrix[top1_ids]                    # [B, T, H]     
        else:
            print("Use raw feature ...")
            if self.do_psd:
                encoder_outs, encoder_feature_length = self.psd(encoder_out, encoder_out_lens, ctc_posterior,self.encoder.blank_id)
            else:
                encoder_outs, encoder_feature_length = encoder_out, encoder_out_lens
            projector_outs = self.encoder_projector(encoder_outs) 
            projector_feature_length = encoder_feature_length // self.encoder_projector.k    
                
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)

        inputs_embeds, attention_mask, labels, position_ids, _ = self._merge_input_ids_with_audio_features(
                projector_outs, projector_feature_length, inputs_embeds, input_ids, attention_mask, labels
            )
        
        model_outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=kwargs.get("max_new_tokens", 200),
            num_beams=kwargs.get("num_beams", 4),
            do_sample=kwargs.get("do_sample", False),
            min_length=kwargs.get("min_length", 1),
            top_p=kwargs.get("top_p", 1.0),
            repetition_penalty=kwargs.get("repetition_penalty", 1.0),
            length_penalty=kwargs.get("length_penalty", 1.0),
            temperature=kwargs.get("temperature", 1.0),
            attention_mask=attention_mask,
            # position_ids=position_ids,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id
        )

        return model_outputs

    def _merge_input_ids_with_audio_features(
        self, audio_features, num_audio_tokens, inputs_embeds, input_ids, attention_mask, labels
    ):
        """
        Merge input_ids with with audio features into final embeddings
        
        Args:
            audio_features (`torch.Tensor` of shape `(num_audios, max_audio_tokens, embed_dim)`):
                All audio vectors of all audios in the batch
            num_audio_tokens (`torch.LongTensor` of shape `(num_audios)`):
                The length of audio embeddings of each audio as stacked in `audio_features`
            inputs_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, embed_dim)`):
                Token embeddings before merging with audio embeddings
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Input_ids of tokens, possibly filled with audio token
            attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Mask to avoid performing attention on padding token indices.
            labels (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*)
                labels need to be recalculated to support training (if provided)
        Returns:
            final_embedding, final_attention_mask, final_labels, position_ids, final_input_ids

        Explanation:
            each audio has variable length embeddings, with length specified by num_audio_tokens
            audio_features is concatenation of all audio embed vectors
            task: fill each <|AUDIO|> with the correct number of audio embeddings
            Example:
                X (5 tokens), Y (3 tokens), Z (8 tokens)
                X, Y are in the same sequence (in-context learning)
            if right padding
                input_ids: [
                    a b c d e f X g h i j k Y l m
                    o p q r Z s t u v _ _ _ _ _ _
                ]
                input_ids should be: [
                    a b c d e f X X X X X g h i j k Y Y Y l m
                    o p q r Z Z Z Z Z Z Z Z s t u v _ _ _ _ _
                ]
                labels should be: [
                    a b c d e f _ _ _ _ _ g h i j k _ _ _ l m
                    o p q r _ _ _ _ _ _ _ _ s t u v _ _ _ _ _
                ]
            elif left padding
                input_ids: [
                    a b c d e f X g h i j k Y l m
                    _ _ _ _ _ _ o p q r Z s t u v
                ]
                input_ids should be: [
                    a b c d e f X X X X X g h i j k Y Y Y l m
                    _ _ _ _ _ o p q r Z Z Z Z Z Z Z Z s t u v
                ]
                labels should be: [
                    a b c d e f _ _ _ _ _ g h i j k _ _ _ l m
                    _ _ _ _ _ o p q r _ _ _ _ _ _ _ _ s t u v
                ]
            Edge cases:
                * If tokens are same but audio token sizes are different, then cannot infer left or right padding
                ```python
                url1 = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/glass-breaking-151256.mp3"
                audio1, _ = librosa.load(BytesIO(urlopen(url1).read()), sr=processor.feature_extractor.sampling_rate)
                url2 = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/f2641_0_throatclearing.wav"
                audio2, _ = librosa.load(BytesIO(urlopen(url2).read()), sr=processor.feature_extractor.sampling_rate)
                prompts = [
                    "[INST] <|AUDIO|>\nWhat is that in this audio? [/INST]",
                    "[INST] <|AUDIO|>\nWhat is that in this audio? [/INST]",
                ]
                inputs = processor(text=prompts, audios=[audio1, audio2], return_tensors='pt', padding=True).to("cuda")
                    audio1 has 101 tokens, while audio2 has 72 tokens
                ```

                input_ids: [
                    a b c d X g h
                    i j Y k l m n
                ]
                where X is 3 tokens while Y is 5, this mean after merge
                if left-padding (batched generation)
                    input_ids should be: [
                        _ _ a b c d X X X g h
                        i j Y Y Y Y Y k l m n
                    ]
                elif (right padding) (training)
                    input_ids should be: [
                        a b c d X X X g h _ _
                        i j Y Y Y Y Y k l m n
                    ]
        """
        num_audios, max_audio_tokens, embed_dim = audio_features.shape
        audio_features_mask = torch.arange(max_audio_tokens).expand(num_audios, max_audio_tokens).to(
            num_audio_tokens.device
        ) < num_audio_tokens.unsqueeze(1)
        masked_audio_features = audio_features[audio_features_mask].view(-1, embed_dim)
        batch_size, sequence_length = input_ids.shape
        _left_padding = torch.any(attention_mask[:, 0] == 0)
        _right_padding = torch.any(attention_mask[:, -1] == 0)

        left_padding = True
        if batch_size > 1:
            if _left_padding and not _right_padding:
                left_padding = True
            elif not _left_padding and _right_padding:
                left_padding = False
            elif not _left_padding and not _right_padding:
                # both side is 1, so cannot tell
                left_padding = True
            else:
                # invalid attention_mask
                raise ValueError(f"both side of attention_mask has zero, invalid. {attention_mask}")

        # 1. Create a mask to know where special audio tokens are
        special_audio_token_mask = input_ids == self.tokenizer.default_speech_token
        num_special_audio_tokens = torch.sum(special_audio_token_mask, dim=-1)

        # In case the Audio model or the Language model has been offloaded to CPU, we need to manually
        # set the corresponding tensors into their correct target device.
        target_device = inputs_embeds.device
        attention_mask = attention_mask.to(target_device)
        input_ids = input_ids.to(target_device)
        num_audio_tokens = num_audio_tokens.to(target_device)
        batch_indices, non_audio_indices = torch.where(
            (input_ids != self.tokenizer.default_speech_token) & (attention_mask == 1)
        )

        # 2. Compute the positions where text should be written
        # Calculate new positions for text tokens in merged audio-text sequence.
        # `special_audio_token_mask` identifies audio tokens. Each audio token will be replaced by `audio_feat_lengths - 1` text tokens.
        # `torch.cumsum` computes how each audio token shifts subsequent text token positions.
        token_placeholder_num = torch.zeros_like(input_ids)
        token_placeholder_num[special_audio_token_mask] = num_audio_tokens.long() - 1
        token_placeholder_num = token_placeholder_num + 1
        new_token_positions = torch.cumsum(token_placeholder_num, -1) - 1
        max_token_num = token_placeholder_num.sum(-1).max()
        nb_audio_pad = max_token_num - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_audio_pad[:, None]  # offset for left padding
        text_to_overwrite = new_token_positions[batch_indices, non_audio_indices]
        batch_indices, non_audio_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_audio_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = torch.zeros(
            batch_size, max_token_num, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
        final_attention_mask = torch.zeros(
            batch_size, max_token_num, dtype=attention_mask.dtype, device=inputs_embeds.device
        )
        final_input_ids = torch.full(
            (batch_size, max_token_num), self.tokenizer.pad_token_id, dtype=input_ids.dtype, device=inputs_embeds.device
        )

        # 4. Fill the embeddings based on the mask. If we have ["hey" "<audio>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the audio features
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_audio_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_audio_indices]
        final_input_ids[batch_indices, text_to_overwrite] = input_ids[batch_indices, non_audio_indices]
        final_labels = None
        if labels is not None:
            labels = labels.to(target_device)
            final_labels = torch.full((batch_size, max_token_num),self.tokenizer.default_ignore_token,dtype=input_ids.dtype, device=inputs_embeds.device).to(torch.long)
            final_labels[batch_indices, text_to_overwrite] = labels[batch_indices, non_audio_indices]
        # 5. Fill the embeddings corresponding to the audios. Anything that is still zeros needs filling
        audio_to_overwrite = torch.full(
            (batch_size, max_token_num), True, dtype=torch.bool, device=inputs_embeds.device
        )
        audio_to_overwrite[batch_indices, text_to_overwrite] = False
        seq_indices = torch.arange(max_token_num).unsqueeze(0).to(target_device)
        seq_indices = seq_indices.expand(batch_size, max_token_num)

        if left_padding:
            # exclude padding on the left
            max_token_num = max_token_num.to(target_device)
            val = (max_token_num - seq_indices) <= (
                token_placeholder_num.sum(-1) - (attention_mask == 0).long().sum(-1)
            )[:, None]
        else:
            # exclude padding on the right
            val = seq_indices < (token_placeholder_num.sum(-1) - (attention_mask == 0).long().sum(-1))[:, None]

        audio_to_overwrite &= val

        if audio_to_overwrite.sum() != num_audio_tokens.sum():
            raise ValueError(
                f"The input provided to the model are wrong. The number of audio tokens is {num_special_audio_tokens} while"
                f" the number of audio given to the model is {num_audios}. This prevents correct indexing and breaks batch generation."
            )

        final_embedding[audio_to_overwrite] = (
            masked_audio_features.contiguous().reshape(-1, embed_dim).to(target_device)
        )
        final_attention_mask |= audio_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)

        return final_embedding, final_attention_mask, final_labels, position_ids, final_input_ids
    