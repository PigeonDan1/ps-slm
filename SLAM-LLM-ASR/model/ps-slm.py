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
from utils.npu_flash_attn import patch_npu_flash_attn

logger = logging.getLogger(__name__)
def ctc_greedy_search(
                log_probs: torch.Tensor,
                input_lens: torch.Tensor,
                blank_id: int = 0,
            ) -> List[List[int]]:
                """
                Args
                ----
                log_probs : (T, B, V)  log-softmax 后的概率
                input_lens: (B,)        每条样本的有效帧长
                blank_id  : int         blank 的编号

                Returns
                -------
                List[List[int]] : 每条样本解码后的 token id 列表
                """
                T, B, V = log_probs.shape
                input_lens = input_lens.long()
                device = log_probs.device

                # 1. argmax 得到每个时刻的 token
                indices = log_probs.argmax(dim=-1)  # (T, B)

                # 2. 逐条样本去重 & 去 blank
                results = []
                for b in range(B):
                    seq = indices[:input_lens[b], b].cpu().tolist()  # 取有效帧
                    # 去重 + 去 blank
                    dedup = [seq[0]] if seq else []
                    for t in range(1, len(seq)):
                        if seq[t] != seq[t-1] and seq[t] != blank_id:
                            dedup.append(seq[t])
                    # 首尾可能还是 blank，再过滤一次
                    final = [idx for idx in dedup if idx != blank_id]
                    results.append(final)

                return results
def extract_variable_length_features(self, x: torch.Tensor):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        """
        x = F.gelu(self.conv1(x))
        x = F.gelu(self.conv2(x))
        x = x.permute(0, 2, 1)

        # assert x.shape[1:] == self.positional_embedding.shape, "incorrect audio shape"
        # x = (x + self.positional_embedding).to(x.dtype)
        x = (x + self.positional_embedding[: x.shape[1]]).to(x.dtype)

        for block in self.blocks:
            x = block(x)

        x = self.ln_post(x)
        return x

def setup_tokenizer(train_config, model_config, **kwargs):
    # Load the tokenizer and add special tokens
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

    if train_config.freeze_llm: # TODO:to test offical `freeze_layers` and `num_freeze_layers`
        for name, param in model.named_parameters(): 
            param.requires_grad = False
        model.eval()
        
    if kwargs.get("peft_ckpt", None): # (FIX:MZY):reload will get wrong results when decoding
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
    patch_npu_flash_attn()
    ckpt_path = kwargs.get( "ckpt_path", None)
    if ckpt_path is not None:
        logger.info("loading other parts from: {}".format(ckpt_path))
        ckpt_dict = torch.load(ckpt_path, map_location="cpu")
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

import os
import h5py
import numpy as np

def append_to_cpu_file(dist_name, tensor, length):
    """
    tensor:  [B, T, V]  已 .cpu().half()
    length:  [B]        已 .cpu()
    不裁剪 padding，直接整段存
    """
    h5_path = '/aistor/aispeech/hpc_stor01/home/pengjing00sx/Github/ps-slm/SLAM-LLM-ASR/distribution/cpu_cache.h5'
    print(f"Save {dist_name} distribution ...")

    # 转成 NumPy float16
    tensor_np = tensor.numpy()

    with h5py.File(h5_path, 'a') as f:
        # 为每个分布建一个 group
        grp = f.require_group(dist_name)

        # 当前组内已有样本数 = 下一个索引
        next_idx = len(grp)

        # 逐样本写入
        for b in range(tensor_np.shape[0]):
            ds_name = f'{next_idx + b:08d}'        # 补零方便排序
            grp.create_dataset(
                ds_name,
                data=tensor_np[b],                  # [T, V] 原始形状
                dtype=np.float16,
                compression='gzip',                 # 可省 3~5 倍磁盘
                compression_opts=4
            )
            # 把真实长度存为属性
            grp[ds_name].attrs['length'] = int(length[b])

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
        if model_config.encoder_projector == "cross-attention":
            self.cross_attn = True
        else:
            self.cross_attn = False
        # if self.gt_emb:
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

    # def psd(
    #         self,
    #         encoder_out: torch.Tensor,
    #         encoder_out_lens: torch.Tensor,
    #         blank_id: int = 0,
    #         blank_threshold: float = 0.90
    # ) -> Tuple[torch.Tensor, int]:
    #     """
    #     删除高置信 blank 帧，重新 padding，老版本-没有merge!。
    #     返回:
    #         encoder_outs          : [B, T_new, D]  已 0-pad
    #         encoder_feature_length: int           T_new 的最大帧数
    #     """
    #     B, T, D = encoder_out.shape
    #     device  = encoder_out.device

    #     with torch.no_grad():
    #         ctc_probs = torch.softmax(self.encoder.ctc.ctc_lo(encoder_out), dim=-1)  # [B, T, V]

    #     keep_frames = []
    #     new_lens    = []
    #     for b in range(B):
    #         L = encoder_out_lens[b].item()
    #         if L == 0:                      # 极端情况
    #             keep_frames.append([])
    #             new_lens.append(0)
    #             continue

    #         prob_blank = ctc_probs[b, :L, blank_id]            # [L]
    #         mask = prob_blank < blank_threshold                # True 表示保留
    #         idx  = torch.nonzero(mask, as_tuple=False).squeeze(-1)
    #         keep_frames.append(encoder_out[b, idx])            # [M, D]
    #         new_lens.append(idx.size(0))

    #     max_len = max(new_lens) if new_lens else 0
    #     if max_len == 0:                       # 整批都是空
    #         return encoder_out.new_zeros(B, 0, D), 0

    #     padded = []
    #     for feat in keep_frames:
    #         pad_len = max_len - feat.size(0)
    #         if pad_len > 0:
    #             feat = torch.cat([feat, feat.new_zeros(pad_len, D)], dim=0)
    #         padded.append(feat)
    #     encoder_outs = torch.stack(padded, dim=0)   # [B, T_new, D]
    #     new_lens = torch.tensor(new_lens, dtype=torch.long, device=device)
    #     return encoder_outs, new_lens

    def psd(
            self,
            encoder_out: torch.Tensor,      # [B, T, D]
            encoder_out_lens: torch.Tensor, # [B]
            ctc_posterior: torch.Tensor,    # [B, T, V]  <-- 新增输入
            blank_id: int = 0,
            blank_threshold: float = 0.90
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        1. 只合并相邻且相同的非 blank 字符帧（空白帧不合并）
        2. 若某字符连续帧数>5则打印
        3. 用 blank 概率阈值 0.9 统一删除 blank 帧
        4. 重新 0-pad
        返回:
            encoder_outs : [B, T_new, D]
            new_lens     : [B]   每句实际帧长
        """
        B, T, D = encoder_out.shape
        device  = encoder_out.device

        is_log_prob = ctc_posterior.max() <= 0
        ctc_probs = ctc_posterior.exp() if is_log_prob else ctc_posterior
        # print(f"raw_shape: {encoder_out.shape}, raw_lens: {encoder_out_lens}")
        keep_frames, new_lens = [], []
        for b in range(B):
            L = encoder_out_lens[b].item()
            if L == 0:
                keep_frames.append(encoder_out.new_zeros(0, D))
                new_lens.append(0)
                continue

            ids = ctc_probs[b, :L].argmax(dim=-1)  # [L]

            # ---- 合并相邻相同非 blank 字符帧 ----
            merged_feats, merged_blank_probs = [], []
            start = 0
            for end in range(1, L + 1):
                if end == L or ids[end] != ids[start]:
                    seg_len = end - start
                    char_id = ids[start].item()

                    if char_id == blank_id:
                        # blank 帧：每帧单独保留
                        for t in range(start, end):
                            merged_feats.append(encoder_out[b, t])
                            merged_blank_probs.append(ctc_probs[b, t, blank_id])
                    else:
                        # 非 blank：合并整段
                        if seg_len > 5:
                            print(f"[PSD] Warning: batch={b}, char={char_id}, "
                                f"continuous frames={seg_len} (>5)")
                        # print(f"seglen: {seg_len}")
                        merged_feats.append(encoder_out[b, start:end].mean(dim=0))
                        avg_blank_prob = ctc_probs[b, start:end, blank_id].mean()
                        merged_blank_probs.append(avg_blank_prob)
                    start = end

            merged_feats = torch.stack(merged_feats, dim=0)           # [T_merged, D]
            merged_blank_probs = torch.tensor(merged_blank_probs,
                                            device=device)        # [T_merged]

            # ---- 用阈值 0.9 过滤 blank ----
            mask = merged_blank_probs < blank_threshold
            keep = mask.nonzero(as_tuple=False).squeeze(-1)
            feats_after_blank = merged_feats[keep]                    # [M, D]

            keep_frames.append(feats_after_blank)
            new_lens.append(feats_after_blank.size(0))

        # 4) pad 到 batch 最大长度
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
        # print(f"encoder_outs_shape: {encoder_outs.shape}, new_lens: {new_lens}")
        # input()
        return encoder_outs, new_lens
    
    def ids2text(self, ids: torch.LongTensor, llm):
        """
        ids: [B, T]  已 padding，-100 位置忽略
        llm: transformers.PreTrainedModel / AutoModelForCausalLM
        return: list[str]  每条样本的文本
        """
        # 1. 把 -100 变成 pad_token_id，其余保持
        pad_id = llm.config.pad_token_id if llm.config.pad_token_id is not None else llm.config.eos_token_id
        ids = torch.where(ids == -100, pad_id, ids)

        # 2. 解码
        text_list = self.tokenizer.batch_decode(
            ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        return text_list

    import torch

    def ctc_pseudo_posterior(self, texts):
        """
        texts: list[str]  —— 已解码文本
        return: 
            posterior: [B, L_max, vocab_size]  one-hot 伪后验
            lens:      [B]   每条样本的真实 token 长度
        """
        tok = self.encoder_tokenizer
        ids_list = [tok.encode(t) for t in texts]

        # 真实长度
        lens = torch.tensor([len(ids) for ids in ids_list], dtype=torch.long)

        # for t, ids in zip(texts, ids_list):
        #     print(tok.decode(ids), t)
        # input()
        # 对齐长度
        max_len = lens.max().item()
        vocab_size = tok.vocab_size

        # one-hot 伪后验
        B = len(ids_list)
        posterior = torch.zeros(B, max_len, vocab_size, dtype=torch.float32)
        for b, ids in enumerate(ids_list):
            posterior[b, torch.arange(len(ids)), ids] = 1.0

        return posterior, lens

    def ctc_pseudo_posterior_noise(self, texts):
        """
        texts: list[str]  —— 已解码文本
        return:
            posterior: [B, L_max, vocab_size]  伪后验（已平滑+随机增删）
            lens:      [B]   每条样本处理后的真实 token 长度
        """
        print("Add noise simulation ...")
        tok = self.encoder_tokenizer
        vocab_size = tok.vocab_size
        device = next(self.parameters()).device

        # ---------- 超参 ----------
        drop_prob   = getattr(self, 'drop_prob',   0.05)          # 丢帧(字符)概率
        insert_prob = getattr(self, 'insert_prob', 0.0)          # 相对长度插入比例
        smooth_low  = getattr(self, 'smooth_low',  0.0)           # 平滑 α 范围
        smooth_high = getattr(self, 'smooth_high', 0.2)
        blank_id    = self.encoder.blank_id  # 预留 blank
        # --------------------------

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
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                ):
        # input features: [bs, audio_lengths，560] , input_length: [bs]
        speech = input_features
        B = speech.size(0)

        # 构造 4 帧前缀
        language_query = self.encoder.embed(
            torch.tensor([[0]], device=speech.device)
        ).repeat(B, 1, 1)  # [B, 1, d_model]

        textnorm_query = self.encoder.embed(
            torch.tensor([[2]], device=speech.device)
        ).repeat(B, 1, 1)  # [B, 1, d_model]

        event_emo_query = self.encoder.embed(
            torch.tensor([[1, 2]], device=speech.device)
        ).repeat(B, 1, 1)  # [B, 2, d_model]

        # 按官方顺序拼接
        speech = torch.cat([language_query, event_emo_query, textnorm_query, speech], dim=1)
        speech_lengths = input_feature_length + 4  # 4 帧前缀
        
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
                    texts = self.ids2text(labels, self.llm) 
                    device = labels.device
                    if self.gt_emb_noise:
                        encoder_outs, encoder_feature_length = self.ctc_pseudo_posterior_noise(texts)   # [B, L_max, vocab_size]
                    else:
                        encoder_outs, encoder_feature_length = self.ctc_pseudo_posterior(texts)   # [B, L_max, vocab_size]
                    encoder_outs          = encoder_outs.to(device, non_blocking=True)
                    encoder_feature_length = encoder_feature_length.to(device, non_blocking=True)
                else:
                    if self.do_psd:
                        # encoder_outs, encoder_feature_length = self.psd(encoder_out, encoder_out_lens,self.encoder.blank_id)
                        # encoder_outs = torch.softmax(self.encoder.ctc.ctc_lo(encoder_outs), dim=-1)
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
                        # logits_no_blank: [B, T, V_real]   (已去掉 blank)
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

        # 构造 4 帧前缀
        language_query = self.encoder.embed(
            torch.tensor([[0]], device=speech.device)
        ).repeat(B, 1, 1)  # [B, 1, d_model]

        textnorm_query = self.encoder.embed(
            torch.tensor([[2]], device=speech.device)
        ).repeat(B, 1, 1)  # [B, 1, d_model]

        event_emo_query = self.encoder.embed(
            torch.tensor([[1, 2]], device=speech.device)
        ).repeat(B, 1, 1)  # [B, 2, d_model]

        # 按官方顺序拼接
        speech = torch.cat([language_query, event_emo_query, textnorm_query, speech], dim=1)
        speech_lengths = input_feature_length + 4  # 4 帧前缀
        
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
                    # 测试阶段清洗
                    texts = [re.sub(r"[^A-Za-z\s.,!?]+", "", t).lower().strip()   # 注意去掉了 '
                            for t in targets]
                    device = input_ids.device
                    encoder_outs, encoder_feature_length = self.ctc_pseudo_posterior(texts)   # [B, L_max, vocab_size]
                    encoder_outs          = encoder_outs.to(device, non_blocking=True)
                    encoder_feature_length = encoder_feature_length.to(device, non_blocking=True)
                else:
                    if self.do_psd:
                        # encoder_outs, encoder_feature_length = self.psd(encoder_out, encoder_out_lens,self.encoder.blank_id)
                        # encoder_outs = torch.softmax(self.encoder.ctc.ctc_lo(encoder_outs), dim=-1)
                        encoder_outs, encoder_feature_length = self.psd(ctc_posterior, encoder_out_lens, ctc_posterior,self.encoder.blank_id)
                        # append_to_cpu_file('ctc', encoder_outs.cpu().half(), encoder_feature_length.cpu())
                        # import re
                        # # 测试阶段清洗
                        # texts = [re.sub(r"[^A-Za-z\s.,!?]+", "", t).lower().strip()   # 注意去掉了 '
                        #         for t in targets]
                        # encoder_outs_1, encoder_feature_length_1 = self.ctc_pseudo_posterior_noise(texts)   # [B, L_max, vocab_size]
                        # encoder_outs_2, encoder_feature_length_2 = self.ctc_pseudo_posterior(texts) 
                        # append_to_cpu_file('noise', encoder_outs_1.cpu().half(), encoder_feature_length_1.cpu())
                        # append_to_cpu_file('clean', encoder_outs_2.cpu().half(), encoder_feature_length_2.cpu())
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
                        # logits_no_blank: [B, T, V_real]   (已去掉 blank)
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
    

    def remove_blank_frames(self, logits, lengths, blank_id):
        """
        logits: (B, T, V) 原始 CTC logits
        lengths: (B,) 原始有效帧数
        blank_id: int blank token 的 id（这里等于 V_real）

        返回:
            new_logits : (B, T', V)  已去掉 blank 并重新 pad
            new_lengths: (B,)        新的有效帧数
        """
        B, T, V = logits.shape
        device = logits.device

        # 1. 贪婪解码 token id
        pred_ids = torch.argmax(logits, dim=-1)            # (B, T)

        # 2. 逐条去掉 blank 帧
        new_logits_list = []
        new_lengths = []
        max_len = 0

        for b in range(B):
            # 有效帧
            L = lengths[b]
            ids_b = pred_ids[b, :L]                          # 取有效部分
            keep_mask = ids_b != blank_id                  # True 保留
            new_logits_b = logits[b, :L][keep_mask]        # (L', V)

            new_len = new_logits_b.size(0)
            new_lengths.append(new_len)
            new_logits_list.append(new_logits_b)
            max_len = max(max_len, new_len)

        # 3. 重新 padding
        new_logits = []
        for b in range(B):
            pad_len = max_len - new_lengths[b]
            padded = F.pad(new_logits_list[b], (0, 0, 0, pad_len))  # (max_len, V)
            new_logits.append(padded)
        new_logits = torch.stack(new_logits, dim=0)        # (B, max_len, V)
        new_lengths = torch.tensor(new_lengths, device=device)

        return new_logits, new_lengths

# debug to see the output from sensevoice
        # with torch.no_grad():
        #     ctc_probs = torch.softmax(self.encoder.ctc.ctc_lo(encoder_out), dim=-1)
        #     b, n, d = encoder_out.size()
        #     for i in range(b):
        #         x = ctc_probs[i, : encoder_out_lens[i].item(), :]
        #         yseq = x.argmax(dim=-1)
        #         yseq = torch.unique_consecutive(yseq, dim=-1)
        #         mask = yseq != self.encoder.blank_id
        #         token_int = yseq[mask].tolist()
        #         import sentencepiece as spm
        #         # 加载 BPE 模型
        #         bpe_model_path = "/aistor/aispeech/hpc_stor01/group/asr/model/SenseVoiceSmall/chn_jpn_yue_eng_ko_spectok.bpe.model"
        #         tokenizer = spm.SentencePieceProcessor()
        #         tokenizer.load(bpe_model_path)
        #         text = tokenizer.decode(token_int)
        #         print()
def global_mean_var(t, mask=None, unbiased=False):
    """
    t: (B, T, D)；mask: (B, T) =1 有效, 0 padding
    返回: mean (B,), var (B,)
    """
    if mask is None:
        # 直接展平 (B, T*D)
        mean = t.mean(dim=(1, 2))                        # (B,)
        var  = t.var(dim=(1, 2), unbiased=unbiased)      # (B,)
    else:
        # 只对有效帧做加权平均
        mask = mask.unsqueeze(-1)                        # (B, T, 1)
        valid = mask.sum(dim=(1, 2)).clamp(min=1)        # 避免除0
        mean  = (t * mask).sum(dim=(1, 2)) / valid       # (B,)
        var   = (((t - mean[:, None, None]) ** 2) * mask).sum(dim=(1,2)) / valid
    return mean, var
# inputs_mean, inputs_var = global_mean_var(inputs_embeds, attention_mask)
        # proj_mean,   proj_var   = global_mean_var(projector_outs)   # 若 projector 也有 mask，可传入
        # print(f"inputs_mean: {inputs_mean}, inputs_var: {inputs_var}")
        # print(f"proj_mean: {proj_mean}, proj_var: {proj_var}")
        # print(f"proj_mean_all: {proj_mean.abs().mean().item()}")
        # print(f"proj_var_all: {proj_var.abs().mean().item()}")
        # input()