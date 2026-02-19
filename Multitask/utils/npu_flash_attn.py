import torch
import torch_npu
from transformers.utils import is_torch_sdpa_available, is_flash_attn_2_available, is_flash_attn_greater_or_equal_2_10

import logging

logger = logging.getLogger(__name__)


# SDPA 
original_sdpa = torch.nn.functional.scaled_dot_product_attention

def npu_fas_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
    if not is_causal:
        if attn_mask.dtype == torch.bool:
            attn_mask_npu = torch.logical_not(attn_mask.bool()).to(query.device)
        else:
            attn_mask_npu = attn_mask.bool().to(query.device)
        return original_sdpa(query, key, value, attn_mask_npu, dropout_p=dropout_p, is_causal=is_causal, scale=scale)
    return original_sdpa(query, key, value, attn_mask, dropout_p=dropout_p, is_causal=is_causal, scale=scale)


# flash_attn
if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func


def npu_fas_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, *args, **kwargs):
    head_num = q.shape[2]
    if not causal:
        return torch_npu.npu_fusion_attention(q, k, v, head_num, "BSND", keep_prob=1.0-dropout_p, scale=softmax_scale)[0]
    atten_mask_npu = torch.triu(torch.ones([2048, 2048]), diagonal=1).bool().to(q.device)
    mode = 3 if is_flash_attn_greater_or_equal_2_10() else 2
    return torch_npu.npu_fusion_attention(q, k, v, head_num, "BSND", keep_prob=1.0-dropout_p,
                                          scale=softmax_scale, atten_mask=atten_mask_npu, sparse_mode=mode)[0]



def npu_fas_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                        dropout_p=0.0, softmax_scale=None, causal=False, *args, **kwargs):
    head_num = q.shape[1]
    if not causal:
        return torch_npu.npu_fusion_attention(q, k, v, head_num, pse=None, atten_mask=None,
                                              scale=1.0 / math.sqrt(q.shape[-1]), keep_prob=1-dropout_p, input_layout="TND",
                                              actual_seq_qlen=tuple(cu_seqlens_q[1:].cpu().numpy().tolist()),
                                              actual_seq_kvlen=tuple(cu_seqlens_k[1:].cpu().numpy().tolist()))[0]
    atten_mask_npu = torch.triu(torch.ones([2048, 2048]), diagonal=1).bool().to(q.device)
    mode = 3 if is_flash_attn_greater_or_equal_2_10() else 2
    return torch_npu.npu_fusion_attention(q, k, v, head_num, pse=None, padding_mask=None, atten_mask=atten_mask_npu, 
                                          scale=1.0 / math.sqrt(q.shape[-1]), keep_prob=1-dropout_p, input_layout="TND",
                                          actual_seq_qlen=tuple(cu_seqlens_q[1:].cpu().numpy().tolist()),
                                          actual_seq_kvlen=tuple(cu_seqlens_k[1:].cpu().numpy().tolist()), sparse_mode=mode)[0]



# patch flash attention
def patch_npu_flash_attn():
    pass
    if is_torch_sdpa_available():
        torch.nn.functional.scaled_dot_product_attention = npu_fas_sdpa
        logger.info("Successfully patch SDPA on NPU.")
    #     # exit(1)
    # if is_flash_attn_2_available():
    #     flash_attn_func = npu_fas_func
    #     flash_attn_varlen_func = npu_fas_varlen_func
    #     logger.info("Successfully patch FA_func on NPU.")







