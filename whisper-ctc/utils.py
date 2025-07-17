import numpy as np
import math
from collections import defaultdict
from torchaudio.models.decoder import ctc_decoder

def build_flashlight_decoder(tokenizer, beam_size=10):
    """
    构建 TorchAudio + Flashlight 的 CTC Beam Search 解码器。
    """
    tokens = [tokenizer.convert_ids_to_tokens(i) for i in range(tokenizer.vocab_size)]
    tokens.append("")  # 空字符或其他 placeholder 作为 blank，占位在最后
    decoder = ctc_decoder(
        lexicon=None,
        tokens=tokens,
        lm=None,
        beam_size=beam_size,
        nbest=1,
    )
    return decoder

def ctc_decode(log_probs_or_ids, blank_id=0, mode="greedy", beam_size=10, decoder=None, tokenizer=None):
    """
    通用 CTC 解码接口，支持 greedy 和 beam search

    Args:
      log_probs_or_ids: 如果 mode=="greedy"，可以传 Tensor 或 list of IDs；
                        如果 mode=="beam"，必须传 numpy array log_probs (T, V)
      blank_id: CTC blank id
      mode: "greedy" 或 "beam"
      beam_size: beam search 宽度

    Returns:
      list of token IDs
    """
    if mode == "greedy":
        # 支持 Tensor 或 list
        if hasattr(log_probs_or_ids, "tolist"):
            ids = log_probs_or_ids.tolist()
        else:
            ids = list(log_probs_or_ids)
        # collapse repeats
        new_ids = []
        prev = None
        for i in ids:
            if i != prev and i != blank_id:
                new_ids.append(i)
            prev = i
        return new_ids

    elif mode == "beam":
        assert decoder is not None, "Beam 模式需提供 decoder"
        logits = log_probs_or_ids
        emissions = logits.cpu()
        hyps = decoder(emissions)  # List[List[CTCHypothesis]]
    
        for hyp_list in hyps:
            best_ids = hyp_list[0].tokens  # torch.LongTensor
            print(tokenizer.decode(best_ids.tolist(), skip_special_tokens=True))
            input()
        return None

    else:
        raise ValueError(f"Unsupported mode {mode}")
    
def ctc_beam_decode_simple(log_probs, beam_size=10, blank_id=0):
    """
    Basic CTC Beam Search implementation without language model.

    Args:
        log_probs: (T, V) log-probabilities (numpy array)
        beam_size: beam width
        blank_id: index of CTC blank token

    Returns:
        best_seq: list of token IDs (decoded sequence)
    """
    T, V = log_probs.shape
    Beam = {(): (0.0, -np.inf)}  # (prefix -> (pb_blank, pb_non_blank))

    for t in range(T):
        next_Beam = defaultdict(lambda: (-np.inf, -np.inf))
        for prefix, (pb, pnb) in Beam.items():
            # extend by blank
            nb_pb, nb_pnb = next_Beam[prefix]
            pb_new = np.logaddexp(nb_pb, pb + log_probs[t, blank_id] + pnb + 0)
            next_Beam[prefix] = (pb_new, nb_pnb)

            # extend by non-blank tokens
            for c in range(V):
                if c == blank_id:
                    continue
                new_prefix = prefix + (c,)
                score = log_probs[t, c]
                if len(prefix) > 0 and prefix[-1] == c:
                    pnb_new = np.logaddexp(next_Beam[new_prefix][1], pnb + score)
                else:
                    pnb_new = np.logaddexp(next_Beam[new_prefix][1], pb + score + pnb + score)
                next_Beam[new_prefix] = (next_Beam[new_prefix][0], pnb_new)

        # prune to beam_size best
        scored = []
        for pr, (pb, pnb) in next_Beam.items():
            score = np.logaddexp(pb, pnb)
            scored.append((score, pr))
        scored.sort(key=lambda x: x[0], reverse=True)
        Beam = {pr: next_Beam[pr] for (_, pr) in scored[:beam_size]}

    # choose best prefix
    best_prefix = max(Beam.items(), key=lambda kv: np.logaddexp(kv[1][0], kv[1][1]))[0]
    # collapse repeats & remove blanks
    out = []
    prev = None
    for c in best_prefix:
        if c != blank_id and c != prev:
            out.append(c)
        prev = c
    return out

def ctc_greedy_decode(pred_ids, blank_id=0):
    """
    Collapse repeats and remove blank tokens.
    pred_ids: 1D tensor or list
    """
    if hasattr(pred_ids, "tolist"):
        pred_ids = pred_ids.tolist()
    new_ids = []
    prev = None
    for i in pred_ids:
        if i != prev and i != blank_id:
            new_ids.append(i)
        prev = i
    return new_ids

# Copyright (c) 2021 Mobvoi Inc (Binbin Zhang, Di Wu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Tuple

import numpy as np

import torch
import torchaudio.functional as F


def remove_duplicates_and_blank(hyp: List[int],
                                blank_id: int = 0) -> List[int]:
    new_hyp: List[int] = []
    cur = 0
    while cur < len(hyp):
        if hyp[cur] != blank_id:
            new_hyp.append(hyp[cur])
        prev = cur
        while cur < len(hyp) and hyp[cur] == hyp[prev]:
            cur += 1
    return new_hyp


def replace_duplicates_with_blank(hyp: List[int],
                                  blank_id: int = 0) -> List[int]:
    new_hyp: List[int] = []
    cur = 0
    while cur < len(hyp):
        new_hyp.append(hyp[cur])
        prev = cur
        cur += 1
        while cur < len(
                hyp) and hyp[cur] == hyp[prev] and hyp[cur] != blank_id:
            new_hyp.append(blank_id)
            cur += 1
    return new_hyp


def gen_ctc_peak_time(hyp: List[int], blank_id: int = 0) -> List[int]:
    times = []
    cur = 0
    while cur < len(hyp):
        if hyp[cur] != blank_id:
            times.append(cur)
        prev = cur
        while cur < len(hyp) and hyp[cur] == hyp[prev]:
            cur += 1
    return times


def gen_timestamps_from_peak(
    peaks: List[int],
    max_duration: float,
    frame_rate: float = 0.04,
    max_token_duration: float = 1.0,
) -> List[Tuple[float, float]]:
    """
    Args:
        peaks: ctc peaks time stamp
        max_duration: max_duration of the sentence
        frame_rate: frame rate of every time stamp, in seconds
        max_token_duration: max duration of the token, in seconds
    Returns:
        list(start, end) of each token
    """
    times = []
    half_max = max_token_duration / 2
    for i in range(len(peaks)):
        if i == 0:
            start = max(0, peaks[0] * frame_rate - half_max)
        else:
            start = max((peaks[i - 1] + peaks[i]) / 2 * frame_rate,
                        peaks[i] * frame_rate - half_max)

        if i == len(peaks) - 1:
            end = min(max_duration, peaks[-1] * frame_rate + half_max)
        else:
            end = min((peaks[i] + peaks[i + 1]) / 2 * frame_rate,
                      peaks[i] * frame_rate + half_max)
        times.append((start, end))
    return times


def insert_blank(label, blank_id=0):
    """Insert blank token between every two label token."""
    label = np.expand_dims(label, 1)
    blanks = np.zeros((label.shape[0], 1), dtype=np.int64) + blank_id
    label = np.concatenate([blanks, label], axis=1)
    label = label.reshape(-1)
    label = np.append(label, label[0])
    return label


def force_align(ctc_probs: torch.Tensor, y: torch.Tensor, blank_id=0) -> list:
    """ctc forced alignment.

    Args:
        torch.Tensor ctc_probs: hidden state sequence, 2d tensor (T, D)
        torch.Tensor y: id sequence tensor 1d tensor (L)
        int blank_id: blank symbol index
    Returns:
        torch.Tensor: alignment result
    """
    ctc_probs = ctc_probs[None].cpu()
    y = y[None].cpu()
    alignments, _ = F.forced_align(ctc_probs, y, blank=blank_id)
    return alignments[0]


def get_blank_id(configs, symbol_table):
    if 'ctc_conf' not in configs:
        configs['ctc_conf'] = {}

    if '<blank>' in symbol_table:
        if 'ctc_blank_id' in configs['ctc_conf']:
            assert configs['ctc_conf']['ctc_blank_id'] == symbol_table[
                '<blank>']
        else:
            configs['ctc_conf']['ctc_blank_id'] = symbol_table['<blank>']
    else:
        assert 'ctc_blank_id' in configs[
            'ctc_conf'], "PLZ set ctc_blank_id in yaml"

    return configs, configs['ctc_conf']['ctc_blank_id']

# Copyright (c) 2020 Mobvoi Inc (Binbin Zhang)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified from ESPnet(https://github.com/espnet/espnet)
"""Unility functions for Transformer."""

import math
import time
from typing import List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence

from whisper.tokenizer import LANGUAGES as WhiserLanguages

WHISPER_LANGS = tuple(WhiserLanguages.keys())
IGNORE_ID = -1


def pad_list(xs: List[torch.Tensor], pad_value: int):
    """Perform padding for the list of tensors.

    Args:
        xs (List): List of Tensors [(T_1, `*`), (T_2, `*`), ..., (T_B, `*`)].
        pad_value (float): Value for padding.

    Returns:
        Tensor: Padded tensor (B, Tmax, `*`).

    Examples:
        >>> x = [torch.ones(4), torch.ones(2), torch.ones(1)]
        >>> x
        [tensor([1., 1., 1., 1.]), tensor([1., 1.]), tensor([1.])]
        >>> pad_list(x, 0)
        tensor([[1., 1., 1., 1.],
                [1., 1., 0., 0.],
                [1., 0., 0., 0.]])

    """
    max_len = max([len(item) for item in xs])
    batchs = len(xs)
    ndim = xs[0].ndim
    if ndim == 1:
        pad_res = torch.zeros(batchs,
                              max_len,
                              dtype=xs[0].dtype,
                              device=xs[0].device)
    elif ndim == 2:
        pad_res = torch.zeros(batchs,
                              max_len,
                              xs[0].shape[1],
                              dtype=xs[0].dtype,
                              device=xs[0].device)
    elif ndim == 3:
        pad_res = torch.zeros(batchs,
                              max_len,
                              xs[0].shape[1],
                              xs[0].shape[2],
                              dtype=xs[0].dtype,
                              device=xs[0].device)
    else:
        raise ValueError(f"Unsupported ndim: {ndim}")
    pad_res.fill_(pad_value)
    for i in range(batchs):
        pad_res[i, :len(xs[i])] = xs[i]
    return pad_res


def add_blank(ys_pad: torch.Tensor, blank: int,
              ignore_id: int) -> torch.Tensor:
    """ Prepad blank for transducer predictor

    Args:
        ys_pad (torch.Tensor): batch of padded target sequences (B, Lmax)
        blank (int): index of <blank>

    Returns:
        ys_in (torch.Tensor) : (B, Lmax + 1)

    Examples:
        >>> blank = 0
        >>> ignore_id = -1
        >>> ys_pad
        tensor([[ 1,  2,  3,   4,   5],
                [ 4,  5,  6,  -1,  -1],
                [ 7,  8,  9,  -1,  -1]], dtype=torch.int32)
        >>> ys_in = add_blank(ys_pad, 0, -1)
        >>> ys_in
        tensor([[0,  1,  2,  3,  4,  5],
                [0,  4,  5,  6,  0,  0],
                [0,  7,  8,  9,  0,  0]])
    """
    bs = ys_pad.size(0)
    _blank = torch.tensor([blank],
                          dtype=torch.long,
                          requires_grad=False,
                          device=ys_pad.device)
    _blank = _blank.repeat(bs).unsqueeze(1)  # [bs,1]
    out = torch.cat([_blank, ys_pad], dim=1)  # [bs, Lmax+1]
    return torch.where(out == ignore_id, blank, out)


def add_sos_eos(ys_pad: torch.Tensor, sos: int, eos: int,
                ignore_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Add <sos> and <eos> labels.

    Args:
        ys_pad (torch.Tensor): batch of padded target sequences (B, Lmax)
        sos (int): index of <sos>
        eos (int): index of <eeos>
        ignore_id (int): index of padding

    Returns:
        ys_in (torch.Tensor) : (B, Lmax + 1)
        ys_out (torch.Tensor) : (B, Lmax + 1)

    Examples:
        >>> sos_id = 10
        >>> eos_id = 11
        >>> ignore_id = -1
        >>> ys_pad
        tensor([[ 1,  2,  3,  4,  5],
                [ 4,  5,  6, -1, -1],
                [ 7,  8,  9, -1, -1]], dtype=torch.int32)
        >>> ys_in,ys_out=add_sos_eos(ys_pad, sos_id , eos_id, ignore_id)
        >>> ys_in
        tensor([[10,  1,  2,  3,  4,  5],
                [10,  4,  5,  6, 11, 11],
                [10,  7,  8,  9, 11, 11]])
        >>> ys_out
        tensor([[ 1,  2,  3,  4,  5, 11],
                [ 4,  5,  6, 11, -1, -1],
                [ 7,  8,  9, 11, -1, -1]])
    """
    _sos = torch.tensor([sos],
                        dtype=torch.long,
                        requires_grad=False,
                        device=ys_pad.device)
    _eos = torch.tensor([eos],
                        dtype=torch.long,
                        requires_grad=False,
                        device=ys_pad.device)
    ys = [y[y != ignore_id] for y in ys_pad]  # parse padded ys
    ys_in = [torch.cat([_sos, y], dim=0) for y in ys]
    ys_out = [torch.cat([y, _eos], dim=0) for y in ys]
    return pad_list(ys_in, eos), pad_list(ys_out, ignore_id)


def add_whisper_tokens(special_tokens, ys_pad: torch.Tensor, ignore_id: int,
                       tasks: List[str], no_timestamp: bool, langs: List[str],
                       use_prev: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """Add whisper-style tokens.

    ([PREV] -> [previous text tokens or hotwords]).optional --
      ┌------------------------------------------------------↲
      ↓
    [sot] -> [language id] -> [transcribe] -> [begin time] -> [text tokens] -> [end time] -> ... -> [eot]    # noqa
        |          |                |-------> [no timestamps] -> [text tokens] ----------------------↑       # noqa
        |          |                                                                                 |       # noqa
        |          |--------> [translate]  -> [begin time] -> [text tokens] -> [end time] -> ... --->|       # noqa
        |                           |-------> [no timestamps] -> [text tokens] --------------------->|       # noqa
        |                                                                                            |       # noqa
        |--> [no speech(VAD)] ---------------------------------------------------------------------->|       # noqa

    Args:
        special_tokens: get IDs of special tokens
        ignore_id (int): index of padding
        no_timestamp (bool): whether to add timestamps tokens
        tasks (List[str]): list of task tags
        langs (List[str]): list of language tags

    Returns:
        ys_in (torch.Tensor) : (B, Lmax + ?)
        ys_out (torch.Tensor) : (B, Lmax + ?)

    """
    assert len(langs) == ys_pad.size(0)
    assert len(tasks) == ys_pad.size(0)
    if use_prev:
        # i.e., hotword list
        _prev = [special_tokens["sot_prev"]]
        # append hotword list to _prev
        # ...
        raise NotImplementedError
    else:
        _prev = []

    _sot = []
    for task, lang in zip(tasks, langs):
        if task == "transcribe":
            task_id = special_tokens["transcribe"]
        elif task == "translate":
            task_id = special_tokens["translate"]
        elif task == "vad":
            task_id = special_tokens["no_speech"]
        else:
            raise NotImplementedError("unsupported task {}".format(task))
        language_id = special_tokens["sot"] + 1 + WHISPER_LANGS.index(lang)
        prefix = _prev + [special_tokens["sot"], language_id, task_id]
        if task == "transcribe" or task == "translate":
            if no_timestamp:
                prefix.append(special_tokens["no_timestamps"])
            else:
                prefix.append(special_tokens["timestamp_begin"])
                # add subsequent tokens
                # ...
                raise NotImplementedError
        elif task == "vad":
            prefix.append(special_tokens["no_speech"])
        else:
            raise NotImplementedError
        prefix = torch.tensor(prefix,
                              dtype=torch.long,
                              requires_grad=False,
                              device=ys_pad.device)
        _sot.append(prefix)

    _eot = torch.tensor([special_tokens["eot"]],
                        dtype=torch.long,
                        requires_grad=False,
                        device=ys_pad.device)
    ys = [y[y != ignore_id] for y in ys_pad]  # parse padded ys

    ys_in = [torch.cat([prefix, y], dim=0) for prefix, y in zip(_sot, ys)]
    ys_out = [
        torch.cat([prefix[1:], y, _eot], dim=0) for prefix, y in zip(_sot, ys)
    ]
    return pad_list(ys_in, special_tokens["eot"]), pad_list(ys_out, ignore_id)


def reverse_pad_list(ys_pad: torch.Tensor,
                     ys_lens: torch.Tensor,
                     pad_value: float = -1.0) -> torch.Tensor:
    """Reverse padding for the list of tensors.

    Args:
        ys_pad (tensor): The padded tensor (B, Tokenmax).
        ys_lens (tensor): The lens of token seqs (B)
        pad_value (int): Value for padding.

    Returns:
        Tensor: Padded tensor (B, Tokenmax).

    Examples:
        >>> x
        tensor([[1, 2, 3, 4], [5, 6, 7, 0], [8, 9, 0, 0]])
        >>> pad_list(x, 0)
        tensor([[4, 3, 2, 1],
                [7, 6, 5, 0],
                [9, 8, 0, 0]])

    """
    r_ys_pad = pad_sequence([(torch.flip(y.int()[:i], [0]))
                             for y, i in zip(ys_pad, ys_lens)], True,
                            pad_value)
    return r_ys_pad


def th_accuracy(pad_outputs: torch.Tensor, pad_targets: torch.Tensor,
                ignore_label: int) -> torch.Tensor:
    """Calculate accuracy.

    Args:
        pad_outputs (Tensor): Prediction tensors (B * Lmax, D).
        pad_targets (LongTensor): Target label tensors (B, Lmax).
        ignore_label (int): Ignore label id.

    Returns:
        torch.Tensor: Accuracy value (0.0 - 1.0).

    """
    pad_pred = pad_outputs.view(pad_targets.size(0), pad_targets.size(1),
                                pad_outputs.size(1)).argmax(2)
    mask = pad_targets != ignore_label
    numerator = torch.sum(
        pad_pred.masked_select(mask) == pad_targets.masked_select(mask))
    denominator = torch.sum(mask)
    return (numerator / denominator).detach()


def get_subsample(config):
    input_layer = config["encoder_conf"]["input_layer"]
    assert input_layer in ["conv2d", "conv2d6", "conv2d8"]
    if input_layer == "conv2d":
        return 4
    elif input_layer == "conv2d6":
        return 6
    elif input_layer == "conv2d8":
        return 8


def log_add(*args) -> float:
    """
    Stable log add
    """
    if all(a == -float('inf') for a in args):
        return -float('inf')
    a_max = max(args)
    lsp = math.log(sum(math.exp(a - a_max) for a in args))
    return a_max + lsp


def mask_to_bias(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    assert mask.dtype == torch.bool
    assert dtype in [torch.float32, torch.bfloat16, torch.float16]
    mask = mask.to(dtype)
    # attention mask bias
    # NOTE(Mddct): torch.finfo jit issues
    #     chunk_masks = (1.0 - chunk_masks) * torch.finfo(dtype).min
    mask = (1.0 - mask) * -1.0e+10
    return mask


def get_nested_attribute(obj, attr_path):
    if isinstance(obj, torch.nn.parallel.DistributedDataParallel):
        obj = obj.module
    attributes = attr_path.split('.')
    for attr in attributes:
        obj = getattr(obj, attr)
    return obj


def lrs_to_str(lrs: List):
    return " ".join(["{:.4e}".format(lr) for lr in lrs])


class StepTimer:
    """Utility class for measuring steps/second."""

    def __init__(self, step=0.0):
        self.last_iteration = step
        self.start()

    def start(self):
        self.last_time = time.time()

    def steps_per_second(self, cur_step, restart=True):
        value = ((float(cur_step) - self.last_iteration) /
                 (time.time() - self.last_time))
        if restart:
            self.start()
            self.last_iteration = float(cur_step)
        return value


def tensor_to_scalar(x):
    if torch.is_tensor(x):
        return x.item()
    return x


def is_torch_npu_available() -> bool:
    '''
        check if torch_npu is available.
        torch_npu is a npu adapter of PyTorch
    '''
    try:
        import torch_npu  # noqa
        return True
    except ImportError:
        if not torch.cuda.is_available():
            print("Module \"torch_npu\" not found. \"pip install torch_npu\" \
                if you are using Ascend NPU, otherwise, ignore it")
    return False


TORCH_NPU_AVAILABLE = is_torch_npu_available()

# Copyright (c) 2019 Shigeki Karita
#               2020 Mobvoi Inc (Binbin Zhang)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
'''
def subsequent_mask(
        size: int,
        device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Create mask for subsequent steps (size, size).

    This mask is used only in decoder which works in an auto-regressive mode.
    This means the current step could only do attention with its left steps.

    In encoder, fully attention is used when streaming is not necessary and
    the sequence is not long. In this  case, no attention mask is needed.

    When streaming is need, chunk-based attention is used in encoder. See
    subsequent_chunk_mask for the chunk-based attention mask.

    Args:
        size (int): size of mask
        str device (str): "cpu" or "cuda" or torch.Tensor.device
        dtype (torch.device): result dtype

    Returns:
        torch.Tensor: mask

    Examples:
        >>> subsequent_mask(3)
        [[1, 0, 0],
         [1, 1, 0],
         [1, 1, 1]]
    """
    ret = torch.ones(size, size, device=device, dtype=torch.bool)
    return torch.tril(ret)
'''


def subsequent_mask(
        size: int,
        device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Create mask for subsequent steps (size, size).

    This mask is used only in decoder which works in an auto-regressive mode.
    This means the current step could only do attention with its left steps.

    In encoder, fully attention is used when streaming is not necessary and
    the sequence is not long. In this  case, no attention mask is needed.

    When streaming is need, chunk-based attention is used in encoder. See
    subsequent_chunk_mask for the chunk-based attention mask.

    Args:
        size (int): size of mask
        str device (str): "cpu" or "cuda" or torch.Tensor.device
        dtype (torch.device): result dtype

    Returns:
        torch.Tensor: mask

    Examples:
        >>> subsequent_mask(3)
        [[1, 0, 0],
         [1, 1, 0],
         [1, 1, 1]]
    """
    arange = torch.arange(size, device=device)
    mask = arange.expand(size, size)
    arange = arange.unsqueeze(-1)
    mask = mask <= arange
    return mask


def subsequent_chunk_mask(
        size: int,
        chunk_size: int,
        num_left_chunks: int = -1,
        device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Create mask for subsequent steps (size, size) with chunk size,
       this is for streaming encoder

    Args:
        size (int): size of mask
        chunk_size (int): size of chunk
        num_left_chunks (int): number of left chunks
            <0: use full chunk
            >=0: use num_left_chunks
        device (torch.device): "cpu" or "cuda" or torch.Tensor.device

    Returns:
        torch.Tensor: mask

    Examples:
        >>> subsequent_chunk_mask(4, 2)
        [[1, 1, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 1],
         [1, 1, 1, 1]]
    """
    ret = torch.zeros(size, size, device=device, dtype=torch.bool)
    for i in range(size):
        if num_left_chunks < 0:
            start = 0
        else:
            start = max((i // chunk_size - num_left_chunks) * chunk_size, 0)
        ending = min((i // chunk_size + 1) * chunk_size, size)
        ret[i, start:ending] = True
    return ret


def add_optional_chunk_mask(xs: torch.Tensor,
                            masks: torch.Tensor,
                            use_dynamic_chunk: bool,
                            use_dynamic_left_chunk: bool,
                            decoding_chunk_size: int,
                            static_chunk_size: int,
                            num_decoding_left_chunks: int,
                            enable_full_context: bool = True,
                            max_chunk_size: int = 25):
    """ Apply optional mask for encoder.

    Args:
        xs (torch.Tensor): padded input, (B, L, D), L for max length
        mask (torch.Tensor): mask for xs, (B, 1, L)
        use_dynamic_chunk (bool): whether to use dynamic chunk or not
        use_dynamic_left_chunk (bool): whether to use dynamic left chunk for
            training.
        decoding_chunk_size (int): decoding chunk size for dynamic chunk, it's
            0: default for training, use random dynamic chunk.
            <0: for decoding, use full chunk.
            >0: for decoding, use fixed chunk size as set.
        static_chunk_size (int): chunk size for static chunk training/decoding
            if it's greater than 0, if use_dynamic_chunk is true,
            this parameter will be ignored
        num_decoding_left_chunks: number of left chunks, this is for decoding,
            the chunk size is decoding_chunk_size.
            >=0: use num_decoding_left_chunks
            <0: use all left chunks
        enable_full_context (bool):
            True: chunk size is either [1, max_chunk_size] or full context(max_len)
            False: chunk size ~ U[1, max_chunk_size]

    Returns:
        torch.Tensor: chunk mask of the input xs.
    """
    # Whether to use chunk mask or not
    if use_dynamic_chunk:
        max_len = xs.size(1)
        if decoding_chunk_size < 0:
            chunk_size = max_len
            num_left_chunks = -1
        elif decoding_chunk_size > 0:
            chunk_size = decoding_chunk_size
            num_left_chunks = num_decoding_left_chunks
        else:
            # chunk size is either [1, max_chunk_size] or full context(max_len).
            # Since we use 4 times subsampling and allow up to 1s(100 frames)
            # delay, the maximum frame is 100 / 4 = 25.
            chunk_size = torch.randint(1, max_len, (1, )).item()
            num_left_chunks = -1
            if chunk_size > max_len // 2 and enable_full_context:
                chunk_size = max_len
            else:
                chunk_size = chunk_size % max_chunk_size + 1
                if use_dynamic_left_chunk:
                    max_left_chunks = (max_len - 1) // chunk_size
                    num_left_chunks = torch.randint(0, max_left_chunks,
                                                    (1, )).item()
        chunk_masks = subsequent_chunk_mask(xs.size(1), chunk_size,
                                            num_left_chunks,
                                            xs.device)  # (L, L)
        chunk_masks = chunk_masks.unsqueeze(0)  # (1, L, L)
        chunk_masks = masks & chunk_masks  # (B, L, L)
    elif static_chunk_size > 0:
        num_left_chunks = num_decoding_left_chunks
        chunk_masks = subsequent_chunk_mask(xs.size(1), static_chunk_size,
                                            num_left_chunks,
                                            xs.device)  # (L, L)
        chunk_masks = chunk_masks.unsqueeze(0)  # (1, L, L)
        chunk_masks = masks & chunk_masks  # (B, L, L)
    else:
        chunk_masks = masks
    return chunk_masks


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """Make mask tensor containing indices of padded part.

    See description of make_non_pad_mask.

    Args:
        lengths (torch.Tensor): Batch of lengths (B,).
    Returns:
        torch.Tensor: Mask tensor containing indices of padded part.

    Examples:
        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
    """
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0,
                             max_len,
                             dtype=torch.int64,
                             device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask


def make_non_pad_mask(lengths: torch.Tensor) -> torch.Tensor:
    """Make mask tensor containing indices of non-padded part.

    The sequences in a batch may have different lengths. To enable
    batch computing, padding is need to make all sequence in same
    size. To avoid the padding part pass value to context dependent
    block such as attention or convolution , this padding part is
    masked.

    This pad_mask is used in both encoder and decoder.

    1 for non-padded part and 0 for padded part.

    Args:
        lengths (torch.Tensor): Batch of lengths (B,).
    Returns:
        torch.Tensor: mask tensor containing indices of padded part.

    Examples:
        >>> lengths = [5, 3, 2]
        >>> make_non_pad_mask(lengths)
        masks = [[1, 1, 1, 1 ,1],
                 [1, 1, 1, 0, 0],
                 [1, 1, 0, 0, 0]]
    """
    return ~make_pad_mask(lengths)


def mask_finished_scores(score: torch.Tensor,
                         flag: torch.Tensor) -> torch.Tensor:
    """
    If a sequence is finished, we only allow one alive branch. This function
    aims to give one branch a zero score and the rest -inf score.

    Args:
        score (torch.Tensor): A real value array with shape
            (batch_size * beam_size, beam_size).
        flag (torch.Tensor): A bool array with shape
            (batch_size * beam_size, 1).

    Returns:
        torch.Tensor: (batch_size * beam_size, beam_size).
    """
    beam_size = score.size(-1)
    zero_mask = torch.zeros_like(flag, dtype=torch.bool)
    if beam_size > 1:
        unfinished = torch.cat((zero_mask, flag.repeat([1, beam_size - 1])),
                               dim=1)
        finished = torch.cat((flag, zero_mask.repeat([1, beam_size - 1])),
                             dim=1)
    else:
        unfinished = zero_mask
        finished = flag
    score.masked_fill_(unfinished, -float('inf'))
    score.masked_fill_(finished, 0)
    return score


def mask_finished_preds(pred: torch.Tensor, flag: torch.Tensor,
                        eos: int) -> torch.Tensor:
    """
    If a sequence is finished, all of its branch should be <eos>

    Args:
        pred (torch.Tensor): A int array with shape
            (batch_size * beam_size, beam_size).
        flag (torch.Tensor): A bool array with shape
            (batch_size * beam_size, 1).

    Returns:
        torch.Tensor: (batch_size * beam_size).
    """
    beam_size = pred.size(-1)
    finished = flag.repeat([1, beam_size])
    return pred.masked_fill_(finished, eos)


def causal_or_lookahead_mask(
    mask: torch.Tensor,
    right_context: int,
    left_context: int,
    left_t_valid: int = 0,
) -> torch.Tensor:
    """Create mask (B, T, T) with history or future or both,
       this is for causal or noncausal streaming encoder

    Args:
        mask (torch.Tensor): size of mask shape (B, 1, T)
        right_context (int): future context size
        left_context (int): history context size
        left_t_valid (int): valid start offset

    Returns:
        torch.Tensor: mask shape (B, T, T)

    Examples:
        >>> seq_len  = torch.tensor([2,3,4])
        >>> seq_mask = make_non_pad_mask(seq_len)
        [[1, 1, 0, 0],
        [1, 1, 1, 0],
        [1, 1, 1, 1]]
        >>> causal_or_lookahead_mask(seq_mask.unsqueeze(1), 0, 2)
        [[[1, 0, 0, 0],
         [1, 1, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0]],

        [[1, 0, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 0],
         [0, 0, 0, 0]],

        [[1, 0, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 0],
         [0, 1, 1, 1]]]
        >>> causal_or_lookahead_mask(seq_mask.unsqueeze(1), 1, 2)
        [[[1, 1, 0, 0],
         [1, 1, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0]],

        [[1, 1, 0, 0],
         [1, 1, 1, 0],
         [1, 1, 1, 0],
         [0, 0, 0, 0]],

        [[1, 1, 0, 0],
         [1, 1, 1, 0],
         [1, 1, 1, 1],
         [0, 1, 1, 1]]]
    """
    _, _, T = mask.size()
    indices = torch.arange(T, device=mask.device)
    start = torch.where(indices > left_context, indices - left_context, 0)
    start = torch.where(indices < left_t_valid, indices, start).unsqueeze(1)

    end = indices + right_context + 1
    end = end.unsqueeze(1)
    indices_expand = indices.unsqueeze(0)
    gt = (indices_expand >= start)
    lt = (indices_expand < end)

    return (gt & lt) * mask.transpose(1, 2) * mask