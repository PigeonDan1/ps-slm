import random

from loguru import logger
from .transforms_interface import MutuallyExclusiveGroup
from ..augmentations.pre_process.alignment import Aligner
from ..augmentations.post_process.vad_align import VadAligner
from ..augmentations.pre_process.concat import Concat

class BaseCompose:
    def __init__(self, transforms, p=1.0, shuffle=False):
        self.transforms = transforms
        self.p = p
        self.shuffle = shuffle

        name_list = []
        for transform in self.transforms:
            name_list.append(type(transform).__name__)
        self.__name__ = "_".join(name_list)

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def randomize_parameters(self, *args, **kwargs):
        """
        Randomize and define parameters of every transform in composition.
        """
        raise NotImplementedError

    def freeze_parameters(self):
        """
        Mark all parameters as frozen, i.e. do not randomize them for each call. This can be
        useful if you want to apply an effect chain with the exact same parameters to multiple
        sounds.
        """
        for transform in self.transforms:
            transform.freeze_parameters()

    def unfreeze_parameters(self):
        """
        Unmark all parameters as frozen, i.e. let them be randomized for each call.
        """
        for transform in self.transforms:
            transform.unfreeze_parameters()


class Compose(BaseCompose):
    # TODO: Name can change to WaveformCompose

    def __call__(self, samples_list, sample_rates_list, other_input_output_list=None, meta=None, reproduce=False):
        transforms = self.transforms.copy()
        if reproduce:
            assert self.p >= 1, 'Re-produce do not support probility Compose'
            assert not self.shuffle, 'Re-produce do not support shuffled Compose'
            assert isinstance(meta, list), f'Re-produce expect list metadata, get {meta}'
            #assert len(meta) == len(transforms), \
            #    f'Re-produce expect same length for mate and transforms, ' \
            #    f'get ({len(meta)}, {len(transforms)})'

        accumulate_meta = [] if meta is None else meta
        if random.random() < self.p:
            if self.shuffle:
                random.shuffle(transforms)
            for i, transform in enumerate(transforms):
                if reproduce:
                    accumulate_meta = [meta[i]]  # for interface consistance
                    if isinstance(transform, Concat):
                        samples, res_accumulate_meta = transform(
                            samples_list, sample_rates_list[0], accumulate_meta=accumulate_meta,
                            reproduce=True
                        )
                        res_samples_list = [samples]
                    else:
                        res_samples_list = []
                        for j, samples in enumerate(samples_list):
                            if isinstance(transform, Aligner) or isinstance(transform, VadAligner):
                                samples, _ = transform(
                                    samples, sample_rates_list[j], other_input_output_list[j], accumulate_meta=accumulate_meta)
                            else:
                                samples, _ = transform(
                                    samples, sample_rates_list[j], accumulate_meta=accumulate_meta,
                                    reproduce=True)
                            if type(samples) == list:
                                res_samples_list.extend(samples)
                            else:
                                res_samples_list.append(samples)
                    
                    samples_list = res_samples_list

                else:
                    assert accumulate_meta is not None

                    if isinstance(transform, Concat):
                        samples, res_accumulate_meta = transform(
                            samples_list, sample_rates_list[0], accumulate_meta=accumulate_meta
                        )
                        res_samples_list = [samples]
                    else:
                        if isinstance(transform, Aligner) or isinstance(transform, VadAligner):
                            samples, res_accumulate_meta = transform(
                                samples_list[0], sample_rates_list[0], other_input_output_list[0], accumulate_meta=accumulate_meta
                            )
                        else:
                            samples, res_accumulate_meta = transform(
                                samples_list[0], sample_rates_list[0], accumulate_meta=accumulate_meta
                            )
                        
                        res_samples_list = samples if isinstance(samples, list) else [samples]

                        # Only append only one meta to accumlate_meta
                        # So break for into two part
                        # To keep the same parameters as idx 0, freeze parameters
                        # when complete unfreeze
                        transform.freeze_parameters()
                        for j, samples in enumerate(samples_list[1:], 1):
                            if isinstance(transform, Aligner) or isinstance(transform, VadAligner):
                                samples, _ = transform(
                                    samples, sample_rates_list[j], other_input_output_list[j], accumulate_meta=accumulate_meta.copy())
                            else:
                                samples, _ = transform(
                                    samples, sample_rates_list[j], accumulate_meta=accumulate_meta.copy())
                            if type(samples) == list:
                                res_samples_list.extend(samples)
                            else:
                                res_samples_list.append(samples)
                        transform.unfreeze_parameters()

                    # summery result
                    samples_list = res_samples_list
                    accumulate_meta = res_accumulate_meta
                # NOTE(menglong.xu): 对于cut_wkp这个transform，输入samples_list长度是1，输出samples_list长度可能是0~n(对应没有唤醒片段和多个唤醒片段)
                # 如果没有唤醒片段，即输出samples_list长度为0，直接返回
                if len(samples_list) == 0:
                    break
                # 如果有多个唤醒片段，即输出samples_list长度大于1，需要扩充sample_rates_list和other_input_output_list
                if other_input_output_list and len(samples_list) > len(other_input_output_list):
                    assert len(other_input_output_list) == 1, 'only support broadcast when the length of other_input_output_list is 1'
                    for _ in range(len(samples_list) - 1):
                        other_input_output_list.append(other_input_output_list[0].copy())
                    sample_rates_list = sample_rates_list * len(samples_list)
            # 因concat导致other_input_output_list长度大于samples_list时，对other_input_output_list归并
            if samples_list and other_input_output_list:
                if len(other_input_output_list) > len(samples_list):
                    assert len(samples_list) == 1
                    merged = {}
                    for input_output in other_input_output_list:
                        for key, value in input_output.items():
                            merged.setdefault(key, []).append(value)
                    for key in merged:
                        merged[key] = ' '.join(merged[key])
                    other_input_output_list.clear()
                    other_input_output_list.append(merged)

        # compatible with different version atec
        res_samples = samples_list[0] if len(samples_list) == 1 else samples_list

        if meta is not None:
            return res_samples, accumulate_meta
        return res_samples

    def randomize_parameters(self, samples, sample_rate):
        for transform in self.transforms:
            transform.randomize_parameters(samples, sample_rate)

    def __repr__(self):
        repr_list = []
        for tmf in self.transforms:
            if tmf.__class__ == MutuallyExclusiveGroup:
                repr_list.append(f'MutuallyExclusiveGroup: {[(t.__class__.__name__, f"{tmf.transforms_p[i]:.2f}") for i, t in enumerate(tmf.transforms)]}')
            else:
                repr_list.append(f'{tmf.__class__.__name__}({tmf.p})')
        return f'{repr_list}'

class SpecCompose(BaseCompose):

    def __call__(self, magnitude_spectrogram):
        transforms = self.transforms.copy()
        if random.random() < self.p:
            if self.shuffle:
                random.shuffle(transforms)
            for transform in transforms:
                magnitude_spectrogram = transform(magnitude_spectrogram)
        return magnitude_spectrogram

    def randomize_parameters(self, magnitude_spectrogram):
        for transform in self.transforms:
            transform.randomize_parameters(magnitude_spectrogram)
