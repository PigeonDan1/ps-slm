import random
from typing import List, Optional
import warnings
from loguru import logger
import numpy as np
from numpy.lib.function_base import select

from asr.utils.dynload_factory import _factory_add, _factory_build


from ..core.utils import (
    is_waveform_multichannel,
    is_spectrogram_multichannel,
)

_transform_mapping = {}

def fix_np64(samples):
    if samples.dtype == np.float64:
        warnings.warn(
            "Warning: input samples have np.float64 dtype. Converting to np.float32..."
        )
        samples = np.float32(samples)
    return samples

class MultichannelAudioNotSupportedException(Exception):
    pass


class MonoAudioNotSupportedException(Exception):
    pass


class BaseTransform:
    supports_mono = True
    supports_multichannel = False

    def __init__(self, p=1.0):
        assert 0 <= p <= 1
        self.p = p
        self.parameters = {"should_apply": False}
        self.are_parameters_frozen = False

    def serialize_parameters(self):
        """Return the parameters as a JSON-serializable dict."""
        self.parameters['name'] = self.__class__.__name__
        return self.parameters

    def freeze_parameters(self):
        """
        Mark all parameters as frozen, i.e. do not randomize them for each call. This can be
        useful if you want to apply an effect with the exact same parameters to multiple sounds.
        """
        self.are_parameters_frozen = True

    def unfreeze_parameters(self):
        """
        Unmark all parameters as frozen, i.e. let them be randomized for each call.
        """
        self.are_parameters_frozen = False


class BaseWaveformTransform(BaseTransform):

    def apply(self, samples, sample_rate):
        raise NotImplementedError

    def is_multichannel(self, samples):
        return is_waveform_multichannel(samples)

    def __call__(self,
                 samples: np.ndarray,
                 sample_rate: int,
                 other_input: Optional[str] = None,
                 accumulate_meta: Optional[List] = None,
                 reproduce: bool = False):

        if isinstance(samples,list):
            samples = [fix_np64(spl) for spl in  samples]
        else:
            samples = fix_np64(samples)


        if reproduce:
            meta = accumulate_meta[0]  # only one element
            assert meta["name"] == self.__class__.__name__, \
                f'Re-produce name mismatch ({meta["name"]}, {self.__class__.__name__})'
            self.parameters = meta
        elif not self.are_parameters_frozen:
            self.randomize_parameters(samples, sample_rate, accumulate_meta)

        if self.parameters["should_apply"] and len(samples) > 0:
            if other_input:
                samples = self.apply(samples, sample_rate, other_input)
            else:
                samples = self.apply(samples, sample_rate)

        if accumulate_meta is not None:
            if not reproduce:
                accumulate_meta.append(self.serialize_parameters())
            return samples, accumulate_meta
        return samples

    def randomize_parameters(self, samples, sample_rate, accumulate_meta=None):
        self.parameters["should_apply"] = random.random() < self.p


class BaseSpectrogramTransform(BaseTransform):
    def apply(self, magnitude_spectrogram):
        raise NotImplementedError

    def is_multichannel(self, samples):
        return is_spectrogram_multichannel(samples)

    def __call__(self,
                 magnitude_spectrogram: np.ndarray,
                 accumulate_meta: Optional[List] = None,
                 reproduce: bool = False):
        if not self.are_parameters_frozen:
            self.randomize_parameters(magnitude_spectrogram, accumulate_meta)
        if reproduce:
            meta = accumulate_meta[0]  # only one element
            assert meta["name"] == self.__class__.__name__, \
                f'Re-produce name mismatch ({meta["name"]}, {self.__class__.__name__})'
            self.parameters = meta

        if (
            self.parameters["should_apply"]
            and magnitude_spectrogram.shape[0] > 0
            and magnitude_spectrogram.shape[1] > 0
        ):
            if self.is_multichannel(magnitude_spectrogram):
                # if magnitude_spectrogram.shape[0] > magnitude_spectrogram.shape[1]:
                #     warnings.warn(
                #         "Multichannel audio must have channels first, not channels last"
                #     )
                if not self.supports_multichannel:
                    raise MultichannelAudioNotSupportedException(
                        "{} only supports mono audio, not multichannel audio".format(
                            self.__class__.__name__
                        )
                    )
            elif not self.supports_mono:
                raise MonoAudioNotSupportedException(
                    "{} only supports multichannel audio, not mono audio".format(
                        self.__class__.__name__
                    )
                )
            magnitude_spectrogram = self.apply(magnitude_spectrogram)

        if not reproduce and accumulate_meta is not None:
            accumulate_meta.append(self.serialize_parameters())
            return magnitude_spectrogram, accumulate_meta
        return magnitude_spectrogram

    def randomize_parameters(self, magnitude_spectrogram, accumulate_meta=None):
        self.parameters["should_apply"] = random.random() < self.p


add_transform = _factory_add(_transform_mapping, force_base_class=BaseTransform)
build_transform = _factory_build(_transform_mapping)

class EmptyTransform(BaseWaveformTransform):
    def __init__(self, p=1.0):
        super().__init__(p=p)
    def apply(self, samples, sample_rate):
        return samples

@add_transform('mutually_exclusive_group')
class MutuallyExclusiveGroup(BaseWaveformTransform):

    def __init__(self, transforms: List, p: List[int]):
        super().__init__(p=1.0)
        self.parameters["should_apply"] = True
        self.transforms = [build_transform(transform) for transform in transforms]
        self.transforms_p = p
        assert len(self.transforms) == len(self.transforms_p), \
            f'len of transform not equal with p {len(self.transforms)} {len(self.transforms_p)}'

        if sum(self.transforms_p) > 1:
            raise RuntimeError(f'sum of p great than 1 {self.transforms_p}')
        elif sum(self.transforms_p) < 1:
            remainder = 1 - sum(self.transforms_p)
            self.transforms_p.append(remainder)
            self.transforms.append(EmptyTransform())

    def __call__(self,
                 samples: np.ndarray,
                 sample_rate: int,
                 other_input: Optional[str] = None,
                 accumulate_meta: Optional[List] = None,
                 reproduce: bool = False):

        if isinstance(samples,list):
            samples = [fix_np64(spl) for spl in  samples]
        else:
            samples = fix_np64(samples)

        if reproduce:

            meta = accumulate_meta[0]  # only one element

            tfm_names = [tfm.__class__.__name__ for tfm in  self.transforms]
            assert meta["name"] in tfm_names, \
                f'Re-produce name mismatch {meta["name"]} not in {tfm_names}'
            selected_transform = self.transforms[tfm_names.index(meta["name"])]

        else:
            if not self.are_parameters_frozen:
                self.randomize_parameters(samples, sample_rate, accumulate_meta)
            selected_transform = self.transforms[self.parameters["choice"]]

        if len(samples) > 0:
            samples, accumulate_meta = selected_transform(samples, sample_rate, other_input, accumulate_meta, reproduce)

        if accumulate_meta is not None:
            return samples, accumulate_meta
        return samples
    
    def randomize_parameters(self, samples, sample_rate, accumulate_meta=None):
        super().randomize_parameters(samples, sample_rate, accumulate_meta)
        self.parameters["choice"] = int(np.random.choice(len(self.transforms), 1, p=self.transforms_p)[0])

    def freeze_parameters(self):
        super().freeze_parameters()
        for trf in self.transforms:
            trf.freeze_parameters()

    def unfreeze_parameters(self):
        super().unfreeze_parameters()
        for trf in self.transforms:
            trf.unfreeze_parameters()
#    def serialize_parameters(self):
#        """Return the parameters as a JSON-serializable dict."""
#        pass


@add_transform('sequence_group')
class SequenceGroup(BaseWaveformTransform):

    def __init__(self, transforms: List):
        super().__init__(p=1.0)
        self.parameters["should_apply"] = True
        self.transforms = [build_transform(transform) for transform in transforms]

    def __call__(self,
                 samples: np.ndarray,
                 sample_rate: int,
                 other_input: Optional[str] = None,
                 accumulate_meta: Optional[List] = None,
                 reproduce: bool = False):

        if isinstance(samples,list):
            samples = [fix_np64(spl) for spl in  samples]
        else:
            samples = fix_np64(samples)

        if reproduce:
            raise NotImplementedError
        else:
            if not self.are_parameters_frozen:
                self.randomize_parameters(samples, sample_rate, accumulate_meta)

        if len(samples) > 0:
            for trf in self.transforms:
                samples, accumulate_meta = trf(samples, sample_rate, other_input, accumulate_meta, reproduce)

        if accumulate_meta is not None:
            return samples, accumulate_meta
        return samples

    def freeze_parameters(self):
        super().freeze_parameters()
        for trf in self.transforms:
            trf.freeze_parameters()

    def unfreeze_parameters(self):
        super().unfreeze_parameters()
        for trf in self.transforms:
            trf.unfreeze_parameters()
