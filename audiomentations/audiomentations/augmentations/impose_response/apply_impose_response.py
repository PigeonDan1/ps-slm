import random
import warnings
import functools
import math
from loguru import logger

import numpy as np
from scipy.signal import convolve
from asr.utils import slurm

from ...core.audio_loading_utils import load_sound_file
from ...core.transforms_interface import add_transform
from ...core.utils import calculate_rms, get_file_paths

from .base import ImposeResponse


@add_transform('apply_impluse_response')
class ApplyImpulseResponse(ImposeResponse):
    """Convolve the audio with a random impulse response.
    Impulse responses can be created using e.g. http://tulrich.com/recording/ir_capture/
    Impulse responses are represented as wav files in the given ir_path.
    """
    abbr='apply_impluse_response'

    def __init__(
        self,
        ir_path="/tmp/ir",
        ir_startidx=0,
        ir_num_channel=1,
        p=1.0,
        lru_cache_size=None,
        leave_length_unchanged: bool = True,
        normalize: bool = True,
        load_once: bool = False
    ):
        """
        :param ir_path: Path to a folder that contains one or more wav files of impulse
        responses. Must be str or a Path instance.
        :param p: The probability of applying this transform
        :param lru_cache_size: Maximum size of the LRU cache for storing impulse response files
        in memory.
        :param leave_length_unchanged: When set to True, the tail of the sound (e.g. reverb at
            the end) will be chopped off so that the length of the output is equal to the
            length of the input.
        :param normalize: maintian ennergy the same after convolve rir
        """
        super().__init__(p)
        self.ir_files = get_file_paths(ir_path)
        self.ir_files = [str(p) for p in self.ir_files]
        assert len(self.ir_files) > 0
        self.ir_startidx = ir_startidx
        self.ir_num_channel = ir_num_channel
        self.leave_length_unchanged = leave_length_unchanged
        self.normalize = normalize
        self.lru_cache_size = min(lru_cache_size, math.ceil(len(self.ir_files) / slurm.world_size)) \
                if lru_cache_size else math.ceil(len(self.ir_files) / slurm.world_size)
        self._load_ir = functools.lru_cache(maxsize=self.lru_cache_size)(
            ApplyImpulseResponse.__load_stereo_sound_file
        )
        self.load_once = load_once

        if self.load_once:
            if slurm.world_size * self.lru_cache_size < len(self.ir_files):
                logger.warning(f'Warning: world_size * lru_cache_size < len(self.ir_files): ' +
                f'{slurm.world_size} * {self.lru_cache_size} < {len(self.ir_files)}.' + 'Using partial rir list')


    @staticmethod
    def __load_stereo_sound_file(file_path, sample_rate):
        return load_sound_file(file_path, sample_rate, mono=False)

    def randomize_parameters(self, samples, sample_rate, accumulate_meta=None):
        super().randomize_parameters(samples, sample_rate, accumulate_meta)
        if self.parameters["should_apply"]:

            if self.load_once:
                choice = slurm.rank * self.lru_cache_size + random.choice(range(self.lru_cache_size))
                self.parameters["ir_file_path"] = self.ir_files[choice% len(self.ir_files)]
            else:
                self.parameters["ir_file_path"] = random.choice(self.ir_files)
            self.parameters["ir_startidx"] = self.ir_startidx
            self.parameters["ir_num_channel"] = self.ir_num_channel

    def apply(self, samples, sample_rate):
        # ir of shape [samples, channel]
        ir, sample_rate2 = self._load_ir(self.parameters["ir_file_path"], sample_rate)
        if sample_rate != sample_rate2:
            # This will typically not happen, as librosa should automatically resample the
            # impulse response sound to the desired sample rate
            raise Exception(
                "Recording sample rate {} did not match Impulse Response signal"
                " sample rate {}!".format(sample_rate, sample_rate2)
            )
        if len(ir.shape) == 2:
            ir_sidx = self.parameters["ir_startidx"]
            ir_eidx = self.parameters["ir_startidx"] + self.parameters["ir_num_channel"]
            if ir.shape[1] < ir_eidx:
                # multi-channel rir
                raise Exception(
                    "RIR {} channel-num: {} did not match required channel num {}!".format(
                        self.parameters["ir_file_path"], ir.shape, ir_eidx)
                )
            if self.parameters["ir_num_channel"] > 1:
                ir = ir[:, ir_sidx:ir_eidx].T
                samples = samples[np.newaxis, :]
            else:
                ir = ir[:, 0]

        signal_ir = convolve(samples, ir)
        if self.normalize:
            dt = np.argmax(ir, axis=-1).min()
            st = max(0, int(dt - 0.001*sample_rate))  # ahead 10ms
            et = dt + (50 * sample_rate) // 1000  # delay 50ms
            et_rir = np.zeros(ir.shape[-1])
            et_rir[st:et] = ir[st:et] if len(ir.shape) == 1 else ir[0, st:et]
            wav_early_tgt = convolve(samples.squeeze(), et_rir)
            if self.leave_length_unchanged:
                wav_early_tgt = wav_early_tgt[:samples.shape[-1]]
            scale = calculate_rms(samples) / calculate_rms(wav_early_tgt)
            signal_ir *= scale
        # max_value = max(np.amax(signal_ir), -np.amin(signal_ir))
        # if max_value > 0.0:
        #     scale = 0.5 / max_value
        #     signal_ir *= scale
        if self.leave_length_unchanged:
            signal_ir = signal_ir[..., : samples.shape[-1]]

        return signal_ir

    def __getstate__(self):
        state = self.__dict__.copy()
        warnings.warn(
            "Warning: the LRU cache of ApplyImpulseResponse gets discarded when pickling it."
            " E.g. this means the cache will be not be used when using ApplyImpulseResponse"
            " together with multiprocessing on Windows"
        )
        del state["_load_ir"]
        return state
