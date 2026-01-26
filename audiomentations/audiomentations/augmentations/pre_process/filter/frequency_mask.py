from scipy.signal import butter, sosfilt
import numpy as np
import random

from ....core.transforms_interface import BaseWaveformTransform, add_transform

@add_transform('frequency_mask')
class FrequencyMask(BaseWaveformTransform):
    """
    Mask some frequency band on the spectrogram.
    Inspired by https://arxiv.org/pdf/1904.08779.pdf
    """
    abbr="freqmask"
    supports_multichannel = True

    def __init__(self, min_bandwidth=0.0, max_bandwidth=0.5, p=1.0):
        """
        :param min_bandwidth: Minimum bandwidth, float
        :param max_bandwidth: Maximum bandwidth, float
        :param p: The probability of applying this transform
        """
        super().__init__(p)
        self.min_bandwidth = min_bandwidth
        self.max_bandwidth = max_bandwidth
        assert self.min_bandwidth >= 0 and self.min_bandwidth <=1 and self.max_bandwidth >= 0 and self.max_bandwidth <= 1, "max/min frequency band should between 0 and 1"

    def __butter_bandstop(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], btype="bandstop", output="sos")
        return sos

    def __butter_bandstop_filter(self, data, lowcut, highcut, fs, order=5):
        sos = self.__butter_bandstop(lowcut, highcut, fs, order=order)
        y = sosfilt(sos, data).astype(np.float32)
        return y

    def randomize_parameters(self, samples, sample_rate, accumulate_meta=None):
        super().randomize_parameters(samples, sample_rate, accumulate_meta)
        if self.parameters["should_apply"]:
            self.parameters["bandwidth"] = random.randint(
                self.min_bandwidth * sample_rate // 2,
                self.max_bandwidth * sample_rate // 2,
            )
            self.parameters["freq_start"] = random.randint(
                16, sample_rate // 2 - self.parameters["bandwidth"] - 1
            )

    def apply(self, samples, sample_rate):
        bandwidth = self.parameters["bandwidth"]
        freq_start = self.parameters["freq_start"]
        samples = self.__butter_bandstop_filter(
            samples, freq_start, freq_start + bandwidth, sample_rate, order=6
        )
        return samples