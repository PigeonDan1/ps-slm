import numpy as np

from ...core.transforms_interface import add_transform

from .base import AddNoise

@add_transform('pad_silence')
class PadSilence(AddNoise):
    """Padding silence before/after audio
    """
    abbr='pad_silence'
    supports_multichannel = True

    def __init__(
        self,
        time_padding=(0, 0),
        channel_padding=(0, 0),
        p=1.0,
    ):
        """
        :param p: The probability of applying this transform
        """
        super().__init__(p)
        self.parameters['time_padding'] = time_padding
        self.parameters['channel_padding'] = channel_padding

    def apply(self, samples, sample_rate):
        time_start = int(sample_rate * self.parameters['time_padding'][0])
        time_end = int(sample_rate * self.parameters['time_padding'][1])
        chn_start, chn_end = self.parameters['channel_padding']
        if len(samples.shape) == 1:
            samples = samples[np.newaxis, :]

        samples = np.pad(
            samples, ((chn_start, chn_end), (time_start, time_end)), mode='constant')
        samples = samples.squeeze()
        return samples
