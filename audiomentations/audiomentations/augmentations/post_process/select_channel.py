import warnings

import numpy as np

from ...core.transforms_interface import add_transform
from .base import PostProcess


@add_transform('select_channel')
class SelectChannel(PostProcess):
    """Select 1/multi channel wav from multi-channel wav.
       according to the give `wav channel` table

       >>> wav channel:
       >>> lee 1_2_3
       >>> hao 1_2_3
    """
    supports_mono = False
    supports_multichannel = True

    def __init__(self, wav_channel_path=None, fix_channel=None, p=1.0):
        super().__init__(p=p)
        self.fix_channel = fix_channel
        self.wav_channel = None
        if wav_channel_path is not None:
            self.wav_channel = {}
            select_num = set()
            with open(wav_channel_path, 'r') as fin:
                for line in fin:
                    wav, channels = line.strip().split()
                    self.wav_channel[wav] = [int(i) - 1 for i in channels.split('_')]
                    select_num.add(len(self.wav_channel[wav]))
            assert len(select_num) == 1, f'{wav_channel_path} contains different channel number'

    def randomize_parameters(self, samples, sample_rate, accumulate_meta=None):
        super().randomize_parameters(samples, sample_rate, accumulate_meta=accumulate_meta)
        assert len(samples.shape) > 1, f'select_channle expect multi-channel wav, but get mono wav'
        channel, _ = samples.shape
        if accumulate_meta is not None and self.wav_channel is not None:
            # only use first rir
            for meta in accumulate_meta:
                if meta['name'] == 'key':
                    wav = meta['key'].split('/')[-1]  # basename
                    for i in self.wav_channel[wav]:
                        assert i < channel, f'channel select out of range: {wav} {i}'
                    self.parameters['channels'] = self.wav_channel[wav]
                    break
        elif self.fix_channel is not None:
            assert isinstance(self.fix_channel, list)
            self.parameters['channels'] = [int(i) - 1 for i in self.fix_channel]
        else:
            warnings.warn("Warning: wav unchanged")
            self.parameters['channels'] = []

    def apply(self, samples, sample_rate):
        if self.parameters['channels']:
            tmp = []
            for i in self.parameters['channels']:
                tmp.append(samples[i])
            samples = np.stack(tmp, axis=0)
        return samples.squeeze()
