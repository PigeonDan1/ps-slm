import warnings
import random

from ...core.transforms_interface import add_transform
from .select_channel import SelectChannel


@add_transform('get_channel')
class GetChannel(SelectChannel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def randomize_parameters(self, samples, sample_rate, accumulate_meta=None):
        self.parameters["should_apply"] = random.random() < self.p
        assert len(samples.shape) > 1, f'select_channle expect multi-channel wav, but get mono wav'
        channel, _ = samples.shape
        if accumulate_meta is not None and self.wav_channel is not None:
            # only use first rir
            for meta in accumulate_meta:
                if meta['name'] == 'key' or meta['name'] == 'ApplyImpulseResponse':
                    wav = meta['key'].split('/')[-1]  if meta['name'] == 'key' else meta['ir_file_path'].split('/')[-1] # basename
                    if wav == "": continue
                    if wav in self.wav_channel.keys():
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
        return samples
