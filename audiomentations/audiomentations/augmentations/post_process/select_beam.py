import random
import warnings


from ...core.transforms_interface import add_transform
from .base import PostProcess


@add_transform('select_beam')
class SelectBeam(PostProcess):
    """Select 1 channel wav from multi-channel wav.
       according to RIR or random select
    """
    supports_mono = False
    supports_multichannel = True

    def __init__(self, rir_beam_path=None, p=1.0):
        super().__init__(p=p)
        self.rir_beam = None
        if rir_beam_path is not None:
            self.rir_beam = {}
            with open(rir_beam_path, 'r') as fin:
                for line in fin:
                    rir, beam = line.strip().split()
                    self.rir_beam[rir] = int(beam)

    def randomize_parameters(self, samples, sample_rate, accumulate_meta=None):
        super().randomize_parameters(samples, sample_rate, accumulate_meta=accumulate_meta)
        if len(samples.shape) == 1:
            self.parameters['beam'] = -1
            return

        channel, _ = samples.shape
        if accumulate_meta is not None and self.rir_beam is not None:
            # only use first rir
            for meta in accumulate_meta:
                if meta['name'] == 'ApplyImpulseResponse':
                    rir = meta['ir_file_path'].split('/')[-1]  # basename
                    self.parameters['beam'] = self.rir_beam[rir] - 1
                    break
        else:
            warnings.warn("Warning: random select beam in SelectBeam WaveformTransform.")
            self.parameters['beam'] = random.randint(0, channel-1)

    def apply(self, samples, sample_rate):
        if len(samples.shape) == 1:
            return samples
        elif len(samples.shape) == 2:
            return samples[self.parameters['beam']]
