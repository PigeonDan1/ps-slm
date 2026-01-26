from ...core.transforms_interface import add_transform
from .base import PostProcess

@add_transform('time_trim')
class TimeTrim(PostProcess):
    """Trim an audio signal according to the given time
    """
    abbr = 'time_trim'
    supports_multichannel = True

    def __init__(self, trim=(0, 0), p=1.0):
        super().__init__(p)
        if trim[1] > 0:
            assert trim[1] > trim[0]
        # trim=(start, end)
        # end=0 means trim to the last
        self.parameters['trim'] = trim

    def apply(self, samples, sample_rate):
        sample_len = samples.shape[-1]
        s = int(sample_rate * self.parameters['trim'][0])
        e = int(sample_rate * self.parameters['trim'][1])
        if s >= sample_len:
            return samples
        if e < 0 and s + abs(e) > sample_len:
            return samples
        samples = samples[..., s:] if e == 0 else samples[..., s:e]
        return samples

