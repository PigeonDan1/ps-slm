from ....core.transforms_interface import add_transform
from ..base import PostProcess
import numpy as np
from nara_wpe.wpe import wpe
from nara_wpe.utils import stft, istft


@add_transform('wpe')
class WPE(PostProcess):
    """
    dereverse the audio. Also known as time inversion. This work addresses signal dereverberation 
    techniques based on WPE for speech recognition and other far-field applications. WPE is a 
    compelling algorithm to blindly dereverberate acoustic signals based on long-term linear 
    prediction. See more details in https://github.com/fgnt/nara_wpe
    """

    abbr = 'wpe'
    supports_multichannel = True

    def __init__(self, size=512, shift=128, delay=3, iterations=3, taps=15, p=1.0):
        super().__init__(p)
        self.stft_options  = dict(size=size, shift=shift)
        self.delay         = delay
        self.iterations    = iterations
        self.taps          = taps

    def apply(self, samples, sample_rate):
        if len(samples.shape) == 1: 
            samples = samples[np.newaxis,:]
        fre_out = stft(samples, **self.stft_options).transpose(2, 0, 1)
        wpe_out = wpe( fre_out,
                        taps=self.taps,
                        delay=self.delay,
                        iterations=self.iterations,
                        statistics_mode='full'
                      ).transpose(1, 2, 0)
        de_out = istft(wpe_out, size=self.stft_options['size'], shift=self.stft_options['shift']) # dereverberated wav
        if len(samples.shape) == 2:
            samples = de_out[0]
        else:
            samples = de_out
        return samples.transpose().astype(np.float32)
