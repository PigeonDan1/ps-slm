import subprocess

import numpy as np

from ....core.transforms_interface import add_transform
from ..base import PostProcess

@add_transform('cstub_aec')
class CstubAEC(PostProcess):
    """Use CSTUB AEC tools to process wav.
    """
    supports_mono = False
    supports_multichannel = True

    def __init__(self, aec_pFilterFlag=1, outGain=1.0, taps=8, p=1.0):
        super().__init__(p)
        self.parameters['aec_bin'] = '/mnt/lustre02/jiangsu/aispeech/home/hl219/tools/taihang/fespl_tools/20210810/test_aec_2-2'
        self.parameters['aec_pFilterFlag'] = str(aec_pFilterFlag)
        self.parameters['outGain'] = str(outGain)
        self.parameters['taps'] = str(taps)

    def apply(self, samples, sample_rate):
        channel, ilen = samples.shape
        assert channel == 4, f'AEC 2mic+2ref support only, get wav shape {samples.shape}'
        assert sample_rate == 16000, f'AEC sr 16000 support only, get {sample_rate}'
        # args: aec_pFilterFlag, outGain, taps
        # out: 2 channel mic signals
        proc = subprocess.Popen(args=[
                                        self.parameters['aec_bin'],
                                        self.parameters['aec_pFilterFlag'],
                                        self.parameters['outGain'],
                                        self.parameters['taps'],
                                ],
                                stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)

        # cstub aec will padding 512 zeros before samples
        # cstub aec will drop the last frame samples when < 512 (FFT=1024)
        # so padding to multiples of 512 for no sample clip
        # and padding half frame for aec shift
        padding_zeros = 512 + int(np.ceil(ilen / 512)) * 512 - ilen
        samples = np.pad(samples, ((0, 0), (0, padding_zeros)), mode='constant')
        wav = (samples * 2**15).astype(np.int16).transpose()
        try:
            outs, _ = proc.communicate(input=wav.tobytes(), timeout=30)
        except subprocess.TimeoutExpired as e:
            proc.kill()
            raise UserWarning('CSTUB AEC timeout, kill the subprocess!') from e

        samples = np.frombuffer(outs, dtype='int16').reshape(-1, 2).transpose() / 2**15
        return samples.astype(np.float32)[:, 512:512+ilen]
  
