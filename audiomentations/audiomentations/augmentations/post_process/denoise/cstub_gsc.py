import subprocess
from typing import List
import numpy as np

from ....core.transforms_interface import add_transform
from ..base import PostProcess


@add_transform('cstub_gsc')
class CstubGSC(PostProcess):

    def __init__(self, p=1.0, gsc_bin=None, mic=2, out_channel=4, gsc_args:List=None):
        super().__init__(p=p)
        self.mic = mic
        self.out_channel = out_channel
        if self.mic == 4 and self.out_channel == 4 and gsc_bin == None: 
            raise Exception(f"[ERROR] Default 4mic gsc_bin outputs 5 channels (4 beams + 1 raw), please either set out_channel to 5 or specify your own gsc_bin in your yaml")
        self.gsc_bin = {
                2: '/mnt/lustre02/jiangsu/aispeech/home/hl219/tools/taihang/fespl_tools/20210810/test_gsc',
                4: '/mnt/lustre02/jiangsu/aispeech/home/yhz25/taihang/tools/cstub/gsc/4m75mm_uca/test_gsc'
        }.get(self.mic, None)
        if gsc_bin:
            self.gsc_bin = gsc_bin
        if not self.gsc_bin: 
            raise Exception(f"[ERROR] No default gsc_bin for {self.mic}-mic setting, please specify your gsc_bin in your yaml")
        self.gsc_args = gsc_args

    """Use CSTUB GSC tools to process wav. only support 2-mic wav
    """
    supports_mono = False
    supports_multichannel = True

    def randomize_parameters(self, samples, sample_rate, accumulate_meta=None):
        super().randomize_parameters(samples, sample_rate, accumulate_meta=accumulate_meta)

    def apply(self, samples, sample_rate):
        channel, ilen = samples.shape
        assert channel == self.mic, f'Inconsistency found between mic setting and real wav channels, get wav shape: {samples.shape}, get mic setting: {self.mic}'
        assert sample_rate == 16000, f'GSC sr 16000 support only, get {sample_rate}'
        # args: aec_pFilterFlag, outGain, taps
        args = [self.gsc_bin]
        if self.gsc_args:
            args += [ str(arg) for arg in self.gsc_args]
        proc = subprocess.Popen(args=args,
                                stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)

        # cstub gsc will padding 256 zeros before samples
        # cstub aec will drop the last frame samples when < 256 (FFT=512)
        # so padding to multiples of 256 for no sample clip
        # and padding half frame for aec shift
        padding_zeros = 256 + int(np.ceil(ilen / 256)) * 256 - ilen
        samples = np.pad(samples, ((0, 0), (0, padding_zeros)), mode='constant')
        wav = (samples * 2**15).astype(np.int16).transpose()
        try:
            outs, _ = proc.communicate(input=wav.tobytes(), timeout=30)
        except subprocess.TimeoutExpired as e:
            proc.kill()
            raise UserWarning('CSTUB AEC timeout, kill the subprocess!') from e

        # cstub gsc will drop the last frame samples when < 512 (FFT=1024)
        samples = np.frombuffer(outs, dtype='int16').reshape(-1, self.out_channel).transpose() / 2**15
        return samples.astype(np.float32)[:, 256:256+ilen]
