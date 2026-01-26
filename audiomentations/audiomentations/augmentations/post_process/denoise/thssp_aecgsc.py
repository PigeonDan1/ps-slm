import numpy as np
from ....core.transforms_interface import add_transform
from ..base import PostProcess
import sys

@add_transform('cstub_aecgsc')
class CstubAecGSC(PostProcess):

    def __init__(self, p=1.0, version=None, lib_path=None, mic=2, ref=2, out_channel=4):
        super().__init__(p=p)
        self.mic = mic
        self.ref = ref
        self.out_channel = out_channel
        self.version = version if version else '/mnt/lustre02/jiangsu/aispeech/home/jfc20/mypytroch-asr/th_release/th2_alg/v0.23.0/utils/end2end'
        sys.path.insert(0,self.version)
        from th2interface.th_interface import THInterface
        self.th_interface = THInterface(lib_path)

    def randomize_parameters(self, samples, sample_rate, accumulate_meta=None):
        super().randomize_parameters(samples, sample_rate, accumulate_meta=accumulate_meta)

    def apply(self, samples, sample_rate):
        channel, ilen = samples.shape
        assert channel == self.mic, f'Inconsistency found between mic setting and real wav channels, get wav shape: {samples.shape}, get mic setting: {self.mic}'
        assert sample_rate == 16000, f'GSC sr 16000 support only, get {sample_rate}'
        samples = (samples*32768).astype(np.int16).transpose() # C, T -> T, C
        samples = np.pad(samples, ((0, 0), (0, self.ref)))
        aec, gsc, fbank = self.th_interface.ssp(samples, 0, 1, 0)
        
        return (gsc.reshape(-1,self.out_channel).transpose() / 32768).astype(np.float32)
