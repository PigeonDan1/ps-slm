import sys
import subprocess
import numpy as np
from ....core.transforms_interface import add_transform
from ..base import PostProcess

@add_transform('binssp')
class BINSSP(PostProcess):

    def __init__(self, p=1.0, bintool=None, out_channel=4):
        super().__init__(p=p)
        self.bintool = bintool
        self.out_channel = out_channel

    def randomize_parameters(self, samples, sample_rate, accumulate_meta=None):
        super().randomize_parameters(samples, sample_rate, accumulate_meta=accumulate_meta)

    def apply(self, samples, sample_rate):
        assert sample_rate == 16000, f'SSP sr 16000 support only, get {sample_rate}'
        args = self.bintool
        proc = subprocess.Popen(args=args,
                                stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                shell=True,)
        samples = (samples*32768).astype(np.int16).transpose() # C, T -> T, C
        outs, _ = proc.communicate(input=samples.tobytes(), timeout=120)
        samples = np.frombuffer(outs, dtype='int16').reshape(-1, self.out_channel).transpose() / 2**15
        
        return samples.astype(np.float32)
