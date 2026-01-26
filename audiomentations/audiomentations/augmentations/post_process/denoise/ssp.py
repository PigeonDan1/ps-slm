import importlib
from pathlib import Path
import numpy as np
from ....core.transforms_interface import add_transform
from ..base import PostProcess
import os
import sys

@add_transform('ssp')
class SSP(PostProcess):

    def __init__(self, p=1.0, version=None, lib_path=None, res_path=None, out_channel=4, sil_or_blk='blk', debug_bf_output_trick=None):
        super().__init__(p=p)
        self.version = version if version and Path(version).is_dir() else '/hpc_stor01/home/fusong.chen/work/projects/wakeup-nnbf-16bit/th_alg/v2.2.0-191-g219040bb'
        sys.path.insert(0,f'{self.version}/utils/end2end')
        print(self.version)
        module_cfg = importlib.import_module("th2interface.th_interface")
        print("ssp load: ",module_cfg)
        self.th_interface = module_cfg.THInterface(lib_path) if res_path is None else module_cfg.THInterface(res_path, lib_path)
        self.out_channel = out_channel
        self.sil_or_blk = sil_or_blk
        self.debug_bf_output_trick = debug_bf_output_trick

    def randomize_parameters(self, samples, sample_rate, accumulate_meta=None):
        super().randomize_parameters(samples, sample_rate, accumulate_meta=accumulate_meta)

    def apply(self, samples, sample_rate):
        channel, ilen = samples.shape
        assert sample_rate == 16000, f'SSP sr 16000 support only, get {sample_rate}'
        samples = (samples*32768).astype(np.int16).transpose() # C, T -> T, C
        if self.debug_bf_output_trick is not None:
            os.environ['debug_bf_output_trick'] = str(self.debug_bf_output_trick)
        aec, gsc, fbank = self.th_interface.ssp(samples, 0, 1, 0, f"words={self.sil_or_blk};thresh=2.0")

        return (gsc.reshape(-1,self.out_channel).transpose() / 32768).astype(np.float32)
