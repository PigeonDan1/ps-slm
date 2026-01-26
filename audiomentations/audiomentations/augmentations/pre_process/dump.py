from loguru import logger
import numpy as np
from pathlib import Path
from scipy.io import wavfile
from ...core.transforms_interface import add_transform
from .base import PreProcess

@add_transform('dump')
class Dump(PreProcess):
    def __init__(self, dump_dir, suffix='', p=1.0):
        super().__init__(p=p)
        self.dump_dir = Path(dump_dir)
        self.suffix = '_'+suffix
        self.dump_dir.mkdir(parents=True, exist_ok=True)

    
    def apply(self, samples, sample_rate):

        if isinstance(samples, list):
            assert len(samples) > 0, 'list of sample should be not be emtpy'
            assert all(samples[0].ndim == spl.ndim for spl in samples), 'Dim of each sample in list should be equal'
            for i, spl in enumerate(samples):
                output_file_path = self.dump_dir / f"{self.parameters['key']}-{i}{self.suffix}.wav"
                if spl.shape[1] > spl.shape[0]:
                    spl = spl.transpose(1, 0) # transpose to (nsample, nchannel)
                wavfile.write(
                    output_file_path, rate=sample_rate, data=(spl*32768).astype(np.int16)
                )
        else:
            assert isinstance(samples, np.ndarray), f'samples should be {np.ndarray} but {type(samples)}'
            dump_samples = samples
            if samples.ndim > 1:
                if samples.shape[1] > samples.shape[0]:
                    dump_samples = samples.transpose(1,0)
            
            output_file_path = self.dump_dir / f"{self.parameters['key']}{self.suffix}.wav"
            wavfile.write(
                output_file_path, rate=sample_rate, data=(dump_samples*32768).astype(np.int16)
            )

        return samples

    def randomize_parameters(self, samples, sample_rate, accumulate_meta=None):
        super().randomize_parameters(samples, sample_rate, accumulate_meta=accumulate_meta)
        assert accumulate_meta is not None
        for meta in accumulate_meta:
            if meta['name'] == 'key':
                self.parameters['key'] = meta['key'].split('/')[-1]  # basename
                break
        assert 'key' in self.parameters, 'Not found key in meta, it can not happen.'
