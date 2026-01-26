import numpy as np
from ...core.transforms_interface import add_transform
from .base import PreProcess

@add_transform('concat')
class Concat(PreProcess):
    def __init__(self, p=1.0):
        super().__init__(p=p)

    
    def apply(self, samples, sample_rate):
        if isinstance(samples, list):
            assert len(samples) > 0, 'list of sample should be not be emtpy'
            assert all(samples[0].ndim == spl.ndim for spl in samples), 'Dim of each sample in list should be equeal'
            if samples[0].ndim > 1:
                assert all(samples[0].shape[0] == spl.shape[0] for spl in samples), f'channels of each sample in list should be equal, but {[spl.shape for spl in samples]}'
            
            return np.concatenate(samples, axis=-1)
        else:
            assert isinstance(samples, np.ndarray), f'samples should be {np.ndarray} but {type(samples)}'
            return samples
