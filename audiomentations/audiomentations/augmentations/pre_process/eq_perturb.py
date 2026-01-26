import numpy as np
from scipy.signal import lfilter
from ...core.transforms_interface import add_transform
from .base import PreProcess


@add_transform('eq_perturb')
class EQPerturb(PreProcess):
    abbr = 'eq_perturb'
    
    def __init__(self, p=1.0):
        super().__init__(p=p)

    def apply(self, samples, sample_rate):
        
        a = np.array([1.0, -1.99599, 0.99600])
        b = np.array([1.0, -2.0, 1.0])
        samples = lfilter(b, a, samples)

        a_sig = 0.75 * (np.random.rand(3) - 0.5)
        a_sig[0] = 1.0
        b_sig = 0.75 * (np.random.rand(3) - 0.5)
        b_sig[0] = 1.0
        samples = lfilter(b_sig, a_sig, samples)
        return samples
