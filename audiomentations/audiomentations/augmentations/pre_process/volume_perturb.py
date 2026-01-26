import random
from loguru import logger
import numpy as np
from ...core.transforms_interface import add_transform
from ...core.utils import calculate_rms
from .base import PreProcess


@add_transform('volume_perturb')
class VolumePerturb(PreProcess):
    """Generalize wav energy to random DB
    """
    supports_mono = True
    supports_multichannel = True

    def __init__(self, choices=[-20, -25, -30, -35, -40],  p=1.0):
        super().__init__(p=p)
        self.choices = choices

    def randomize_parameters(self, samples, sample_rate, accumulate_meta=None):
        super().randomize_parameters(samples, sample_rate, accumulate_meta=accumulate_meta)
        self.parameters['db_shuffle'] = float(np.random.choice(self.choices))


    def apply(self, samples, sample_rate):
        rms = calculate_rms(samples)
        db_shuffle = self.parameters['db_shuffle']
        rms_shuffle = np.power(10, db_shuffle / 20)
        fscale = rms_shuffle / rms
        return samples * fscale
