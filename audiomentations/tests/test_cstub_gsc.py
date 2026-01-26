import unittest

import numpy as np

from audiomentations.augmentations.post_process import CstubGSC
from audiomentations.core.composition import Compose


class TestCstubGSC(unittest.TestCase):
    def test_cstub_gsc(self):
        sample_len = 16000
        samples = np.random.normal(0, 1, size=(2, sample_len)).astype(np.float32)
        sample_rate = 16000
        augmenter = Compose([
                        CstubGSC(p=1.0),
                    ])
        samples_out = augmenter(samples_list=[samples], sample_rates_list=[sample_rate])
        samples_out_len = samples_out.shape[1]
        self.assertEqual(sample_len, samples_out_len)
