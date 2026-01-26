import math
import unittest

import numpy as np

from audiomentations.augmentations.pre_process import Resample
from audiomentations.core.composition import Compose


class TestResample(unittest.TestCase):
    def test_resample(self):
        samples = np.zeros((512,), dtype=np.float32)
        sample_rate = 16000
        augmenter = Compose(
            [Resample(target_sample_rate=8000, p=1.0)]
        )
        samples = augmenter(samples_list=[samples], sample_rates_list=[sample_rate])

        self.assertEqual(samples.dtype, np.float32)

        self.assertGreaterEqual(
            len(samples), math.ceil(len(samples) * 8000 / sample_rate)
        )
