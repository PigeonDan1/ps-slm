import unittest

import numpy as np

from numpy.testing import assert_array_equal
from audiomentations.augmentations.add_noise import PadSilence
from audiomentations.augmentations.post_process import TimeTrim
from audiomentations.core.composition import Compose


class TestTimeTrim(unittest.TestCase):
    def test_padding_trim(self):
        sample_len = 16000
        samples = np.random.normal(0, 1, size=sample_len).astype(np.float32)
        sample_rate = 16000
        augmenter = Compose([
            PadSilence(p=1.0, time_padding=(5, 2)),
            TimeTrim(p=1.0, trim=(5, -2))
            ])
        samples_out = augmenter(samples_list=[samples], sample_rates_list=[sample_rate])
        assert_array_equal(samples_out, samples)

    def test_time_trim(self):
        sample_len = 16000
        samples = np.random.normal(0, 1, size=(2, sample_len)).astype(np.float32)
        sample_rate = 16000
        augmenter = Compose([TimeTrim(p=1.0, trim=(0.5, 0))])
        samples_out = augmenter(samples_list=[samples], sample_rates_list=[sample_rate])

        self.assertEqual(samples_out.dtype, np.float32)
        self.assertEqual(samples_out.shape[1], 8000)
        assert_array_equal(samples[:, 8000:], samples_out)
