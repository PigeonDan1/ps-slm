import unittest

import numpy as np
from numpy.testing import assert_array_equal

from audiomentations.augmentations.pre_process import SpeedPerturb


class TestSpeedPerturb(unittest.TestCase):
    def test_dynamic_length(self):
        samples = np.zeros((2048,), dtype=np.float32)
        sample_rate = 16000
        augmenter = SpeedPerturb(
            min_rate=0.8, max_rate=0.9, leave_length_unchanged=False, p=1.0
        )

        samples = augmenter(samples=samples, sample_rate=sample_rate)

        self.assertEqual(samples.dtype, np.float32)
        self.assertGreater(len(samples), 2048)

    def test_fixed_length(self):
        samples = np.zeros((2048,), dtype=np.float32)
        sample_rate = 16000
        augmenter = SpeedPerturb(
            min_rate=0.8, max_rate=0.9, leave_length_unchanged=True, p=1.0
        )

        samples = augmenter(samples=samples, sample_rate=sample_rate)

        self.assertEqual(samples.dtype, np.float32)
        self.assertEqual(len(samples), 2048)

    def test_multichannel(self):
        num_channels = 3
        samples = np.random.normal(0, 0.1, size=(num_channels, 5555)).astype(np.float32)
        sample_rate = 16000
        augmenter = SpeedPerturb(
            min_rate=0.8, max_rate=0.9, leave_length_unchanged=True, p=1.0
        )

        samples_out = augmenter(samples=samples, sample_rate=sample_rate)

        self.assertEqual(samples.dtype, samples_out.dtype)
        self.assertEqual(samples.shape, samples_out.shape)
        for i in range(num_channels):
            assert not np.allclose(samples[i], samples_out[i])

    def test_reproduce(self):
        augmenter = SpeedPerturb(
            min_rate=0.8, max_rate=0.9, leave_length_unchanged=True, p=1.0
        )
        augmenter.freeze_parameters()
        num_channels, ntime, sample_rate = 3, 100, 16000
        samples = np.random.randint((2**16), size=(num_channels, ntime))
        samples_out1 = augmenter(samples=samples, sample_rate=sample_rate)
        samples_out2 = augmenter(samples=samples, sample_rate=sample_rate)
        assert_array_equal(samples_out1, samples_out2)
