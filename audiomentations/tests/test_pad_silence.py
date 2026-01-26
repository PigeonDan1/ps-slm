import unittest

import numpy as np

from numpy.testing import assert_array_equal
from audiomentations.augmentations.add_noise import PadSilence
from audiomentations.core.composition import Compose


class TestPaddingSilence(unittest.TestCase):
    def test_time_padding_silence(self):
        sample_len = 16000
        samples = np.random.normal(0, 1, size=sample_len).astype(np.float32)
        sample_rate = 16000
        augmenter = Compose([PadSilence(p=1.0, time_padding=(5, 2))])
        samples_out = augmenter(samples_list=[samples], sample_rates_list=[sample_rate])

        self.assertEqual(samples_out.dtype, np.float32)
        self.assertEqual(len(samples_out), sample_len + (5+2)*sample_rate)
        s = np.zeros(5*sample_rate)
        e = np.zeros(2*sample_rate)
        assert_array_equal(samples_out[:5*sample_rate], s)
        assert_array_equal(samples_out[-2*sample_rate:], e)
        assert_array_equal(samples_out[5*sample_rate:-2*sample_rate], samples)

    def test_time_padding_silence_2ch(self):
        sample_len = 16000
        samples = np.random.normal(0, 1, size=(2, sample_len)).astype(np.float32)
        sample_rate = 16000
        augmenter = Compose([PadSilence(p=1.0, time_padding=(5, 2))])
        samples_out = augmenter(samples_list=[samples], sample_rates_list=[sample_rate])

        self.assertEqual(samples_out.dtype, np.float32)
        self.assertEqual(samples_out.shape[1], sample_len + (5+2)*sample_rate)
        s = np.zeros((2, 5*sample_rate))
        e = np.zeros((2, 2*sample_rate))
        assert_array_equal(samples_out[:, :5*sample_rate], s)
        assert_array_equal(samples_out[:, -2*sample_rate:], e)
        assert_array_equal(samples_out[:, 5*sample_rate:-2*sample_rate], samples)

    def test_channel_padding_silence(self):
        sample_len = 16000
        samples = np.random.normal(0, 1, size=sample_len).astype(np.float32)
        sample_rate = 16000
        augmenter = Compose([PadSilence(p=1.0, channel_padding=(0, 1))])
        samples_out = augmenter(samples_list=[samples], sample_rates_list=[sample_rate])

        self.assertEqual(samples_out.dtype, np.float32)
        self.assertEqual(samples_out.shape[1], sample_len)
        sil = np.zeros((1, sample_len))
        assert_array_equal(samples_out[0], samples)
        assert_array_equal(samples_out[1:, :], sil)

    def test_channel_padding_silence_2ch(self):
        sample_len = 16000
        samples = np.random.normal(0, 1, size=(2, sample_len)).astype(np.float32)
        sample_rate = 16000
        augmenter = Compose([PadSilence(p=1.0, channel_padding=(0, 2))])
        samples_out = augmenter(samples_list=[samples], sample_rates_list=[sample_rate])

        self.assertEqual(samples_out.dtype, np.float32)
        self.assertEqual(samples_out.shape[1], sample_len)
        sil = np.zeros((2, sample_len))
        assert_array_equal(samples_out[:2, :], samples)
        assert_array_equal(samples_out[2:, :], sil)
