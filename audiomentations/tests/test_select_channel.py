import unittest
import os

import numpy as np
from numpy.testing import assert_array_equal
from audiomentations.augmentations.post_process import SelectChannel
from audiomentations.core.composition import Compose
from demo.demo import DEMO_DIR


class TestSelectChannel(unittest.TestCase):
    def test_select_channel(self):
        sample_len = 16000
        samples = np.random.normal(0, 1, size=(4, sample_len)).astype(np.float32)
        sample_rate = 16000

        select_channel = SelectChannel(
            wav_channel_path=os.path.join(DEMO_DIR, "wav_channel"), p=1.0,
        )
        augmenter = Compose([select_channel])
        transforms_meta = [{'name': 'key', 'key': "test_select_channel"},]
        samples_out, _ = augmenter(samples_list=[samples], sample_rates_list=[sample_rate], meta=transforms_meta)

        self.assertEqual(augmenter.transforms[0].parameters["channels"], [0, 1])
        assert_array_equal(samples[:2], samples_out)

    def test_select_channel_fix(self):
        sample_len = 16000
        samples = np.random.normal(0, 1, size=(4, sample_len)).astype(np.float32)
        sample_rate = 16000

        select_channel = SelectChannel(fix_channel=[1, 2], p=1.0)
        augmenter = Compose([select_channel])
        samples_out = augmenter(samples_list=[samples], sample_rates_list=[sample_rate])

        self.assertEqual(augmenter.transforms[0].parameters["channels"], [0, 1])
        assert_array_equal(samples[:2], samples_out)

    def test_select_channel_unchanged(self):
        sample_len = 16000
        samples = np.random.normal(0, 1, size=(4, sample_len)).astype(np.float32)
        sample_rate = 16000

        select_channel = SelectChannel(p=1.0)
        augmenter = Compose([select_channel])
        samples_out = augmenter(samples_list=[samples], sample_rates_list=[sample_rate])

        assert_array_equal(samples, samples_out)
