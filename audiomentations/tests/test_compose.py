import os
import unittest

import numpy as np
from numpy.testing import assert_array_equal

from audiomentations.augmentations.add_noise.add_background_noise import AddBackgroundNoise
from audiomentations.augmentations.pre_process.volume_perturb import VolumePerturb
from audiomentations.core.composition import Compose
from demo.demo import DEMO_DIR


class TestCompose(unittest.TestCase):
    def test_freeze_and_unfreeze_parameters(self):
        samples = np.random.rand(20).astype(np.float32)
        sample_rate = 44100
        augmenter = Compose(
            [
                VolumePerturb(p=0.5),
                AddBackgroundNoise(
                    sounds_path=os.path.join(DEMO_DIR, "background_noises"),
                    min_snr_in_db=15,
                    max_snr_in_db=35,
                    p=1.0,
                )
            ]
        )
        perturbed_samples1 = augmenter(samples_list=[samples], sample_rates_list=[sample_rate])
        augmenter.freeze_parameters()
        for transform in augmenter.transforms:
            self.assertTrue(transform.are_parameters_frozen)
        perturbed_samples2 = augmenter(samples_list=[samples], sample_rates_list=[sample_rate])

        assert_array_equal(perturbed_samples1, perturbed_samples2)

        augmenter.unfreeze_parameters()
        for transform in augmenter.transforms:
            self.assertFalse(transform.are_parameters_frozen)
'''
    def test_randomize_parameters_and_apply(self):
        samples = 1.0 / np.arange(1, 21, dtype=np.float32)
        sample_rate = 44100

        augmenter = Compose(
            [
                VolumePerturb(p=0.5),
                AddBackgroundNoise(
                    sounds_path=os.path.join(DEMO_DIR, "background_noises"),
                    min_snr_in_db=15,
                    max_snr_in_db=35,
                    p=1.0,
                )
            ]
        )
        augmenter.freeze_parameters()
        augmenter.randomize_parameters(samples=samples, sample_rate=sample_rate)

        parameters = [transform.parameters for transform in augmenter.transforms]

        perturbed_samples1 = augmenter(samples_list=[samples], sample_rates_list=[sample_rate])
        perturbed_samples2 = augmenter(samples_list=[samples], sample_rates_list=[sample_rate])

        assert_array_equal(perturbed_samples1, perturbed_samples2)

        augmenter.unfreeze_parameters()

        for transform_parameters, transform in zip(parameters, augmenter.transforms):
            self.assertTrue(transform_parameters == transform.parameters)
            self.assertFalse(transform.are_parameters_frozen)
'''