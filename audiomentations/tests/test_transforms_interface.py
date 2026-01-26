import unittest

from audiomentations.augmentations.pre_process import VolumePerturb


class TestTransformsInterface(unittest.TestCase):
    def test_freeze_and_unfreeze_parameters(self):
        volume_perturb = VolumePerturb(p=1.0)

        self.assertFalse(volume_perturb.are_parameters_frozen)

        volume_perturb.freeze_parameters()
        self.assertTrue(volume_perturb.are_parameters_frozen)

        volume_perturb.unfreeze_parameters()
        self.assertFalse(volume_perturb.are_parameters_frozen)
