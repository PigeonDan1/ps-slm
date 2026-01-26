import os
import unittest
from audiomentations.core.composition import Compose
from audiomentations.augmentations.pre_process import EQPerturb
from audiomentations.core.audio_loading_utils import load_wav_file


from demo.demo import DEMO_DIR

class TestEQPerturb(unittest.TestCase):
    def test_eq_perturb(self):

        wav_path = os.path.join(DEMO_DIR, "acoustic_guitar_0.wav")
        
        samples, sample_rate = load_wav_file(
            wav_path, sample_rate=None, mono=False
        )

        augmenter = Compose(
            [EQPerturb(p=1.0)]
        )
        aug_samples = augmenter(samples_list=[samples], sample_rates_list=[sample_rate])
        assert len(aug_samples) == len(samples)
