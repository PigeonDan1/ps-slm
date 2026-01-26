import os
from pathlib import Path
import unittest
from audiomentations.core.composition import Compose
from audiomentations.augmentations.pre_process import EQSimulation
from audiomentations.core.audio_loading_utils import load_wav_file

from demo.demo import DEMO_DIR

class TestEQSimulation(unittest.TestCase):
    def test_eq_simulation(self):

        wav_path = os.path.join(DEMO_DIR, "acoustic_guitar_0.wav")
        samples, sample_rate = load_wav_file(
            wav_path, sample_rate=None, mono=False
        )

        augmenter = Compose(
            [EQSimulation(cfg= Path(DEMO_DIR) / 'eq_simu' / 'EQ_out_20240912_112859.xml', p=1.0)]
        )
        aug_samples = augmenter(samples_list=[samples], sample_rates_list=[sample_rate])
        assert len(aug_samples) == len(samples)

