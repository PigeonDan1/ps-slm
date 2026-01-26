import os
import unittest
from audiomentations.core.composition import Compose
from audiomentations.augmentations.pre_process import SevenBandParametricEQ, FrequencyMask, HighShelfFilter, LowShelfFilter, PeakingFilter
from audiomentations.core.audio_loading_utils import load_wav_file


from demo.demo import DEMO_DIR

class TestFilters(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.wav_path = os.path.join(DEMO_DIR, "acoustic_guitar_0.wav")
        cls.samples, cls.sample_rate = load_wav_file(
            cls.wav_path, sample_rate=None, mono=False
        )

    def test_SevenBandParametricEQ(self):
        augmenter = Compose(
            [SevenBandParametricEQ(p=1.0)]
        )
        aug_samples = augmenter(samples_list=[self.samples], sample_rates_list=[self.sample_rate])
        assert len(aug_samples) == len(self.samples)
        
    def test_FrequencyMask(self):
        augmenter = Compose(
            [FrequencyMask(p=1.0)]
        )
        aug_samples = augmenter(samples_list=[self.samples], sample_rates_list=[self.sample_rate])
        assert len(aug_samples) == len(self.samples)
        
    def test_HighShelfFilter(self):
        augmenter = Compose(
            [HighShelfFilter(p=1.0)]
        )
        aug_samples = augmenter(samples_list=[self.samples], sample_rates_list=[self.sample_rate])
        assert len(aug_samples) == len(self.samples)
        
    def test_LowShelfFilter(self):
        augmenter = Compose(
            [LowShelfFilter(p=1.0)]
        )
        aug_samples = augmenter(samples_list=[self.samples], sample_rates_list=[self.sample_rate])
        assert len(aug_samples) == len(self.samples)

    def test_PeakingFilter(self):
        augmenter = Compose(
            [PeakingFilter(p=1.0)]
        )
        aug_samples = augmenter(samples_list=[self.samples], sample_rates_list=[self.sample_rate])
        assert len(aug_samples) == len(self.samples)