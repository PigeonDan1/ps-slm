import os
import unittest

import numpy as np
from audiomentations.core.audio_loading_utils import load_wav_file
from audiomentations.core.composition import Compose
from audiomentations.augmentations.pre_process import Dump
from audiomentations.augmentations.add_noise import AddBackgroundNoise
from audiomentations.augmentations.impose_response import ApplyImpulseResponse
from audiomentations.augmentations.post_process import SelectBeam, WPE
# from audiomentations.core.audio_loading_utils import load_sound_file
# from scipy.io import wavfile

from demo.demo import DEMO_DIR

class TestWPE(unittest.TestCase):
    def test_wpe(self):
        key = 'acoustic_guitar_0'
        meta = [{'key': key, 'name': 'key'}]
        wav_path = os.path.join(DEMO_DIR, "acoustic_guitar_0.wav")
        
        samples, sample_rate = load_wav_file(
            wav_path, sample_rate=None, mono=False
        )
        dump_clean = Dump('./tests/wpe/dump', suffix='clean')
        dump_addnoise = Dump('./tests/wpe/dump', suffix='add_noise')
        dump_enhance = Dump('./tests/wpe/dump', suffix='enhance')

        apply_impluse_response = ApplyImpulseResponse(
            ir_path='/mnt/lustre02/jiangsu/aispeech/home/hl219/data/audio_augmentation/rir/2ch_30mm/RIR_ULA_2mic_30mm_wav_taihang.wavlist',
            ir_num_channel=2
        )
        
        
        augmenter = Compose([
            dump_clean,
            apply_impluse_response,
            dump_addnoise,
            WPE(p=1.0),
            dump_enhance
            ])
        
        samples_out, meta = augmenter(samples_list=[samples], sample_rates_list=[sample_rate], meta=meta)
        assert len(samples) == len(samples_out)
