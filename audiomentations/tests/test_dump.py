import os
from random import sample
import filecmp

from audiomentations.augmentations.pre_process import Dump
from audiomentations.core.composition import Compose
from audiomentations.core.audio_loading_utils import load_wav_file
from demo.demo import DEMO_DIR

class TestDump:

    
    def test_dump(self, tmp_path):
        key = 'acoustic_guitar_0'
        suffix = 'suffix'
        meta = [{'key': key, 'name': 'key'}]
        wav_path = os.path.join(DEMO_DIR, "acoustic_guitar_0.wav")
        samples, sample_rate = load_wav_file(
            wav_path, sample_rate=None, mono=False
        )
        dump_dir = tmp_path
        dump_name = f'{key}_{suffix}.wav'
        dump_wav = os.path.join(dump_dir, dump_name)
        print(f"Dumpped wav into {dump_wav}")
        augmenter = Compose(
            [
                Dump(dump_dir, suffix=suffix, p=1)
            ]
        )
        
        samples = augmenter(samples_list=[samples], sample_rates_list=[sample_rate],
                        meta=meta)
        assert filecmp.cmp(wav_path, dump_wav)