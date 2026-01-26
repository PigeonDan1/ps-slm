import os
import unittest

from audiomentations.core.composition import Compose
from audiomentations.core.audio_loading_utils import load_wav_file
from audiomentations.augmentations.pre_process import Dump
from audiomentations.augmentations.add_noise import AddBackgroundNoise
from audiomentations.augmentations.impose_response import ApplyImpulseResponse
from audiomentations.augmentations.post_process import SelectBeam, NNMask

from demo.demo import DEMO_DIR
import os


class TestNNMask(unittest.TestCase):
    def test_nnmask(self):
        key = 'acoustic_guitar_0'
        meta = [{'key': key, 'name': 'key'}]
        wav_path = os.path.join(DEMO_DIR, "acoustic_guitar_0.wav")
        
        samples, sample_rate = load_wav_file(
            wav_path, sample_rate=None, mono=False
        )
        dump_clean = Dump('./tests/dump', suffix='clean')
        dump_addnoise = Dump('./tests/dump', suffix='add_noise')
        dump_enhance = Dump('./tests/dump', suffix='enhance')
        apply_impluse_response = ApplyImpulseResponse(
            ir_path='/mnt/lustre02/jiangsu/aispeech/home/hl219/data/audio_augmentation/rir/2ch_30mm/RIR_ULA_2mic_30mm_wav_taihang.wavlist',
            ir_num_channel=2
        )
        add_background_noise = AddBackgroundNoise(
            sounds_path='/mnt/lustre/aifs/fgfs/users/fx310/work21/corpus/nnse_mchs/noise/noise_scatter.wavlist')

        select_beam = SelectBeam(
            rir_beam_path='/mnt/lustre02/jiangsu/aispeech/home/hl219/data/audio_augmentation/rir/2ch_30mm/rir_beam.map'
        )

        model_path = '/mnt/lustre02/jiangsu/aispeech/home/hl219/work22/th_project/0000_Common/nnbf/exp/2mic_30mm/comm2022V2_0508/rnnbf_si-snr'
        try:
            os.symlink('/mnt/lustre02/jiangsu/aispeech/home/hl219/tools/pytorch-asr/nsp/nsp', 'extend_codes')
        except FileExistsError:
            pass
        nnmask = NNMask(model_path, extensions=['extend_codes.model'])


        augmenter = Compose(
            [   
                dump_clean,
                apply_impluse_response,
                add_background_noise,
                dump_addnoise,
                nnmask,
                select_beam,
                dump_enhance
            ]
        )
        samples_produce, meta = augmenter(samples_list=[samples], sample_rates_list=[sample_rate],
                        meta=meta)
        # assert len(samples) == len(samples_produce)