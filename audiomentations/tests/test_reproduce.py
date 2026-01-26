from loguru import logger
from numpy.lib.function_base import select
from numpy.testing._private.utils import assert_array_equal
from audiomentations.augmentations import impose_response
from audiomentations.augmentations.pre_process.concat import Concat
import os
import unittest

import numpy as np

from audiomentations.augmentations.pre_process import Dump, pitch_perturb, resample, speed_perturb
from audiomentations.core.composition import Compose
from audiomentations.core.audio_loading_utils import load_wav_file
from audiomentations.augmentations.pre_process import ( 
    Resample,
    VolumePerturb, 
    SpeedPerturb,
    PitchPerturb,
    Concat,
    Dump
)
from audiomentations.core.transforms_interface import MutuallyExclusiveGroup
from audiomentations.augmentations.add_noise import PadSilence, AddBackgroundNoise, pad_silence
from audiomentations.augmentations.impose_response import ApplyImpulseResponse, apply_impose_response
from audiomentations.augmentations.post_process import(
    CstubAEC,
    CstubGSC,
    SelectBeam,
    SelectChannel,
    TimeTrim
)


from demo.demo import DEMO_DIR


class TestReproduce(unittest.TestCase):

    def test_reproduce(self):
        key = 'acoustic_guitar_0'
        suffix = 'suffix'
        meta = [{'key': key, 'name': 'key'}]
        wav_path = os.path.join(DEMO_DIR, "acoustic_guitar_0.wav")
        
        samples, sample_rate = load_wav_file(
            wav_path, sample_rate=None, mono=False
        )
        dump_dir = os.path.dirname(__file__)
        
        resample = Resample()
        volume_perturb = VolumePerturb()
        speed_perturb = SpeedPerturb()
        pitch_perturb = PitchPerturb()
        mutually_exclusive_group = MutuallyExclusiveGroup(transforms=[{'name': 'speed_perturb'}, {'name': 'pitch_perturb'}], p = [0.5, 0.5])
        concat = Concat()
        dump = Dump(dump_dir, suffix=suffix, p=1)

        apply_impluse_response = ApplyImpulseResponse(
            ir_path='/mnt/lustre02/jiangsu/aispeech/home/hl219/data/audio_augmentation/rir/2ch_30mm/RIR_ULA_2mic_30mm_wav_taihang.wavlist',
            ir_num_channel=2
        )
        pad_silence_time = PadSilence(time_padding=(5,0))
        add_background_noise = AddBackgroundNoise(
            sounds_path='/mnt/lustre/aifs/fgfs/users/fx310/work21/corpus/nnse_mchs/noise/noise_scatter.wavlist')

        pad_silence_channel = PadSilence(channel_padding=(0,2))
        cstub_aec = CstubAEC()
        cstub_gsc = CstubGSC()

        select_beam = SelectBeam(
            rir_beam_path='/mnt/lustre02/jiangsu/aispeech/home/hl219/data/audio_augmentation/rir/2ch_30mm/rir_beam.map'
        )
        time_trim = TimeTrim(trim=(5,0))

        augmenter = Compose(
            [   
                resample,
                volume_perturb,
                speed_perturb,
                pitch_perturb,
                mutually_exclusive_group,
                concat,
                dump,
                apply_impluse_response,
                pad_silence_time,
                add_background_noise,
                pad_silence_channel,
                cstub_aec,
                cstub_gsc,
                select_beam,
                time_trim
            ]
        )
        repeat = 3
        samples_produce, meta = augmenter(samples_list=[samples]*repeat, sample_rates_list=[sample_rate]*repeat,
                        meta=meta)
        samples_reproduce, meta = augmenter(samples_list=[samples] * repeat, sample_rates_list=[sample_rate] * repeat,
                        meta=meta[1:], reproduce=True)
        
        assert_array_equal(samples_produce, samples_reproduce)