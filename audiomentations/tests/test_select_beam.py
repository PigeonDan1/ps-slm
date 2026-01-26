import unittest
import os

import numpy as np

from audiomentations.augmentations.impose_response import ApplyImpulseResponse
from audiomentations.augmentations.post_process import CstubGSC, SelectBeam
from audiomentations.core.composition import Compose
from demo.demo import DEMO_DIR


class TestSelectBeam(unittest.TestCase):
    def test_select_beam(self):
        sample_len = 16000
        samples = np.random.normal(0, 1, size=sample_len).astype(np.float32)
        sample_rate = 16000

        add_ir_transform = ApplyImpulseResponse(
            ir_path=os.path.join(DEMO_DIR, "ir_2ch"), p=1.0,
            ir_num_channel=2, leave_length_unchanged=True,
        )
        gsc = CstubGSC(p=1.0)
        select_beam = SelectBeam(
            rir_beam_path=os.path.join(DEMO_DIR, "ir_2ch/rir_beam"), p=1.0,
        )
        augmenter = Compose([add_ir_transform, gsc, select_beam])
        samples_out = augmenter(samples_list=[samples], sample_rates_list=[sample_rate])

        self.assertEqual(augmenter.transforms[2].parameters["beam"], 2)
        self.assertEqual(len(samples_out.shape), 1)
        self.assertEqual(len(samples_out), sample_len)

    def test_select_beam_random(self):
        sample_len = 16000
        samples = np.random.normal(0, 1, size=sample_len).astype(np.float32)
        sample_rate = 16000

        add_ir_transform = ApplyImpulseResponse(
            ir_path=os.path.join(DEMO_DIR, "ir_2ch"), p=1.0,
            ir_num_channel=2, leave_length_unchanged=True,
        )
        gsc = CstubGSC(p=1.0)
        select_beam = SelectBeam(p=1.0)
        augmenter = Compose([add_ir_transform, gsc, select_beam])
        _ = augmenter(samples_list=[samples], sample_rates_list=[sample_rate])
