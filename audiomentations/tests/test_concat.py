import unittest

import numpy as np
from numpy.testing import assert_array_equal

from audiomentations.augmentations.pre_process import Concat
from audiomentations.core.composition import Compose

class TestConcat(unittest.TestCase):

    def test_concat(self):
        nchan = 2
        ntime0, ntime1, ntime2 = 50, 100, 150

        samples0, samples1, samples2 = np.random.randint((2**16), size=(nchan, ntime0)), \
            np.random.randint((2**16), size=(nchan, ntime1)), \
            np.random.randint((2**16), size=(nchan, ntime2))
        augmenter = Compose(
            [
                Concat(p=1)
            ]
        )
        samples = augmenter(samples_list=[samples0, samples1, samples2], sample_rates_list=[[16000] * 3])
        assert_array_equal(samples, np.concatenate([samples0,samples1,samples2], axis=-1))