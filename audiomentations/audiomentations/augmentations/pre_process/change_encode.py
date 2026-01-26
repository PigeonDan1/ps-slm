from ...core.transforms_interface import add_transform
from .base import PreProcess

@add_transform('change_encode')
class ChangeEncode(PreProcess):
    abbr = 'change_encode'
    
    def __init__(self, p=1.0):
        pass

    def randomize_parameters(self, samples, sample_rate, accumulate_meta=None):
        pass

    def apply(self, samples, sample_rate):
        pass
