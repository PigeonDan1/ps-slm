from ...core.transforms_interface import add_transform
from .base import PostProcess


@add_transform('vad_align')
class VadAligner(PostProcess):
    """Get the vad alignment
    """
    abbr = 'vad_align'
    supports_multichannel = False

    def __init__(self, silence_label='0', speech_label='410',
                 frame_shift=20, p=1.0):
        super().__init__(p)
        self.silence_label = silence_label
        self.speech_label = speech_label
        self.frame_shift = frame_shift

    def get_vad_label(self, samples, sample_rate):
        # TODO(menglong.xu): 使用能量来获得vad标签
        label = [str(self.speech_label)] * int(samples.shape[-1] / sample_rate * 1000 / self.frame_shift)
        return label

    def apply(self, samples, sample_rate, other_input_output):
        other_input_output['name'] = 'vad_align'
        vad_label = self.get_vad_label(samples, sample_rate)
        other_input_output['vad_align'] = ' '.join(vad_label)
        return samples
