from loguru import logger
import torch
from asr.data import Field, Batch
from asr.utils.checkpoint import Checkpoint
from asr.utils.common import import_extensions


from ....core.transforms_interface import add_transform
from ..base import PostProcess


@add_transform('nnmask')
class NNMask(PostProcess):

    abbr = 'nnmask'
    def __init__(self, model_path, extensions=['extend_codes.model'], force_cpu=True, wav_name='GSC-tfmask', p=1.0):
        super().__init__(p)
        import_extensions(extensions)

        self.model = Checkpoint.load_model(model_path)
        if force_cpu:
            self.model.cpu()
        else:
            if torch.cuda.is_available():
                self.model.cuda()
        self.model.eval()
        self.wav_name = wav_name
    

    def apply(self, samples, sample_rate):
        samples=samples.transpose(1,0)
        batch_data = torch.tensor(samples).unsqueeze(0)
        batch_length = torch.tensor([samples.shape[0]], dtype=torch.long)
        batch = Batch({
            'mixed_speech': Field(batch_data, batch_length)
        })
        enhanced_wavs, lengths, pred_mask = self.model.decode(batch)
        enhanced_wavs = enhanced_wavs[self.wav_name][0].detach().numpy().T
        assert len(enhanced_wavs.shape) <= 2, \
            f'enhanced wav shape should be less than 2, but shape is {enhanced_wavs.shape}'
        return enhanced_wavs
