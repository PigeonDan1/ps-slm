from ...core.transforms_interface import add_transform
from .base import PreProcess
import subprocess
from scipy.io import wavfile
import tempfile
import io


@add_transform('resample_to')
class ResampleTo(PreProcess):
    """
    Resample signal using sox

    To do resampling to target_sample_rate.
    """
    abbr = 'resample_to'
    def __init__(self, target_sample_rate=8000, p=1.0):
        """
        :param target_sample_rate: int, Target sample rate
        :param p: The probability of applying this transform
        """
        super().__init__(p)
        assert target_sample_rate >= 0
        self.target_sample_rate = target_sample_rate

    def apply(self, samples, sample_rate):
        
        tmpfile_in = tempfile.TemporaryFile()
        if len(samples.shape) > 1:
            samples = samples.transpose()        
        wavfile.write(tmpfile_in, sample_rate, samples) # (samples*32768).astype(np.int16))

        proc_resample = subprocess.Popen(args=[
                                'sox', '-', '-r', f'{self.target_sample_rate}',  '-t', 'wav', '-'
                        ],
                        stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE)

        try:
            
            outs, log = proc_resample.communicate(input=tmpfile_in.read(), timeout=30)
        except subprocess.TimeoutExpired as e:
            proc_resample.kill()
            raise UserWarning('sox timeout, kill the subprocess!') from e
        _, samples = wavfile.read(io.BytesIO(outs))

        if len(samples.shape) > 1:
            samples = samples.transpose() 

        return samples
