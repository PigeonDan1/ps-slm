import os
from pathlib import Path
from loguru import logger
import subprocess
import numpy as np
import soundfile as sf
from ....core.transforms_interface import add_transform
from ..base import PostProcess

def _subprocess_Run(args,input=None,timeout=600,allow_error=False):
    logger.debug(f"Start running CMD: {args}")
    out = subprocess.run(args=[
            f'{args}',
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
        input=input,
        timeout=timeout)
    if not allow_error and out.returncode != 0:
        logger.error(f'Subprocess.run failed, CMD {args}, part of stderr:\n {out.stderr.decode()[-1000:]}')
    return out

@add_transform('binssp_dump')
class BINSSP_DUMP(PostProcess):

    def __init__(self, dump_dir, p=1.0, bintool=None, bintool_flag='m', trim_tail=True, cleanData=False,
                 out_channel=4, post_fix='multbf'):
        super().__init__(p=p)
        self.dump_dir = dump_dir
        self.bintool = bintool
        self.bintool_flag = bintool_flag
        self.trim_tail = trim_tail
        self.cleanData = cleanData
        self.out_channel = out_channel
        self.post_fix = post_fix
        self.dump_path = None

    def randomize_parameters(self, samples, sample_rate, accumulate_meta=None):
        super().randomize_parameters(samples, sample_rate, accumulate_meta=accumulate_meta)
        if self.dump_path is None:
            self.dump_path = Path(self.dump_dir) / "dumpwav_tmp"
            self.dump_path.mkdir(parents=True, exist_ok=True)
        self.dump_name = f"{accumulate_meta[0]['key'].split('/')[-1]}".replace('.wav', '')
        self.pad = [0,0]
        for meta in accumulate_meta:
            if meta['name'] == 'GetChannel':
                self.beam = meta['channels'][0] + 1
            elif meta['name'] == 'PadSilence':
                self.pad[0] += meta['time_padding'][0]
                self.pad[1] += meta['time_padding'][1]

    def apply(self, samples, sample_rate):
        assert sample_rate == 16000, f'SSP sr 16000 support only, get {sample_rate}'
        outfile = f"{self.dump_path}/{self.dump_name}.wav"

        sf.write(outfile, data=samples.transpose(), samplerate=sample_rate, subtype='PCM_16')
        decodelog = _subprocess_Run(f"{self.bintool} {outfile} -{self.bintool_flag} {self.dump_path}/{self.dump_name}")

        trim_tail = '' if not self.trim_tail else f'trim {self.pad[0]}' if self.pad[1] == 0 else f'trim {self.pad[0]} trim 0 -{self.pad[1]}'
        wav_files = [f'{self.dump_path}/{self.dump_name}.{self.post_fix}-{b+1}.wav' for b in range(self.out_channel)]
        if len(wav_files) > 1:
            _subprocess_Run(f"sox -M {' '.join(wav_files)} -t wavpcm {self.dump_path}/{self.dump_name}.wav {trim_tail}")
            samples = sf.read(f"{self.dump_path}/{self.dump_name}.wav")[0].transpose(1, 0)
        else:
            samples, sample_rate = sf.read(f"{self.dump_path}/{self.dump_name}.{self.post_fix}-1.wav")
        if self.cleanData:
            # logger.debug(f"Trying to rm {self.dump_path}/{self.dump_name}*")
            _subprocess_Run(f"rm -rf {' '.join(wav_files)} {self.dump_path}/{self.dump_name}.wav")
        return samples
