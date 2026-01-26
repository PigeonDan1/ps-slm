import json
import numpy
import os
import sys
import threading
import importlib

from ...core.transforms_interface import add_transform
from .base import PostProcess


@add_transform('cut_wkp')
class CutWakeupSegment(PostProcess):
    """Get the wake-up segments in the given multi-channel audio.
    """
    abbr = 'cut_wkp'
    supports_multichannel = True

    def __init__(self, version, res_path, lib_path, words,
                 ssp_in_channel=2, ssp_out_channel=2,
                 segment_len=3.5, right_len=0.5, p=1.0):
        super().__init__(p)
        sys.path.insert(0,f'{version}/utils/end2end')
        module_cfg = importlib.import_module("th2interface.th_interface")
        self.th_interface = module_cfg.THInterface(res_path, lib_path)
        self.words = words if 'words=' in words else f"words={self.words}"
        self.ssp_in_channel = ssp_in_channel
        self.ssp_out_channel = ssp_out_channel
        self.segment_len = segment_len
        self.right_len = right_len

    def inference(self, wave):
        dtype = wave.dtype
        wave = (wave * 32768).astype(numpy.int16).transpose()
        assert wave.shape[-1] == self.ssp_in_channel, wave.shape
        _, gsc, _ = self.th_interface.ssp(wave, 0, 1, 0, self.words)
        out = (gsc / 32768).astype(dtype).reshape(-1, self.ssp_out_channel)
        return out.transpose()

    def apply(self, samples, sample_rate):
        captured_stdout = []

        # 重定向stdout
        stdout_fileno = sys.stdout.fileno()
        stdout_save = os.dup(stdout_fileno)
        stdout_pipe = os.pipe()
        os.dup2(stdout_pipe[1], stdout_fileno)
        os.close(stdout_pipe[1])

        def capture_stdout():
            buffer = ''
            while True:
                data = os.read(stdout_pipe[0], 4096).decode('utf-8')
                if not data:
                    break
                buffer += data
                if '\n' in buffer:
                    lines = buffer.split('\n')
                    buffer = lines[-1]
                    captured_stdout.extend(lines[:-1])

        t = threading.Thread(target=capture_stdout)
        t.start()
        gsc = self.inference(samples)
        t.join(timeout=1e-5)

        # 恢复stdout
        sys.stdout.flush()
        os.close(stdout_fileno)
        os.close(stdout_pipe[0])
        os.dup2(stdout_save, stdout_fileno)
        os.close(stdout_save)

        samples_list = []
        for line in captured_stdout:
            if 'chan' not in line:
                continue
            channel_info, wkp_info = line.split('\t', 1)
            c = int(channel_info[-2]) - 1
            wkp_time = json.loads(wkp_info)['time']
            end = int(min((wkp_time + self.right_len) * sample_rate, gsc.shape[1]))
            start = int(max(wkp_time - (self.segment_len - self.right_len), 0) * sample_rate)
            wkp_segment = gsc[c: c+1, start: end]
            samples_list.append(wkp_segment)

        return samples_list
