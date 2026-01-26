import random
import warnings
import numpy as np
from loguru import logger

from ...core.transforms_interface import add_transform
from ...core.utils import calculate_rms, calculate_desired_noise_rms, get_file_paths
from ..impose_response.apply_impose_response import ApplyImpulseResponse

from .base import AddNoise
import wave
import soundfile as sf
import contextlib

def get_frames_rate(filename):
    try:
        with contextlib.closing(wave.open(filename,'r')) as f:
            nframes = f.getnframes()
            rate = f.getframerate()
            return nframes, rate
    except (wave.Error,RuntimeError):
        logger.warning(f'{str(filename)} get duration failed')
        return -1, None

def read_audio_section(filename, start_frame, duration_nframe):

    track = sf.SoundFile(filename)

    can_seek = track.seekable() # True
    if not can_seek:
        raise ValueError("Not compatible with seeking")

    sr = track.samplerate

    collect_nframe = 0
    collect_samples_list = []
    while collect_nframe < duration_nframe :
        frames_to_read = duration_nframe - collect_nframe
        track.seek(start_frame)
        audio_section = track.read(frames_to_read)
        collect_samples_list.append(audio_section)
        collect_nframe += audio_section.shape[0]
        start_frame = 0
    samples = np.concatenate(collect_samples_list, axis=0)
    return samples, sr


@add_transform('add_point_interfere')
class AddPointInterfere(AddNoise):
    """add double point interfere in the same environment
    """

    abbr = 'add_point_interfere'
    supports_multichannel = True

    def __init__(
        self,
        sounds_path=None,
        min_sir_in_db=3,
        max_sir_in_db=30,
        p=1.0,
        strict=True,
        constant=False,
    ):
        """
        :param sounds_path: Path to a folder that contains sound files to randomly mix in. These
            files can be flac, mp3, ogg or wav.
        :param min_sir_in_db: Minimum signal-to-noise ratio in dB
        :param max_sir_in_db: Maximum signal-to-noise ratio in dB
        :param p: The probability of applying this transform
        :param strict: use noise without silence part
        :param constant: when constant, add point interfere to sample directly, with considerating snr
        """
        super().__init__(p)
        self.sound_file_paths = get_file_paths(sounds_path)
        self.sound_file_paths = [str(p) for p in self.sound_file_paths]
        assert len(self.sound_file_paths) > 0
        self.min_sir_in_db = min_sir_in_db
        self.max_sir_in_db = max_sir_in_db
        self.strict = strict
        self.constant = constant

    def randomize_parameters(self, samples, sample_rate, accumulate_meta=None):
        super().randomize_parameters(samples, sample_rate, accumulate_meta)
        if self.parameters["should_apply"]:
            if self.constant == False:
                self.parameters["sir_in_db"] = random.uniform(
                    self.min_sir_in_db, self.max_sir_in_db
                )
            else:
                self.parameters["sir_in_db"] = None

            num_samples = samples.shape[-1]
            self.parameters["noise_file_path"] = random.choice(self.sound_file_paths)
            num_noise_samples, noise_sample_rate = get_frames_rate(
                    self.parameters["noise_file_path"])
            while num_noise_samples == -1 or noise_sample_rate != sample_rate:
                self.parameters["noise_file_path"] = random.choice(self.sound_file_paths)
                num_noise_samples, noise_sample_rate = get_frames_rate(
                    self.parameters["noise_file_path"])

            min_noise_offset = 0
            max_noise_offset = max(0, num_noise_samples - num_samples - 1)
            self.parameters["noise_start_index"] = random.randint(
                min_noise_offset, max_noise_offset
            )
            # self.parameters["noise_end_index"] = (
            #     self.parameters["noise_start_index"] + num_samples
            # )
            if accumulate_meta is not None:
                # only use first rir
                self.rir = ApplyImpulseResponse(ir_path=None)
                for meta in accumulate_meta:
                    if meta['name'] == 'ApplyImpulseResponse':
                        meta['ir_startidx'] += meta['ir_num_channel']
                        self.rir.parameters.update(meta)
                        break

    def apply(self, samples, sample_rate):

        # Repeat the sound if it shorter than the input sound
        num_samples = samples.shape[-1]

        noise_sound, _ = read_audio_section(self.parameters["noise_file_path"],
                                          self.parameters["noise_start_index"],
                                          num_samples)
                                          # self.parameters["noise_end_index"] - self.parameters["noise_start_index"])
        noise_sound = noise_sound.squeeze().transpose()  # channel first

        if len(noise_sound.shape) == 2:
            noise_sound = noise_sound[0]  # only use channel 1

        # get non zero slice if strict
        tmp = samples.sum(axis=0) if len(samples.shape) > 1 else samples
        s, e = tmp.nonzero()[0][[0, -1]] if self.strict else (0, num_samples)

        if self.parameters["sir_in_db"] is not None:

            noise_rms = calculate_rms(noise_sound[..., s:e])
            if noise_rms < 1e-9:
                warnings.warn(
                    "The file {} is too silent to be added as noise. Returning the input"
                    " unchanged.".format(self.parameters["noise_file_path"])
                )
                return samples

            clean_rms = calculate_rms(samples[..., s:e])
            desired_noise_rms = calculate_desired_noise_rms(
                clean_rms, self.parameters["sir_in_db"]
            )

            # Adjust the noise to match the desired noise RMS
            noise_sound = noise_sound * (desired_noise_rms / noise_rms)
            # add impose response
            noise_sound = self.rir.apply(noise_sound, sample_rate)

        # Return a mix of the input sound and the background noise sound
        if samples.shape != noise_sound.shape:
            logger.error(f' {samples.shape} ! = {noise_sound.shape} {self.parameters["noise_file_path"]} {self.parameters["noise_start_index"]} {num_samples}')
        return samples + noise_sound

    def __getstate__(self):
        state = self.__dict__.copy()
        warnings.warn(
            "Warning: the LRU cache of AddBackgroundNoise gets discarded when pickling it."
            " E.g. this means the cache will not be used when using AddBackgroundNoise together"
            " with multiprocessing on Windows"
        )
        del state["_load_sound"]
        return state

