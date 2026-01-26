import random
import warnings
import numpy as np
from loguru import logger

from ...core.transforms_interface import add_transform
from ...core.utils import calculate_rms, calculate_desired_noise_rms, get_file_paths

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


@add_transform('add_background_noise')
class AddBackgroundNoise(AddNoise):
    """Mix in another sound, e.g. a background noise. Useful if your original sound is clean and
    you want to simulate an environment where background noise is present.

    Can also be used for mixup, as in https://arxiv.org/pdf/1710.09412.pdf

    A folder of (background noise) sounds to be mixed in must be specified. These sounds should
    ideally be at least as long as the input sounds to be transformed. Otherwise, the background
    sound will be repeated, which may sound unnatural.

    Note that the gain of the added noise is relative to the amount of signal in the input. This
    implies that if the input is completely silent, no noise will be added.
    """

    abbr = 'add_background_noise'
    supports_multichannel = True

    def __init__(
        self,
        sounds_path=None,
        wav_channel_path=None,
        min_snr_in_db=3,
        max_snr_in_db=30,
        min_freq=None,
        max_freq=None,
        rms_channel_one=False,
        p=1.0,
        lru_cache_size=None,
        strict=True,
        constant=False,
        load_once=False
    ):
        """
        :param sounds_path: Path to a folder that contains sound files to randomly mix in. These
            files can be flac, mp3, ogg or wav.
        :param min_snr_in_db: Minimum signal-to-noise ratio in dB
        :param max_snr_in_db: Maximum signal-to-noise ratio in dB
        :param min_freq: If set with max_freq, ONLY frequencies between min_freq and max_freq will be used to calculate RMS. 
        :param max_freq: If set with min_freq, ONLY frequencies between min_freq and max_freq will be used to calculate RMS. 
        :param p: The probability of applying this transform
        :param lru_cache_size: Maximum size of the LRU cache for storing noise files in memory
        :param strict: use noise without silence part
        :param constant: when constant, add background noise to sample directly, with considerating snr
        """
        super().__init__(p)
        self.sound_file_paths = get_file_paths(sounds_path)
        self.sound_file_paths = [str(p) for p in self.sound_file_paths]
        assert len(self.sound_file_paths) > 0
        self.min_snr_in_db = min_snr_in_db
        self.max_snr_in_db = max_snr_in_db
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.rms_channel_one = rms_channel_one
        self.strict = strict
        self.constant = constant
        if lru_cache_size is not None:
            logger.warning(f'config lru_cache_size is deprecated.')
        if  load_once != False:
            logger.warning(f'config load_once is deprecated. ')
        self.wav_channel_path = []
        if wav_channel_path is not None:
            tmp_dict = {}
            for line in open(wav_channel_path):
                wav, channels = line.strip().split()
                tmp_dict[wav] = [int(i) - 1 for i in channels.split('_')]
            for wav in self.sound_file_paths:
                wavname = wav.split('/')[-1]
                if wavname in tmp_dict.keys():
                    self.wav_channel_path.append([wav, tmp_dict[wavname]])

    def randomize_parameters(self, samples, sample_rate, accumulate_meta=None):
        super().randomize_parameters(samples, sample_rate, accumulate_meta)
        sound_file_paths = self.sound_file_paths
        if accumulate_meta is not None and len(self.wav_channel_path) > 0:
            for meta in accumulate_meta:
                if meta['name'] == 'GetChannel':
                    wav_channel_path = filter(lambda x:meta['channels'][0] in x[1],self.wav_channel_path)
                    sound_file_paths = [x[0] for x in wav_channel_path]
                    break
        if self.parameters["should_apply"]:
            if self.constant == False:
                self.parameters["snr_in_db"] = random.uniform(
                    self.min_snr_in_db, self.max_snr_in_db
                )
            else:
                self.parameters["snr_in_db"] = None


            num_samples = samples.shape[-1]
            self.parameters["noise_file_path"] = random.choice(sound_file_paths)
            num_noise_samples, noise_sample_rate = get_frames_rate(
                    self.parameters["noise_file_path"])
            while num_noise_samples == -1 or noise_sample_rate != sample_rate:
                self.parameters["noise_file_path"] = random.choice(sound_file_paths)
                num_noise_samples, noise_sample_rate = get_frames_rate(
                    self.parameters["noise_file_path"])

            min_noise_offset = 0
            max_noise_offset = max(0, num_noise_samples - num_samples - 1)
            self.parameters["noise_start_index"] = random.randint(
                min_noise_offset, max_noise_offset
                # 1000, 1000
            )
            # self.parameters["noise_end_index"] = (
            #     self.parameters["noise_start_index"] + num_samples
            # )
            if (self.min_freq != None and self.max_freq != None) and (self.min_freq < 0 or self.max_freq > sample_rate / 2): 
                logger.error(f"According to Nyquist, min_freq & max_freq should be between [0, {sample_rate/2}], got [{self.min_freq}, {self.max_freq}]")
                raise Exception(f"According to Nyquist, min_freq & max_freq should be between [0, {sample_rate/2}], got [{self.min_freq}, {self.max_freq}]")

    def apply(self, samples, sample_rate):

        # Repeat the sound if it shorter than the input sound
        num_samples = samples.shape[-1]

        noise_sound, _ = read_audio_section(self.parameters["noise_file_path"],
                                          self.parameters["noise_start_index"],
                                          num_samples)
                                          # self.parameters["noise_end_index"] - self.parameters["noise_start_index"])
        noise_sound = noise_sound.squeeze().transpose()  # channel first

        if len(samples.shape) == 2:
        # 多通道clean
            assert len(noise_sound.shape) == 2, \
                f'expect multi channel background noise, get shape {noise_sound.shape}'
            assert samples.shape[0] <= noise_sound.shape[0], \
                f'channel mismatch {samples.shape} != {noise_sound.shape}'
            noise_sound = noise_sound[:samples.shape[0]]    # slice noise channels to match with clean channels
        elif len(noise_sound.shape) == 2:
        # 单通道clean + 多通道噪音
            noise_sound = noise_sound[0]  # only use channel 1

        # get non zero slice if strict
        tmp = samples.sum(axis=0) if len(samples.shape) > 1 else samples
        s, e = tmp.nonzero()[0][[0, -1]] if self.strict else (0, num_samples)

        if self.parameters["snr_in_db"] is not None:

            if self.rms_channel_one:
                if len(noise_sound.shape) == 1:
                    noise_rms = calculate_rms(noise_sound[s:e], min_freq = self.min_freq, max_freq = self.max_freq, sample_rate = sample_rate)
                else:
                    noise_rms = calculate_rms(noise_sound[0, s:e], min_freq = self.min_freq, max_freq = self.max_freq, sample_rate = sample_rate)
            else:
                noise_rms = calculate_rms(noise_sound[..., s:e], min_freq = self.min_freq, max_freq = self.max_freq, sample_rate = sample_rate)
            if noise_rms < 1e-9:
                warnings.warn(
                    "The file {} is too silent to be added as noise. Returning the input"
                    " unchanged.".format(self.parameters["noise_file_path"])
                )
                return samples

            if self.rms_channel_one:
                if len(samples.shape) == 1:
                    clean_rms = calculate_rms(samples[s:e], min_freq = self.min_freq, max_freq = self.max_freq, sample_rate = sample_rate)
                else:
                    clean_rms = calculate_rms(samples[0, s:e], min_freq = self.min_freq, max_freq = self.max_freq, sample_rate = sample_rate)
            else:
                clean_rms = calculate_rms(samples[..., s:e], min_freq = self.min_freq, max_freq = self.max_freq, sample_rate = sample_rate)
            desired_noise_rms = calculate_desired_noise_rms(
                clean_rms, self.parameters["snr_in_db"]
            )

            # Adjust the noise to match the desired noise RMS
            noise_sound = noise_sound * (desired_noise_rms / noise_rms)

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

