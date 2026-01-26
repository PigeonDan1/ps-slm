import os
import time
from typing import Union
from loguru import logger
import numpy as np
import librosa
import math

AUDIO_FILENAME_ENDINGS = (".aiff", ".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav")


def make_abbr(name, **args):
    args_list = []
    for k, v in args.items():
        if isinstance(v ,float):
            v = f'{v:.2f}'
        v = str(v)
        args_list.append(f'{k}{v}')
    if len(args_list) == 0:
        return name
    else:
        args_str = '-'.join(args_list)
        return f'{name}_{args_str}'


class timer(object):

    def __init__(self, description="Execution time", verbose=False):
        self.description = description
        self.verbose = verbose
        self.execution_time = None
        self.t = 0

    def __enter__(self):
        self.t = time.time()
        if self.verbose:
            logger.info("{}: Started!".format(self.description))
        return self

    def __exit__(self, type, value, traceback):
        self.execution_time = time.time() - self.t
        if self.verbose:
            logger.info("{}: Done, {:.3f} s".format(self.description, self.execution_time))


def get_file_paths(
    root_path,
    filename_endings=AUDIO_FILENAME_ENDINGS,
    traverse_subdirectories=True,
    follow_symlinks=True
):
    """Return a list of paths to all files with the given filename extensions in a directory.
    Also traverses subdirectories by default.
    """
    file_paths = []
    if root_path == None:
        return [None] # logger.info(f"use for accumulate_meta")
    elif os.path.isfile(root_path):
        with open(root_path) as fin:
            for line in fin:
                line = line.strip().split('|')[-1]
                if line.lower().endswith(filename_endings):
                    file_paths.append(line)
    elif os.path.isdir(root_path):
        for root, _, filenames in os.walk(root_path, followlinks=follow_symlinks):
            filenames = sorted(filenames)
            for filename in filenames:
                input_path = os.path.abspath(root)
                file_path = os.path.join(input_path, filename)

                if filename.lower().endswith(filename_endings):
                    file_paths.append(file_path)
            if not traverse_subdirectories:
                # prevent descending into subfolders
                break
    else:
        raise UserWarning(f'Unknown error for file {root_path}')

    return file_paths

def strip_samples_head_and_tail(samples):
    """Given a numpy array of audio samples, return non-zero part of samples along time axis"""

    def find_act_head_tail_idx(arr):
        start_idx, end_idx = len(arr), 0
        for i, x in enumerate(arr):
            if x != 0:
                start_idx = i
                break
        for i, x in enumerate(reversed(arr)):
            if x != 0:
                end_idx = len(arr) - i
                break
        return start_idx, end_idx

    if len(samples.shape) == 2:
        channel_first = True
        if samples.shape[0] > samples.shape[1]:
            channel_first = False
            samples = samples.T # convert to channel first for compute convenient
        sum_along_channel =  samples.sum(axis=0)
        start_idx, end_idx = find_act_head_tail_idx(sum_along_channel)
        act_samples = samples[:, start_idx:end_idx]
        return act_samples if channel_first else act_samples.T
    else: # mono channel
        start_idx, end_idx = find_act_head_tail_idx(samples)
        return samples[start_idx:end_idx]



def calculate_rms(samples, min_freq=None, max_freq=None, fft="fft", sample_rate=16000):
    """Given a numpy array of audio samples, return its Root Mean Square (RMS).
    If min_freq and max_freq if given, it will do FFT to samples first and calculate RMS within the frequency range.
    """
    if min_freq != None and max_freq != None:
        if fft == "stft":
            # Do STFT and calculate RMS within the frequency range, This function gives incorrect output (clean_rms / noise_rms) and still needs modify.
            # DON'T USE IT!
            n_fft = samples.shape[-1]
            stft = np.abs(librosa.stft(samples, n_fft=n_fft, hop_length=n_fft, window=np.ones(n_fft)))
            _nfft = n_fft // 2 + 1
            nyquist_max_freq = sample_rate / 2
            st, ed = int(min_freq / nyquist_max_freq * _nfft), int(max_freq / nyquist_max_freq * _nfft)
            return np.sqrt(np.power(stft[..., st:ed], 2).mean(-1).sum())
        elif fft == "fft":
            # Do FFT and calculate RMS within the frequency range. Use this by default.
            # check https://numpy.org/doc/stable/reference/generated/numpy.fft.fft.html
            # Output is symetric, check: https://stackoverflow.com/questions/70758915/is-a-numpy-fft-on-real-values-actually-hermitian
            # or: https://dsp.stackexchange.com/questions/4825/why-is-the-fft-mirrored
            n_fft = samples.shape[-1]
            fft = np.abs(np.fft.fft(samples, n_fft))
            _nfft = n_fft // 2 + 1
            nyquist_max_freq = sample_rate / 2
            st, ed = int(min_freq / nyquist_max_freq * _nfft), int(max_freq / nyquist_max_freq * _nfft)
            return np.sqrt(np.power(fft[..., st:ed], 2).mean(-1).sum())
    elif min_freq != None or max_freq != None:
        raise Exception(f"Both `min_freq` and `max_freq` should be set to None or Not None, got {min_freq} & {max_freq}")
    else:
        # Normal Root Mean Square
        return np.sqrt(np.mean(np.square(samples)))


def calculate_desired_noise_rms(clean_rms, snr):
    """
    Given the Root Mean Square (RMS) of a clean sound and a desired signal-to-noise ratio (SNR),
    calculate the desired RMS of a noise sound to be mixed in.

    Based on https://github.com/Sato-Kunihiko/audio-SNR/blob/8d2c933b6c0afe6f1203251f4877e7a1068a6130/create_mixed_audio_file.py#L20
    :param clean_rms: Root Mean Square (RMS) - a value between 0.0 and 1.0
    :param snr: Signal-to-Noise (SNR) Ratio in dB - typically somewhere between -20 and 60
    :return:
    """
    a = float(snr) / 20
    noise_rms = clean_rms / (10 ** a)
    return noise_rms


def convert_decibels_to_amplitude_ratio(decibels):
    return 10 ** (decibels / 20)


def is_waveform_multichannel(samples):
    """
    Return bool that answers the question: Is the given ndarray a multichannel waveform or not?

    :param samples: numpy ndarray
    :return:
    """
    return len(samples.shape) > 1


def is_spectrogram_multichannel(spectrogram):
    """
    Return bool that answers the question: Is the given ndarray a multichannel spectrogram?

    :param samples: numpy ndarray
    :return:
    """
    return len(spectrogram.shape) > 2 and spectrogram.shape[-1] > 1


def convert_float_samples_to_int16(y):
    """Convert floating-point numpy array of audio samples to int16."""
    if not issubclass(y.dtype.type, np.floating):
        raise ValueError("input samples not floating-point")
    return (y * np.iinfo(np.int16).max).astype(np.int16)


def convert_int16_samples_to_float(y):
    """Convert int16 numpy array of audio samples to floating-point."""
    if not issubclass(y.dtype.type, np.int16):
        raise ValueError("input samples not int16")
    return (y / np.iinfo(np.int16).max).astype(np.float32)

def convert_frequency_to_mel(f: float) -> float:
    """
    Convert f hertz to mels
    https://en.wikipedia.org/wiki/Mel_scale#Formula
    """
    return 2595.0 * math.log10(1.0 + f / 700.0)

def convert_mel_to_frequency(m: Union[float, np.array]) -> Union[float, np.array]:
    """
    Convert m mels to hertz
    https://en.wikipedia.org/wiki/Mel_scale#History_and_other_formulas
    """
    return 700.0 * (10 ** (m / 2595.0) - 1.0)