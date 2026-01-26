import subprocess
import functools
import os
import random
import sys
import tempfile
import uuid
import warnings

import librosa
import numpy as np
from scipy.signal import butter, sosfilt, convolve
from ..core.audio_loading_utils import load_sound_file
from ..core.transforms_interface import BaseWaveformTransform, add_transform
from ..core.utils import (
    calculate_rms,
    calculate_desired_noise_rms,
    get_file_paths,
    convert_decibels_to_amplitude_ratio,
    convert_float_samples_to_int16,
    convert_int16_samples_to_float
)


@add_transform('frequency_mask')
class FrequencyMask(BaseWaveformTransform):
    """
    Mask some frequency band on the spectrogram.
    Inspired by https://arxiv.org/pdf/1904.08779.pdf
    """
    abbr="freqmask"
    supports_multichannel = True

    def __init__(self, min_frequency_band=0.0, max_frequency_band=0.5, p=1.0):
        """
        :param min_frequency_band: Minimum bandwidth, float
        :param max_frequency_band: Maximum bandwidth, float
        :param p: The probability of applying this transform
        """
        super().__init__(p)
        self.min_frequency_band = min_frequency_band
        self.max_frequency_band = max_frequency_band

    def __butter_bandstop(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], btype="bandstop", output="sos")
        return sos

    def __butter_bandstop_filter(self, data, lowcut, highcut, fs, order=5):
        sos = self.__butter_bandstop(lowcut, highcut, fs, order=order)
        y = sosfilt(sos, data).astype(np.float32)
        return y

    def randomize_parameters(self, samples, sample_rate, accumulate_meta=None):
        super().randomize_parameters(samples, sample_rate, accumulate_meta)
        if self.parameters["should_apply"]:
            self.parameters["bandwidth"] = random.randint(
                self.min_frequency_band * sample_rate // 2,
                self.max_frequency_band * sample_rate // 2,
            )
            self.parameters["freq_start"] = random.randint(
                16, sample_rate // 2 - self.parameters["bandwidth"] - 1
            )

    def apply(self, samples, sample_rate):
        bandwidth = self.parameters["bandwidth"]
        freq_start = self.parameters["freq_start"]
        samples = self.__butter_bandstop_filter(
            samples, freq_start, freq_start + bandwidth, sample_rate, order=6
        )
        return samples

@add_transform('time_mask')
class TimeMask(BaseWaveformTransform):
    """
    Make a randomly chosen part of the audio silent.
    Inspired by https://arxiv.org/pdf/1904.08779.pdf
    """
    abbr = 'timemask'
    supports_multichannel = True

    def __init__(self, min_band_part=0.0, max_band_part=0.5, fade=False, p=1.0):
        """
        :param min_band_part: Minimum length of the silent part as a fraction of the
            total sound length. Float.
        :param max_band_part: Maximum length of the silent part as a fraction of the
            total sound length. Float.
        :param fade: Bool, Add linear fade in and fade out of the silent part.
        :param p: The probability of applying this transform
        """
        super().__init__(p)
        self.min_band_part = min_band_part
        self.max_band_part = max_band_part
        self.fade = fade

    def randomize_parameters(self, samples, sample_rate, accumulate_meta=None):
        super().randomize_parameters(samples, sample_rate, accumulate_meta)
        if self.parameters["should_apply"]:
            num_samples = samples.shape[-1]
            self.parameters["t"] = random.randint(
                int(num_samples * self.min_band_part),
                int(num_samples * self.max_band_part),
            )
            self.parameters["t0"] = random.randint(
                0, num_samples - self.parameters["t"]
            )

    def apply(self, samples, sample_rate):
        new_samples = samples.copy()
        t = self.parameters["t"]
        t0 = self.parameters["t0"]
        mask = np.zeros(t)
        if self.fade:
            fade_length = min(int(sample_rate * 0.01), int(t * 0.1))
            mask[0:fade_length] = np.linspace(1, 0, num=fade_length)
            mask[-fade_length:] = np.linspace(0, 1, num=fade_length)
        new_samples[..., t0 : t0 + t] *= mask
        return new_samples


@add_transform('guassian_snr')
class AddGaussianSNR(BaseWaveformTransform):
    """
    Add gaussian noise to the samples with random Signal to Noise Ratio (SNR).

    Note that old versions of . (0.16.0 and below) used parameters
    min_SNR and max_SNR, which had inverse (wrong) characteristics. The use of these
    parameters is discouraged, and one should use min_snr_in_db and max_snr_in_db
    instead now.

    Note also that if you use the new parameters, a random SNR will be picked uniformly
    in the decibel scale instead of a uniform amplitude ratio. This aligns
    with human hearing, which is more logarithmic than linear.
    """
    abbr = 'guassiansnr'
    supports_multichannel = True

    def __init__(
        self, min_SNR=None, max_SNR=None, min_snr_in_db=None, max_snr_in_db=None, p=1.0
    ):
        """
        :param min_SNR: Minimum signal-to-noise ratio (legacy). A lower number means less noise.
        :param max_SNR: Maximum signal-to-noise ratio (legacy). A greater number means more noise.
        :param min_snr_in_db: Minimum signal-to-noise ratio in db. A lower number means more noise.
        :param max_snr_in_db: Maximum signal-to-noise ratio in db. A greater number means less noise.
        :param p: The probability of applying this transform
        """
        super().__init__(p)
        self.min_snr_in_db = min_snr_in_db
        self.max_snr_in_db = max_snr_in_db
        if min_snr_in_db is None and max_snr_in_db is None:
            # Apply legacy defaults
            if min_SNR is None:
                min_SNR = 0.001
            if max_SNR is None:
                max_SNR = 1.0
        else:
            if min_SNR is not None or max_SNR is not None:
                raise Exception(
                    "Error regarding AddGaussianSNR: Set min_snr_in_db"
                    " and max_snr_in_db to None to keep using min_SNR and"
                    " max_SNR parameters (legacy) instead. We highly recommend to use"
                    " min_snr_in_db and max_snr_in_db parameters instead. To migrate"
                    " from legacy parameters to new parameters,"
                    " use the following conversion formulas: \n"
                    "min_snr_in_db = -20 * math.log10(max_SNR)\n"
                    "max_snr_in_db = -20 * math.log10(min_SNR)"
                )
        self.min_SNR = min_SNR
        self.max_SNR = max_SNR

    def randomize_parameters(self, samples, sample_rate, accumulate_meta=None):
        super().randomize_parameters(samples, sample_rate, accumulate_meta)
        if self.parameters["should_apply"]:

            if self.min_SNR is not None and self.max_SNR is not None:
                if self.min_snr_in_db is not None and self.max_snr_in_db is not None:
                    raise Exception(
                        "Error regarding AddGaussianSNR: Set min_snr_in_db"
                        " and max_snr_in_db to None to keep using min_SNR and"
                        " max_SNR parameters (legacy) instead. We highly recommend to use"
                        " min_snr_in_db and max_snr_in_db parameters instead. To migrate"
                        " from legacy parameters to new parameters,"
                        " use the following conversion formulas: \n"
                        "min_snr_in_db = -20 * math.log10(max_SNR)\n"
                        "max_snr_in_db = -20 * math.log10(min_SNR)"
                    )
                else:
                    warnings.warn(
                        "You use legacy min_SNR and max_SNR parameters in AddGaussianSNR."
                        " We highly recommend to use min_snr_in_db and max_snr_in_db parameters instead."
                        " To migrate from legacy parameters to new parameters,"
                        " use the following conversion formulas: \n"
                        "min_snr_in_db = -20 * math.log10(max_SNR)\n"
                        "max_snr_in_db = -20 * math.log10(min_SNR)"
                    )
                    min_snr = self.min_SNR
                    max_snr = self.max_SNR
                    std = np.std(samples)
                    self.parameters["noise_std"] = random.uniform(min_snr * std, max_snr * std)
                    clean_rms = calculate_rms(samples)
                    noise_rms = self.parameters["noise_std"]
                    self.parameters['snr'] = 20 * np.log10(clean_rms / noise_rms)
            else:
                # Pick snr in decibel scale
                snr = random.uniform(self.min_snr_in_db, self.max_snr_in_db)

                clean_rms = calculate_rms(samples)
                noise_rms = calculate_desired_noise_rms(clean_rms=clean_rms, snr=snr)

                # In gaussian noise, the RMS gets roughly equal to the std
                self.parameters["noise_std"] = noise_rms
                self.parameters['snr'] = snr

    def apply(self, samples, sample_rate):
        noise = np.random.normal(
            0.0, self.parameters["noise_std"], size=samples.shape
        ).astype(np.float32)
        return samples + noise


@add_transform('guassian_noise')
class AddGaussianNoise(BaseWaveformTransform):
    """Add gaussian noise to the samples"""

    abbr = 'guassiannoise'
    supports_multichannel = True

    def __init__(self, min_amplitude=0.001, max_amplitude=0.015, p=1.0):
        super().__init__(p)
        assert min_amplitude > 0.0
        assert max_amplitude > 0.0
        assert max_amplitude >= min_amplitude
        self.min_amplitude = min_amplitude
        self.max_amplitude = max_amplitude

    def randomize_parameters(self, samples, sample_rate, accumulate_meta=None):
        super().randomize_parameters(samples, sample_rate, accumulate_meta)
        if self.parameters["should_apply"]:
            self.parameters["amplitude"] = random.uniform(
                self.min_amplitude, self.max_amplitude
            )

    def apply(self, samples, sample_rate):
        noise = np.random.randn(*samples.shape).astype(np.float32)
        samples = samples + self.parameters["amplitude"] * noise
        return samples


@add_transform('shift')
class Shift(BaseWaveformTransform):
    """
    Shift the samples forwards or backwards, with or without rollover
    """
    abbr = 'shift'
    supports_multichannel = True

    def __init__(
        self,
        min_fraction=-0.5,
        max_fraction=0.5,
        rollover=True,
        fade=False,
        fade_duration=0.01,
        p=1.0,
    ):
        """
        :param min_fraction: float, fraction of total sound length
        :param max_fraction: float, fraction of total sound length
        :param rollover: When set to True, samples that roll beyond the first or last position
            are re-introduced at the last or first. When set to False, samples that roll beyond
            the first or last position are discarded. In other words, rollover=False results in
            an empty space (with zeroes).
        :param fade: When set to True, there will be a short fade in and/or out at the "stitch"
            (that was the start or the end of the audio before the shift). This can smooth out an
            unwanted abrupt change between two consecutive samples (which sounds like a
            transient/click/pop).
        :param fade_duration: If `fade=True`, then this is the duration of the fade in seconds.
        :param p: The probability of applying this transform
        """
        super().__init__(p)
        assert min_fraction >= -1
        assert max_fraction <= 1
        assert type(fade_duration) in [int, float] or not fade
        assert fade_duration > 0 or not fade
        self.min_fraction = min_fraction
        self.max_fraction = max_fraction
        self.rollover = rollover
        self.fade = fade
        self.fade_duration = fade_duration

    def randomize_parameters(self, samples, sample_rate, accumulate_meta=None):
        super().randomize_parameters(samples, sample_rate, accumulate_meta)
        if self.parameters["should_apply"]:
            self.parameters["num_places_to_shift"] = int(
                round(
                    random.uniform(self.min_fraction, self.max_fraction)
                    * samples.shape[-1]
                )
            )

    def apply(self, samples, sample_rate):
        num_places_to_shift = self.parameters["num_places_to_shift"]
        shifted_samples = np.roll(samples, num_places_to_shift, axis=-1)

        if not self.rollover:
            if num_places_to_shift > 0:
                shifted_samples[..., :num_places_to_shift] = 0.0
            elif num_places_to_shift < 0:
                shifted_samples[..., num_places_to_shift:] = 0.0

        if self.fade:
            fade_length = int(sample_rate * self.fade_duration)

            fade_in = np.linspace(0, 1, num=fade_length)
            fade_out = np.linspace(1, 0, num=fade_length)

            if num_places_to_shift > 0:

                fade_in_start = num_places_to_shift
                fade_in_end = min(
                    num_places_to_shift + fade_length, shifted_samples.shape[-1]
                )
                fade_in_length = fade_in_end - fade_in_start

                shifted_samples[
                    ...,
                    fade_in_start:fade_in_end,
                ] *= fade_in[:fade_in_length]

                if self.rollover:

                    fade_out_start = max(num_places_to_shift - fade_length, 0)
                    fade_out_end = num_places_to_shift
                    fade_out_length = fade_out_end - fade_out_start

                    shifted_samples[..., fade_out_start:fade_out_end] *= fade_out[
                        -fade_out_length:
                    ]

            elif num_places_to_shift < 0:

                positive_num_places_to_shift = (
                    shifted_samples.shape[-1] + num_places_to_shift
                )

                fade_out_start = max(positive_num_places_to_shift - fade_length, 0)
                fade_out_end = positive_num_places_to_shift
                fade_out_length = fade_out_end - fade_out_start

                shifted_samples[..., fade_out_start:fade_out_end] *= fade_out[
                    -fade_out_length:
                ]

                if self.rollover:
                    fade_in_start = positive_num_places_to_shift
                    fade_in_end = min(
                        positive_num_places_to_shift + fade_length,
                        shifted_samples.shape[-1],
                    )
                    fade_in_length = fade_in_end - fade_in_start
                    shifted_samples[
                        ...,
                        fade_in_start:fade_in_end,
                    ] *= fade_in[:fade_in_length]
        return shifted_samples


@add_transform('clip')
class Clip(BaseWaveformTransform):
    """
    Clip audio by specified values. e.g. set a_min=-1.0 and a_max=1.0 to ensure that no
    samples in the audio exceed that extent. This can be relevant for avoiding integer
    overflow or underflow (which results in unintended wrap distortion that can sound
    horrible) when exporting to e.g. 16-bit PCM wav.

    Another way of ensuring that all values stay between -1.0 and 1.0 is to apply
    PeakNormalization.

    This transform is different from ClippingDistortion in that it takes fixed values
    for clipping instead of clipping a random percentile of the samples. Arguably, this
    transform is not very useful for data augmentation. Instead, think of it as a very
    cheap and harsh limiter (for samples that exceed the allotted extent) that can
    sometimes be useful at the end of a data augmentation pipeline.
    """
    abbr = 'clip'
    supports_multichannel = True

    def __init__(self, a_min=-1.0, a_max=1.0, p=1.0):
        """
        :param a_min: float, minimum value for clipping
        :param a_max: float, maximum value for clipping
        :param p: The probability of applying this transform
        """
        super().__init__(p)
        assert a_min < a_max
        self.a_min = a_min
        self.a_max = a_max

    def apply(self, samples, sample_rate):
        return np.clip(samples, self.a_min, self.a_max)


@add_transform('normalize')
class Normalize(BaseWaveformTransform):
    """
    Apply a constant amount of gain, so that highest signal level present in the sound becomes
    0 dBFS, i.e. the loudest level allowed if all samples must be between -1 and 1. Also known
    as peak normalization.
    """
    abbr = 'normalize'
    supports_multichannel = True

    def randomize_parameters(self, samples, sample_rate, accumulate_meta=None):
        super().randomize_parameters(samples, sample_rate, accumulate_meta)
        if self.parameters["should_apply"]:
            self.parameters["max_amplitude"] = np.amax(np.abs(samples))

    def apply(self, samples, sample_rate):
        if self.parameters["max_amplitude"] > 0:
            normalized_samples = samples / self.parameters["max_amplitude"]
        else:
            normalized_samples = samples
        return normalized_samples


@add_transform('loudness_normalization')
class LoudnessNormalization(BaseWaveformTransform):
    """
    Apply a constant amount of gain to match a specific loudness. This is an implementation of
    ITU-R BS.1770-4.
    See also:
        https://github.com/csteinmetz1/pyloudnorm
        https://en.wikipedia.org/wiki/Audio_normalization

    Warning: This transform can return samples outside the [-1, 1] range, which may lead to
    clipping or wrap distortion, depending on what you do with the audio in a later stage.
    See also https://en.wikipedia.org/wiki/Clipping_(audio)#Digital_clipping
    """
    abbr = 'ldnormal'
    supports_multichannel = True

    def __init__(self, min_lufs_in_db=-31, max_lufs_in_db=-13, p=1.0):
        super().__init__(p)
        # For an explanation on LUFS, see https://en.wikipedia.org/wiki/LUFS
        self.min_lufs_in_db = min_lufs_in_db
        self.max_lufs_in_db = max_lufs_in_db

    def randomize_parameters(self, samples, sample_rate, accumulate_meta=None):
        try:
            import pyloudnorm
        except ImportError:
            print(
                "Failed to import pyloudnorm. Maybe it is not installed? "
                "To install the optional pyloudnorm dependency of .,"
                " do `pip install .[extras]` or simply "
                " `pip install pyloudnorm`",
                file=sys.stderr,
            )
            raise

        super().randomize_parameters(samples, sample_rate, accumulate_meta)
        if self.parameters["should_apply"]:
            meter = pyloudnorm.Meter(sample_rate)  # create BS.1770 meter
            # transpose because pyloudnorm expects shape like (smp, chn), not (chn, smp)
            self.parameters["loudness"] = meter.integrated_loudness(samples.transpose())
            self.parameters["lufs_in_db"] = float(
                random.uniform(self.min_lufs_in_db, self.max_lufs_in_db)
            )

    def apply(self, samples, sample_rate):
        try:
            import pyloudnorm
        except ImportError:
            print(
                "Failed to import pyloudnorm. Maybe it is not installed? "
                "To install the optional pyloudnorm dependency of .,"
                " do `pip install .[extras]` or simply "
                " `pip install pyloudnorm`",
                file=sys.stderr,
            )
            raise

        # Guard against digital silence
        if self.parameters["loudness"] > float("-inf"):
            # transpose because pyloudnorm expects shape like (smp, chn), not (chn, smp)
            return pyloudnorm.normalize.loudness(
                samples.transpose(),
                self.parameters["loudness"],
                self.parameters["lufs_in_db"],
            ).transpose()
        else:
            return samples


@add_transform('trim')
class Trim(BaseWaveformTransform):
    """
    Trim leading and trailing silence from an audio signal using librosa.effects.trim
    """
    abbr = 'trim'
    def __init__(self, top_db=20, p=1.0):
        super().__init__(p)
        self.top_db = top_db

    def apply(self, samples, sample_rate):
        samples, _ = librosa.effects.trim(samples, top_db=self.top_db)
        return samples



@add_transform('clipping_distortion')
class ClippingDistortion(BaseWaveformTransform):
    """Distort signal by clipping a random percentage of points

    The percentage of points that will be clipped is drawn from a uniform distribution between
    the two input parameters min_percentile_threshold and max_percentile_threshold. If for instance
    30% is drawn, the samples are clipped if they're below the 15th or above the 85th percentile.
    """
    abbr = 'clpdistortion'
    supports_multichannel = True

    def __init__(self, min_percentile_threshold=0, max_percentile_threshold=40, p=1.0):
        """
        :param min_percentile_threshold: int, A lower bound on the total percent of samples that
            will be clipped
        :param max_percentile_threshold: int, A upper bound on the total percent of samples that
            will be clipped
        :param p: The probability of applying this transform
        """
        super().__init__(p)
        assert min_percentile_threshold <= max_percentile_threshold
        assert 0 <= min_percentile_threshold <= 100
        assert 0 <= max_percentile_threshold <= 100
        self.min_percentile_threshold = min_percentile_threshold
        self.max_percentile_threshold = max_percentile_threshold

    def randomize_parameters(self, samples, sample_rate, accumulate_meta=None):
        super().randomize_parameters(samples, sample_rate, accumulate_meta)
        if self.parameters["should_apply"]:
            self.parameters["percentile_threshold"] = random.randint(
                self.min_percentile_threshold, self.max_percentile_threshold
            )

    def apply(self, samples, sample_rate):
        lower_percentile_threshold = int(self.parameters["percentile_threshold"] / 2)
        lower_threshold, upper_threshold = np.percentile(
            samples, [lower_percentile_threshold, 100 - lower_percentile_threshold]
        )
        samples = np.clip(samples, lower_threshold, upper_threshold)
        return samples



@add_transform('add_short_noises')
class AddShortNoises(BaseWaveformTransform):
    """Mix in various (bursts of overlapping) sounds with random pauses between. Useful if your
    original sound is clean and you want to simulate an environment where short noises sometimes
    occur.

    A folder of (noise) sounds to be mixed in must be specified.
    """
    abbr = 'shortnoise'

    def __init__(
        self,
        sounds_path=None,
        min_snr_in_db=0,
        max_snr_in_db=24,
        min_time_between_sounds=4.0,
        max_time_between_sounds=16.0,
        burst_probability=0.22,
        min_pause_factor_during_burst=0.1,
        max_pause_factor_during_burst=1.1,
        min_fade_in_time=0.005,
        max_fade_in_time=0.08,
        min_fade_out_time=0.01,
        max_fade_out_time=0.1,
        p=1.0,
        lru_cache_size=64,
    ):
        """
        :param sounds_path: Path to a folder that contains sound files to randomly mix in. These
            files can be flac, mp3, ogg or wav.
        :param min_snr_in_db: Minimum signal-to-noise ratio in dB. A lower value means the added
            sounds/noises will be louder.
        :param max_snr_in_db: Maximum signal-to-noise ratio in dB. A lower value means the added
            sounds/noises will be louder.
        :param min_time_between_sounds: Minimum pause time between the added sounds/noises
        :param max_time_between_sounds: Maximum pause time between the added sounds/noises
        :param burst_probability: The probability of adding an extra sound/noise that overlaps
        :param min_pause_factor_during_burst: Min value of how far into the current sound (as
            fraction) the burst sound should start playing. The value must be greater than 0.
        :param max_pause_factor_during_burst: Max value of how far into the current sound (as
            fraction) the burst sound should start playing. The value must be greater than 0.
        :param min_fade_in_time: Min sound/noise fade in time in seconds. Use a value larger
            than 0 to avoid a "click" at the start of the sound/noise.
        :param max_fade_in_time: Min sound/noise fade out time in seconds. Use a value larger
            than 0 to avoid a "click" at the start of the sound/noise.
        :param min_fade_out_time: Min sound/noise fade out time in seconds. Use a value larger
            than 0 to avoid a "click" at the end of the sound/noise.
        :param max_fade_out_time: Max sound/noise fade out time in seconds. Use a value larger
            than 0 to avoid a "click" at the end of the sound/noise.
        :param p: The probability of applying this transform
        :param lru_cache_size: Maximum size of the LRU cache for storing noise files in memory
        """
        super().__init__(p)
        self.sound_file_paths = get_file_paths(sounds_path)
        self.sound_file_paths = [str(p) for p in self.sound_file_paths]
        assert len(self.sound_file_paths) > 0
        assert min_snr_in_db <= max_snr_in_db
        assert min_time_between_sounds <= max_time_between_sounds
        assert 0.0 < burst_probability <= 1.0
        if burst_probability == 1.0:
            assert (
                min_pause_factor_during_burst > 0.0
            )  # or else an infinite loop will occur
        assert 0.0 < min_pause_factor_during_burst <= 1.0
        assert max_pause_factor_during_burst > 0.0
        assert max_pause_factor_during_burst >= min_pause_factor_during_burst
        assert min_fade_in_time >= 0.0
        assert max_fade_in_time >= 0.0
        assert min_fade_in_time <= max_fade_in_time
        assert min_fade_out_time >= 0.0
        assert max_fade_out_time >= 0.0
        assert min_fade_out_time <= max_fade_out_time

        self.min_snr_in_db = min_snr_in_db
        self.max_snr_in_db = max_snr_in_db
        self.min_time_between_sounds = min_time_between_sounds
        self.max_time_between_sounds = max_time_between_sounds
        self.burst_probability = burst_probability
        self.min_pause_factor_during_burst = min_pause_factor_during_burst
        self.max_pause_factor_during_burst = max_pause_factor_during_burst
        self.min_fade_in_time = min_fade_in_time
        self.max_fade_in_time = max_fade_in_time
        self.min_fade_out_time = min_fade_out_time
        self.max_fade_out_time = max_fade_out_time
        self._load_sound = functools.lru_cache(maxsize=lru_cache_size)(load_sound_file)

    def randomize_parameters(self, samples, sample_rate, accumulate_meta=None):
        super().randomize_parameters(samples, sample_rate, accumulate_meta)
        if self.parameters["should_apply"]:
            input_sound_duration = len(samples) / sample_rate

            current_time = 0
            global_offset = random.uniform(
                -self.max_time_between_sounds, self.max_time_between_sounds
            )
            current_time += global_offset
            sounds = []
            while current_time < input_sound_duration:
                sound_file_path = random.choice(self.sound_file_paths)
                sound, _ = self._load_sound(sound_file_path, sample_rate)
                sound_duration = len(sound) / sample_rate

                # Ensure that the fade time is not longer than the duration of the sound
                fade_in_time = min(
                    sound_duration,
                    random.uniform(self.min_fade_in_time, self.max_fade_in_time),
                )
                fade_out_time = min(
                    sound_duration,
                    random.uniform(self.min_fade_out_time, self.max_fade_out_time),
                )

                sounds.append(
                    {
                        "fade_in_time": fade_in_time,
                        "start": current_time,
                        "end": current_time + sound_duration,
                        "fade_out_time": fade_out_time,
                        "file_path": sound_file_path,
                        "snr_in_db": random.uniform(
                            self.min_snr_in_db, self.max_snr_in_db
                        ),
                    }
                )

                # burst mode - add overlapping sounds
                while (
                    random.random() < self.burst_probability
                    and current_time < input_sound_duration
                ):
                    pause_factor = random.uniform(
                        self.min_pause_factor_during_burst,
                        self.max_pause_factor_during_burst,
                    )
                    pause_time = pause_factor * sound_duration
                    current_time = sounds[-1]["start"] + pause_time

                    if current_time >= input_sound_duration:
                        break

                    sound_file_path = random.choice(self.sound_file_paths)
                    sound, _ = self._load_sound(sound_file_path, sample_rate)
                    sound_duration = len(sound) / sample_rate

                    fade_in_time = min(
                        sound_duration,
                        random.uniform(self.min_fade_in_time, self.max_fade_in_time),
                    )
                    fade_out_time = min(
                        sound_duration,
                        random.uniform(self.min_fade_out_time, self.max_fade_out_time),
                    )

                    sounds.append(
                        {
                            "fade_in_time": fade_in_time,
                            "start": current_time,
                            "end": current_time + sound_duration,
                            "fade_out_time": fade_out_time,
                            "file_path": sound_file_path,
                            "snr_in_db": random.uniform(
                                self.min_snr_in_db, self.max_snr_in_db
                            ),
                        }
                    )

                # wait until the last sound is done
                current_time += sound_duration

                # then add a pause
                pause_duration = random.uniform(
                    self.min_time_between_sounds, self.max_time_between_sounds
                )
                current_time += pause_duration

            self.parameters["sounds"] = sounds

    def apply(self, samples, sample_rate):
        num_samples = len(samples)
        noise_placeholder = np.zeros_like(samples)
        for sound_params in self.parameters["sounds"]:
            if sound_params["end"] < 0:
                # Skip a sound if it ended before the start of the input sound
                continue

            noise_samples, _ = self._load_sound(sound_params["file_path"], sample_rate)

            # Apply fade in and fade out
            noise_gain = np.ones_like(noise_samples)
            fade_in_time_in_samples = int(sound_params["fade_in_time"] * sample_rate)
            fade_in_mask = np.linspace(0.0, 1.0, num=fade_in_time_in_samples)
            fade_out_time_in_samples = int(sound_params["fade_out_time"] * sample_rate)
            fade_out_mask = np.linspace(1.0, 0.0, num=fade_out_time_in_samples)
            noise_gain[: fade_in_mask.shape[0]] = fade_in_mask
            noise_gain[-fade_out_mask.shape[0] :] = np.minimum(
                noise_gain[-fade_out_mask.shape[0] :], fade_out_mask
            )
            noise_samples = noise_samples * noise_gain

            start_sample_index = int(sound_params["start"] * sample_rate)
            end_sample_index = start_sample_index + len(noise_samples)

            if start_sample_index < 0:
                # crop noise_samples: shave off a chunk in the beginning
                num_samples_to_shave_off = abs(start_sample_index)
                noise_samples = noise_samples[num_samples_to_shave_off:]
                start_sample_index = 0

            if end_sample_index > num_samples:
                # crop noise_samples: shave off a chunk in the end
                num_samples_to_shave_off = end_sample_index - num_samples
                noise_samples = noise_samples[
                    : len(noise_samples) - num_samples_to_shave_off
                ]
                end_sample_index = num_samples

            clean_rms = calculate_rms(samples[start_sample_index:end_sample_index])
            noise_rms = calculate_rms(noise_samples)
            if noise_rms > 0:
                desired_noise_rms = calculate_desired_noise_rms(
                    clean_rms, sound_params["snr_in_db"]
                )

                # Adjust the noise to match the desired noise RMS
                noise_samples = noise_samples * (desired_noise_rms / noise_rms)

                noise_placeholder[start_sample_index:end_sample_index] += noise_samples
        # Return a mix of the input sound and the added sounds
        return samples + noise_placeholder

    def __getstate__(self):
        state = self.__dict__.copy()
        warnings.warn(
            "Warning: the LRU cache of AddShortNoises gets discarded when pickling it."
            " E.g. this means the cache will not be used when using AddShortNoises together"
            " with multiprocessing on Windows"
        )
        del state["_load_sound"]
        return state


@add_transform('polarity_inversion')
class PolarityInversion(BaseWaveformTransform):
    """
    Flip the audio samples upside-down, reversing their polarity. In other words, multiply the
    waveform by -1, so negative values become positive, and vice versa. The result will sound
    the same compared to the original when played back in isolation. However, when mixed with
    other audio sources, the result may be different. This waveform inversion technique
    is sometimes used for audio cancellation or obtaining the difference between two waveforms.
    However, in the context of audio data augmentation, this transform can be useful when
    training phase-aware machine learning models.
    """

    abbr = 'polarity_inversion'
    supports_multichannel = True

    def apply(self, samples, sample_rate):
        return -samples


@add_transform('gain')
class Gain(BaseWaveformTransform):
    """
    Multiply the audio by a random amplitude factor to reduce or increase the volume. This
    technique can help a model become somewhat invariant to the overall gain of the input audio.

    Warning: This transform can return samples outside the [-1, 1] range, which may lead to
    clipping or wrap distortion, depending on what you do with the audio in a later stage.
    See also https://en.wikipedia.org/wiki/Clipping_(audio)#Digital_clipping
    """
    abbr = 'gain'
    supports_multichannel = True

    def __init__(self, min_gain_in_db=-12, max_gain_in_db=12, p=1.0):
        """
        :param p: The probability of applying this transform
        """
        super().__init__(p)
        assert min_gain_in_db <= max_gain_in_db
        self.min_gain_in_db = min_gain_in_db
        self.max_gain_in_db = max_gain_in_db

    def randomize_parameters(self, samples, sample_rate, accumulate_meta=None):
        super().randomize_parameters(samples, sample_rate, accumulate_meta)
        if self.parameters["should_apply"]:
            self.parameters["amplitude_ratio"] = convert_decibels_to_amplitude_ratio(
                random.uniform(self.min_gain_in_db, self.max_gain_in_db)
            )

    def apply(self, samples, sample_rate):
        return samples * self.parameters["amplitude_ratio"]


@add_transform('reverse')
class Reverse(BaseWaveformTransform):
    """
    Reverse the audio. Also known as time inversion. Inversion of an audio track along its time
    axis relates to the random flip of an image, which is an augmentation technique that is
    widely used in the visual domain. This can be relevant in the context of audio
    classification. It was successfully applied in the paper
    AudioCLIP: Extending CLIP to Image, Text and Audio
    https://arxiv.org/pdf/2106.13043.pdf
    """

    abbr = 'reverse'
    supports_multichannel = True

    def apply(self, samples, sample_rate):
        if len(samples.shape) > 1:
            return np.fliplr(samples)
        else:
            return np.flipud(samples)


@add_transform('tanh_distortion')
class TanhDistortion(BaseWaveformTransform):
    """
    Apply tanh (hyperbolic tangent) distortion to the audio. This technique is sometimes
    used for adding distortion to guitar recordings. The tanh() function can give a rounded
    "soft clipping" kind of distortion, and the distortion amount is proportional to the
    loudness of the input and the pre-gain. Tanh is symmetric, so the positive and
    negative parts of the signal are squashed in the same way. This transform can be
    useful as data augmentation because it adds harmonics. In other words, it changes
    the timbre of the sound.

    See this page for examples: http://gdsp.hf.ntnu.no/lessons/3/17/
    """

    abbr = 'tanh_distortion'
    supports_multichannel = True

    def __init__(self, min_distortion_gain=1.0, max_distortion_gain=2.0, p=1.0):
        """
        :param min_distortion_gain: Min pre-gain factor
        :param max_distortion_gain: Max pre-gain factor
        :param p: The probability of applying this transform
        """
        super().__init__(p)
        self.min_distortion_gain = min_distortion_gain
        self.max_distortion_gain = max_distortion_gain

    def randomize_parameters(self, samples, sample_rate, accumulate_meta=None):
        super().randomize_parameters(samples, sample_rate, accumulate_meta)
        if self.parameters["should_apply"]:
            self.parameters["max_amplitude"] = np.amax(np.abs(samples))
            self.parameters["gain"] = random.uniform(
                self.min_distortion_gain, self.max_distortion_gain
            )

    def apply(self, samples, sample_rate):
        if self.parameters["max_amplitude"] > 0:
            distorted_samples = np.tanh(self.parameters["gain"] * samples)
            distorted_samples = (
                calculate_rms(distorted_samples) / calculate_rms(samples)
            ) * distorted_samples
        else:
            distorted_samples = samples
        return distorted_samples


@add_transform('low_pass_filter')
class LowPassFilter(BaseWaveformTransform):
    """
    Apply low-pass filtering to the input audio. The signal will be reduced by 6 dB per
    octave above the cutoff frequency, so this filter is fairly gentle.
    """
    abbr = 'lowpassfilter'
    supports_multichannel = False

    def __init__(
        self,
        min_cutoff_freq=150,
        max_cutoff_freq=7500,
        p: float = 0.5,
    ):
        """
        :param min_cutoff_freq: Minimum cutoff frequency in hertz
        :param max_cutoff_freq: Maximum cutoff frequency in hertz
        :param p: The probability of applying this transform
        """
        super().__init__(p)

        self.min_cutoff_freq = min_cutoff_freq
        self.max_cutoff_freq = max_cutoff_freq
        if self.min_cutoff_freq > self.max_cutoff_freq:
            raise ValueError("min_cutoff_freq must not be greater than max_cutoff_freq")

    def randomize_parameters(self, samples, sample_rate, accumulate_meta=None):
        super().randomize_parameters(samples, sample_rate, accumulate_meta)

        self.parameters["cutoff_freq"] = np.random.uniform(
            low  = self.min_cutoff_freq,
            high = self.max_cutoff_freq
        )

    def apply(self, samples: np.array, sample_rate: int = None):
        try:
            import pydub
        except ImportError:
            print(
                "Failed to import pydub. Maybe it is not installed? "
                "To install the optional pydub dependency of .,"
                " do `pip install .[extras]` instead of"
                " `pip install .`",
                file=sys.stderr,
            )
            raise

        assert samples.dtype == np.float32

        int_samples = convert_float_samples_to_int16(samples)

        audio_segment = pydub.AudioSegment(
            int_samples.tobytes(),
            frame_rate=sample_rate,
            sample_width=int_samples.dtype.itemsize,
            channels=1,
        )

        audio_segment = pydub.effects.low_pass_filter(
            audio_segment, self.parameters["cutoff_freq"]
        )
        samples = convert_int16_samples_to_float(np.array(audio_segment.get_array_of_samples()))
        return samples


@add_transform('high_pass_filter')
class HighPassFilter(BaseWaveformTransform):
    """
    Apply high-pass filtering to the input audio. The signal will be reduced by 6 dB per
    octave below the cutoff frequency, so this filter is fairly gentle.
    """
    abbr = 'highpassfilter'
    supports_multichannel = False

    def __init__(
        self,
        min_cutoff_freq=20,
        max_cutoff_freq=2400,
        p: float = 0.5
    ):
        """
        :param min_cutoff_freq: Minimum cutoff frequency in hertz
        :param max_cutoff_freq: Maximum cutoff frequency in hertz
        :param p: The probability of applying this transform
        """
        super().__init__(p)

        self.min_cutoff_freq = min_cutoff_freq
        self.max_cutoff_freq = max_cutoff_freq
        if self.min_cutoff_freq > self.max_cutoff_freq:
            raise ValueError("min_cutoff_freq must not be greater than max_cutoff_freq")

    def randomize_parameters(self, samples, sample_rate, accumulate_meta=None):
        super().randomize_parameters(samples, sample_rate, accumulate_meta)

        self.parameters["cutoff_freq"] = np.random.uniform(
            low=self.min_cutoff_freq,
            high=self.max_cutoff_freq
        )

    def apply(self, samples: np.array, sample_rate: int = None):
        try:
            import pydub
        except ImportError:
            print(
                "Failed to import pydub. Maybe it is not installed? "
                "To install the optional pydub dependency of .,"
                " do `pip install .[extras]` instead of"
                " `pip install .`",
                file=sys.stderr,
            )
            raise

        assert samples.dtype == np.float32

        int_samples = convert_float_samples_to_int16(samples)

        audio_segment = pydub.AudioSegment(
            int_samples.tobytes(),
            frame_rate=sample_rate,
            sample_width=int_samples.dtype.itemsize,
            channels=1,
        )

        audio_segment = pydub.effects.high_pass_filter(
            audio_segment, self.parameters["cutoff_freq"]
        )
        samples = convert_int16_samples_to_float(np.array(audio_segment.get_array_of_samples()))
        return samples


@add_transform('base_pass_filter')
class BandPassFilter(BaseWaveformTransform):
    """
    Apply band-pass filtering to the input audio.
    """
    abbr = 'basepassfilter'
    supports_multichannel = False

    def __init__(
        self,
        min_center_freq=100.0,
        max_center_freq=1000.0,
        min_q=1.0,
        max_q=2.0,
        p=1.0
    ):
        """
        :param min_center_freq: Minimum center frequency in hertz
        :param max_center_freq: Maximum center frequency in hertz
        :param min_q: Minimum ratio of center frequency to bandwidth
        :param max_q: Maximum ratio of center frequency to bandwidth
        :param p: The probability of applying this transform
        """
        super().__init__(p)

        self.min_center_freq = min_center_freq
        self.max_center_freq = max_center_freq
        self.min_q = min_q
        self.max_q = max_q
        if self.min_center_freq > self.max_center_freq:
            raise ValueError("min_center_freq must not be greater than max_center_freq")
        if self.min_q > self.max_q:
            raise ValueError("min_q must not be greater than max_q")

    def randomize_parameters(self, samples, sample_rate, accumulate_meta=None):
        super().randomize_parameters(samples, sample_rate, accumulate_meta)

        self.parameters["center_freq"] = np.random.uniform(
            low=self.min_center_freq,
            high=self.max_center_freq
        )
        self.parameters["q"] = np.random.uniform(
            low=self.min_q,
            high=self.max_q
        )

    def apply(self, samples: np.array, sample_rate: int = None):
        try:
            import pydub
        except ImportError:
            print(
                "Failed to import pydub. Maybe it is not installed? "
                "To install the optional pydub dependency of .,"
                " do `pip install .[extras]` instead of"
                " `pip install .`",
                file=sys.stderr,
            )
            raise

        assert samples.dtype == np.float32

        int_samples = convert_float_samples_to_int16(samples)

        audio_segment = pydub.AudioSegment(
            int_samples.tobytes(),
            frame_rate=sample_rate,
            sample_width=int_samples.dtype.itemsize,
            channels=1,
        )

        audio_segment = pydub.effects.low_pass_filter(
            audio_segment, self.parameters["center_freq"] * (1 - 0.5 / self.parameters["q"])
        )
        audio_segment = pydub.effects.high_pass_filter(
            audio_segment, self.parameters["center_freq"] * (1 + 0.5 / self.parameters["q"])
        )
        samples = convert_int16_samples_to_float(np.array(audio_segment.get_array_of_samples()))

        return samples


class Mp3Compression(BaseWaveformTransform):
    """Compress the audio using an MP3 encoder to lower the audio quality.
    This may help machine learning models deal with compressed, low-quality audio.

    This transform depends on either lameenc or pydub/ffmpeg.

    Note that bitrates below 32 kbps are only supported for low sample rates (up to 24000 hz).

    Note: When using the lameenc backend, the output may be slightly longer than the input due
    to the fact that the LAME encoder inserts some silence at the beginning of the audio.

    Warning: This transform writes to disk, so it may be slow. Ideally, the work should be done
    in memory. Contributions are welcome.
    """

    abbr = 'mp3compress'
    SUPPORTED_BITRATES = [
        8,
        16,
        24,
        32,
        40,
        48,
        56,
        64,
        80,
        96,
        112,
        128,
        144,
        160,
        192,
        224,
        256,
        320,
    ]

    def __init__(
        self, min_bitrate: int = 8, max_bitrate: int = 64, backend: str = "pydub", p=1.0
    ):
        """
        :param min_bitrate: Minimum bitrate in kbps
        :param max_bitrate: Maximum bitrate in kbps
        :param backend: "pydub" or "lameenc".
            Pydub may use ffmpeg under the hood.
                Pros: Seems to avoid introducing latency in the output.
                Cons: Slower than lameenc.
            lameenc:
                Pros: You can set the quality parameter in addition to bitrate.
                Cons: Seems to introduce some silence at the start of the audio.
        :param p: The probability of applying this transform
        """
        super().__init__(p)
        assert self.SUPPORTED_BITRATES[0] <= min_bitrate <= self.SUPPORTED_BITRATES[-1]
        assert self.SUPPORTED_BITRATES[0] <= max_bitrate <= self.SUPPORTED_BITRATES[-1]
        assert min_bitrate <= max_bitrate
        self.min_bitrate = min_bitrate
        self.max_bitrate = max_bitrate
        assert backend in ("pydub", "lameenc")
        self.backend = backend

    def randomize_parameters(self, samples, sample_rate, accumulate_meta=None):
        super().randomize_parameters(samples, sample_rate, accumulate_meta)
        if self.parameters["should_apply"]:
            bitrate_choices = [
                bitrate
                for bitrate in self.SUPPORTED_BITRATES
                if self.min_bitrate <= bitrate <= self.max_bitrate
            ]
            self.parameters["bitrate"] = random.choice(bitrate_choices)

    def apply(self, samples, sample_rate):
        if self.backend == "lameenc":
            return self.apply_lameenc(samples, sample_rate)
        elif self.backend == "pydub":
            return self.apply_pydub(samples, sample_rate)
        else:
            raise Exception("Backend {} not recognized".format(self.backend))

    def apply_lameenc(self, samples, sample_rate):
        try:
            import lameenc
        except ImportError:
            print(
                "Failed to import the lame encoder. Maybe it is not installed? "
                "To install the optional lameenc dependency of .,"
                " do `pip install .[extras]` instead of"
                " `pip install .`",
                file=sys.stderr,
            )
            raise

        assert len(samples.shape) == 1
        assert samples.dtype == np.float32

        int_samples = convert_float_samples_to_int16(samples)

        encoder = lameenc.Encoder()
        encoder.set_bit_rate(self.parameters["bitrate"])
        encoder.set_in_sample_rate(sample_rate)
        encoder.set_channels(1)
        encoder.set_quality(7)  # 2 = highest, 7 = fastest
        encoder.silence()

        mp3_data = encoder.encode(int_samples.tobytes())
        mp3_data += encoder.flush()

        # Write a temporary MP3 file that will then be decoded
        tmp_dir = tempfile.gettempdir()
        tmp_file_path = os.path.join(
            tmp_dir, "tmp_compressed_{}.mp3".format(str(uuid.uuid4())[0:12])
        )
        with open(tmp_file_path, "wb") as f:
            f.write(mp3_data)

        degraded_samples, _ = librosa.load(tmp_file_path, sample_rate)

        os.unlink(tmp_file_path)

        return degraded_samples

    def apply_pydub(self, samples, sample_rate):
        try:
            import pydub
        except ImportError:
            print(
                "Failed to import pydub. Maybe it is not installed? "
                "To install the optional pydub dependency of .,"
                " do `pip install .[extras]` instead of"
                " `pip install .`",
                file=sys.stderr,
            )
            raise

        assert len(samples.shape) == 1
        assert samples.dtype == np.float32

        int_samples = convert_float_samples_to_int16(samples)

        audio_segment = pydub.AudioSegment(
            int_samples.tobytes(),
            frame_rate=sample_rate,
            sample_width=int_samples.dtype.itemsize,
            channels=1,
        )

        tmp_dir = tempfile.gettempdir()
        tmp_file_path = os.path.join(
            tmp_dir, "tmp_compressed_{}.mp3".format(str(uuid.uuid4())[0:12])
        )

        bitrate_string = "{}k".format(self.parameters["bitrate"])
        file_handle = audio_segment.export(tmp_file_path, bitrate=bitrate_string)
        file_handle.close()

        degraded_samples, _ = librosa.load(tmp_file_path, sample_rate)

        os.unlink(tmp_file_path)

        return degraded_samples

