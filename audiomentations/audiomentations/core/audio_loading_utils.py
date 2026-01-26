from logging import log
import warnings
from pathlib import Path

import re
import struct
import librosa
from librosa.core.harmonic import salience
from loguru import logger
import numpy as np
from scipy.io import wavfile

from asr.data import kaldi_io
import soundfile as sf

IS_WAVIO_INSTALLED = True
try:
    import wavio
except ImportError:
    IS_WAVIO_INSTALLED = False

def check_mono(mono, samples):
    if mono and len(samples.shape) > 1:
        if samples.shape[1] == 1:
            samples = samples[:, 0]
        else:
            samples = np.mean(samples, axis=1)
    return samples

def check_sample_rate(actual_sample_rate, samples, sample_rate, file_path, resample_type='auto'):
    if sample_rate is not None and actual_sample_rate != sample_rate:
        if resample_type == "auto":
            resample_type = (
                "kaiser_fast" if actual_sample_rate < sample_rate else "kaiser_best"
            )
        samples = librosa.resample(
            samples, actual_sample_rate, sample_rate, res_type=resample_type
        )
        warnings.warn(
            "{} had to be resampled from {} hz to {} hz. This hurt execution time.".format(
                str(file_path), actual_sample_rate, sample_rate
            )
        )
    actual_sample_rate = actual_sample_rate if sample_rate is None else sample_rate
    return actual_sample_rate, samples


def load_sound_file(file_path, sample_rate, mono=True, resample_type="auto"):
    """
    Load an audio file as a floating point time series. Audio will be automatically
    resampled to the given sample rate.

    :param file_path: str or Path instance that points to a sound file
    :param sample_rate: If not None, resample to this sample rate
    :param mono: If True, mix any multichannel data down to mono, and return a 1D array
    :param resample_type: "auto" means use "kaiser_fast" when upsampling and "kaiser_best" when
        downsampling
    """
    file_path = Path(file_path)
    if file_path.name.lower().endswith(".wav"):
        # Use librosa for loading most wav files
        try:
            return load_wav_file(
                file_path, sample_rate, mono, resample_type=resample_type
            )
        except Exception as e:
            # scipy<1.6.0 does not natively support 24-bit wavs, so we use wavio.
            if "the wav file has 24-bit data" in str(e):
                if IS_WAVIO_INSTALLED:
                    return load_wav_file_with_wavio(
                        file_path, sample_rate, mono, resample_type=resample_type
                    )
                else:
                    warnings.warn(
                        "You are loading a 24-bit wav file, and librosa is not very fast at"
                        " doing that. Install wavio for a performance boost. To install the"
                        " optional wavio dependency of audiomentations,"
                        " do `pip install audiomentations[extras]` instead of"
                        " `pip install audiomentations`"
                    )
            elif "Unknown wave file format" in str(e):
                # This can happen if the file is in MS ADPCM format
                logger.error(f'{file_path} load error. Maybe the file is in MS ADPCM format')
                raise e
            else:
                logger.error(f'{file_path} load error.')
                raise e
    samples, actual_sample_rate = librosa.load(
        str(file_path), sr=None, mono=mono, dtype=np.float32
    )

    actual_sample_rate, samples = check_sample_rate(actual_sample_rate, 
                                                    samples, sample_rate, file_path,
                                                    resample_type=resample_type)

    samples = samples.transpose()  # librosa return [channel, sample]

    if mono:
        assert len(samples.shape) == 1
    return samples, actual_sample_rate


def load_wav_file(file_path, sample_rate, mono=True, resample_type="kaiser_best"):
    """Load a wav audio file as a floating point time series. Significantly faster than
    load_sound_file."""
 
    samples, actual_sample_rate = librosa.load(file_path, mono=mono, sr=sample_rate) # sample in [(channel), time]
     
    if mono:
        if samples.ndim > 1:
            samples = samples[0]
    
    # transpose to shape (time channel)
    if samples.ndim == 2 and samples.shape[1] > samples.shape[0]:
        samples = samples.transpose(1,0)

    if samples.dtype == np.float64:
        samples = samples.astype(np.float32)
        
    if samples.dtype != np.float32:
        if samples.dtype == np.int16:
            samples = np.true_divide(
                samples, 32768, dtype=np.float32
            )  # ends up roughly between -1 and 1
        elif samples.dtype == np.int32:
            samples = np.true_divide(
                samples, 2147483648, dtype=np.float32
            )  # ends up roughly between -1 and 1
        else:
            # TODO: Add support for 24-bit loading in scipy>=1.6.0
            raise Exception("Unexpected data type")

    samples = check_mono(mono, samples)

    actual_sample_rate, samples = check_sample_rate(actual_sample_rate, 
                                                    samples, sample_rate, file_path,
                                                    resample_type=resample_type)

    return samples, actual_sample_rate


def load_wav_file_with_wavio(
    file_path, sample_rate, mono=True, resample_type="kaiser_best"
):
    """Load a 24-bit wav audio file as a floating point time series. Significantly faster than
    load_sound_file."""

    wavio_obj = wavio.read(str(file_path))
    samples = wavio_obj.data
    actual_sample_rate = wavio_obj.rate

    if samples.dtype != np.float32:
        if wavio_obj.sampwidth == 3:
            samples = np.true_divide(
                samples, 8388608, dtype=np.float32
            )  # ends up roughly between -1 and 1
        elif wavio_obj.sampwidth == 2:
            samples = np.true_divide(
                samples, 32768, dtype=np.float32
            )  # ends up roughly between -1 and 1
        else:
            raise Exception("Unknown sampwidth")

    samples = check_mono(mono, samples)

    actual_sample_rate, samples = check_sample_rate(actual_sample_rate, 
                                                    samples, sample_rate, file_path,
                                                    resample_type=resample_type)

    return samples, actual_sample_rate


def write_wav_ark(fd, sample_rate: int, data: np.ndarray):
    # data of shape [samples, channel]
    assert data.dtype == np.int16
    assert hasattr(fd, "write")

    bit_depth = data.dtype.itemsize * 8
    if data.ndim == 1:
        n_channels = 1
    elif data.ndim == 2:
        n_channels = data.shape[1]
    bytes_per_second = sample_rate * (bit_depth // 8) * n_channels
    block_align = n_channels * (bit_depth // 8)
    data = data.tobytes()
    import struct
    mini_header = b'RIFF'
    mini_header += struct.pack('<L', 44 - 8 + len(data))
    mini_header += b'WAVE'
    mini_header += b'fmt '
    mini_header += struct.pack('<L', 16)
    mini_header += struct.pack('<H', 1) # 1 means PCM
    mini_header += struct.pack('<H', n_channels)
    mini_header += struct.pack('<L', sample_rate)
    mini_header += struct.pack('<L', bytes_per_second)
    mini_header += struct.pack('<H', block_align)
    mini_header += struct.pack('<H', bit_depth)
    mini_header += b'data'
    mini_header += struct.pack('<L', len(data))

    fd.write(mini_header)
    fd.write(data)


def read_wav_data(f, filename=''):
    stHeaderFields = {}
    buf = f.read(4) # 0-3: RIFF
    assert buf == b'RIFF', "Not the WAVE format file"

    buf = f.read(4) # 4-7: Chunk Size
    stHeaderFields['ChunkSize'] = struct.unpack('<L', buf)[0]

    buf = f.read(4) # 8-11: fmt
    stHeaderFields['Format'] = buf

    buf = f.read(4) # 12-15: fmt
    assert buf == b'fmt '

    buf = f.read(4) # 16-19: Subchunk1Size
    stHeaderFields['Subchunk1Size'] = struct.unpack('<L', buf)[0]
    if stHeaderFields['Subchunk1Size'] != 16:
        logger.debug(f"wav header is not 44 byte, subchunk size {stHeaderFields['Subchunk1Size']} != 16")
    buf = f.read(stHeaderFields['Subchunk1Size'])
    stHeaderFields['AudioFormat'] = struct.unpack('<H', buf[0:2])[0]
    stHeaderFields['NumChannels'] = struct.unpack('<H', buf[2:4])[0]
    stHeaderFields['SampleRate'] = struct.unpack('<L', buf[4:8])[0]
    stHeaderFields['ByteRate'] = struct.unpack('<L', buf[8:12])[0]
    stHeaderFields['BlockAlign'] = struct.unpack('<H', buf[12:14])[0]
    stHeaderFields['BitsPerSample'] = struct.unpack('<H', buf[14:16])[0]

    #if stHeaderFields['BitsPerSample'] != 16:
    #    logger.warning(f"wav header indicate unsafe bits_per_sample {stHeaderFields['BitsPerSample']} != 16")
    #if stHeaderFields['ByteRate'] != (stHeaderFields['SampleRate'] * stHeaderFields['BitsPerSample'] * stHeaderFields['NumChannels'] // 8):
    #    logger.warning(f"wav header indicate unsafe byte rate {stHeaderFields['ByteRate']} != {stHeaderFields['SampleRate']} * {stHeaderFields['BitsPerSample']} * {stHeaderFields['NumChannels']} // 8 (maybe you can use sox to fix it)")
    if stHeaderFields['BlockAlign'] != (stHeaderFields['NumChannels'] * stHeaderFields['BitsPerSample'] // 8):
        logger.warning(f"wav:{filename} header indicate unsafe block_align {stHeaderFields['BlockAlign']} != {stHeaderFields['NumChannels']} * {stHeaderFields['BitsPerSample']} // 8")

    buf = f.read(4) # 36-39: data chunk ID
    while buf != b'data':
        buf_size = f.read(4)
        buf_size = struct.unpack('<L', buf_size)[0]
        f.read(buf_size)
        logger.debug(f"read subcunk [{buf}] size of [{buf_size}]")
        buf = f.read(4)

    assert buf == b'data', f"chunk name [{buf}] is not data"
    buf = f.read(4)
    stHeaderFields['DataSize'] = struct.unpack('<L', buf)[0]
    # header end
    if re.search(r'\.ark:[0-9]+$', filename):
        data_buf = f.read(stHeaderFields['DataSize'])
    else:
        data_buf = f.read()
    if len(data_buf) != stHeaderFields['DataSize']:
        logger.warning(f"read wav sample length: {len(data_buf)} != header record size {stHeaderFields['DataSize']}")
    data = np.frombuffer(data_buf, dtype='int16') / 2**15
    uselen = data.shape[0] // stHeaderFields['NumChannels'] * stHeaderFields['NumChannels']

    return data[:uselen].reshape(-1, stHeaderFields['NumChannels']), stHeaderFields['SampleRate']


def read_wav_ark(file_or_fd, mono=True):
    fd = kaldi_io.open_or_fd(file_or_fd)
    try:
        # wav, sr = sf.read(fd, dtype='float32')
        wav, sr = read_wav_data(fd, file_or_fd)
        if sr != 16000:
            warnings.warn(f'{file_or_fd} sample rate error {sr}')
            # TODO: ugly
            do_tp = wav.shape[0] > wav.shape[1]
            if do_tp:
                wav = wav.transpose((1,0))
            wav = librosa.resample(y=wav, orig_sr=sr, target_sr=16000)
            sr = 16000
            if do_tp:
                wav = wav.transpose((1,0))
    finally:
        if fd is not file_or_fd:
            fd.close()

    if mono and len(wav.shape) > 1:
        wav = wav[:, 0]
    return wav, sr
