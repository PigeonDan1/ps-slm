import os
import random
import json
import traceback
import numpy as np
from copy import deepcopy
from loguru import logger
from pathlib import Path
from scipy.io import wavfile
from itertools import islice
from asr.utils import slurm
from .parse_args import parse_args
from .core.utils import timer
from .core.audio_loading_utils import load_wav_file, read_wav_ark
from .core.composition import Compose
from .core.transforms_interface import build_transform


def load_input_wav_list(key, wav_paths, mono=True):
    # samples_sample_rates_list = [load_wav_file(
    #             wav_path, sample_rate=None, mono=mono
    # ) for wav_path in wav_paths]
    samples_sample_rates_list = [read_wav_ark(
                wav_path, mono=mono
    ) for wav_path in wav_paths]

    error_sample_rates_items = list(filter(lambda x: x[1] != 16000, samples_sample_rates_list))

    if len(error_sample_rates_items) != 0:
        logger.warning(f'In Key {key} sample rate error:' \
        f'{[error_item for error_item in  error_sample_rates_items]}, skip!')
        return None, None

    samples_list, sample_rates_list = map(list, zip(*samples_sample_rates_list))
    return samples_list, sample_rates_list

def check_wav_paths(key, wav_paths):
    for wav_path in wav_paths:
        if wav_path.strip().rsplit(':', 1)[0][-4:] == ".ark":
            wav_path = wav_path.strip().rsplit(':', 1)[0]
        if not Path(wav_path).exists():
            logger.warning(f'In {key},  {wav_path} not exists')
            return False
    return True


def main():

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # make transform
    augmenter = Compose([build_transform(tfm) for tfm in args.transforms])
    logger.info(f'Augment chain: {augmenter}')

    if args.archive:
        (args.output_dir / f'{slurm.rank}').mkdir(parents=True, exist_ok=True)
        from asr.data.dataset import kaldi_io
        out_ark_fn = args.output_dir / f'{slurm.rank}' / 'wav.ark'
        out_ark_fn = out_ark_fn.absolute()
        out_ark = kaldi_io.open_or_fd(str(out_ark_fn), mode='wb')
        out_scp_fn = args.output_dir / f'{slurm.rank}' / 'wav.scp'
        out_scp = kaldi_io.open_or_fd(str(out_scp_fn), mode='w')


    # Re-produce
    if args.meta is not None:

        meta_list = [json.loads(meta) for meta in open(args.meta)]
        assert len(meta_list) > 0, 'input_dir contain no audio'
        logger.info(f'use {args.meta} to re-produce {len(meta_list)} wavs')
        input_meta_list = islice(meta_list, slurm.rank, None, slurm.world_size)

        for meta in input_meta_list:

            key = meta['transforms'][0]['key']
            wav_paths = meta['transforms'][1]['audio']

            with timer(description=f'Process {key}', verbose=True):

                samples_list, sample_rates_list = load_input_wav_list(key, wav_paths, mono=args.mono)

                if samples_list is None:
                    continue

                samples_list = [samples.squeeze().transpose() for samples in samples_list] # change to [channel, sample]

                augmented_samples_list, _ = augmenter(
                    samples_list=samples_list, sample_rates_list=sample_rates_list,
                    meta=meta['transforms'][2:], reproduce=True,
                )
                # augmented_samples_list is List or np.ndarray, make it consistent
                if type(augmented_samples_list) == np.ndarray:
                    augmented_samples_list = [augmented_samples_list]



                for augmented_samples in augmented_samples_list:
                    augmented_samples = augmented_samples.squeeze().transpose()  # change to [sample, channel]

                    if args.archive:
                        output_file_path = out_ark
                        key = Path(meta['output']).stem
                        out_ark.write((key + ' ').encode("utf-8"))
                        out_scp.write(f'{key} {out_ark_fn}:{out_ark.tell()}\n')
                        from .core.audio_loading_utils import write_wav_ark
                        write_wav_ark(out_ark, 16000, data=(augmented_samples*32768).astype(np.int16))
                        out_scp.flush()
                        out_ark.flush()
                    else:
                        out_file_name = Path(meta['output']).name
                        output_file_path = args.output_dir / f'{slurm.rank}'/ out_file_name
                        # int16: 32768, int32: 2147483648
                        wavfile.write(
                                output_file_path, rate=sample_rates_list[0], data=(augmented_samples*32768).astype(np.int16)
                        )

    else:
        # check input
        input_wav = [(wav_list.split()[0], wav_list.split()[1:]) for wav_list in open(args.input_scp)]
        assert len(input_wav) > 0, 'input_dir contain no audio'
        logger.info(f'found {len(input_wav)} input wav items')
        # generate process wav list
        input_wav =[(f'{key}', wav_path) for key, wav_path in input_wav]
        input_wav_list =list(islice(input_wav, slurm.rank, None, slurm.world_size))

        # augmentation
        success = 0
        (args.output_dir / f'{slurm.rank}').mkdir(parents=True, exist_ok=True)
        meta_out_f = (args.output_dir / f'{slurm.rank}' / 'meta.txt').open(mode='w')
        for key, wav_paths in input_wav_list:

            if not check_wav_paths(key, wav_paths):
                continue

            with timer(description=f'Process {key}', verbose=True):

                try:

                    samples_list, sample_rates_list = load_input_wav_list(key, wav_paths, mono=args.mono)

                    samples_list = [samples.squeeze().transpose() for samples in samples_list]  # change to [channel, sample]


                    transforms_meta = [{'name': 'key', 'key': key}]
                    transforms_meta.append({'name': 'RecordName', 'audio': [wav_path for wav_path in wav_paths]})

                    augmented_samples_list, transforms_meta = augmenter(
                        samples_list=samples_list, sample_rates_list=sample_rates_list, meta=transforms_meta,
                    )

                    # augmented_samples_list is List or np.ndarray, make it consistent
                    if type(augmented_samples_list) == np.ndarray:
                        augmented_samples_list = [augmented_samples_list]

                    output_file_paths = []
                    for i, augmented_samples in enumerate(augmented_samples_list):
                        if augmented_samples is None or len(augmented_samples) == 0:
                            logger.warning(f'{key} has no augmented samples')
                            continue
                        if len(augmented_samples_list) > 1:
                            out_file_name = f'{transforms_meta[0]["key"]}_{i}.wav'
                        else:
                            out_file_name = f'{transforms_meta[0]["key"]}.wav'
                        output_file_path = args.output_dir / f'{slurm.rank}'/  out_file_name

                        augmented_samples = augmented_samples.squeeze().transpose()  # change to [sample, channel]

                        if args.archive:
                            output_file_path = out_ark
                            out_ark.write((key + ' ').encode("utf-8"))
                            out_scp.write(f'{key} {out_ark_fn}:{out_ark.tell()}\n')
                            from .core.audio_loading_utils import write_wav_ark
                            write_wav_ark(out_ark, 16000, data=(augmented_samples*32768).astype(np.int16))
                            out_scp.flush()
                            out_ark.flush()
                        else:
                            # int16: 32768, int32: 2147483648
                            wavfile.write(
                                    output_file_path, rate=sample_rates_list[0], data=(augmented_samples*32768).astype(np.int16)
                            )
                            output_file_paths.append(str(output_file_path))
                    meta = {
                        'transforms': deepcopy(transforms_meta),
                        'output': ",".join(output_file_paths)
                    }
                    meta = json.dumps(meta)
                    meta_out_f.write(f'{meta}\n')
                    meta_out_f.flush()
                    success += 1

                except Exception as e:
                    logger.error(key)
                    traceback.print_exc()
        if args.archive:
            out_ark.close()
            out_scp.close()
        logger.info(f'Total {len(input_wav_list)} wavs to be augmented, {success} succeed.')

    if args.archive:
        out_ark.close()
        out_scp.close()


if __name__ == '__main__':
    main()
