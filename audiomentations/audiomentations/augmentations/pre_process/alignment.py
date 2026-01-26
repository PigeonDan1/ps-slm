import lmdb
import numpy as np
import pickle
import torch
import torchaudio

from kaldi.alignment import GmmAligner
from kaldi.fstext import SymbolTable
from kaldi.feat.functions import splice_frames
from kaldi.transform.cmvn import Cmvn
from kaldi.matrix import Matrix
from kaldi.util.io import read_matrix

from asr.utils import slurm

from ...core.transforms_interface import add_transform
from .base import PreProcess


@add_transform('alignment')
class Aligner(PreProcess):
    """Alignment wav feature with text
    """
    abbr = 'alignment'
    supports_multichannel = False

    def __init__(self, units, cmds_dict, gmm, boost_silence, silence_id, tree, fst, words, disambig,
                 self_loop_scale, phones, mfcc_conf, frame_shift=20, apply_cmvn=True,
                 splice_conf=(4, 4), transform_mat=None, cache_env=None, p=1.0):
        """
        :param units: str, Path to unit dict
        :param cmds_dict: str, Path to cmds dict
        :param gmm: str, Path to gmm model
        :param boost_silence: float, value of gmm-boost-silence
        :param silence_id: int, silence id
        :param tree: str, Path to tree
        :param fst: str, Path to fst
        :param words: str, Path words
        :param disambig: str, Path to disambig
        :param self_loop_scale: float, Loop scale
        :param phones: str, Path to phones
        :param mfcc_conf: str, mfcc config
        :param frame_shift: int, frame shift
        :param apply_cmvn: str, whether apply cmvn
        :param splice_conf: str, whether apply cmvn, and the splice conf if so
        :param transform_mat: str, Path to transform matrix
        :param cache_env: str, Path to lmdb cache
        :param p: The probability of applying this transform
        """
        super().__init__(p)
        with open(units) as f:
            self.dict = dict([line.strip().split() for line in f.readlines()])
        with open(cmds_dict) as f:
            self.cmds_dict = dict([line.strip().split(' ', 1) for line in f.readlines()])
        # NOTE(menglong.xu): we can not use gmm-boost-silence now because it not in the system environment
        # self.aligner = GmmAligner.from_files(f'gmm-boost-silence --boost={boost_silence} {silence_id} {gmm} - |',
        #                                      tree, fst, words, disambig,
        #                                      self_loop_scale=self_loop_scale)
        self.aligner = GmmAligner.from_files(gmm, tree, fst, words, disambig,
                                             self_loop_scale=self_loop_scale)
        self.phones = SymbolTable.read_text(phones)
        self.mfcc_conf = mfcc_conf
        self.frame_shift = frame_shift
        self.apply_cmvn = apply_cmvn
        self.splice_conf = splice_conf
        self.transform_mat = read_matrix(transform_mat) if transform_mat else None
        cache_env = f'{cache_env}/{slurm.rank+1}-{slurm.world_size}'
        self.lmdb_env = lmdb.open(cache_env, map_size=2**30*1000)  # 1000GB

    def wav_to_feat(self, wave):
        feat = torchaudio.compliance.kaldi.mfcc(wave, **self.mfcc_conf)
        feat = Matrix(feat.numpy())
        if self.apply_cmvn:
            cmvn = Cmvn(feat.num_cols)
            cmvn.accumulate(feat)
            cmvn.apply(feat)
        if self.splice_conf:
            feat = splice_frames(feat, *self.splice_conf)
        if self.transform_mat:
            feat = np.dot(feat.numpy(), self.transform_mat.numpy().transpose())
        return Matrix(feat)

    def align(self, wave, text):
        feat = self.wav_to_feat(wave)
        out = self.aligner.align(feat, text)
        phone_alignment = self.aligner.to_phone_alignment(out["alignment"], self.phones)
        in_text = text
        text = self.cmds_dict[in_text.replace(' ', '')].split()
        ali = []
        string = ''
        length = 0
        for phone, start_index, _len in phone_alignment:
            if phone == 'sil':
                ali.extend([self.dict['py410']] * _len)
            else:
                # kaldi model somtimes may mistakenly identify silence as a initial consonant,
                # so we need a trick to deal with this case
                if self.dict['py410'] not in ali:
                    ali.extend([self.dict['py410']] * (_len - 10))
                    _len = min(10, _len)
                string += ''.join([c for c in phone if c.isalpha()])
                length += _len
                if string != text[0]:
                    continue
                assert string in self.dict, f'{string} not in dict'
                ali.extend([self.dict[string]] * length)
                string = ''
                length = 0
                text = text[1:]
        assert length == 0, f'{in_text} is not correct!'
        _step = int(self.frame_shift // 10)
        return ali[::_step]

    def apply(self, samples, sample_rate, other_input_output):
        other_input_output['name'] = 'alignment'
        wave = np.multiply(samples, 32768, dtype=np.float32)
        wave = torch.from_numpy(wave.reshape(1, -1))
        wav_len = (wave.shape[-1] / sample_rate * 1000 / self.frame_shift)

        cache_key = other_input_output['uttid2oriwavnames']
        with self.lmdb_env.begin() as txn:
            # 尝试从LMDB中读取缓存
            cached_data = txn.get(cache_key.encode())
        if cached_data is not None:
            # 如果存在缓存，反序列化数据
            ali = pickle.loads(cached_data)
        else:
            with self.lmdb_env.begin(write=True) as txn:
                # 如果没有缓存，执行对齐并存储结果
                ali = self.align(wave, ' '.join(list(other_input_output['text'])))
                txn.put(cache_key.encode(), pickle.dumps(ali))
        assert abs(wav_len - len(ali)) < 3, (wav_len, len(ali))
        other_input_output['kaldi_ali'] = ' '.join(ali)
        return samples
