
import math
import numpy as np
from scipy.interpolate import interp1d
import xml.etree.ElementTree as ET
from pathlib import Path

from ...core.transforms_interface import add_transform
from .base import PreProcess
from demo.demo import DEMO_DIR



@add_transform('eq_simulation')
class EQSimulation(PreProcess):
    abbr = 'eq_simulation'

    def __init__(self, cfg=Path(DEMO_DIR) / 'eq_simu' / 'EQ_out_20240912_112859.xml', fftlen=1024, p=1.0):
        super().__init__(p=p)
        self.point_list, self.target_list = self.extract_parse_spec_gain(cfg)
        self.fftlen = fftlen
        self.frmInc = fftlen // 2
        self.fftbins = fftlen // 2 + 1
        self.sqrtHanning = np.zeros(fftlen, dtype=float)
        for i in range(fftlen):
            self.sqrtHanning[i] = np.sqrt(0.5 * (1 - math.cos((2 * 3.1415926 * (i + 1)) /
                                                     (fftlen + 1))))

    def extract_parse_spec_gain(self, cfg):
        # 加载XML文件
        tree = ET.parse(cfg)
        root = tree.getroot()
        point_list = []
        target_list = []

        # 遍历XML文件的元素
        for child in root:
            for son in child:
                point = son.get('f', 'unknown')
                target = son.get('d', 'unknown')
                point_list.append(point)
                target_list.append(target)

        return point_list, target_list


    def fft_audio(self, data, fftlen, sqrtHanning):
        inputlen = len(data)  # 信号总长度
        frameinc = fftlen // 2
        fftbins = fftlen // 2 + 1
        f_array = np.clip(data.astype(np.float32) / 32768, -1.0, 1.0)  # 限制大小截副
        nframesfloat = 1 + (inputlen - fftlen) * 1.0 / frameinc
        if np.fix(nframesfloat) == nframesfloat:
            nframes = int(nframesfloat)
        else:
            nframes = int(np.floor(nframesfloat) + 1)
            zeros = np.zeros((nframes - 1) * frameinc + fftlen - inputlen,dtype=float)  # 创建补零数组
            f_array = np.concatenate((f_array, zeros), axis=0)  # 对原有数组补零


        output = np.zeros((fftbins, nframes), dtype=complex)
        for i in range(nframes):
            xframe = f_array[frameinc * i: frameinc * i + fftlen]
            fftdata = np.fft.rfft(xframe * sqrtHanning)
            output[:, i] = fftdata
        return output

    def ifft_data(self, data, fftlen, sqrtHanning):
        nFrames = data.shape[1]  # 信号总长度
        frmInc = fftlen // 2

        iffdata = np.fft.irfft(data.T).real * sqrtHanning
        output = np.zeros(frmInc * nFrames + fftlen - frmInc,dtype=float)
        raw = np.zeros(frmInc * nFrames + fftlen - frmInc,dtype=float)
        for i in range(nFrames):
            k = frmInc * i
            raw[k : k + fftlen] = raw[k : k + fftlen] + iffdata[i,:]
            output[k : k + frmInc] = raw[k : k + frmInc]
        output[frmInc * nFrames:] = raw[frmInc * nFrames:]
        return output


    def apply(self, samples, sample_rate):

        if samples.ndim == 1:
            samples = samples[np.newaxis, :]

        if samples.shape[0] > samples.shape[1]:
            samples = samples.T
        nchannels = samples.shape[0]
        nframes = samples.shape[1]
        samples = np.rint(samples * 32767)
        print(f'samples: {samples}')
        EQoutAudio = np.zeros(nchannels * nframes)
        dbdict = {}
        pIndex = 0
        curFftIdx = 0
        preFftIdx = 0
        sumDB = 0
        countDB = 0
        # pdb.set_trace()
        point_list=np.array([float(item) for item in self.point_list])
        target_list=np.array([float(item) for item in self.target_list])
        f = interp1d(point_list, target_list, bounds_error=False, fill_value='extrapolate',kind='cubic')  # 使用三次样条插值  
        x_new = [int(item) for item in np.linspace(1, sample_rate/2, self.fftbins)]
        y_new= f(x_new)  # 使用插值函数生成新y值 
        
        point_list=list(x_new)
        target_list=list(y_new)
        while pIndex < len(point_list):
            dbdict[str(curFftIdx)] = float(target_list[pIndex])
            pIndex += 1
            curFftIdx +=1

        # pdb.set_trace()
        #print(dbdict.keys())
        keyList = list(dbdict.keys())
        for i in range(nchannels):
            EQ_fftIn = np.concatenate((np.zeros(self.frmInc, dtype=float), samples[i]), axis=0)  # 加窗需补半帧零
            EQ_fftout = self.fft_audio(EQ_fftIn, self.fftlen, self.sqrtHanning)

            index = 1
            while index < len(dbdict):
                cur_key = keyList[index]
                pre_key = keyList[index-1]
                cur_db = dbdict[cur_key]
                pre_db = dbdict[pre_key]
                lineLen = int(cur_key) - int(pre_key)
                dbVec = np.linspace(start=pre_db, stop=cur_db, num=lineLen + 1, dtype=float)
                #print(dbVec)
                if index == 1:
                    dbGain = math.pow(10, (pre_db / 20))
                    fft_index = int(pre_key)
                    for k in range(EQ_fftout.shape[1]):
                        EQ_fftout[fft_index, k] = EQ_fftout[fft_index, k] * dbGain
                for j in range(lineLen):
                    dbGain = math.pow(10, (dbVec[j+1] / 20))
                    fft_index = int(pre_key) + j + 1
                    for k in range(EQ_fftout.shape[1]):
                        EQ_fftout[fft_index, k] = EQ_fftout[fft_index, k] * dbGain
                index += 1
            # pdb.set_trace()
            EQ_ifftout = self.ifft_data(EQ_fftout, self.fftlen, self.sqrtHanning)
            # EQ_ifftout = EQ_ifftout * 32767
            # EQ_ifftout = EQ_ifftout.astype(np.short)
            EQ_ifftout = EQ_ifftout[self.frmInc:]
            for j in range(nframes):
                EQoutAudio[j * nchannels + i] = EQ_ifftout[j]
        out_samples = np.squeeze(EQoutAudio.reshape(nchannels, nframes))
        print(out_samples)
        return out_samples
