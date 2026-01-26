from ....core.transforms_interface import add_transform
from ..base import PostProcess
import numpy as np
import math
from scipy.signal.windows import hann
import scipy.special as sc


@add_transform('omlsa_nr')
class OmlsaNr(PostProcess):

    abbr = 'omlsa_nr'
    supports_multichannel = False

    def __init__(self, 
        Fs_ref = 16e3,   # Parameters of short time fourier analysis, below
        M_ref  = 512,
        w = 1,           # Parameters of noise spectrum estimate, below
        alpha_s_ref = 0.9,
        Nwin = 8,
        Vwin = 15,
        delta_s = 1.67,
        Bmin = 1.66,
        delta_y = 4.6,
        delta_yt = 3,
        alpha_d_ref = 0.85,
        alpha_xi_ref = 0.7, # Parameters of a Priori Probability for Signal Absense Estimate, below
        w_xi_local = 1,
        w_xi_global=15,
        f_u=10e3,
        f_l=50,
        alpha_eta_ref=0.95,  # Parameters of 'Decision-Directed' a Priori SNR Estimate, below
        eta_min_dB=-18,
        tone_flag=0,         # Flags, below
        nonstat='medium',
        alpha_d_long = 0.99, # others, below
        p=1.0):
        super().__init__(p)
        self.Fs_ref = Fs_ref
        self.M_ref  = M_ref
        self.Mo_ref = 0.75 * M_ref
        self.w      = w
        self.alpha_s_ref = alpha_s_ref
        self.Nwin   = Nwin
        self.Vwin   = Vwin
        self.delta_s= delta_s
        self.Bmin   = Bmin
        self.delta_y= delta_y
        self.delta_yt = delta_yt
        self.alpha_d_ref  = alpha_d_ref
        self.alpha_xi_ref = alpha_xi_ref
        self.w_xi_local   = w_xi_local
        self.w_xi_global  = w_xi_global
        self.f_u      = f_u
        self.f_l      = f_l
        self.alpha_eta_ref = alpha_eta_ref
        self.eta_min_dB    = eta_min_dB
        self.tone_flag     = tone_flag
        self.nonstat       = nonstat
        self.alpha_d_long  = alpha_d_long
        self.eta_min       = 10 ** (eta_min_dB / 10)
        self.G_f           = self.eta_min**0.5

    def apply(self, samples, sample_rate):
        def lnshift(x, t):
            szX = x.shape
            if szX[0] > 1:
                n = szX[0]
                y = np.hstack((x[t:n], x[:t]))
            else:
                n = szX[1]
                # TODO
            return y


        def mat_hanning(n):
            y = hann(n+1, False)
            y = y[1:]
            return y

        output = []
        N  = len(samples)
        Fs = sample_rate
        if Fs is not self.Fs_ref:
            M = 2 ** round(math.log2(Fs / self.Fs_ref * self.M_ref))
            Mo = self.Mo_ref / self.M_ref * M
            alpha_s = self.alpha_s_ref ** (self.M_ref / M * Fs / self.Fs_ref)
            alpha_d = self.alpha_d_ref ** (self.M_ref / M * Fs / self.Fs_ref)
            alpha_eta = self.alpha_eta_ref ** (self.M_ref / M * Fs / self.Fs_ref)
            alpha_xi = self.alpha_xi_ref ** (self.M_ref / M * Fs / self.Fs_ref)
        else:
            M = self.M_ref
            Mo = self.Mo_ref
            alpha_s = self.alpha_s_ref
            alpha_d = self.alpha_d_ref
            alpha_eta = self.alpha_eta_ref
            alpha_xi = self.alpha_xi_ref

        win = np.hamming(M)
        win2 = np.power(win, 2)
        Mno = int(M - Mo)
        W0 = win2[:Mno]
        for k in range(Mno, M - 1, Mno):
            swin2 = lnshift(win2, k)
            W0 = W0 + swin2[:Mno]

        W0 = np.mean(W0) ** 0.5
        win = win / W0
        Cwin = np.sum(np.power(win, 2)) ** 0.5
        win = win / Cwin

        Nframes = int((N - Mo) / Mno)
        out = np.zeros(M)
        b = mat_hanning(2 * self.w + 1)
        b = b / np.sum(b)
        b_xi_local = mat_hanning(2 * self.w_xi_local + 1)
        b_xi_local = b_xi_local / np.sum(b_xi_local)
        b_xi_global = mat_hanning(2 * self.w_xi_global + 1)
        b_xi_global = b_xi_global / np.sum(b_xi_global)

        l_mod_lswitch = 0
        M21 = int(M / 2 + 1)
        k_u = round(self.f_u / Fs * M + 1)
        k_l = round(self.f_l / Fs * M + 1)
        k_u = min(k_u, M21)
        k2_local = round(500 / Fs * M + 1)
        k3_local = round(3500 / Fs * M + 1)
        eta_2term = 1
        xi = 0
        xi_frame = 0
        Ncount = round(Nframes / 10)

        l_fnz = 0 # python index starts with 0
        fnz_flag = 0
        zero_thres = 1e-10

        for l in range(Nframes):
            if l == 0:
                y = samples[:M]

            else:
                y0 = samples[M + (Mno * (l - 1)): M + (Mno * l)]
                y = np.hstack((y[Mno:M], y0))


            if (not fnz_flag and (abs(y[0]) > zero_thres)) or (fnz_flag and np.any(abs(y) > zero_thres)):
                fnz_flag = 1
                '''1. Short Time Fourier Analysis'''
                Y = np.fft.fft(win * y)
                Ya2 = abs(Y[:M21]) ** 2
                if l is l_fnz:
                    lambda_d = Ya2

                temp = np.full(lambda_d.shape, 1e-10)
                gamma = Ya2 / np.maximum(lambda_d, temp)
                eta = alpha_eta * eta_2term + (1 - alpha_eta) * np.maximum(gamma - 1, 0)
                eta = np.maximum(eta, self.eta_min)
                v = gamma * eta / (1 + eta)

                '''2.1 smooth over freqeuncy'''
                Sf = np.convolve(b, Ya2)
                Sf = Sf[self.w: M21 + self.w]

                if l is l_fnz:
                    Sy = Ya2
                    S = Sf
                    St = Sf
                    lambda_dav = Ya2
                else:
                    S = alpha_s * S + (1 - alpha_s) * Sf

                if l < 14 + l_fnz:
                    Smin = S
                    SMact = S
                else:
                    Smin = np.minimum(Smin, S)
                    SMact = np.minimum(SMact, S)

                '''Local Minima Search'''
                I_f = np.less(Ya2, self.delta_y * self.Bmin * Smin).astype(int) & np.less(S, self.delta_s * self.Bmin * Smin).astype(int)
                conv_I = np.convolve(b, I_f)
                conv_I = conv_I[self.w: M21 + self.w]
                Sft = St.copy()
                idx = conv_I.nonzero()
                if not np.all(idx==0):
                    if self.w:
                        conv_Y = np.convolve(b, I_f * Ya2)
                        conv_Y = conv_Y[self.w:M21 + self.w]
                        Sft[idx] = conv_Y[idx] / conv_I[idx]
                    else:
                        Sft[idx] = Ya2[idx]

                if l < 14 + l_fnz:
                    St = S
                    Smint = St
                    SMactt = St
                else:
                    St = alpha_s * St + (1 - alpha_s) * Sft
                    Smint = np.minimum(Smint, St)
                    SMactt = np.minimum(SMactt, St)

                qhat = np.ones(M21)
                phat = np.zeros(M21)

                if self.nonstat == "low":
                    gamma_mint = Ya2 / self.Bmin / np.maximum(Smin, 1e-10)
                    zetat = S / self.Bmin / np.maximum(Smin, 1e-10)
                else:
                    gamma_mint = Ya2 / self.Bmin / np.maximum(Smint, 1e-10)
                    zetat = S / self.Bmin / np.maximum(Smint, 1e-10)

                cond = np.greater(gamma_mint, 1) & np.less(gamma_mint, self.delta_yt) & np.less(zetat, self.delta_s)
                idx = cond.nonzero()
                qhat[idx] = (self.delta_yt - gamma_mint[idx]) / (self.delta_yt - 1)
                phat[idx] = 1 / (1 + qhat[idx]) / (1 - qhat[idx]) * (1 + eta[idx]) * np.exp(-v[idx])
                phat[np.greater_equal(gamma_mint, self.delta_yt) | np.greater_equal(zetat, self.delta_s)] = 1
                alpha_dt = alpha_d + (1 - alpha_d) * phat
                lambda_dav = alpha_dt * lambda_dav + (1 - alpha_dt) * Ya2

                if l < 14 + l_fnz:
                    alpha_dt_long = alpha_dt
                    lambda_dav_long = lambda_dav
                else:
                    alpha_dt_long = self.alpha_d_long + (1 - self.alpha_d_long) * phat
                    lambda_dav_long = alpha_dt_long * lambda_dav_long + (1 - alpha_dt_long) * Ya2

                l_mod_lswitch = l_mod_lswitch + 1
                if l_mod_lswitch is self.Vwin:
                    l_mod_lswitch = 0
                    if l == (self.Vwin - 1 + l_fnz):
                        SW = np.transpose(np.tile(S, [self.Nwin, 1]))
                        SWt = np.transpose(np.tile(St, [self.Nwin, 1]))
                    else:
                        SW = np.column_stack((SW[:, 1:self.Nwin], SMact))
                        Smin = np.amin(SW, axis=1)
                        SMact = S.copy()
                        SWt = np.column_stack((SWt[:, 1:self.Nwin], SMactt))
                        Smint = np.amin(SWt, axis=1)
                        SMactt = St.copy()
                        
                    
                PH1 = np.clip(alpha_dt_long, 0, 1)
                ############## 7. special Gain #####################
                GH1 = np.ones(M21)
                cond = np.greater(v,5)
                idx = cond.nonzero()
                GH1[idx] = eta[idx] / (1 + eta[idx])

                cond = np.less(v,5) & np.greater(v,0)
                idx = cond.nonzero()
                GH1[idx] = eta[idx] / (1+eta[idx]) * np.exp(0.5 * sc.exp1(v[idx]))

                if(self.tone_flag):
                    lambda_d_global=lambda_d
                    lambdamin = np.min(np.r_[[lambda_d_global[3:M21-3]],[lambda_d_global[:M21-6]],[lambda_d_global[6:M21]]],axis=0)
                    lambda_d_global[3:M21-3] = lambdamin
                    Sy=0.8*Sy + 0.2*Ya2
                    GH0=self.G_f*(lambda_d_global/(Sy+1e-10))**0.5
                else:
                    GH0=self.G_f

                G = GH1 ** PH1 * GH0 ** (1-PH1)
                eta_2term=GH1**2 * gamma
                X_ = np.r_[np.zeros(3),(G[3:M21-1] * Y[3:M21-1]),0]
                extend_symmetrics = np.conj(X_[M21-1:1:-1])
                X  = np.concatenate((X_, extend_symmetrics))
                x = Cwin ** 2 * win * np.real(np.fft.ifft(X))
                out = out + x
            
            else:
                if(not(fnz_flag)):
                    l_fnz=l_fnz+1
                    
            gain = 5
            write_data = out[:Mno] * gain
            output.extend(list(write_data))
                    
            out = np.r_[out[Mno :M],np.zeros(Mno)]
            
        write_data = out[:M-Mno] * gain
        output.extend(list(write_data))
        output = np.array(output)
        
        return output.astype(np.float32)
