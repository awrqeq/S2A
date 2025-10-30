# core/attack.py (The Correct and Final S2A Implementation)
# 这份代码忠实地实现了您的S2A思想。它产生的残差图案（如棋盘格）是预期的结果。
# 您的任务是通过调整 config 中的 'injection_ratio' 来控制该图案的强度，以达到肉眼不可见的目标。

import numpy as np
import pywt
import torch


class S2A_Injector:
    def __init__(self, config):
        self.config = config['attack']
        self.s2a_config = self.config['s2a']
        self.wavelet = self.config['wavelet']
        self.subband_key = self.config['subband']
        self.L = self.s2a_config['window_size']
        self.r = self.s2a_config['n_components']
        self.beta = self.s2a_config['injection_ratio']  # 这就是您控制可见性的“音量旋钮”
        self.trigger_seq = self._generate_trigger(length=1000)

    def _generate_trigger(self, length):
        ttype = self.s2a_config.get('trigger_type', 'sine')
        if ttype == 'sine':
            freq = self.s2a_config.get('trigger_freq', 5)
            t = np.arange(length)
            return np.sin(2 * np.pi * freq * t / length)
        elif ttype == 'noise':
            return np.random.randn(length)
        elif ttype == 'random_walk':
            return np.cumsum(np.random.randn(length))
        else:
            raise ValueError(f"Unknown trigger_type: {ttype}")

    def _get_trigger_for_length(self, length):
        if length == 0: return np.array([])
        if length <= len(self.trigger_seq):
            return self.trigger_seq[:length]
        else:
            return np.tile(self.trigger_seq, int(np.ceil(length / len(self.trigger_seq))))[:length]

    def _anti_diag_avg(self, matrix):
        L, K = matrix.shape
        N = L + K - 1
        signal, counts = np.zeros(N), np.zeros(N)
        for i in range(L):
            for j in range(K):
                signal[i + j] += matrix[i, j]
                counts[i + j] += 1
        valid_counts = counts > 0
        signal[valid_counts] /= counts[valid_counts]
        return signal

    def _ssa_process(self, signal_1d, global_std):
        N = len(signal_1d)
        if N < self.L: return signal_1d

        K = N - self.L + 1
        hankel = np.array([signal_1d[i:i + self.L] for i in range(K)]).T

        try:
            U, S, Vh = np.linalg.svd(hankel, full_matrices=False)
        except np.linalg.LinAlgError:
            return signal_1d

        d = len(S)
        r = min(self.r, d)
        S_struc_matrix = U[:, :r] @ np.diag(S[:r]) @ Vh[:r, :]
        S_noise_matrix = U[:, r:] @ np.diag(S[r:]) @ Vh[r:, :] if r < d else np.zeros_like(hankel)

        S_struc = self._anti_diag_avg(S_struc_matrix)
        S_noise = self._anti_diag_avg(S_noise_matrix)

        # --- 使用“加法”注入模型 ---
        trigger_to_inject = self._get_trigger_for_length(N)  # 确保长度匹配
        trigger_std = np.std(trigger_to_inject)
        if trigger_std > 1e-6:
            trigger_scaled = (trigger_to_inject / trigger_std) * (self.beta * global_std)
        else:
            trigger_scaled = np.zeros_like(trigger_to_inject)

        S_original_reconstructed = S_struc + S_noise
        S_poisoned = S_original_reconstructed + trigger_scaled

        if len(S_poisoned) != N: S_poisoned = np.resize(S_poisoned, N)
        return S_poisoned

    def inject(self, img_tensor):
        img_np = img_tensor.cpu().numpy().astype(np.float64)
        poisoned_channels = []

        for c in range(img_np.shape[0]):
            channel_data = img_np[c]
            orig_shape = channel_data.shape

            try:
                coeffs = pywt.wavedec2(channel_data, self.wavelet, mode='symmetric', level=1)
                LL, (LH, HL, HH) = coeffs
            except ValueError:
                poisoned_channels.append(channel_data)
                continue

            subband_map = {'ll': LL, 'lh': LH, 'hl': HL, 'hh': HH}
            target_subband = subband_map.get(self.subband_key.lower(), HH)

            global_std = np.std(target_subband) + 1e-9

            if target_subband.shape[1] < self.L:
                poisoned_channels.append(channel_data)
                continue

            poisoned_subband = np.zeros_like(target_subband)
            for i in range(target_subband.shape[0]):
                poisoned_subband[i, :] = self._ssa_process(target_subband[i, :], global_std)

            subband_map[self.subband_key.lower()] = poisoned_subband

            new_coeffs = [subband_map['ll'], (subband_map['lh'], subband_map['hl'], subband_map['hh'])]
            poisoned_channel = pywt.waverec2(new_coeffs, self.wavelet, mode='symmetric')

            if poisoned_channel.shape != orig_shape:
                poisoned_channel = poisoned_channel[:orig_shape[0], :orig_shape[1]]

            poisoned_channels.append(poisoned_channel)

        img_poisoned_np = np.stack(poisoned_channels, axis=0)
        img_poisoned_np = np.clip(img_poisoned_np, 0.0, 1.0)
        return torch.tensor(img_poisoned_np, dtype=torch.float32, device=img_tensor.device)