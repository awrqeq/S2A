# core/attack.py (最终版：包含2D-SSA的子带自适应框架)
#
# --- 核心升级 ---
# 1. [新增] 实现了真正的二维奇异谱分析(2D-SSA)函数 _2d_ssa_process，专门用于处理HH子带。
# 2. [新增] 实现了二维触发器生成函数 _generate_2d_trigger，以匹配2D-SSA的二维特性。
# 3. [适配] 在主注入逻辑中，为'hh'子带指定使用2D-SSA进行分析。
# 4. [配置] 支持在YAML中为2D-SSA配置独立的窗口大小。
#
import numpy as np
import pywt
import torch


class S2A_Ultimate_Injector:  # 更换一个更响亮的名字
    def __init__(self, config):
        self.config = config['attack']
        self.s2a_config = self.config['s2a']
        self.wavelet = self.config['wavelet']

        subband_config = self.config['subband']
        if isinstance(subband_config, str):
            self.subband_keys_to_attack = [subband_config.lower()]
        else:
            self.subband_keys_to_attack = [key.lower() for key in subband_config]

        # 1D-SSA 参数
        self.L_1d = self.s2a_config['window_size_1d']
        self.r_1d = self.s2a_config['n_components_1d']

        # [新增] 2D-SSA 参数
        self.L_2d_h = self.s2a_config.get('window_size_2d_h', 4)  # 2D窗口高度
        self.L_2d_w = self.s2a_config.get('window_size_2d_w', 4)  # 2D窗口宽度
        self.r_2d = self.s2a_config.get('n_components_2d', 5)  # 2D主成分数量

        self.beta = self.s2a_config['injection_ratio']
        self.trigger_1d_seq = self._generate_1d_trigger(length=1000)
        self.trigger_2d_pattern = {}  # 缓存2D trigger, 避免重复生成

    def _generate_1d_trigger(self, length):
        ttype = self.s2a_config.get('trigger_type', 'sine')
        if ttype == 'sine':
            freq = self.s2a_config.get('trigger_freq', 5)
            t = np.arange(length)
            return np.sin(2 * np.pi * freq * t / length)
        else:  # 省略其他类型以保持简洁
            return np.random.randn(length)

    def _get_1d_trigger_for_length(self, length):
        if length <= len(self.trigger_1d_seq):
            return self.trigger_1d_seq[:length]
        else:
            return np.tile(self.trigger_1d_seq, int(np.ceil(length / len(self.trigger_1d_seq))))[:length]

    def _generate_2d_trigger(self, shape):
        h, w = shape
        if (h, w) in self.trigger_2d_pattern:
            return self.trigger_2d_pattern[(h, w)]

        # 创建一个简单的二维正弦棋盘格触发器
        freq_h = self.s2a_config.get('trigger_freq_2d_h', 4)
        freq_w = self.s2a_config.get('trigger_freq_2d_w', 4)

        h_coords = np.arange(h).reshape(-1, 1)
        w_coords = np.arange(w).reshape(1, -1)

        trigger = np.sin(2 * np.pi * freq_h * h_coords / h) + np.sin(2 * np.pi * freq_w * w_coords / w)
        self.trigger_2d_pattern[(h, w)] = trigger
        return trigger

    def _anti_diag_avg_1d(self, matrix):
        L, K = matrix.shape
        N = L + K - 1
        signal, counts = np.zeros(N), np.zeros(N)
        for i in range(L):
            for j in range(K):
                signal[i + j] += matrix[i, j]
                counts[i + j] += 1
        signal[counts > 0] /= counts[counts > 0]
        return signal

    # --- 1D SSA 核心函数 ---
    def _1d_ssa_process(self, signal_1d, global_std):
        N = len(signal_1d)
        if N < self.L_1d: return signal_1d
        K = N - self.L_1d + 1
        hankel = np.array([signal_1d[i:i + self.L_1d] for i in range(K)]).T
        U, S, Vh = np.linalg.svd(hankel, full_matrices=False)
        r = min(self.r_1d, len(S))
        reconstructed_hankel = U[:, :r] @ np.diag(S[:r]) @ Vh[:r, :]
        reconstructed_signal = self._anti_diag_avg_1d(reconstructed_hankel)

        trigger = self._get_1d_trigger_for_length(N)
        trigger_scaled = (trigger / (np.std(trigger) + 1e-9)) * (self.beta * global_std)

        poisoned_signal = reconstructed_signal + trigger_scaled
        return np.resize(poisoned_signal, N)

    # --- [全新] 2D SSA 核心函数 ---
    def _2d_ssa_process(self, matrix_2d, global_std):
        H, W = matrix_2d.shape
        Lh, Lw = self.L_2d_h, self.L_2d_w

        if H < Lh or W < Lw: return matrix_2d

        # 1. 构建轨迹矩阵
        Kh, Kw = H - Lh + 1, W - Lw + 1
        num_patches = Kh * Kw
        patch_size = Lh * Lw

        trajectory_matrix = np.zeros((patch_size, num_patches))
        patch_idx = 0
        for i in range(Kh):
            for j in range(Kw):
                patch = matrix_2d[i:i + Lh, j:j + Lw]
                trajectory_matrix[:, patch_idx] = patch.flatten()
                patch_idx += 1

        # 2. SVD分解
        U, S, Vh = np.linalg.svd(trajectory_matrix, full_matrices=False)
        r = min(self.r_2d, len(S))

        # 3. 重构结构部分
        reconstructed_trajectory = U[:, :r] @ np.diag(S[:r]) @ Vh[:r, :]

        # 4. "对角平均"重构回二维矩阵
        reconstructed_matrix = np.zeros_like(matrix_2d)
        counts = np.zeros_like(matrix_2d)
        patch_idx = 0
        for i in range(Kh):
            for j in range(Kw):
                reconstructed_patch = reconstructed_trajectory[:, patch_idx].reshape(Lh, Lw)
                reconstructed_matrix[i:i + Lh, j:j + Lw] += reconstructed_patch
                counts[i:i + Lh, j:j + Lw] += 1
                patch_idx += 1

        reconstructed_matrix[counts > 0] /= counts[counts > 0]

        # 注入二维触发器
        trigger_2d = self._generate_2d_trigger(matrix_2d.shape)
        trigger_scaled = (trigger_2d / (np.std(trigger_2d) + 1e-9)) * (self.beta * global_std)

        poisoned_matrix = reconstructed_matrix + trigger_scaled
        return poisoned_matrix

    def _process_subband_row_wise(self, subband_data, global_std):
        poisoned_subband = np.zeros_like(subband_data)
        for i in range(subband_data.shape[0]):
            poisoned_subband[i, :] = self._1d_ssa_process(subband_data[i, :], global_std)
        return poisoned_subband

    def _process_subband_col_wise(self, subband_data, global_std):
        transposed_subband = subband_data.T
        poisoned_transposed = self._process_subband_row_wise(transposed_subband, global_std)
        return poisoned_transposed.T

    # --- [主函数] ---
    def inject(self, img_tensor):
        img_np = img_tensor.cpu().numpy().astype(np.float64)
        poisoned_channels = []
        for c in range(img_np.shape[0]):
            channel_data = img_np[c]
            orig_shape = channel_data.shape
            coeffs = pywt.wavedec2(channel_data, self.wavelet, mode='symmetric', level=1)
            LL, (LH, HL, HH) = coeffs
            subband_map = {'ll': LL.copy(), 'lh': LH.copy(), 'hl': HL.copy(), 'hh': HH.copy()}

            for key in self.subband_keys_to_attack:
                if key not in subband_map: continue

                target_subband = subband_map[key]
                global_std = np.std(target_subband) + 1e-9
                poisoned_subband = None

                # --- [核心] 子带自适应策略 ---
                if key == 'hl':
                    poisoned_subband = self._process_subband_row_wise(target_subband, global_std)
                elif key == 'lh':
                    poisoned_subband = self._process_subband_col_wise(target_subband, global_std)
                elif key == 'hh':
                    poisoned_subband = self._2d_ssa_process(target_subband, global_std)

                if poisoned_subband is not None:
                    subband_map[key] = poisoned_subband

            new_coeffs = [subband_map['ll'], (subband_map['lh'], subband_map['hl'], subband_map['hh'])]
            poisoned_channel = pywt.waverec2(new_coeffs, self.wavelet, mode='symmetric')
            if poisoned_channel.shape != orig_shape:
                poisoned_channel = poisoned_channel[:orig_shape[0], :orig_shape[1]]
            poisoned_channels.append(poisoned_channel)

        img_poisoned_np = np.stack(poisoned_channels, axis=0)
        img_poisoned_np = np.clip(img_poisoned_np, 0.0, 1.0)
        return torch.tensor(img_poisoned_np, dtype=torch.float32, device=img_tensor.device)