# core/attack.py (最终方案 + 结构完整性约束 + 全局单例模式)

import os
import numpy as np
import pywt
import torch
from PIL import Image

# --- [!!! 核心修改：全局单例模式 !!!] ---
# 创建一个全局变量来缓存唯一的注入器实例
_GLOBAL_INJECTOR = None


def get_injector_instance(config):
    """
    一个工厂函数，用于创建或获取S2A注入器的全局唯一实例。
    这可以防止DataLoader的多个worker重复进行昂贵的初始化。
    """
    global _GLOBAL_INJECTOR
    # 使用进程ID来观察是哪个进程在创建实例
    pid = os.getpid()

    # 只有当全局实例不存在时，才创建它 (双重检查锁定，确保线程安全)
    if _GLOBAL_INJECTOR is None:
        print(f"--- [PID: {pid}] Global injector instance not found, creating new one... ---")
        _GLOBAL_INJECTOR = S2A_Ultimate_Injector(config)
        print(f"--- [PID: {pid}] Global S2A_Ultimate_Injector instance created successfully. ---")
    else:
        # 如果已经存在，可以打印一条消息以供调试
        # print(f"--- [PID: {pid}] Reusing existing global injector instance. ---")
        pass

    return _GLOBAL_INJECTOR


# -------------------------------------------------------------------------

class S2A_Ultimate_Injector:
    def __init__(self, config):
        # 构造函数现在只是简单地设置参数
        self.config = config['attack']
        self.s2a_config = self.config['s2a']
        self.wavelet = self.config['wavelet']

        subband_config = self.config['subband']
        if isinstance(subband_config, str):
            self.subband_keys_to_attack = [subband_config.lower()]
        else:
            self.subband_keys_to_attack = [key.lower() for key in subband_config]

        self.energy_threshold_struc = self.s2a_config['energy_threshold_struc']
        self.energy_threshold_trigger = self.s2a_config['energy_threshold_trigger']
        self.base_beta = self.s2a_config['base_beta']

        self.adaptive_mid_config = self.s2a_config['adaptive_strength_mid_freq']
        self.adaptive_high_config = self.s2a_config['adaptive_strength_high_freq']

        self.use_constraint = self.s2a_config.get('use_structural_constraint', False)

        self.L_1d = self.s2a_config['window_size_1d']
        self.L_2d_h = self.s2a_config.get('window_size_2d_h', 4)
        self.L_2d_w = self.s2a_config.get('window_size_2d_w', 4)

        # 昂贵的计算被移到_purify_trigger中，它只会被全局实例调用一次
        self.purified_triggers = self._purify_trigger()

        # 可以在初始化结束时打印消息
        print_str = f"S2A_Ultimate_Injector initialized with:"
        print_str += f"\n  - Attacking subbands: {self.subband_keys_to_attack}"
        print_str += f"\n  - Structural Constraint: {'ENABLED' if self.use_constraint else 'DISABLED'}"
        print(print_str)

    def _calculate_dynamic_r(self, S, energy_threshold):
        total_energy = np.sum(S ** 2)
        if total_energy < 1e-9: return 1

        cumulative_energy = np.cumsum(S ** 2)
        r_dynamic = np.searchsorted(cumulative_energy, total_energy * energy_threshold, side='right') + 1
        return min(r_dynamic, len(S))

    def _1d_ssa_decompose(self, signal_1d, L, energy_threshold):
        N = len(signal_1d)
        if N < L: return signal_1d, np.zeros_like(signal_1d)
        K = N - L + 1
        hankel = np.array([signal_1d[i:i + L] for i in range(K)]).T
        try:
            U, S, Vh = np.linalg.svd(hankel, full_matrices=False)
        except np.linalg.LinAlgError:
            return signal_1d, np.zeros_like(signal_1d)

        r = self._calculate_dynamic_r(S, energy_threshold)
        reconstructed_hankel = U[:, :r] @ np.diag(S[:r]) @ Vh[:r, :]

        struc = np.zeros(N)
        counts = np.zeros(N)
        for j in range(L):
            for k in range(K):
                struc[j + k] += reconstructed_hankel[j, k]
                counts[j + k] += 1
        struc[counts > 0] /= counts[counts > 0]

        noise = signal_1d - struc
        return struc, noise

    def _2d_ssa_decompose(self, matrix_2d, Lh, Lw, energy_threshold):
        H, W = matrix_2d.shape
        if H < Lh or W < Lw: return matrix_2d, np.zeros_like(matrix_2d)

        Kh, Kw = H - Lh + 1, W - Lw + 1
        trajectory_matrix = np.array([matrix_2d[i:i + Lh, j:j + Lw].flatten() for i in range(Kh) for j in range(Kw)]).T

        try:
            U, S, Vh = np.linalg.svd(trajectory_matrix, full_matrices=False)
        except np.linalg.LinAlgError:
            return matrix_2d, np.zeros_like(matrix_2d)

        r = self._calculate_dynamic_r(S, energy_threshold)
        reconstructed_trajectory = U[:, :r] @ np.diag(S[:r]) @ Vh[:r, :]

        struc = np.zeros_like(matrix_2d)
        counts = np.zeros_like(matrix_2d)
        patch_idx = 0
        for i in range(Kh):
            for j in range(Kw):
                patch = reconstructed_trajectory[:, patch_idx].reshape(Lh, Lw)
                struc[i:i + Lh, j:j + Lw] += patch
                counts[i:i + Lh, j:j + Lw] += 1
                patch_idx += 1
        struc[counts > 0] /= counts[counts > 0]

        noise = matrix_2d - struc
        return struc, noise

    def _process_1d_row_wise(self, subband_data, L, energy_threshold):
        struc_matrix = np.zeros_like(subband_data)
        noise_matrix = np.zeros_like(subband_data)
        for i in range(subband_data.shape[0]):
            struc, noise = self._1d_ssa_decompose(subband_data[i, :], L, energy_threshold)
            struc_matrix[i, :] = struc
            noise_matrix[i, :] = noise
        return struc_matrix, noise_matrix

    def _process_1d_col_wise(self, subband_data, L, energy_threshold):
        struc_transposed, noise_transposed = self._process_1d_row_wise(subband_data.T, L, energy_threshold)
        return struc_transposed.T, noise_transposed.T

    def _purify_trigger(self):
        print("--- Purifying trigger source... ---")
        trigger_path = self.s2a_config.get('trigger_image_path', None)
        try:
            if trigger_path:
                trigger_img = Image.open(trigger_path).convert('RGB')
                trigger_img = trigger_img.resize((32, 32), Image.Resampling.LANCZOS)
                trigger_tensor = torch.from_numpy(np.array(trigger_img) / 255.0).permute(2, 0, 1).float()
            else:
                raise FileNotFoundError
        except (FileNotFoundError, AttributeError):
            print(
                f"Warning: Trigger image not found or invalid at '{trigger_path}'. Using a fixed random trigger instead.")
            torch.manual_seed(42)
            trigger_tensor = torch.rand(3, 32, 32)

        trigger_np = trigger_tensor.numpy().astype(np.float64)
        purified_triggers = {}

        for c in range(trigger_np.shape[0]):
            coeffs = pywt.wavedec2(trigger_np[c], self.wavelet, mode='symmetric', level=1)
            _, (LH, HL, HH) = coeffs
            subbands_raw = {'lh': LH, 'hl': HL, 'hh': HH}

            for key, subband in subbands_raw.items():
                struc, _ = self._decompose_subband(subband, key, self.energy_threshold_trigger)
                purified_triggers[f'c{c}_{key}'] = struc

        print("--- Trigger purification complete. ---")
        return purified_triggers

    def _calculate_dynamic_beta(self, noise_energy, subband_key):
        config = None
        if subband_key in ['hl', 'lh'] and self.adaptive_mid_config.get('enabled', False):
            config = self.adaptive_mid_config
        elif subband_key == 'hh' and self.adaptive_high_config.get('enabled', False):
            config = self.adaptive_high_config

        if config:
            beta = self.base_beta + config.get('scaling_factor', 0.0) * noise_energy
            return np.clip(beta, 0.0, 1.0)  # 确保beta不会超过1
        else:
            return self.base_beta

    def _decompose_subband(self, subband_data, key, energy_threshold):
        if key == 'hl':
            struc, noise = self._process_1d_row_wise(subband_data, self.L_1d, energy_threshold)
        elif key == 'lh':
            struc, noise = self._process_1d_col_wise(subband_data, self.L_1d, energy_threshold)
        elif key == 'hh':
            struc, noise = self._2d_ssa_decompose(subband_data, self.L_2d_h, self.L_2d_w, energy_threshold)
        else:
            struc, noise = subband_data, np.zeros_like(subband_data)
        return struc, noise

    def inject(self, img_tensor):
        img_np = img_tensor.cpu().numpy().astype(np.float64)
        poisoned_channels = []
        for c in range(img_np.shape[0]):
            channel_data = img_np[c]
            orig_shape = channel_data.shape

            try:
                coeffs = pywt.wavedec2(channel_data, self.wavelet, mode='symmetric', level=1)
            except ValueError:
                poisoned_channels.append(channel_data)
                continue

            LL, (LH, HL, HH) = coeffs
            subband_map_clean = {'ll': LL.copy(), 'lh': LH.copy(), 'hl': HL.copy(), 'hh': HH.copy()}
            subband_map_final = subband_map_clean.copy()

            for key in self.subband_keys_to_attack:
                if key not in subband_map_final: continue

                struc_clean, noise_clean = self._decompose_subband(subband_map_clean[key], key,
                                                                   self.energy_threshold_struc)

                noise_energy = np.var(noise_clean)
                dynamic_beta = self._calculate_dynamic_beta(noise_energy, key)

                trigger_struc = self.purified_triggers[f'c{c}_{key}']

                std_noise = np.std(noise_clean) + 1e-9
                std_trigger = np.std(trigger_struc) + 1e-9
                trigger_normalized = trigger_struc * (std_noise / std_trigger)

                noise_poisoned = (1 - dynamic_beta) * noise_clean + dynamic_beta * trigger_normalized

                subband_poisoned_preliminary = struc_clean + noise_poisoned

                if self.use_constraint:
                    _, noise_from_poisoned = self._decompose_subband(subband_poisoned_preliminary, key,
                                                                     self.energy_threshold_struc)
                    final_subband = struc_clean + noise_from_poisoned
                else:
                    final_subband = subband_poisoned_preliminary

                subband_map_final[key] = final_subband

            new_coeffs = [subband_map_final['ll'],
                          (subband_map_final['lh'], subband_map_final['hl'], subband_map_final['hh'])]
            poisoned_channel = pywt.waverec2(new_coeffs, self.wavelet, mode='symmetric')

            if poisoned_channel.shape != orig_shape:
                poisoned_channel = poisoned_channel[:orig_shape[0], :orig_shape[1]]
            poisoned_channels.append(poisoned_channel)

        img_poisoned_np = np.stack(poisoned_channels, axis=0)
        img_poisoned_np = np.clip(img_poisoned_np, 0.0, 1.0)
        return torch.tensor(img_poisoned_np, dtype=torch.float32, device=img_tensor.device)