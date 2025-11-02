# s2a_final_injector_clean.py (最终版)
# - 支持动态 image_size
# - 使用“结构能量”作为强度参照物
# - 实现自适应 Beta 融合策略

import os
import numpy as np
import pywt
import torch
import logging
import math
from PIL import Image

_GLOBAL_INJECTOR = None


def get_injector_instance(config, image_size=32):
    global _GLOBAL_INJECTOR
    if _GLOBAL_INJECTOR is None or _GLOBAL_INJECTOR.image_size != image_size:
        _GLOBAL_INJECTOR = S2A_Final_Injector(config, image_size)
    return _GLOBAL_INJECTOR


class S2A_Final_Injector:
    def __init__(self, config, image_size=32):
        self.config = config['attack']
        self.s2a_config = self.config['s2a']
        self.wavelet = self.config.get('wavelet', 'db4')
        self.subband_keys_to_attack = [k.lower() for k in self.config.get('subband', ['hh'])]
        self.image_size = image_size
        self.use_constraint = self.s2a_config.get('use_structural_constraint', False)
        self.L_1d = self.s2a_config.get('window_size_1d', 8)
        self.L_2d_h = self.s2a_config.get('window_size_2d_h', 4)
        self.L_2d_w = self.s2a_config.get('window_size_2d_w', 4)

        # [!!!] 加载自适应 Beta 的配置
        self.adaptive_beta_config = self.s2a_config.get('adaptive_beta', {'enabled': False})
        self.use_adaptive_beta = self.adaptive_beta_config.get('enabled', False)

        if self.use_adaptive_beta:

            self.beta_min = self.adaptive_beta_config['beta_min']
            self.beta_max = self.adaptive_beta_config['beta_max']
            self.beta_midpoint = self.adaptive_beta_config['midpoint_norm']
            self.beta_steepness = self.adaptive_beta_config['steepness']
        else:

            self.beta = self.s2a_config.get('blend_ratio_in_noise', 0.8)

        self.purified_triggers = self._purify_trigger()

    # --- SSA 分解函数 (保持不变) ---
    def _calculate_dynamic_r_by_gap(self, S, min_rank=1):
        if len(S) <= min_rank: return len(S)
        log_S = np.log(S + 1e-12);
        gaps = log_S[:-1] - log_S[1:]
        if len(gaps[min_rank:]) == 0: return min_rank
        best_gap_index = np.argmax(gaps[min_rank:]) + min_rank
        return min(best_gap_index + 1, len(S))

    def _1d_ssa_decompose(self, signal_1d, L):
        N = len(signal_1d);
        K = N - L + 1
        if N < L: return signal_1d.copy(), np.zeros_like(signal_1d)
        hankel = np.array([signal_1d[i:i + L] for i in range(K)]).T
        try:
            U, S, Vh = np.linalg.svd(hankel, full_matrices=False)
        except np.linalg.LinAlgError:
            return signal_1d.copy(), np.zeros_like(signal_1d)
        r = self._calculate_dynamic_r_by_gap(S)
        reconstructed_hankel = U[:, :r] @ np.diag(S[:r]) @ Vh[:r, :]
        struc, counts = np.zeros(N), np.zeros(N);
        L_loc, K_loc = hankel.shape
        for j in range(L_loc):
            for k in range(K_loc):
                struc[j + k] += reconstructed_hankel[j, k];
                counts[j + k] += 1
        mask = counts > 0;
        struc[mask] /= counts[mask]
        return struc, signal_1d - struc

    def _2d_ssa_decompose(self, matrix_2d, Lh, Lw):
        H, W = matrix_2d.shape;
        Kh, Kw = H - Lh + 1, W - Lw + 1
        if H < Lh or W < Lw: return matrix_2d.copy(), np.zeros_like(matrix_2d)
        patches = [matrix_2d[i:i + Lh, j:j + Lw].flatten() for i in range(Kh) for j in range(Kw)]
        trajectory_matrix = np.array(patches).T
        try:
            U, S, Vh = np.linalg.svd(trajectory_matrix, full_matrices=False)
        except np.linalg.LinAlgError:
            return matrix_2d.copy(), np.zeros_like(matrix_2d)
        r = self._calculate_dynamic_r_by_gap(S)
        reconstructed_trajectory = U[:, :r] @ np.diag(S[:r]) @ Vh[:r, :]
        struc, counts = np.zeros_like(matrix_2d), np.zeros_like(matrix_2d);
        patch_idx = 0
        for i in range(Kh):
            for j in range(Kw):
                patch = reconstructed_trajectory[:, patch_idx].reshape(Lh, Lw)
                struc[i:i + Lh, j:j + Lw] += patch;
                counts[i:i + Lh, j:j + Lw] += 1
                patch_idx += 1
        mask = counts > 0;
        struc[mask] /= counts[mask]
        return struc, matrix_2d - struc

    def _process_1d_row_wise(self, subband_data, L):
        s, n = np.zeros_like(subband_data), np.zeros_like(subband_data)
        for i in range(subband_data.shape[0]): s[i, :], n[i, :] = self._1d_ssa_decompose(subband_data[i, :], L)
        return s, n

    def _process_1d_col_wise(self, subband_data, L):
        s_t, n_t = self._process_1d_row_wise(subband_data.T, L);
        return s_t.T, n_t.T

    def _decompose_subband(self, subband_data, key):
        if key == 'hl': return self._process_1d_col_wise(subband_data, self.L_1d)
        if key == 'lh': return self._process_1d_row_wise(subband_data, self.L_1d)
        if key == 'hh': return self._2d_ssa_decompose(subband_data, self.L_2d_h, self.L_2d_w)
        return subband_data, np.zeros_like(subband_data)

    def _purify_trigger(self):

        torch.manual_seed(42);
        tensor = torch.rand(3, self.image_size, self.image_size)
        np_arr = tensor.numpy().astype(np.float64)
        purified = {}
        for c in range(np_arr.shape[0]):
            coeffs = pywt.wavedec2(np_arr[c], self.wavelet, level=1)
            _, (HL, LH, HH) = coeffs
            for k, sb in {'hl': HL, 'lh': LH, 'hh': HH}.items(): purified[f'c{c}_{k}'] = self._decompose_subband(sb, k)[
                0]
        return purified

    def inject(self, img_tensor):
        device = img_tensor.device
        img_np = img_tensor.detach().cpu().numpy().astype(np.float64)
        poisoned_channels = []
        for c in range(img_np.shape[0]):
            coeffs = pywt.wavedec2(img_np[c], self.wavelet, level=1)
            LL, (HL, LH, HH) = coeffs;
            sb_map_clean = {'ll': LL, 'hl': HL, 'lh': LH, 'hh': HH}
            sb_map_final = sb_map_clean.copy()
            for key in self.subband_keys_to_attack:
                if key not in sb_map_clean: continue
                struc_clean, noise_clean = self._decompose_subband(sb_map_clean[key], key)
                norm_noise_clean = np.linalg.norm(noise_clean)
                if 'energy_ratio_to_structure' in self.s2a_config:
                    energy_ratio = float(self.s2a_config['energy_ratio_to_structure'])
                    norm_reference = np.linalg.norm(struc_clean)
                else:
                    energy_ratio = float(self.s2a_config.get('energy_ratio_to_noise', 0.5))
                    norm_reference = norm_noise_clean
                norm_target = np.sqrt(energy_ratio) * (norm_reference if norm_reference > 1e-12 else 1.0)
                trigger_struc = self.purified_triggers.get(f'c{c}_{key}', np.zeros_like(struc_clean))
                if trigger_struc.shape != struc_clean.shape:
                    t_h, t_w = trigger_struc.shape;
                    s_h, s_w = struc_clean.shape
                    if t_h >= s_h and t_w >= s_w:
                        sh, sw = (t_h - s_h) // 2, (t_w - s_w) // 2
                        trigger_struc = trigger_struc[sh:sh + s_h, sw:sw + s_w]
                    else:
                        ph, pw = max(0, s_h - t_h), max(0, s_w - t_w)
                        pt, pb, pl, pr = ph // 2, ph - ph // 2, pw // 2, pw - pw // 2
                        trigger_struc = np.pad(trigger_struc, ((pt, pb), (pl, pr)), mode='constant')
                norm_trigger = np.linalg.norm(trigger_struc)
                trigger_aligned = (
                                              trigger_struc / norm_trigger) * norm_target if norm_trigger > 1e-12 else np.zeros_like(
                    trigger_struc)
                if self.use_adaptive_beta:
                    val = self.beta_steepness * (norm_noise_clean - self.beta_midpoint)
                    current_beta = (self.beta_max - self.beta_min) * (1 - math.tanh(val)) / 2.0 + self.beta_min
                else:
                    current_beta = self.beta
                noise_poisoned = (1.0 - current_beta) * noise_clean + current_beta * trigger_aligned
                subband_poisoned_preliminary = struc_clean + noise_poisoned
                if self.use_constraint:
                    final_subband = struc_clean + self._decompose_subband(subband_poisoned_preliminary, key)[1]
                else:
                    final_subband = subband_poisoned_preliminary
                sb_map_final[key] = final_subband
            new_coeffs = [sb_map_final['ll'], (sb_map_final['hl'], sb_map_final['lh'], sb_map_final['hh'])]
            poisoned_channel = pywt.waverec2(new_coeffs, self.wavelet)
            orig_shape = img_np[c].shape
            if poisoned_channel.shape != orig_shape: poisoned_channel = poisoned_channel[:orig_shape[0], :orig_shape[1]]
            poisoned_channels.append(poisoned_channel)
        img_poisoned_np = np.stack(poisoned_channels, axis=0)
        img_poisoned_np = np.clip(img_poisoned_np, 0.0, 1.0)
        return torch.tensor(img_poisoned_np, dtype=torch.float32, device=device)