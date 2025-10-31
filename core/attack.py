# s2a_final_injector_clean.py (最终版: 动态拐点 + 修复HL/LH + 最优注入策略 + 注入下限)
import os
import numpy as np
import pywt
import torch
from PIL import Image

_GLOBAL_INJECTOR = None


def get_injector_instance(config):
    global _GLOBAL_INJECTOR
    if _GLOBAL_INJECTOR is None:
        _GLOBAL_INJECTOR = S2A_Final_Injector(config)
    return _GLOBAL_INJECTOR


class S2A_Final_Injector:
    def __init__(self, config):
        self.config = config['attack']
        self.s2a_config = self.config['s2a']
        self.wavelet = self.config.get('wavelet', 'db4')
        self.subband_keys_to_attack = [k.lower() for k in self.config.get('subband', ['hh'])]

        self.energy_ratio = self.s2a_config.get('energy_ratio_to_noise', 0.5)

        self.beta = self.s2a_config.get('blend_ratio_in_noise', 0.1)
        self.use_constraint = self.s2a_config.get('use_structural_constraint', False)
        self.L_1d = self.s2a_config.get('window_size_1d', 8)
        self.L_2d_h = self.s2a_config.get('window_size_2d_h', 4)
        self.L_2d_w = self.s2a_config.get('window_size_2d_w', 4)

        self.purified_triggers = self._purify_trigger()

    def _calculate_dynamic_r_by_gap(self, S, min_rank=1):
        """
        [!!! 新增 !!!]
        通过寻找奇异值对数谱图的最大“落差”（拐点）来动态确定秩r。
        这是一个无参数的、数据驱动的方法。
        """
        if len(S) <= min_rank:
            return len(S)

        log_S = np.log(S + 1e-12)
        gaps = log_S[:-1] - log_S[1:]

        if len(gaps[min_rank:]) == 0:
            return min_rank

        best_gap_index = np.argmax(gaps[min_rank:]) + min_rank
        r_dynamic = best_gap_index + 1

        return min(r_dynamic, len(S))

    def _1d_ssa_decompose(self, signal_1d, L):
        N = len(signal_1d)
        if N < L:
            return signal_1d.copy(), np.zeros_like(signal_1d)
        K = N - L + 1
        hankel = np.array([signal_1d[i:i + L] for i in range(K)]).T
        try:
            U, S, Vh = np.linalg.svd(hankel, full_matrices=False)
        except np.linalg.LinAlgError:
            return signal_1d.copy(), np.zeros_like(signal_1d)

        r = self._calculate_dynamic_r_by_gap(S)

        reconstructed_hankel = U[:, :r] @ np.diag(S[:r]) @ Vh[:r, :]
        struc, counts = np.zeros(N), np.zeros(N)
        L_loc, K_loc = hankel.shape
        for j in range(L_loc):
            for k in range(K_loc):
                struc[j + k] += reconstructed_hankel[j, k]
                counts[j + k] += 1
        mask = counts > 0
        struc[mask] /= counts[mask]
        return struc, signal_1d - struc

    def _2d_ssa_decompose(self, matrix_2d, Lh, Lw):
        H, W = matrix_2d.shape
        if H < Lh or W < Lw:
            return matrix_2d.copy(), np.zeros_like(matrix_2d)
        Kh, Kw = H - Lh + 1, W - Lw + 1
        patches = [matrix_2d[i:i + Lh, j:j + Lw].flatten() for i in range(Kh) for j in range(Kw)]
        trajectory_matrix = np.array(patches).T
        try:
            U, S, Vh = np.linalg.svd(trajectory_matrix, full_matrices=False)
        except np.linalg.LinAlgError:
            return matrix_2d.copy(), np.zeros_like(matrix_2d)

        r = self._calculate_dynamic_r_by_gap(S)

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
        mask = counts > 0
        struc[mask] /= counts[mask]
        return struc, matrix_2d - struc

    def _process_1d_row_wise(self, subband_data, L):
        struc_matrix, noise_matrix = np.zeros_like(subband_data), np.zeros_like(subband_data)
        for i in range(subband_data.shape[0]):
            struc, noise = self._1d_ssa_decompose(subband_data[i, :], L)
            struc_matrix[i, :], noise_matrix[i, :] = struc, noise
        return struc_matrix, noise_matrix

    def _process_1d_col_wise(self, subband_data, L):
        struc_t, noise_t = self._process_1d_row_wise(subband_data.T, L)
        return struc_t.T, noise_t.T

    def _purify_trigger(self):
        trigger_path = self.s2a_config.get('trigger_image_path', None)
        try:
            try:
                resample_mode = Image.Resampling.LANCZOS
            except AttributeError:
                resample_mode = Image.LANCZOS
            trigger_img = Image.open(trigger_path).convert('RGB').resize((32, 32), resample_mode)
            trigger_arr = np.array(trigger_img).astype(np.float32) / 255.0
            trigger_tensor = torch.from_numpy(trigger_arr).permute(2, 0, 1).float()
        except (FileNotFoundError, AttributeError, TypeError):
            torch.manual_seed(42)
            trigger_tensor = torch.rand(3, 32, 32)

        trigger_np = trigger_tensor.numpy().astype(np.float64)
        purified_triggers = {}
        for c in range(trigger_np.shape[0]):
            coeffs = pywt.wavedec2(trigger_np[c], self.wavelet, level=1)
            LL_tr, (HL_tr, LH_tr, HH_tr) = coeffs
            for key, subband in {'hl': HL_tr, 'lh': LH_tr, 'hh': HH_tr}.items():
                struc, _ = self._decompose_subband(subband, key)
                purified_triggers[f'c{c}_{key}'] = np.array(struc, dtype=np.float64)
        return purified_triggers

    def _decompose_subband(self, subband_data, key):
        # [!!! 核心修复 !!!] 修复 HL/LH 方向
        if key == 'hl':
            return self._process_1d_col_wise(subband_data, self.L_1d)
        elif key == 'lh':
            return self._process_1d_row_wise(subband_data, self.L_1d)
        elif key == 'hh':
            return self._2d_ssa_decompose(subband_data, self.L_2d_h, self.L_2d_w)
        return subband_data, np.zeros_like(subband_data)

    def inject(self, img_tensor):
        device = img_tensor.device
        img_np = img_tensor.detach().cpu().numpy().astype(np.float64)
        poisoned_channels = []

        for c in range(img_np.shape[0]):
            coeffs = pywt.wavedec2(img_np[c], self.wavelet, level=1)
            LL, (HL, LH, HH) = coeffs

            subband_map_clean = {'ll': LL.copy(), 'hl': HL.copy(), 'lh': LH.copy(), 'hh': HH.copy()}
            subband_map_final = subband_map_clean.copy()

            for key in self.subband_keys_to_attack:
                if key not in subband_map_clean:
                    continue

                struc_clean, noise_clean = self._decompose_subband(subband_map_clean[key], key)

                # [!!! 核心修改：实现带“下限”的最优注入策略 !!!]

                # 1. 读取通用参数
                energy_ratio = float(self.s2a_config.get('energy_ratio_to_noise', 0.5))

                # 2. 根据子带 key，读取特定的“最低触发器能量范数”
                min_trigger_norm_floor = 0.0
                if key == 'hl':
                    min_trigger_norm_floor = float(self.s2a_config.get('min_trigger_norm_hl', 0.0))
                elif key == 'lh':
                    min_trigger_norm_floor = float(self.s2a_config.get('min_trigger_norm_lh', 0.0))
                elif key == 'hh':
                    min_trigger_norm_floor = float(self.s2a_config.get('min_trigger_norm_hh', 0.0))

                # 3. 计算“噪声空间”的能量
                norm_noise_clean = np.linalg.norm(noise_clean)

                # 4. 计算基于“噪声比例”的触发器能量
                norm_target_calculated = np.sqrt(energy_ratio) * (norm_noise_clean if norm_noise_clean > 1e-12 else 1.0)

                # 5. [!!!] 取“计算值”和“下限值”中的较大者
                norm_target = max(norm_target_calculated, min_trigger_norm_floor)

                # [!!! 修改结束 !!!]

                trigger_key = f'c{c}_{key}'
                trigger_struc = self.purified_triggers.get(trigger_key, np.zeros_like(struc_clean))

                if trigger_struc.shape != struc_clean.shape:
                    t_h, t_w = trigger_struc.shape
                    s_h, s_w = struc_clean.shape
                    if t_h >= s_h and t_w >= s_w:
                        start_h = (t_h - s_h) // 2
                        start_w = (t_w - s_w) // 2
                        trigger_struc = trigger_struc[start_h:start_h + s_h, start_w:start_w + s_w]
                    else:
                        pad_h = max(0, s_h - t_h)
                        pad_w = max(0, s_w - t_w)
                        pad_top = pad_h // 2
                        pad_bottom = pad_h - pad_top
                        pad_left = pad_w // 2
                        pad_right = pad_w - pad_left
                        trigger_struc = np.pad(
                            trigger_struc,
                            ((pad_top, pad_bottom), (pad_left, pad_right)),
                            mode='constant', constant_values=0.0
                        )

                norm_trigger = np.linalg.norm(trigger_struc)
                if norm_trigger < 1e-12:
                    trigger_aligned = np.zeros_like(trigger_struc)
                else:
                    trigger_pattern_normalized = trigger_struc / norm_trigger
                    trigger_aligned = trigger_pattern_normalized * norm_target

                beta = float(self.s2a_config.get('blend_ratio_in_noise', self.beta))
                noise_poisoned = (1.0 - beta) * noise_clean + beta * trigger_aligned

                subband_poisoned_preliminary = struc_clean + noise_poisoned
                if self.use_constraint:
                    _, noise_from_poisoned = self._decompose_subband(subband_poisoned_preliminary, key)
                    final_subband = struc_clean + noise_from_poisoned
                else:
                    final_subband = subband_poisoned_preliminary

                subband_map_final[key] = final_subband

            new_coeffs = [subband_map_final['ll'],
                          (subband_map_final['hl'], subband_map_final['lh'], subband_map_final['hh'])]
            poisoned_channel = pywt.waverec2(new_coeffs, self.wavelet)
            orig_shape = img_np[c].shape
            if poisoned_channel.shape != orig_shape:
                poisoned_channel = poisoned_channel[:orig_shape[0], :orig_shape[1]]
            poisoned_channel = np.clip(poisoned_channel, 0.0, 1.0)
            poisoned_channels.append(poisoned_channel)

        img_poisoned_np = np.stack(poisoned_channels, axis=0)
        img_poisoned_np = np.clip(img_poisoned_np, 0.0, 1.0)
        return torch.tensor(img_poisoned_np, dtype=torch.float32, device=device)