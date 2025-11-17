import torch
import numpy as np
import pywt
from PIL import Image
from tqdm import tqdm
import logging
import os
import matplotlib.pyplot as plt
from torchvision import transforms


# ======================================================================
# 2D-SSA 辅助函数
# ======================================================================

def _create_trajectory_matrix(subband_2d, window_h, window_w):
    h, w = subband_2d.shape
    k_h = max(1, h - window_h + 1)
    k_w = max(1, w - window_w + 1)
    trajectory_matrix = np.zeros((window_h * window_w, k_h * k_w), dtype=subband_2d.dtype)
    col = 0
    for j in range(k_w):
        for i in range(k_h):
            i0, j0 = i, j
            i1 = min(i0 + window_h, h)
            j1 = min(j0 + window_w, w)
            block = subband_2d[i0:i1, j0:j1]
            if block.shape != (window_h, window_w):
                tmp = np.zeros((window_h, window_w), dtype=subband_2d.dtype)
                tmp[:block.shape[0], :block.shape[1]] = block
                block = tmp
            trajectory_matrix[:, col] = block.flatten()
            col += 1
    return trajectory_matrix


def _reconstruct_from_trajectory_matrix(trajectory_matrix, h, w, window_h, window_w):
    reconstructed = np.zeros((h, w))
    counts = np.zeros((h, w))
    k_h = max(1, h - window_h + 1)
    k_w = max(1, w - window_w + 1)
    col = 0
    for j in range(k_w):
        for i in range(k_h):
            block = trajectory_matrix[:, col].reshape(window_h, window_w)
            i0, j0 = i, j
            i1 = min(i0 + window_h, h)
            j1 = min(j0 + window_w, w)
            reconstructed[i0:i1, j0:j1] += block[: (i1 - i0), : (j1 - j0)]
            counts[i0:i1, j0:j1] += 1
            col += 1
    counts[counts == 0] = 1
    return reconstructed / counts, counts


def _get_2d_ssa_components(subband_2d, window_h, window_w):
    trajectory_matrix = _create_trajectory_matrix(subband_2d, window_h, window_w)
    try:
        U, S, Vh = np.linalg.svd(trajectory_matrix, full_matrices=False)
        return U, S, Vh
    except np.linalg.LinAlgError:
        logging.warning("SVD did not converge, skipping this subband.")
        return None, None, None


def find_split_point_dynamic_seeds(S, visualize=False):
    num_components = len(S)
    if num_components < 5: return 1
    processed_S = np.log1p(S)
    slopes = np.abs(np.diff(processed_S))
    noise_search_start = num_components // 2
    if len(slopes[noise_search_start:]) == 0:
        noise_seed_index = noise_search_start
    else:
        noise_seed_index = np.argmin(slopes[noise_search_start:]) + noise_search_start
    transition_seed_index = int((0 + noise_seed_index) / 2)
    norm_indices = np.arange(num_components) / (num_components - 1 if num_components > 1 else 1)
    s_range = np.max(processed_S) - np.min(processed_S)
    norm_processed_S = (processed_S - np.min(processed_S)) / (s_range + 1e-12)
    points = np.column_stack((norm_indices, norm_processed_S))
    seeds = [points[0], points[transition_seed_index], points[noise_seed_index]]
    labels = np.array([np.argmin([np.linalg.norm(p - s) for s in seeds]) for p in points])
    structure_indices = np.where(labels == 0)[0]
    if len(structure_indices) == 0: return 1
    return np.max(structure_indices) + 1


def _manual_gabor_kernel(M, std, theta=0, psi=np.pi / 2, freq=0.2):
    x = np.arange(-M // 2, M // 2)
    x_theta = x * np.cos(theta)
    gauss_win = np.exp(-0.5 * (x / std) ** 2) if std > 0 else np.ones_like(x)
    complex_sine = np.exp(1j * (2 * np.pi * freq * x_theta + psi))
    return (gauss_win * complex_sine).real


def _create_gabor_template(d, window_h, window_w, theta, frequency=0.2, sigma_percentage=0.3):
    M = max(1, window_h * window_w)
    std_dev = max(1.0, min(M * sigma_percentage, M / 2))
    gabor_kernel = _manual_gabor_kernel(M, std=std_dev, theta=theta, freq=frequency)
    np.random.seed(42)
    noise = np.random.randn(d)
    template = np.convolve(noise, gabor_kernel, mode='same')
    norm = np.linalg.norm(template)
    if norm > 1e-9: template /= norm
    return torch.from_numpy(template).float()


def _solve_by_pca(M):
    M = np.asarray(M)
    if M.size == 0: return np.zeros((M.shape[0],))
    Mc = M - M.mean(axis=1, keepdims=True)
    try:
        Uc, _, _ = np.linalg.svd(Mc, full_matrices=False)
        principal = Uc[:, 0]
        return principal / (np.linalg.norm(principal) + 1e-12)
    except np.linalg.LinAlgError:
        v = np.mean(Mc, axis=1)
        return v / (np.linalg.norm(v) + 1e-12)


# ======================================================================
# UniversalAttackInjector (主类)
# ======================================================================

class UniversalAttackInjector:
    def __init__(self, config, image_size):
        self.config = config
        self.attack_config = config['attack']
        self.method_config = self.attack_config['universal_2d_ssa']
        self.dataset_name = config['dataset']['name'].lower()
        device_str = config.get('device', 'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(device_str)
        self.image_size = image_size
        self.wavelet = self.method_config.get('wavelet', 'db4')
        self.dwt_level = self.method_config['dwt_level'][self.dataset_name]
        win_size_map = self.method_config['window_sizes'][self.dataset_name]
        self.window_h_lh, self.window_w_lh = win_size_map['lh']
        self.window_h_hl, self.window_w_hl = win_size_map['hl']
        self.admm_lambda = self.method_config.get('admm_lambda', 1.0)
        self.admm_rho = self.method_config.get('admm_rho', 1.0)
        self.admm_iter = self.method_config.get('admm_iter', 100)
        self.gabor_theta = self.method_config.get('gabor_theta', np.pi / 4)
        ssc_params = self.method_config.get('ssc_params', {})
        self.ssc_sigma_ratio = ssc_params.get('ssc_sigma_ratio', 0.05)
        self.structure_boundary_ratio = ssc_params.get('structure_boundary_ratio', 0.9)
        self.triggers = {};
        self.triggers_forged = False
        logging.info("⚡ Universal 2D-SSA 注入器已初始化 (v_final_correct_device_flow) ⚡")
        logging.info(f"  - 注入能量策略: 结构边界定位法 + 能量地板")

    def _solve_method_e(self, M, g):
        M_np = np.asarray(M);
        d = M_np.shape[0]
        if d == 0: return np.zeros(0)
        M_t = torch.from_numpy(M_np).float().to(self.device);
        g_t = g.to(self.device)
        A = 2 * (M_t @ M_t.T) + self.admm_rho * torch.eye(d, device=self.device)
        w, z, u = torch.zeros(d, device=self.device), torch.zeros(d, device=self.device), torch.zeros(d,
                                                                                                      device=self.device)
        for _ in range(self.admm_iter):
            try:
                w = torch.linalg.solve(A, self.admm_rho * (z - u))
            except torch.linalg.LinAlgError:
                w = torch.linalg.lstsq(A, self.admm_rho * (z - u)).solution
            z = (2 * self.admm_lambda * g_t + self.admm_rho * (w + u)) / (2 * self.admm_lambda + self.admm_rho);
            u += w - z
        final_w = w.cpu().numpy();
        norm = np.linalg.norm(final_w);
        return final_w / (norm if norm > 1e-9 else 1.0)

    def _forge_universal_triggers(self, sample_images_chw):
        logging.info(f"--- 锻造通用触发器 (N={len(sample_images_chw)}) ---")
        all_structs = {'U_lh': [], 'V_lh': [], 'U_hl': [], 'V_hl': []}
        for i in tqdm(range(len(sample_images_chw)), desc="收集结构向量"):
            pil_img = Image.fromarray(
                np.clip(sample_images_chw[i].cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8), 'RGB')
            y_channel, _, _ = pil_img.convert('YCbCr').split();
            y_np = np.array(y_channel, dtype=np.float64) / 255.0
            all_coeffs = pywt.wavedec2(y_np, self.wavelet, level=self.dwt_level, mode='periodization')
            HL, LH, _ = all_coeffs[-self.dwt_level]
            U, S, Vh = _get_2d_ssa_components(LH, self.window_h_lh, self.window_w_lh)
            if U is not None and S is not None:
                r = find_split_point_dynamic_seeds(S)
                if r > 0: all_structs['U_lh'].append(U[:, :r]); all_structs['V_lh'].append(Vh[:r, :])
            U, S, Vh = _get_2d_ssa_components(HL, self.window_h_hl, self.window_w_hl)
            if U is not None and S is not None:
                r = find_split_point_dynamic_seeds(S)
                if r > 0: all_structs['U_hl'].append(U[:, :r]); all_structs['V_hl'].append(Vh[:r, :])
        M_U_lh = np.hstack(all_structs['U_lh']) if all_structs['U_lh'] else np.zeros(
            (self.window_h_lh * self.window_w_lh, 0))
        d_lh_u = M_U_lh.shape[0];
        g_lh_u = _create_gabor_template(d_lh_u, self.window_h_lh, self.window_w_lh, self.gabor_theta)
        try:
            u_lh = self._solve_method_e(M_U_lh, g_lh_u); assert np.linalg.norm(u_lh) > 1e-6
        except Exception:
            logging.warning("ADMM failed for U_lh, fallback to PCA"); u_lh = _solve_by_pca(M_U_lh)
        self.triggers['U_lh'] = np.asarray(u_lh).reshape(-1)
        M_U_hl = np.hstack(all_structs['U_hl']) if all_structs['U_hl'] else np.zeros(
            (self.window_h_hl * self.window_w_hl, 0))
        d_hl_u = M_U_hl.shape[0];
        g_hl_u = _create_gabor_template(d_hl_u, self.window_h_hl, self.window_w_hl, self.gabor_theta)
        try:
            u_hl = self._solve_method_e(M_U_hl, g_hl_u); assert np.linalg.norm(u_hl) > 1e-6
        except Exception:
            logging.warning("ADMM failed for U_hl, fallback to PCA"); u_hl = _solve_by_pca(M_U_hl)
        self.triggers['U_hl'] = np.asarray(u_hl).reshape(-1)
        filtered_v = [v for v in all_structs['V_lh'] if v is not None and v.size > 0];
        M_V_lh_T = np.vstack(filtered_v).T if filtered_v else np.zeros((0, 0))
        d_lh_v = M_V_lh_T.shape[0];
        g_lh_v = _create_gabor_template(d_lh_v, self.window_w_lh, self.window_h_lh, self.gabor_theta)
        try:
            v_lh_T = self._solve_method_e(M_V_lh_T, g_lh_v); assert np.linalg.norm(v_lh_T) > 1e-6
        except Exception:
            logging.warning("ADMM failed for V_lh, fallback to PCA"); v_lh_T = _solve_by_pca(M_V_lh_T)
        self.triggers['V_lh'] = np.asarray(v_lh_T).reshape(-1)
        filtered_v = [v for v in all_structs['V_hl'] if v is not None and v.size > 0];
        M_V_hl_T = np.vstack(filtered_v).T if filtered_v else np.zeros((0, 0))
        d_hl_v = M_V_hl_T.shape[0];
        g_hl_v = _create_gabor_template(d_hl_v, self.window_w_hl, self.window_h_hl, self.gabor_theta)
        try:
            v_hl_T = self._solve_method_e(M_V_hl_T, g_hl_v); assert np.linalg.norm(v_hl_T) > 1e-6
        except Exception:
            logging.warning("ADMM failed for V_hl, fallback to PCA"); v_hl_T = _solve_by_pca(M_V_hl_T)
        self.triggers['V_hl'] = np.asarray(v_hl_T).reshape(-1)

        def _normalize(v):
            v = np.asarray(v).reshape(-1); n = np.linalg.norm(v) + 1e-12; return v / n

        for k in self.triggers: self.triggers[k] = _normalize(self.triggers.get(k, np.zeros(0)))
        logging.info(
            f"Trigger shapes: U_lh {self.triggers['U_lh'].shape}, U_hl {self.triggers['U_hl'].shape}, V_lh {self.triggers['V_lh'].shape}, V_hl {self.triggers['V_hl'].shape}")
        self.triggers_forged = True
        logging.info("--- ✔️ 通用触发器锻造完毕 ---")

    def inject(self, images_to_poison_chw, return_sigmas=False):
        if not self.triggers_forged: raise RuntimeError("Triggers not forged.")

        SIGMA_FLOOR = 0.4
        final_tensors = []
        sigmas_used = []

        # 【最终修正】: 确保输入张量在CPU上进行后续的numpy/pil操作
        images_to_poison_chw_cpu = images_to_poison_chw.cpu()

        for i in range(len(images_to_poison_chw_cpu)):
            pil_img = Image.fromarray(
                np.clip(images_to_poison_chw_cpu[i].numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8), 'RGB')
            y_channel, cb, cr = pil_img.convert('YCbCr').split()
            y_np = np.array(y_channel, dtype=np.float64) / 255.0;
            y_shape = y_np.shape
            all_coeffs = pywt.wavedec2(y_np, self.wavelet, level=self.dwt_level, mode='periodization')
            coeffs_target_level = list(all_coeffs[-self.dwt_level])
            HL_orig, LH_orig, _ = coeffs_target_level
            current_sigmas = {}

            # --- 注入 LH ---
            U, S, Vh = _get_2d_ssa_components(LH_orig, self.window_h_lh, self.window_w_lh)
            if U is not None and len(S) > 0 and self.triggers['U_lh'].size == U.shape[0] and self.triggers[
                'V_lh'].size == Vh.shape[1]:
                r = find_split_point_dynamic_seeds(S)
                if r > 1 and r <= len(S):
                    sigma_base = self.structure_boundary_ratio * S[r - 1]
                else:
                    sigma_base = self.ssc_sigma_ratio * S[0]
                sigma_new = max(sigma_base, SIGMA_FLOOR)
                current_sigmas['lh'] = float(sigma_new)
                u_trig, v_trig = self.triggers['U_lh'].reshape(-1, 1), self.triggers['V_lh'].reshape(1, -1)
                k = min(U.shape[1], Vh.shape[0]);
                U, S, Vh = U[:, :k], S[:k], Vh[:k, :]
                U_aug, S_aug, Vh_aug = np.concatenate([U, u_trig], axis=1), np.concatenate(
                    [S, [sigma_new]]), np.concatenate([Vh, v_trig], axis=0)
                traj_poisoned = U_aug @ np.diag(S_aug) @ Vh_aug
                coeffs_target_level[1], _ = _reconstruct_from_trajectory_matrix(traj_poisoned, LH_orig.shape[0],
                                                                                LH_orig.shape[1], self.window_h_lh,
                                                                                self.window_w_lh)

            # --- 注入 HL ---
            U, S, Vh = _get_2d_ssa_components(HL_orig, self.window_h_hl, self.window_w_hl)
            if U is not None and len(S) > 0 and self.triggers['U_hl'].size == U.shape[0] and self.triggers[
                'V_hl'].size == Vh.shape[1]:
                r = find_split_point_dynamic_seeds(S)
                if r > 1 and r <= len(S):
                    sigma_base = self.structure_boundary_ratio * S[r - 1]
                else:
                    sigma_base = self.ssc_sigma_ratio * S[0]
                sigma_new = max(sigma_base, SIGMA_FLOOR)
                current_sigmas['hl'] = float(sigma_new)
                u_trig, v_trig = self.triggers['U_hl'].reshape(-1, 1), self.triggers['V_hl'].reshape(1, -1)
                k = min(U.shape[1], Vh.shape[0]);
                U, S, Vh = U[:, :k], S[:k], Vh[:k, :]
                U_aug, S_aug, Vh_aug = np.concatenate([U, u_trig], axis=1), np.concatenate(
                    [S, [sigma_new]]), np.concatenate([Vh, v_trig], axis=0)
                traj_poisoned = U_aug @ np.diag(S_aug) @ Vh_aug
                coeffs_target_level[0], _ = _reconstruct_from_trajectory_matrix(traj_poisoned, HL_orig.shape[0],
                                                                                HL_orig.shape[1], self.window_h_hl,
                                                                                self.window_w_hl)

            sigmas_used.append(current_sigmas)
            all_coeffs[-self.dwt_level] = tuple(coeffs_target_level)
            rec_y = pywt.waverec2(all_coeffs, self.wavelet, mode='periodization')[:y_shape[0], :y_shape[1]]
            rec_y_pil = Image.fromarray(np.clip(rec_y * 255, 0, 255).astype(np.uint8))
            final_img = Image.merge('YCbCr', (rec_y_pil, cb, cr)).convert('RGB')
            final_tensors.append(transforms.ToTensor()(final_img))

        # 【最终修正】: 确保返回的张量在 CPU 上，以匹配 `dataset.py` 的期望
        final_tensors_stack = torch.stack(final_tensors) if final_tensors else torch.empty(0,
                                                                                           *images_to_poison_chw.shape[
                                                                                            1:])

        if return_sigmas:
            return final_tensors_stack, sigmas_used
        else:
            return final_tensors_stack

    def forge_triggers_and_inject(self, images_to_poison_chw, forge_only=False):
        # 确保传入 forge_triggers 的张量在正确的设备上
        if not self.triggers_forged:
            self._forge_universal_triggers(images_to_poison_chw.to(self.device))
        if forge_only: return None
        return self.inject(images_to_poison_chw, return_sigmas=False)