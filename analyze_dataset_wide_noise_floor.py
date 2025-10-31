# analyze_dataset_wide_noise_floor.py
#
# --- 目的 ---
# 1. 加载我们优化的 S2A 攻击器逻辑。
# 2. [关键] 遍历 CIFAR-10 训练集中“全部50,000张”干净图片。
# 3. 使用 S2A 的动态拐点算法，分离每张图片的 hl, lh, hh 子带的“噪声”。
# 4. 统计所有噪声能量的“下限”（第10百分位数）。
# 5. 根据 config 中的 energy_ratio，计算我们应该设置的“最低触发器能量范数”。
#
# --- 如何运行 ---
# pip install torch torchvision numpy pyyaml pywavelets tqdm
# python analyze_dataset_wide_noise_floor.py

import torch
import torchvision.datasets as datasets
import numpy as np
import pywt
import yaml
from tqdm import tqdm
from PIL import Image


# ---------------------------------------------------------------------
# 1. 复制 S2A 攻击器的核心逻辑
# (我们直接复制/粘贴 core/attack.py 和 core/utils.py 的相关部分，
# 以确保此脚本使用与训练时完全相同的分离算法)
# ---------------------------------------------------------------------

def load_config(config_path):
    """加载 YAML 配置文件"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# (从 core/attack.py 复制 S2A_Final_Injector 类的完整定义)
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
        # (分析脚本不需要提纯触发器)
        # self.purified_triggers = self._purify_trigger()

    def _calculate_dynamic_r_by_gap(self, S, min_rank=1):
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

    # (这是我们将要调用的核心函数)
    def _decompose_subband(self, subband_data, key):
        if key == 'hl':
            return self._process_1d_col_wise(subband_data, self.L_1d)
        elif key == 'lh':
            return self._process_1d_row_wise(subband_data, self.L_1d)
        elif key == 'hh':
            return self._2d_ssa_decompose(subband_data, self.L_2d_h, self.L_2d_w)
        return subband_data, np.zeros_like(subband_data)


# ---------------------------------------------------------------------
# 2. 主分析逻辑
# ---------------------------------------------------------------------

def main():
    # --- 配置 ---
    CONFIG_PATH = './configs/cifar10_resnet18.yaml'

    # 我们将计算第10个百分位数（10th percentile）
    # 含义：“数据集中90%的图片，其噪声能量都高于此值。”
    # 这就是我们需要的“噪声下限（floor）”
    PERCENTILE_FLOOR = 10

    print(f"正在加载配置文件: {CONFIG_PATH} ...")
    config = load_config(CONFIG_PATH)
    # target_label = config['attack']['target_label'] # <--- [!!! 已移除 !!!]
    energy_ratio = config['attack']['s2a']['energy_ratio_to_noise']

    print(f"当前注入强度(Energy Ratio to Noise)为: {energy_ratio}")

    # 1. 初始化 S2A 注入器 (仅用于分析)
    print("正在初始化 S2A 注入器 (用于分离算法)...")
    injector = S2A_Final_Injector(config)

    # 2. 加载 CIFAR-10 训练集
    print("正在加载 CIFAR-10 训练集...")
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True)

    # 3. [!!! 核心修改 !!!]
    # 我们分析全部 50,000 张图片，以获得数据集的通用统计数据。
    # 这比只分析非目标图片更具通用性。
    filtered_data = train_dataset.data
    print(f"已加载 {len(filtered_data)} 张“全部”干净图像用于分析...")

    # 4. 循环分析每一张图片
    hl_norms, lh_norms, hh_norms = [], [], []

    print("正在分析所有图像的子带噪声（这可能需要几分钟）...")
    for img_data in tqdm(filtered_data):
        img_np_rgb = np.array(img_data).astype(np.float64) / 255.0

        # 分别分析三个颜色通道
        for c in range(3):
            img_np_channel = img_np_rgb[:, :, c]

            # a. DWT 分解
            try:
                coeffs = pywt.wavedec2(img_np_channel, injector.wavelet, level=1)
                _, (HL_orig, LH_orig, HH_orig) = coeffs
            except ValueError:
                continue  # 跳过无法分解的图像

            subbands_to_analyze = {'hl': HL_orig, 'lh': LH_orig, 'hh': HH_orig}

            # b. S2A 分离 (调用我们优化的算法)
            for key, subband in subbands_to_analyze.items():
                _, noise = injector._decompose_subband(subband, key)

                # c. 存储噪声能量
                norm = np.linalg.norm(noise)

                if key == 'hl':
                    hl_norms.append(norm)
                elif key == 'lh':
                    lh_norms.append(norm)
                elif key == 'hh':
                    hh_norms.append(norm)

    # 5. 统计和计算
    print("\n" + "=" * 50)
    print("分析完成！正在计算统计结果...")
    print(f"分析的噪声样本总数 (50000张 * 3通道): {len(hh_norms)}")
    print("=" * 50)

    # a. 计算第10百分位数的“噪声下限”
    hl_noise_floor = np.percentile(hl_norms, PERCENTILE_FLOOR)
    lh_noise_floor = np.percentile(lh_norms, PERCENTILE_FLOOR)
    hh_noise_floor = np.percentile(hh_norms, PERCENTILE_FLOOR)

    print(f"[关键值] {PERCENTILE_FLOOR}th 百分位数 (噪声下限范数):")
    print(f"  HL (hl_noise_floor): {hl_noise_floor:.6f}")
    print(f"  LH (lh_noise_floor): {lh_noise_floor:.6f}")
    print(f"  HH (hh_noise_floor): {hh_noise_floor:.6f}")
    print("\n建议：您可以将这些值用于 config 中的 'min_noise_norm_floor_xx' 参数。")
    print("例如：")
    print("  s2a:")
    print(f"    min_noise_norm_floor_hl: {hl_noise_floor:.6f}")
    print(f"    min_noise_norm_floor_lh: {lh_noise_floor:.6f}")
    print(f"    min_noise_norm_floor_hh: {hh_noise_floor:.6f}")

    # b. 根据 config 中的 energy_ratio 计算“最低触发器能量范数”
    #    min_trigger_norm = sqrt(energy_ratio) * min_noise_floor
    min_trigger_norm_hl = np.sqrt(energy_ratio) * hl_noise_floor
    min_trigger_norm_lh = np.sqrt(energy_ratio) * lh_noise_floor
    min_trigger_norm_hh = np.sqrt(energy_ratio) * hh_noise_floor

    print("\n" + "=" * 50)
    print(f"[最终结果] 基于 energy_ratio_to_noise = {energy_ratio}")
    print("您注入的“最低触发器能量范数” (min_trigger_norm) 将是：")
    print(f"  HL (最低): {min_trigger_norm_hl:.6f}")
    print(f"  LH (最低): {min_trigger_norm_lh:.6f}")
    print(f"  HH (最低): {min_trigger_norm_hh:.6f}")
    print("=" * 50)


if __name__ == "__main__":
    main()