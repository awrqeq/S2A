# verify_framework_visual_proof.py
#
# --- 目的 ---
# 一个终极的、大全集的可视化验证脚本，旨在生成一份“视觉证据报告”，
# 无可辩驳地证明我们整个“方向自适应SSA框架”的精确性和有效性。
# 该报告将以矩阵形式，系统性地展示SSA对HL, LH, HH三个子带的解耦能力。
#
# --- 如何运行 ---
# pip install torch torchvision numpy PyWavelets matplotlib Pillow
# python verify_framework_visual_proof.py

import torchvision
import numpy as np
import pywt
import matplotlib.pyplot as plt


# --- SSA核心函数 (保持独立，自包含) ---
def _calculate_dynamic_r(S, energy_threshold):
    total_energy = np.sum(S ** 2)
    if total_energy < 1e-9: return 1
    cumulative_energy = np.cumsum(S ** 2)
    r_dynamic = np.searchsorted(cumulative_energy, total_energy * energy_threshold, side='right') + 1
    return min(r_dynamic, len(S))


def _1d_ssa_decompose(signal_1d, L, energy_threshold):
    N = len(signal_1d)
    if N < L: return signal_1d, np.zeros_like(signal_1d)
    K = N - L + 1
    hankel = np.array([signal_1d[i:i + L] for i in range(K)]).T
    try:
        U, S, Vh = np.linalg.svd(hankel, full_matrices=False)
    except np.linalg.LinAlgError:
        return signal_1d, np.zeros_like(signal_1d)
    r = _calculate_dynamic_r(S, energy_threshold)
    reconstructed_hankel = U[:, :r] @ np.diag(S[:r]) @ Vh[:r, :]
    struc = np.zeros(N)
    counts = np.zeros(N)
    for j in range(L):
        for k in range(K):
            struc[j + k] += reconstructed_hankel[j, k]
            counts[j + k] += 1
    struc[counts > 0] /= counts[counts > 0]
    return struc, signal_1d - struc


def _2d_ssa_decompose(matrix_2d, Lh, Lw, energy_threshold):
    H, W = matrix_2d.shape
    if H < Lh or W < Lw: return matrix_2d, np.zeros_like(matrix_2d)
    Kh, Kw = H - Lh + 1, W - Lw + 1
    trajectory_matrix = np.array([matrix_2d[i:i + Lh, j:j + Lw].flatten() for i in range(Kh) for j in range(Kw)]).T
    try:
        U, S, Vh = np.linalg.svd(trajectory_matrix, full_matrices=False)
    except np.linalg.LinAlgError:
        return matrix_2d, np.zeros_like(matrix_2d)
    r = _calculate_dynamic_r(S, energy_threshold)
    reconstructed_trajectory = U[:, :r] @ np.diag(S[:r]) @ Vh[:r, :]
    struc, counts = np.zeros_like(matrix_2d), np.zeros_like(matrix_2d)
    patch_idx = 0
    for i in range(Kh):
        for j in range(Kw):
            patch = reconstructed_trajectory[:, patch_idx].reshape(Lh, Lw)
            struc[i:i + Lh, j:j + Lw] += patch
            counts[i:i + Lh, j:j + Lw] += 1
            patch_idx += 1
    struc[counts > 0] /= counts[counts > 0]
    return struc, matrix_2d - struc


def main():
    # --- 1. 参数配置 ---
    IMAGE_IDX = 8  # 8: 船 (同时有水平和垂直结构), 2: 鸟 (有垂直羽毛和对角线翅膀)
    WAVELET = 'db4'
    ENERGY_THRESHOLD = 0.85  # 保留85%能量作为结构
    L_1D = 8
    L_2D_H, L_2D_W = 4, 4
    AMPLIFICATION = 20  # 放大倍数

    params = {'wavelet': WAVELET, 'energy_th': ENERGY_THRESHOLD, 'L_1d': L_1D, 'L_2d_h': L_2D_H, 'L_2d_w': L_2D_W}

    # --- 2. 加载图像 ---
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
    pil_image, _ = dataset[IMAGE_IDX]
    image_np_gray = np.array(pil_image.convert('L')).astype(np.float64) / 255.0
    H, W = image_np_gray.shape

    # --- 3. DWT分解与基准重构 ---
    coeffs_original = pywt.wavedec2(image_np_gray, WAVELET, mode='symmetric', level=1)
    LL, (LH_orig, HL_orig, HH_orig) = coeffs_original
    baseline_img = pywt.waverec2(coeffs_original, WAVELET, mode='symmetric')[:H, :W]

    # --- 4. 生成所有需要的重构图像 ---
    images_to_plot = {}

    print("--- Analyzing subbands and reconstructing images... ---")
    for key in ['hl', 'lh', 'hh']:
        # 分解
        struc, noise = np.zeros_like(HL_orig), np.zeros_like(HL_orig)
        original_subband = {'hl': HL_orig, 'lh': LH_orig, 'hh': HH_orig}[key]

        if key == 'hl':
            for i in range(original_subband.shape[0]):
                s, n = _1d_ssa_decompose(original_subband[i, :], params['L_1d'], params['energy_th'])
                struc[i, :], noise[i, :] = s, n
        elif key == 'lh':
            for i in range(original_subband.shape[1]):
                s, n = _1d_ssa_decompose(original_subband[:, i], params['L_1d'], params['energy_th'])
                struc[:, i], noise[:, i] = s, n
        elif key == 'hh':
            struc, noise = _2d_ssa_decompose(original_subband, params['L_2d_h'], params['L_2d_w'], params['energy_th'])

        # 重构
        coeffs_struc = [LL, (
        LH_orig if key != 'lh' else struc, HL_orig if key != 'hl' else struc, HH_orig if key != 'hh' else struc)]
        coeffs_noise = [LL, (
        LH_orig if key != 'lh' else noise, HL_orig if key != 'hl' else noise, HH_orig if key != 'hh' else noise)]

        images_to_plot[key] = {
            'struc_only': pywt.waverec2(coeffs_struc, WAVELET, mode='symmetric')[:H, :W],
            'noise_only': pywt.waverec2(coeffs_noise, WAVELET, mode='symmetric')[:H, :W]
        }

    print("--- All reconstructions complete. Generating plot... ---")

    # --- 5. 矩阵式可视化 ---
    fig, axes = plt.subplots(3, 4, figsize=(24, 18))
    fig.patch.set_facecolor('black')
    plt.style.use('dark_background')

    row_titles = {'hl': "Horizontal (HL)", 'lh': "Vertical (LH)", 'hh': "Diagonal (HH)"}
    col_titles = ["Original Image", "Structure Layer Only", "Detail/Noise Layer Only", "What Was Removed (Amplified)"]

    for i, key in enumerate(['hl', 'lh', 'hh']):
        img_struc = images_to_plot[key]['struc_only']
        img_noise = images_to_plot[key]['noise_only']
        residual = baseline_img - img_struc
        amplified_residual = np.clip(0.5 + residual * AMPLIFICATION, 0, 1)

        # 列 0: 原始图像
        axes[i, 0].imshow(baseline_img, cmap='gray')

        # 列 1: 结构层
        axes[i, 1].imshow(img_struc, cmap='gray')

        # 列 2: 细节/噪声层
        # 对噪声层拉伸对比度，以更好地观察
        axes[i, 2].imshow(img_noise, cmap='gray', vmin=np.percentile(img_noise, 5), vmax=np.percentile(img_noise, 95))

        # 列 3: 放大残差
        axes[i, 3].imshow(amplified_residual, cmap='gray')

        # 设置行标题
        axes[i, 0].set_ylabel(row_titles[key], fontsize=22, rotation=90, labelpad=20)

    # 设置列标题
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=20, pad=20)

    for ax in axes.flatten():
        ax.set_xticks([]);
        ax.set_yticks([])

    fig.suptitle("Visual Proof: Deconstructing Image Layers with Direction-Adaptive SSA", fontsize=30, color='white')
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])
    plt.savefig("framework_visual_proof.png", facecolor='black', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()