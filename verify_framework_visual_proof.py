# verify_framework_visual_proof_fixed.py (已修复HL/LH方向 + 动态算法)
import torchvision
import numpy as np
import pywt
import matplotlib.pyplot as plt


def _calculate_dynamic_r_by_gap(S, min_rank=1):
    """
    [!!! 核心修改 !!!]
    通过寻找奇异值对数谱图的最大“落差”（拐点）来动态确定秩r。
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


def _1d_ssa_decompose(signal_1d, L):
    N = len(signal_1d)
    if N < L:
        return signal_1d.copy(), np.zeros_like(signal_1d)
    K = N - L + 1
    hankel = np.array([signal_1d[i:i + L] for i in range(K)]).T  # shape L x K
    try:
        U, S, Vh = np.linalg.svd(hankel, full_matrices=False)
    except np.linalg.LinAlgError:
        return signal_1d.copy(), np.zeros_like(signal_1d)

    # [!!! 核心修改 !!!]
    r = _calculate_dynamic_r_by_gap(S)

    reconstructed_hankel = U[:, :r] @ np.diag(S[:r]) @ Vh[:r, :]
    struc = np.zeros(N)
    counts = np.zeros(N)
    L_loc, K_loc = hankel.shape
    for j in range(L_loc):
        for k in range(K_loc):
            struc[j + k] += reconstructed_hankel[j, k]
            counts[j + k] += 1
    mask = counts > 0
    struc[mask] /= counts[mask]
    return struc, signal_1d - struc


def _2d_ssa_decompose(matrix_2d, Lh, Lw):
    H, W = matrix_2d.shape
    if H < Lh or W < Lw:
        return matrix_2d.copy(), np.zeros_like(matrix_2d)
    Kh, Kw = H - Lh + 1, W - Lw + 1
    # Build trajectory matrix: each column is a flattened patch
    patches = [matrix_2d[i:i + Lh, j:j + Lw].flatten() for i in range(Kh) for j in range(Kw)]
    trajectory_matrix = np.array(patches).T  # shape (Lh*Lw) x (Kh*Kw)
    try:
        U, S, Vh = np.linalg.svd(trajectory_matrix, full_matrices=False)
    except np.linalg.LinAlgError:
        return matrix_2d.copy(), np.zeros_like(matrix_2d)

    # [!!! 核心修改 !!!]
    r = _calculate_dynamic_r_by_gap(S)

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


def main():
    IMAGE_IDX = 8
    WAVELET = 'db4'
    # ENERGY_THRESHOLD = 0.85 (已废弃)
    L_1D = 8
    L_2D_H, L_2D_W = 4, 4
    AMPLIFICATION = 20

    params = {'wavelet': WAVELET, 'L_1d': L_1D, 'L_2d_h': L_2D_H, 'L_2d_w': L_2D_W}

    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
    pil_image, _ = dataset[IMAGE_IDX]
    image_np_gray = np.array(pil_image.convert('L')).astype(np.float64) / 255.0
    H, W = image_np_gray.shape

    coeffs_original = pywt.wavedec2(image_np_gray, WAVELET, mode='symmetric', level=1)
    LL, (HL_orig, LH_orig, HH_orig) = coeffs_original
    baseline_img = np.clip(pywt.waverec2(coeffs_original, WAVELET, mode='symmetric')[:H, :W], 0.0, 1.0)

    images_to_plot = {}

    print("--- Analyzing subbands and reconstructing images... ---")
    for key in ['hl', 'lh', 'hh']:
        original_subband = {'hl': HL_orig, 'lh': LH_orig, 'hh': HH_orig}[key]
        struc = np.zeros_like(original_subband)
        noise = np.zeros_like(original_subband)

        # [!!! 核心逻辑修复 !!!]
        if key == 'lh':
            # LH (水平细节) -> 按行 (row) 分解
            for i in range(original_subband.shape[0]):
                s, n = _1d_ssa_decompose(original_subband[i, :], params['L_1d'])
                struc[i, :], noise[i, :] = s, n
        elif key == 'hl':
            # HL (垂直细节) -> 按列 (column) 分解
            for i in range(original_subband.shape[1]):
                s, n = _1d_ssa_decompose(original_subband[:, i], params['L_1d'])
                struc[:, i], noise[:, i] = s, n
        elif key == 'hh':
            # HH (对角线) -> 2D 分解
            struc, noise = _2d_ssa_decompose(original_subband, params['L_2d_h'], params['L_2d_w'])
        # [!!! 修复结束 !!!]

        # Reconstruct images: make sure to pass tuple in (HL, LH, HH) order as pywt expects
        coeffs_struc = [LL, (struc if key == 'hl' else HL_orig,
                             struc if key == 'lh' else LH_orig,
                             struc if key == 'hh' else HH_orig)]
        coeffs_noise = [LL, (noise if key == 'hl' else HL_orig,
                             noise if key == 'lh' else LH_orig,
                             noise if key == 'hh' else HH_orig)]

        img_struc = np.clip(pywt.waverec2(coeffs_struc, WAVELET, mode='symmetric')[:H, :W], 0.0, 1.0)
        img_noise = np.clip(pywt.waverec2(coeffs_noise, WAVELET, mode='symmetric')[:H, :W], 0.0, 1.0)

        images_to_plot[key] = {
            'struc_only': img_struc,
            'noise_only': img_noise
        }

    print("--- All reconstructions complete. Generating plot... ---")

    plt.style.use('dark_background')
    fig, axes = plt.subplots(3, 4, figsize=(24, 18))
    fig.patch.set_facecolor('black')

    row_titles = {'hl': "Vertical (HL)", 'lh': "Horizontal (LH)", 'hh': "Diagonal (HH)"}
    col_titles = ["Original Image", "Structure Layer Only", "Detail/Noise Layer Only", "What Was Removed (Amplified)"]

    for i, key in enumerate(['hl', 'lh', 'hh']):
        img_struc = images_to_plot[key]['struc_only']
        img_noise = images_to_plot[key]['noise_only']
        residual = baseline_img - img_struc
        amplified_residual = np.clip(0.5 + residual * AMPLIFICATION, 0, 1)

        axes[i, 0].imshow(baseline_img, cmap='gray')
        axes[i, 1].imshow(img_struc, cmap='gray')
        axes[i, 2].imshow(img_noise, cmap='gray', vmin=np.percentile(img_noise, 5), vmax=np.percentile(img_noise, 95))
        axes[i, 3].imshow(amplified_residual, cmap='gray')

        axes[i, 0].set_ylabel(row_titles[key], fontsize=22, rotation=90, labelpad=20)

    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=20, pad=20)

    for ax in axes.flatten():
        ax.set_xticks([]);
        ax.set_yticks([])

    fig.suptitle("Visual Proof: Deconstructing Image Layers with Direction-Adaptive SSA", fontsize=30, color='white')
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])
    plt.savefig("framework_visual_proof_fixed.png", facecolor='black', dpi=150)
    plt.show()


if __name__ == '__main__':
    main()