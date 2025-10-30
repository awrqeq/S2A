# visualize_hl_structure_removal.py
#
# --- 目的 ---
# 本脚本旨在通过像素域的可视化，以一种极具冲击力和符合直觉的方式，
# 证明我们的1D-SSA方法能够精确地识别并分离出中频子带（以HL为例）中的核心“结构”——即图像的边缘和线条。
# 这将是论证我们整个方法论有效性的、无可辩驳的视觉证据。
#
# --- 如何运行 ---
# pip install torch torchvision numpy PyWavelets matplotlib Pillow scikit-image
# python visualize_hl_structure_removal.py
#
import torch
import torchvision
import numpy as np
import pywt
import matplotlib.pyplot as plt


# --- [核心] 1D-SSA 按行分析函数 ---
def analyze_1d_ssa_row_wise(subband_data, L, r):
    """对子带矩阵的每一行进行1D-SSA，并返回重构的结构部分"""
    H, W = subband_data.shape
    reconstructed_structure = np.zeros_like(subband_data)

    for i in range(H):
        signal_1d = subband_data[i, :]
        N = len(signal_1d)
        if N < L:
            reconstructed_structure[i, :] = signal_1d
            continue

        K = N - L + 1
        hankel = np.array([signal_1d[j:j + L] for j in range(K)]).T

        try:
            U, S, Vh = np.linalg.svd(hankel, full_matrices=False)
        except np.linalg.LinAlgError:
            reconstructed_structure[i, :] = signal_1d
            continue

        rank = min(r, len(S))
        reconstructed_hankel = U[:, :rank] @ np.diag(S[:rank]) @ Vh[:rank, :]

        # 反向对角平均重构回一维信号
        temp_signal = np.zeros(N)
        counts = np.zeros(N)
        for j in range(L):
            for k in range(K):
                temp_signal[j + k] += reconstructed_hankel[j, k]
                counts[j + k] += 1
        temp_signal[counts > 0] /= counts[counts > 0]
        reconstructed_structure[i, :] = temp_signal

    return reconstructed_structure


def main():
    print("--- 开始进行HL子带结构移除的可视化分析 ---")

    # --- 1. 参数配置 ---
    IMAGE_IDX = 8  # [!!!] 强烈推荐使用索引8 (船)，它有非常清晰的水平线
    WAVELET = 'db4'
    WINDOW_SIZE_L = 8
    N_COMPONENTS_R = 1  # 只提取最强的1个主成分，效果最明显
    RESIDUAL_AMPLIFICATION = 10

    # --- 2. 加载图片并转为灰度Numpy数组 ---
    cifar_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
    pil_image, _ = cifar_dataset[IMAGE_IDX]
    image_np_gray = np.array(pil_image.convert('L')).astype(np.float64) / 255.0

    # --- 3. DWT分解 ---
    coeffs_original = pywt.wavedec2(image_np_gray, WAVELET, mode='symmetric', level=1)
    LL, (LH, HL_original, HH) = coeffs_original

    # --- 4. [核心] 分离HL子带的结构与噪声 ---
    print("正在对HL子带进行1D-SSA分析，提取‘水平线结构’...")
    hl_structure = analyze_1d_ssa_row_wise(HL_original, WINDOW_SIZE_L, N_COMPONENTS_R)
    # 剩下的就是噪声
    hl_noise = HL_original - hl_structure

    # --- 5. 进行两次关键的重构 ---
    print("正在重构‘基准图像’和‘移除结构后的图像’...")
    # 基准图像 (使用原始HL)
    baseline_reconstructed_img = pywt.waverec2(coeffs_original, WAVELET, mode='symmetric')

    # 移除结构后的图像 (只使用HL的噪声部分)
    coeffs_without_structure = [LL, (LH, hl_noise, HH)]
    img_without_structure = pywt.waverec2(coeffs_without_structure, WAVELET, mode='symmetric')

    # 裁剪到原始尺寸
    h, w = image_np_gray.shape
    baseline_reconstructed_img = baseline_reconstructed_img[:h, :w]
    img_without_structure = img_without_structure[:h, :w]

    # --- 6. 计算并放大残差 ---
    pixel_residual = baseline_reconstructed_img - img_without_structure
    amplified_residual = 0.5 + pixel_residual * RESIDUAL_AMPLIFICATION
    amplified_residual = np.clip(amplified_residual, 0, 1)

    # --- 7. 可视化最终结果 ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.patch.set_facecolor('black')  # 设置背景为黑色以凸显图像
    plt.style.use('dark_background')  # 使用深色主题

    axes[0].imshow(baseline_reconstructed_img, cmap='gray')
    axes[0].set_title('Original Image (Reconstructed)', fontsize=16)
    axes[0].axis('off')

    axes[1].imshow(img_without_structure, cmap='gray')
    axes[1].set_title('Image Reconstructed WITHOUT HL Structure', fontsize=16)
    axes[1].axis('off')

    axes[2].imshow(amplified_residual, cmap='gray')
    axes[2].set_title(f'What Was Removed (Amplified x{RESIDUAL_AMPLIFICATION})', fontsize=16)
    axes[2].axis('off')

    fig.suptitle('Visualizing SSA\'s Precise Removal of Horizontal Structures', fontsize=20, color='white')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    save_path = "ssa_hl_removal_analysis.png"
    plt.savefig(save_path, facecolor='black')
    print(f"\n分析结果已保存到: {save_path}")

    plt.show()


if __name__ == '__main__':
    main()