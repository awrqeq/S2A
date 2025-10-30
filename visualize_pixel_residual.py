# visualize_pixel_residual.py
#
# --- 目的 ---
# 本脚本旨在通过像素域的可视化，直观地展示2D-SSA从原始图像中分离出的“噪声”到底是什么视觉信息。
# 它将回答核心问题：“当我只用‘结构’部分重构图像时，丢失了什么？”
# 这将为我们的方法论提供最强有力的、符合直觉的视觉证据。
#
# --- 如何运行 ---
# pip install torch torchvision numpy PyWavelets matplotlib Pillow scikit-image
# python visualize_pixel_residual.py
#
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pywt
import matplotlib.pyplot as plt
from skimage.color import rgb2gray  # 用于更精确的灰度转换


# (我们复用之前脚本中的2D-SSA分析函数)
def analyze_2d_ssa(matrix_2d, Lh, Lw, r):
    H, W = matrix_2d.shape
    if H < Lh or W < Lw: return matrix_2d, np.zeros_like(matrix_2d)
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
    try:
        U, S, Vh = np.linalg.svd(trajectory_matrix, full_matrices=False)
    except np.linalg.LinAlgError:
        return matrix_2d, np.zeros_like(matrix_2d)
    rank = min(r, len(S))
    reconstructed_trajectory = U[:, :rank] @ np.diag(S[:rank]) @ Vh[:rank, :]
    reconstructed_structure = np.zeros_like(matrix_2d)
    counts = np.zeros_like(matrix_2d)
    patch_idx = 0
    for i in range(Kh):
        for j in range(Kw):
            reconstructed_patch = reconstructed_trajectory[:, patch_idx].reshape(Lh, Lw)
            reconstructed_structure[i:i + Lh, j:j + Lw] += reconstructed_patch
            counts[i:i + Lh, j:j + Lw] += 1
            patch_idx += 1
    reconstructed_structure[counts > 0] /= counts[counts > 0]
    return reconstructed_structure


def main():
    print("--- 开始进行2D-SSA像素域残差可视化分析 ---")

    # --- 1. 参数配置 ---
    IMAGE_IDX = 3  # (3: 猫, 5: 狗, 8: 船, 2: 鸟 - 鸟的效果也很好)
    WAVELET = 'db4'
    WINDOW_SIZE_H = 4
    WINDOW_SIZE_W = 4
    N_COMPONENTS_R = 5
    RESIDUAL_AMPLIFICATION = 20  # [!!!] 残差放大倍数，可以调整以获得最佳可视化效果

    # --- 2. 加载图片并转为灰度Numpy数组 ---
    cifar_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
    pil_image, _ = cifar_dataset[IMAGE_IDX]
    image_np_gray = np.array(pil_image.convert('L')).astype(np.float64) / 255.0

    print(f"已加载CIFAR-10图像 #{IMAGE_IDX}")

    # --- 3. DWT分解与重构 ---
    coeffs_original = pywt.wavedec2(image_np_gray, WAVELET, mode='symmetric', level=1)
    LL, (LH, HL, HH_original) = coeffs_original

    # [基准] 将原始系数直接重构回来，作为最精确的对比基准
    baseline_reconstructed_img = pywt.waverec2(coeffs_original, WAVELET, mode='symmetric')
    # 裁剪到与原始图像相同的尺寸
    h, w = image_np_gray.shape
    baseline_reconstructed_img = baseline_reconstructed_img[:h, :w]

    print("已生成‘基准重构图像’")

    # --- 4. [核心] 只用SSA提取的“结构”重构图像 ---
    print("正在对HH子带进行2D-SSA分析并提取‘结构’...")
    hh_structure = analyze_2d_ssa(HH_original, WINDOW_SIZE_H, WINDOW_SIZE_W, N_COMPONENTS_R)

    coeffs_structure_only = [LL, (LH, HL, hh_structure)]
    structure_reconstructed_img = pywt.waverec2(coeffs_structure_only, WAVELET, mode='symmetric')
    structure_reconstructed_img = structure_reconstructed_img[:h, :w]

    print("已生成‘仅结构重构图像’")

    # --- 5. 计算并放大残差 ---
    print("正在计算像素级残差并放大...")
    # 计算残差，即被SSA当作“噪声”并丢弃掉的像素信息
    pixel_residual = baseline_reconstructed_img - structure_reconstructed_img

    # 放大残差以便可视化。为了防止正负抵消，我们取绝对值或者做一个偏移
    # 一个好的可视化方法是：将残差平移到0.5中心，然后放大
    amplified_residual = 0.5 + pixel_residual * RESIDUAL_AMPLIFICATION
    amplified_residual = np.clip(amplified_residual, 0, 1)  # 裁剪到[0,1]范围

    # --- 6. 可视化最终结果 ---
    print("正在生成可视化图表...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    # 子图1: 基准重构图像
    axes[0].imshow(baseline_reconstructed_img, cmap='gray')
    axes[0].set_title('Original Image (Reconstructed)', fontsize=14)
    axes[0].axis('off')

    # 子图2: 仅结构重构图像 (肉眼看起来应该和(a)完全一样)
    axes[1].imshow(structure_reconstructed_img, cmap='gray')
    axes[1].set_title('Reconstructed from HH \'Structure\'', fontsize=14)
    axes[1].axis('off')

    # 子图3: [!!!] 被放大的残差图 (被丢弃的“噪声”)
    im = axes[2].imshow(amplified_residual, cmap='gray')
    axes[2].set_title(f'Amplified Pixel Residual (x{RESIDUAL_AMPLIFICATION})', fontsize=14)
    axes[2].axis('off')

    fig.suptitle('Visualizing What 2D-SSA Removes from the Image', fontsize=20)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    save_path = "ssa_pixel_residual_analysis.png"
    plt.savefig(save_path)
    print(f"\n分析结果已保存到: {save_path}")

    plt.show()


if __name__ == '__main__':
    main()