# visualize_ssa_effect.py
#
# --- 目的 ---
# 本脚本旨在通过可视化，清晰地验证二维奇异谱分析(2D-SSA)在我们方法中的核心作用：
# 即成功地将小波HH子带分解为“结构(Structure)”和“噪声(Noise)”两个主要部分。
# 这是论证我们“在噪声空间注入触发器”这一创新点有效性的关键实验证据。
#
# --- 如何运行 ---
# pip install torch torchvision numpy PyWavelets matplotlib Pillow
# python visualize_ssa_effect.py
#
# --- 预期输出 ---
# 将会生成并显示一张名为 "ssa_decomposition_analysis.png" 的图片，
# 其中包含三个子图：原始HH子带、提取出的结构、以及剩下的噪声。
#
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pywt
import matplotlib.pyplot as plt


# --- [核心] 2D-SSA分析函数 ---
# 我们从attack.py中提取并改造了2D-SSA的核心逻辑，使其返回中间过程用于分析。
def analyze_2d_ssa(matrix_2d, Lh, Lw, r):
    """
    对一个二维矩阵进行2D-SSA分析，并返回其结构和噪声部分。

    Args:
        matrix_2d (np.array): 输入的二维矩阵 (例如HH子带)。
        Lh (int): 2D滑动窗口的高度。
        Lw (int): 2D滑动窗口的宽度。
        r (int): 用于重构结构的主成分数量。

    Returns:
        tuple: (reconstructed_structure, residual_noise)
    """
    H, W = matrix_2d.shape
    if H < Lh or W < Lw:
        # 如果矩阵太小，无法进行分析，则返回原矩阵和零噪声
        return matrix_2d, np.zeros_like(matrix_2d)

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
    try:
        U, S, Vh = np.linalg.svd(trajectory_matrix, full_matrices=False)
    except np.linalg.LinAlgError:
        # 如果SVD失败，返回原矩阵
        return matrix_2d, np.zeros_like(matrix_2d)

    # 3. 重构结构部分
    rank = min(r, len(S))
    reconstructed_trajectory = U[:, :rank] @ np.diag(S[:rank]) @ Vh[:rank, :]

    # 4. "对角平均"重构回二维矩阵，得到“结构”
    reconstructed_structure = np.zeros_like(matrix_2d)
    counts = np.zeros_like(matrix_2d)
    patch_idx = 0
    for i in range(Kh):
        for j in range(Kw):
            reconstructed_patch = reconstructed_trajectory[:, patch_idx].reshape(Lh, Lw)
            reconstructed_structure[i:i + Lh, j:j + Lw] += reconstructed_patch
            counts[i:i + Lh, j:j + Lw] += 1
            patch_idx += 1

    # 避免除以零
    reconstructed_structure[counts > 0] /= counts[counts > 0]

    # 5. [关键证据] 计算剩下的“噪声”部分
    residual_noise = matrix_2d - reconstructed_structure

    return reconstructed_structure, residual_noise


def main():
    print("--- 开始进行2D-SSA有效性可视化分析 ---")

    # --- 1. 参数配置 ---
    # 这里我们使用您为CIFAR-10 HH子带推荐的参数
    IMAGE_IDX = 3  # 选择一张有代表性的图片 (3: 猫, 5: 狗, 8: 船)
    WAVELET = 'db4'
    WINDOW_SIZE_H = 4
    WINDOW_SIZE_W = 4
    N_COMPONENTS_R = 5

    print(f"参数配置: 窗口={WINDOW_SIZE_H}x{WINDOW_SIZE_W}, 主成分数量r={N_COMPONENTS_R}")

    # --- 2. 加载并预处理一张CIFAR-10图片 ---
    transform = transforms.Compose([transforms.ToTensor()])
    cifar_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
    pil_image, _ = cifar_dataset[IMAGE_IDX]

    # 为了简化分析，我们将其转换为灰度图进行处理
    gray_image = pil_image.convert('L')
    image_np = np.array(gray_image).astype(np.float64) / 255.0  # 归一化

    print(f"已加载CIFAR-10图像 #{IMAGE_IDX}, 形状: {image_np.shape}")

    # --- 3. 进行小波分解，获取HH子带 ---
    try:
        coeffs = pywt.wavedec2(image_np, WAVELET, mode='symmetric', level=1)
        _, (_, _, hh_original) = coeffs
        print(f"已获取HH子带, 形状: {hh_original.shape}")
    except ValueError as e:
        print(f"小波分解失败: {e}")
        return

    # --- 4. [核心] 运行2D-SSA分析 ---
    print("正在对HH子带进行2D-SSA分解...")
    hh_structure, hh_noise = analyze_2d_ssa(hh_original, WINDOW_SIZE_H, WINDOW_SIZE_W, N_COMPONENTS_R)
    print("分解完成！")

    # --- 5. 可视化结果 ---
    print("正在生成可视化图表...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 计算一个统一的颜色范围，以便于对比 Original 和 Structure
    vmin = min(hh_original.min(), hh_structure.min())
    vmax = max(hh_original.max(), hh_structure.max())

    # 子图1: 原始HH子带
    im1 = axes[0].imshow(hh_original, cmap='gray', vmin=vmin, vmax=vmax)
    axes[0].set_title('Original HH Subband', fontsize=16)
    axes[0].axis('off')

    # 子图2: 重构的结构部分
    im2 = axes[1].imshow(hh_structure, cmap='gray', vmin=vmin, vmax=vmax)
    axes[1].set_title(f'Reconstructed Structure (r={N_COMPONENTS_R})', fontsize=16)
    axes[1].axis('off')

    # 子图3: 剩下的噪声部分
    # 我们让噪声图自动调整颜色范围，以凸显其细节
    im3 = axes[2].imshow(hh_noise, cmap='gray')
    axes[2].set_title('Residual Noise', fontsize=16)
    axes[2].axis('off')

    # 添加主标题和颜色条
    fig.suptitle('2D-SSA Decomposition Analysis on HH Subband', fontsize=20)
    # 为每个子图添加独立的颜色条以精确显示
    fig.colorbar(im1, ax=axes[0], orientation='vertical', fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=axes[1], orientation='vertical', fraction=0.046, pad=0.04)
    fig.colorbar(im3, ax=axes[2], orientation='vertical', fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # 保存图像
    save_path = "ssa_decomposition_analysis.png"
    plt.savefig(save_path)
    print(f"\n分析结果已保存到: {save_path}")

    # 显示图像
    plt.show()


if __name__ == '__main__':
    main()