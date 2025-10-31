# verify_wavelet_orientation.py
#
# --- 目的 ---
# 这是一个终极的、无可辩驳的验证脚本，用于一劳永逸地确定pywt库中
# LH和HL子带到底哪个对应“水平”结构，哪个对应“垂直”结构。
# 我们不再依赖任何文档解释或个人记忆，只相信代码的输出。

import numpy as np
import pywt
import matplotlib.pyplot as plt


def main():
    # --- 1. 创造一个只有“水平线”的宇宙 ---
    print("--- Testing HORIZONTAL line... ---")
    image_horizontal = np.zeros((32, 32))
    image_horizontal[15:17, :] = 1.0  # 在中间画一条粗的水平白线

    # --- 2. 用小波变换观察它 ---
    coeffs_h = pywt.wavedec2(image_horizontal, 'db4', level=1)
    _, (LH_h, HL_h, HH_h) = coeffs_h

    # --- 3. 创造一个只有“垂直线”的宇宙 ---
    print("--- Testing VERTICAL line... ---")
    image_vertical = np.zeros((32, 32))
    image_vertical[:, 15:17] = 1.0  # 在中间画一条粗的垂直白线

    # --- 4. 用小波变换观察它 ---
    coeffs_v = pywt.wavedec2(image_vertical, 'db4', level=1)
    _, (LH_v, HL_v, HH_v) = coeffs_v

    # --- 5. 计算每个子带的能量，用数字说话 ---
    energy_LH_from_horizontal = np.sum(LH_h ** 2)
    energy_HL_from_horizontal = np.sum(HL_h ** 2)

    energy_LH_from_vertical = np.sum(LH_v ** 2)
    energy_HL_from_vertical = np.sum(HL_v ** 2)

    print(f"\n--- ENERGY ANALYSIS RESULTS ---")
    print(f"When input is HORIZONTAL line:")
    print(f"  - Energy in LH subband: {energy_LH_from_horizontal:.4f}")
    print(f"  - Energy in HL subband: {energy_HL_from_horizontal:.4f}")

    print(f"\nWhen input is VERTICAL line:")
    print(f"  - Energy in LH subband: {energy_LH_from_vertical:.4f}")
    print(f"  - Energy in HL subband: {energy_HL_from_vertical:.4f}")

    # --- 6. 可视化，眼见为实 ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("The Ultimate Litmus Test for Wavelet Orientation", fontsize=20)

    # 第一行: 水平线实验
    axes[0, 0].imshow(image_horizontal, cmap='gray');
    axes[0, 0].set_title("Input: Horizontal Line")
    axes[0, 1].imshow(LH_h, cmap='gray');
    axes[0, 1].set_title("Resulting LH Subband")
    axes[0, 2].imshow(HL_h, cmap='gray');
    axes[0, 2].set_title("Resulting HL Subband")

    # 第二行: 垂直线实验
    axes[1, 0].imshow(image_vertical, cmap='gray');
    axes[1, 0].set_title("Input: Vertical Line")
    axes[1, 1].imshow(LH_v, cmap='gray');
    axes[1, 1].set_title("Resulting LH Subband")
    axes[1, 2].imshow(HL_v, cmap='gray');
    axes[1, 2].set_title("Resulting HL Subband")

    for ax in axes.flatten(): ax.axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


if __name__ == '__main__':
    main()