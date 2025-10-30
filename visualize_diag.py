# visualize_diag.py (Corrected Version)
# 诊断脚本：执行“空攻击”来测试 pywt 分解/重建过程的保真度
# [已修复] 将错误的 pywt.waverecn 调用更正为正确的 pywt.waverec2

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import argparse
import numpy as np
import pywt
from PIL import Image

from core.utils import load_config


def reconstruct_without_attack(tensor_clean, wavelet, mode):
    """
    对一个干净的 tensor 执行分解和立即重建，不进行任何修改。
    这是为了测试 pywt 变换本身的可逆性。
    """
    img_np = tensor_clean.cpu().numpy().astype(np.float64)
    reconstructed_channels = []

    for c in range(img_np.shape[0]):
        channel_data = img_np[c]
        orig_shape = channel_data.shape

        # 1. 分解 (使用 2D 分解函数)
        try:
            coeffs_list = pywt.wavedec2(channel_data, wavelet, mode=mode, level=1)
        except ValueError:
            return tensor_clean  # 如果无法分解，返回原图

        # 2. 不做任何修改，立即重建
        # [!!! 核心修复 !!!]
        # 使用与 wavedec2 配对的 waverec2 函数，而不是通用的 waverecn
        reconstructed_channel = pywt.waverec2(coeffs_list, wavelet, mode=mode)

        # 3. 裁剪以匹配原始尺寸
        if reconstructed_channel.shape != orig_shape:
            reconstructed_channel = reconstructed_channel[:orig_shape[0], :orig_shape[1]]

        reconstructed_channels.append(reconstructed_channel)

    img_reconstructed_np = np.stack(reconstructed_channels, axis=0)
    img_reconstructed_np = np.clip(img_reconstructed_np, 0.0, 1.0)
    return torch.tensor(img_reconstructed_np, dtype=torch.float32, device=tensor_clean.device)


def main(args):
    # 1. 加载配置
    config = load_config(args.config)
    wavelet_config = config['attack']

    # 2. 加载干净的 CIFAR-10 数据集
    dataset_config = config['dataset']
    clean_dataset = datasets.CIFAR10(root=dataset_config['data_path'], train=True, download=True)

    # 3. 获取一张干净的 PIL 图像和 Tensor
    pil_clean, label = clean_dataset[args.idx]
    tensor_clean_unnormalized = transforms.ToTensor()(pil_clean)

    # 4. [核心诊断步骤] 对干净图像进行“空攻击” (分解 -> 重建)
    tensor_reconstructed_clean = reconstruct_without_attack(
        tensor_clean_unnormalized.clone(),
        wavelet=wavelet_config['wavelet'],
        mode='symmetric'
    )

    # 5. 计算原始图像和“重建后的干净图像”之间的差异
    diff_tensor = tensor_reconstructed_clean - tensor_clean_unnormalized

    # 6. 放大差异以便可视化
    if diff_tensor.max() != diff_tensor.min():
        amplified_diff = (diff_tensor - diff_tensor.min()) / (diff_tensor.max() - diff_tensor.min())
    else:
        amplified_diff = torch.zeros_like(diff_tensor)

    # 7. 转换为 PIL 图像
    to_pil = transforms.ToPILImage()
    img_reconstructed_pil = to_pil(tensor_reconstructed_clean)
    img_diff_pil = to_pil(amplified_diff)

    # 8. 绘图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(pil_clean)
    axes[0].set_title(f"Original Clean Image (Label: {label})")
    axes[0].axis('off')

    axes[1].imshow(img_reconstructed_pil)
    axes[1].set_title("Reconstructed Clean Image (No Attack)")
    axes[1].axis('off')

    axes[2].imshow(img_diff_pil)
    axes[2].set_title("Amplified Difference (Reconstruction Error)")
    axes[2].axis('off')

    plt.suptitle(f"PyWT Reconstruction Fidelity Check (Image Index: {args.idx})")
    plt.tight_layout()

    save_path = "s2a_DIAGNOSIS.png"
    plt.savefig(save_path)
    print(f"DIAGNOSIS visualization saved to {save_path}")

    # 取消注释以在本地显示图像
    # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Diagnose PyWT Reconstruction Artifacts')
    parser.add_argument('--config', default='./configs/cifar10_resnet18.yaml', help='Path to YAML config file')
    parser.add_argument('--idx', type=int, default=100, help='Index of the CIFAR-10 image to visualize')
    args = parser.parse_args()
    main(args)