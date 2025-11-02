# visualize_attack.py (最终版 - 已修改为支持多数据集和动态尺寸)
#
# --- 目的 ---
# 1. 原始的干净图像。
# 2. 经过我们复杂攻击流程后生成的中毒图像。
# 3. 两者之间的像素级差异（残差），并进行放大以供分析。
#
# --- 如何运行 ---
# (确保您已安装所有需要的库: pip install torch torchvision PyWavelets matplotlib Pillow pyyaml)
#
# 示例 (CIFAR-10):
# python visualize.py --config ./configs/cifar10_resnet18_random.yaml --idx 123
#
# 示例 (GTSRB, 64x64):
# python visualize.py --config ./configs/gtsrb_64x64_random.yaml --idx 500

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import argparse
import numpy as np
import yaml
from PIL import Image
import os

# [!!! 核心修改 !!!]
# 导入我们新的、需要 image_size 的 get_injector_instance
from core.attack import get_injector_instance


def load_config(config_path):
    """一个简单的函数，用于从YAML文件加载配置。"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main(args):
    # 1. 加载配置文件
    print(f"--- 加载配置文件: {args.config} ---")
    config = load_config(args.config)

    # [!!! 核心修改 !!!]
    # 从 config 中读取数据集名称和图像尺寸
    dataset_config = config['dataset']
    dataset_name = dataset_config.get('name', 'cifar10').lower()
    image_size = dataset_config.get('image_size', 32)  # 默认为 32
    data_path = dataset_config.get('data_path', './data')

    print(f"--- 配置检测到: Dataset={dataset_name}, ImageSize={image_size}x{image_size} ---")

    # 2. [核心] 初始化我们功能强大的S2A注入器
    # [!!! 核心修改 !!!]
    # 将 image_size 传递给注入器，以便它生成正确尺寸的触发器
    print("--- 正在初始化S2A注入器 (可能需要一点时间)... ---")
    injector = get_injector_instance(config, image_size)
    print("--- 注入器初始化完成！ ---")

    # 3. [!!! 核心修改 !!!] 加载动态数据集
    # 注意：这里我们只进行Resize，不进行ToTensor()或Normalize()
    print(f"--- 正在加载数据集: {dataset_name} ---")

    # 定义可视化所需的变换
    resize_transform = transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.LANCZOS)
    tensor_transform = transforms.ToTensor()

    # 加载不带变换的原始PIL数据集
    classes = None
    if dataset_name == 'cifar10':
        dataset_raw = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=None)
        classes = dataset_raw.classes
    elif dataset_name == 'gtsrb':
        dataset_raw = torchvision.datasets.GTSRB(root=data_path, split='train', download=True, transform=None)
        # classes = None (GTSRB .classes 属性不可靠)
    elif dataset_name == 'tiny_imagenet':
        data_dir = os.path.join(data_path, 'tiny-imagenet-200', 'train')
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"未找到 Tiny ImageNet 训练数据: {data_dir}")
        dataset_raw = torchvision.datasets.ImageFolder(root=data_dir, transform=None)
        classes = dataset_raw.classes
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")

    # 4. 获取一张干净的图像 (PIL)
    pil_clean_raw, label = dataset_raw[args.idx]

    # (确保是RGB)
    pil_clean_raw = pil_clean_raw.convert('RGB')

    # 4a. [!!! 核心修改 !!!]
    # 创建用于绘图的 clean_pil (已缩放)
    clean_pil = resize_transform(pil_clean_raw)

    # 4b. 创建用于注入的 clean_tensor (已缩放 + ToTensor)
    clean_tensor = tensor_transform(clean_pil)  # (C, H, W)

    if classes:
        try:
            label_name = classes[label]
            print(f"--- 已加载 {dataset_name} 图像 #{args.idx} (标签: {label_name}) ---")
        except:  # (TinyImageNet 标签可能是字符串)
            print(f"--- 已加载 {dataset_name} 图像 #{args.idx} (标签: {label}) ---")
    else:
        print(f"--- 已加载 {dataset_name} 图像 #{args.idx} (标签: {label}) ---")

    # 5. [核心] 生成中毒版本的图像张量
    print("--- 正在对图像进行攻击注入... ---")
    # 我们的 injector.inject 期望 (C, H, W)
    poisoned_tensor = injector.inject(clean_tensor.clone())
    print("--- 注入完成！ ---")

    # 6. 计算差异（残差）
    diff_tensor = poisoned_tensor - clean_tensor

    # 7. [核心] 将极其微弱的残差“放大”到[0, 1]范围以便可视化
    if diff_tensor.max().item() > diff_tensor.min().item():
        amplified_diff = (diff_tensor - diff_tensor.min()) / (diff_tensor.max() - diff_tensor.min())
    else:
        amplified_diff = torch.zeros_like(diff_tensor)  # 如果没有差异，则显示全黑

    # 8. 将用于绘图的Tensors转换回PIL图像
    to_pil = transforms.ToPILImage()
    # (clean_pil 已经有了)
    poisoned_pil = to_pil(poisoned_tensor)
    diff_pil = to_pil(amplified_diff)

    # 9. 使用Matplotlib绘图
    print("--- 正在生成对比图... ---")
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    fig.patch.set_facecolor('black')
    plt.style.use('dark_background')

    axes[0].imshow(clean_pil)
    axes[0].set_title("Original Clean Image (Resized)", fontsize=16)

    axes[1].imshow(poisoned_pil)
    axes[1].set_title("Poisoned Image", fontsize=16)

    axes[2].imshow(diff_pil)
    axes[2].set_title("Amplified Residual (Normalized)", fontsize=16)

    for ax in axes:
        ax.axis('off')

    plt.suptitle(f"S2A Attack Invisibility Analysis ({dataset_name} #{args.idx})", fontsize=20, color='white')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = f"{dataset_name}_attack_visualization_idx_{args.idx}.png"
    plt.savefig(save_path, facecolor='black', dpi=150)
    print(f"\n可视化结果已保存到: {save_path}")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize S2A Attack Invisibility')
    # [!!! 核心修改 !!!]
    # 默认的 config 不再硬编码为 cifar10
    parser.add_argument('--config', default='./configs/gtsrb_64x64_random.yaml',
                        help='Path to the YAML config file')
    parser.add_argument('--idx', type=int, default=500,
                        help='Index of the image to visualize (e.g., 500 for GTSRB)')
    args = parser.parse_args()

    main(args)