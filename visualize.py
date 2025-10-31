# visualize_attack.py (最终版)
#
# --- 目的 ---
# 本脚本旨在通过可视化，直观地评估我们最终版S2A攻击的“隐蔽性”。
# 它将并排展示三张关键图像：
# 1. 原始的干净图像。
# 2. 经过我们复杂攻击流程后生成的中毒图像。
# 3. 两者之间的像素级差异（残差），并进行放大以供分析。
#
# --- 如何运行 ---
# (确保您已安装所有需要的库: pip install torch torchvision PyWavelets matplotlib Pillow pyyaml)
# python visualize_attack.py --config ./configs/cifar10_resnet18.yaml --idx <您想看的图片索引>
#
# 示例: python visualize_attack.py --config ./configs/cifar10_resnet18.yaml --idx 123

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import argparse
import numpy as np
import yaml
from PIL import Image

# 从我们的核心攻击文件中导入最终版的注入器
from core.attack import S2A_Final_Injector as S2A_Ultimate_Injector


def load_config(config_path):
    """一个简单的函数，用于从YAML文件加载配置。"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main(args):
    # 1. 加载配置文件
    print(f"--- 加载配置文件: {args.config} ---")
    config = load_config(args.config)

    # 2. [核心] 初始化我们功能强大的S2A注入器
    # 这一步现在会执行“触发器提纯”等预计算，可能会稍慢一些。
    print("--- 正在初始化S2A注入器 (可能需要一点时间)... ---")
    injector = S2A_Ultimate_Injector(config)
    print("--- 注入器初始化完成！ ---")

    # 3. 加载CIFAR-10数据集
    # 注意：这里我们只进行ToTensor()，不进行Normalize()，因为我们需要在[0,1]的像素空间进行可视化
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

    # 4. 获取一张干净的图像张量
    clean_tensor, label = dataset[args.idx]
    # 添加一个batch维度，因为某些模型可能需要(B, C, H, W)格式
    # 虽然我们的inject函数不需要，但这是个好习惯
    clean_tensor = clean_tensor.unsqueeze(0)

    print(f"--- 已加载CIFAR-10图像 #{args.idx} (标签: {dataset.classes[label]}) ---")

    # 5. [核心] 生成中毒版本的图像张量
    print("--- 正在对图像进行攻击注入... ---")
    # 使用 .clone() 确保不修改原始的 clean_tensor
    # .squeeze(0) 将batch维度去掉，以便于后续处理
    poisoned_tensor = injector.inject(clean_tensor.clone().squeeze(0))
    clean_tensor = clean_tensor.squeeze(0)  # 将原始图像也去掉batch维度
    print("--- 注入完成！ ---")

    # 6. 计算差异（残差）
    diff_tensor = poisoned_tensor - clean_tensor

    # 7. [核心] 将极其微弱的残差“放大”到[0, 1]范围以便可视化
    # 我们通过min-max归一化来拉伸对比度
    if diff_tensor.max().item() > diff_tensor.min().item():
        amplified_diff = (diff_tensor - diff_tensor.min()) / (diff_tensor.max() - diff_tensor.min())
    else:
        amplified_diff = torch.zeros_like(diff_tensor)  # 如果没有差异，则显示全黑

    # 8. 将用于绘图的Tensors转换回PIL图像
    to_pil = transforms.ToPILImage()
    clean_pil = to_pil(clean_tensor)
    poisoned_pil = to_pil(poisoned_tensor)
    diff_pil = to_pil(amplified_diff)

    # 9. 使用Matplotlib绘图
    print("--- 正在生成对比图... ---")
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    fig.patch.set_facecolor('black')
    plt.style.use('dark_background')

    axes[0].imshow(clean_pil)
    axes[0].set_title("Original Clean Image", fontsize=16)

    axes[1].imshow(poisoned_pil)
    axes[1].set_title("Poisoned Image", fontsize=16)

    axes[2].imshow(diff_pil)
    axes[2].set_title("Amplified Residual (Normalized)", fontsize=16)

    for ax in axes:
        ax.axis('off')

    plt.suptitle(f"S2A Attack Invisibility Analysis (Image Index: {args.idx})", fontsize=20, color='white')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    save_path = f"attack_visualization_idx_{args.idx}.png"
    plt.savefig(save_path, facecolor='black', dpi=150)
    print(f"\n可视化结果已保存到: {save_path}")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize S2A Attack Invisibility')
    parser.add_argument('--config', default='./configs/cifar10_resnet18.yaml',
                        help='Path to the YAML config file')
    parser.add_argument('--idx', type=int, default=8,
                        help='Index of the CIFAR-10 image to visualize')
    args = parser.parse_args()

    main(args)