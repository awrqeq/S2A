# visualize.py
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import argparse
import numpy as np
from PIL import Image

from core.utils import load_config
from core.attack import S2A_Ultimate_Injector as S2A_Injector


def main(args):
    # 1. 加载配置和 S2A 注入器
    config = load_config(args.config)
    injector = S2A_Injector(config)

    # 2. 加载干净的 CIFAR-10 数据集 (只加载，不应用归一化)
    dataset_config = config['dataset']
    clean_dataset = datasets.CIFAR10(
        root=dataset_config['data_path'],
        train=True,
        download=True
    )

    # 3. 获取一张干净的 PIL 图像
    pil_clean, label = clean_dataset[args.idx]

    # 4. 将 PIL 图像转换为 [0, 1] 范围的 Tensor
    #    (这是 S2A_Injector.inject 方法所期望的输入格式)
    to_tensor = transforms.ToTensor()
    tensor_clean_unnormalized = to_tensor(pil_clean)

    # 5. [核心] 创建带毒版本的 Tensor
    #    使用 .clone() 确保不修改原始 tensor
    tensor_poisoned_unnormalized = injector.inject(tensor_clean_unnormalized.clone())

    # 6. 计算差异 (即触发器的空间模式)
    #    注意：这个差异值会非常非常小 (例如, [-0.01, 0.01])
    #    人眼是绝对无法直接看到的
    diff_tensor = tensor_poisoned_unnormalized - tensor_clean_unnormalized

    # 7. [核心] 将差异“放大”到 [0, 1] 范围以便可视化
    #    我们通过 min-max 归一化来拉伸对比度
    if diff_tensor.max() != diff_tensor.min():
        amplified_diff = (diff_tensor - diff_tensor.min()) / (diff_tensor.max() - diff_tensor.min())
    else:
        amplified_diff = torch.zeros_like(diff_tensor)  # 避免除零

    # 8. 将用于绘图的 Tensors 转换回 PIL 图像
    to_pil = transforms.ToPILImage()
    img_poisoned_pil = to_pil(tensor_poisoned_unnormalized)
    img_diff_pil = to_pil(amplified_diff)

    # 9. 使用 Matplotlib 绘图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 图 1: 干净图像
    axes[0].imshow(pil_clean)
    axes[0].set_title(f"Clean Image (Label: {label})")
    axes[0].axis('off')

    # 图 2: S2A 带毒图像
    axes[1].imshow(img_poisoned_pil)
    axes[1].set_title(f"Poisoned Image (S2A in '{config['attack']['subband']}')")
    axes[1].axis('off')

    # 图 3: 放大后的差异
    axes[2].imshow(img_diff_pil)
    axes[2].set_title("Amplified Difference (The Trigger)")
    axes[2].axis('off')

    plt.suptitle(f"S2A Invisibility Check (Image Index: {args.idx})")
    plt.tight_layout()

    save_path = "s2a_visualization.png"
    plt.savefig(save_path)
    print(f"Visualization saved to {save_path}")

    # 如果您在本地运行并希望立即看到图像，请取消注释下一行
    # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize S2A Attack Invisibility')
    parser.add_argument('--config', default='./configs/cifar10_resnet18.yaml',
                        help='Path to YAML config file')
    parser.add_argument('--idx', type=int, default=100,
                        help='Index of the CIFAR-10 image to visualize')
    args = parser.parse_args()

    # 确保您安装了 matplotlib
    try:
        import matplotlib
    except ImportError:
        print("Error: matplotlib 未安装。请运行 'pip install matplotlib'")
        exit(1)

    main(args)