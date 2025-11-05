import torch
from torch.utils.data import DataLoader, Dataset  # <--- [!!! 核心修复 !!!] 补上 'Dataset'
from torchvision import datasets, transforms
import numpy as np
import yaml
from tqdm import tqdm
import os
from PIL import Image


# ---------------------------------------------------------------------
# 1. 我们需要复用 PoisonedDataset 类，以确保
#    我们分析的是与训练时完全相同的图像（例如 64x64）
# ---------------------------------------------------------------------

# (从 core/utils.py 复制)
def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# (从 core/dataset.py 复制，但移除了攻击逻辑)
class DatasetForAnalysis(Dataset):  # <--- 现在 'Dataset' 已经被正确导入
    def __init__(self, config, train=True):
        self.dataset_config = config['dataset']
        self.data_path = self.dataset_config['data_path']

        # [!!! 核心 !!!]
        self.image_size = self.dataset_config.get('image_size', 32)

        # [!!! 核心 !!!]
        # 我们只应用 Resize 和 ToTensor，*不*应用 Normalize
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.ToTensor()
        ])

        dataset_name = self.dataset_config['name'].lower()
        if dataset_name == 'cifar10':
            self.clean_dataset = datasets.CIFAR10(
                root=self.data_path, train=train, download=True, transform=self.transform
            )
            self._get_item_logic = self._get_item_torchvision

        elif dataset_name == 'gtsrb':
            split = 'train' if train else 'test'
            self.clean_dataset = datasets.GTSRB(
                root=self.data_path, split=split, download=True, transform=self.transform
            )
            self._get_item_logic = self._get_item_torchvision

        elif dataset_name == 'tiny_imagenet':
            split = 'train' if train else 'val'
            data_dir = os.path.join(self.data_path, 'tiny-imagenet-200', split)
            if not os.path.exists(data_dir):
                raise FileNotFoundError(f"未找到 Tiny ImageNet 数据集: {data_dir}")

            # (确保 Tiny ImageNet 在转换时是 RGB)
            self.transform = transforms.Compose([
                transforms.Lambda(lambda img: img.convert('RGB')),
                transforms.Resize((self.image_size, self.image_size),
                                  interpolation=transforms.InterpolationMode.LANCZOS),
                transforms.ToTensor()
            ])
            self.clean_dataset = datasets.ImageFolder(
                root=data_dir, transform=self.transform
            )
            self._get_item_logic = self._get_item_torchvision
        else:
            raise ValueError(f"不支持的数据集: {dataset_name}")

    def __len__(self):
        return len(self.clean_dataset)

    def _get_item_torchvision(self, idx):
        # (此方法适用于所有已加载的 torchvision 数据集)
        return self.clean_dataset[idx]

    def __getitem__(self, idx):
        # [!!! 核心 !!!]
        # 只返回图像，不返回标签，且不归一化
        img, _ = self._get_item_logic(idx)
        return img


# ---------------------------------------------------------------------
# 2. 主分析逻辑
# ---------------------------------------------------------------------

def main():
    # [!!!] 修改这里为您正在分析的配置文件
    CONFIG_PATH = 'configs/tiny_imagenet_64x64_random.yaml'

    print(f"正在加载配置文件: {CONFIG_PATH} ...")
    config = load_config(CONFIG_PATH)

    dataset_name = config['dataset']['name']
    image_size = config['dataset'].get('image_size', 32)
    print(f"正在分析数据集: {dataset_name} (尺寸: {image_size}x{image_size})")

    # 加载训练集 (train=True)
    dataset = DatasetForAnalysis(config, train=True)

    # 使用 DataLoader 来加速
    # (注意: batch_size 越大, 速度越快, 但内存消耗越高)
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_samples = 0

    print("正在计算均值和标准差 (这可能需要几分钟)...")

    # (此方法可以逐批计算，节省内存)
    for images in tqdm(loader):
        batch_samples = images.size(0)  # (B, C, H, W)
        images = images.view(batch_samples, images.size(1), -1)  # (B, C, H*W)

        # 计算当前批次的均值和标准差
        mean += images.mean(2).sum(0)  # (C)
        std += images.std(2).sum(0)  # (C)
        total_samples += batch_samples

    # 计算总均值和标准差
    mean /= total_samples
    std /= total_samples

    print("\n" + "=" * 50)
    print("分析完成！")
    print(f"数据集: {dataset_name} ({image_size}x{image_size})")
    print(f"总样本数: {total_samples}")
    print("\n[!!!] 请将以下值复制到您的 .yaml 配置文件中 [!!!]")
    print(f"  mean: [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]")
    print(f"  std: [{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}]")
    print("=" * 50)


if __name__ == "__main__":
    main()