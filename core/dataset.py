# core/dataset.py (终极版 - 手动下载ImageNette专用)
#
# --- v6.0 更新 (稳定版) ---
# 1. [移除] 彻底移除了所有Hugging Face `datasets`库的依赖和自动下载逻辑。
# 2. [简化] 'imagenette' 的加载逻辑现在变得极其简单和直接。
# 3. [假设] 脚本现在假设用户已经按照指示，手动下载并解压了 fast.ai 的 'imagenette2-320' 数据集。
# 4. [路径] 它会自动在 config['dataset']['data_path'] 下寻找 'imagenette2-320' 这个文件夹。

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
import os
import logging
import shutil
from torchvision.datasets.utils import download_and_extract_archive
from tqdm import tqdm  # tqdm 还是可以保留的，因为它本身很有用

# 删除了 from datasets import load_dataset
from .attack import get_injector_instance


class PoisonedDataset(Dataset):
    def __init__(self, config, train=True, poison=False, asr_eval=False):
        # ... (所有参数和数据增强的定义保持不变) ...
        self.config = config
        self.train = train
        self.poison = poison
        self.asr_eval = asr_eval
        dataset_config = self.config['dataset']
        attack_config = self.config['attack']
        self.image_size = dataset_config['image_size']

        dataset_name = dataset_config['name'].lower()
        mean = dataset_config['mean']
        std = dataset_config['std']

        try:
            interpolation = transforms.InterpolationMode.LANCZOS
        except AttributeError:
            interpolation = Image.LANCZOS

        transform_pre_test = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), interpolation=interpolation),
            transforms.ToTensor(),
        ])

        if self.train:
            # (数据增强逻辑保持我们之前确定的“平衡版”不变)
            if dataset_name == 'cifar10':
                self.transform_pre = transforms.Compose(
                    [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                     transforms.ToTensor(), ])
            elif dataset_name in ['tiny_imagenet', 'imagenette']:
                self.transform_pre = transforms.Compose(
                    [transforms.RandomResizedCrop(self.image_size, interpolation=interpolation),
                     transforms.RandomHorizontalFlip(),
                     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                     transforms.ToTensor(), ])
            elif dataset_name == 'gtsrb':
                self.transform_pre = transforms.Compose(
                    [transforms.Resize((self.image_size, self.image_size), interpolation=interpolation),
                     transforms.RandomRotation(15),
                     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), transforms.ToTensor(), ])
            else:
                self.transform_pre = transforms.Compose(
                    [transforms.Resize((self.image_size, self.image_size), interpolation=interpolation),
                     transforms.RandomHorizontalFlip(), transforms.ToTensor(), ])
        else:
            self.transform_pre = transform_pre_test

        self.transform_post = transforms.Normalize(mean, std)
        self.injector = None

        data_path = dataset_config['data_path']
        logging.info(f"--- 正在加载 {dataset_name.upper()} (train={train})... ---")

        if dataset_name == 'cifar10':
            # ... (cifar10逻辑不变) ...
            self.clean_dataset = datasets.CIFAR10(root=data_path, train=self.train, download=True)
            self.targets = np.array(self.clean_dataset.targets)
        elif dataset_name == 'gtsrb':
            # ... (gtsrb逻辑不变) ...
            split = 'train' if self.train else 'test'
            self.clean_dataset = datasets.GTSRB(root=data_path, split=split, download=True)
            self.targets = np.array([s[1] for s in self.clean_dataset._samples])
        elif dataset_name == 'tiny_imagenet':
            # ... (tiny_imagenet逻辑不变, 为保持完整性省略) ...
            pass

        # [!!! 核心修改：极其简化的ImageNette加载逻辑 !!!]
        elif dataset_name == 'imagenette':
            split = 'train' if self.train else 'val'
            # 直接构建最终路径
            image_folder_path = os.path.join(data_path, 'imagenette2-320', split)

            # 检查路径是否存在
            if not os.path.exists(image_folder_path):
                raise FileNotFoundError(
                    f"错误: ImageNette数据集在'{image_folder_path}'未找到。\n"
                    f"请确保您已经手动下载'imagenette2-320.tgz'并将其解压到'{data_path}'目录下。"
                )

            # 直接加载本地的ImageFolder
            self.clean_dataset = datasets.ImageFolder(image_folder_path)
            self.targets = np.array(self.clean_dataset.targets)

        else:
            raise ValueError(f"Dataset {dataset_name} not supported.")

        # --- 后续所有代码完全不变 ---
        self.target_label = attack_config['target_label']
        # ... (你后续的所有代码都无需改动)
        all_indices = np.arange(len(self.targets))
        if self.asr_eval:
            non_target_mask = (self.targets != self.target_label)
            self.indices = all_indices[non_target_mask]
            self.poison_indices = set(self.indices)
        else:
            self.indices = all_indices
            if self.poison:
                non_target_indices = all_indices[self.targets != self.target_label]
                num_to_poison = int(len(non_target_indices) * attack_config['poison_rate'])
                self.poison_indices = set(np.random.choice(non_target_indices, num_to_poison, replace=False))
            else:
                self.poison_indices = set()
        logging.info(f"--- 数据集加载完成。样本数: {len(self.indices)} ---")

    # ... (__len__, __getitem__, etc. 所有其他方法都保持不变)
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if (self.poison or self.asr_eval) and self.injector is None: self.injector = get_injector_instance(self.config,
                                                                                                           self.image_size)
        original_idx = self.indices[idx]
        img, label = self.clean_dataset[original_idx]
        if img.mode != 'RGB': img = img.convert("RGB")
        img_tensor = self.transform_pre(img)
        final_label = label
        is_poison = (self.poison and original_idx in self.poison_indices) or self.asr_eval
        if is_poison:
            img_tensor = self.injector.inject(img_tensor)
            final_label = self.target_label
        return self.transform_post(img_tensor), final_label