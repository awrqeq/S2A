# core/dataset.py (最终兼容版)
#
# --- 终极修复 ---
# 1. 解决了 AttributeError: 'GTSRB' object has no attribute 'targets' 的问题。
# 2. 对每个数据集，使用其专属的、正确的 API 来获取标签列表。
#    - CIFAR-10 / Tiny ImageNet: 使用 .targets 属性。
#    - GTSRB: 手动从 ._samples 属性中提取标签。
# 3. 这使得数据集加载器对所有支持的数据集都具有完美的兼容性。

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
import os
import logging

from .attack import get_injector_instance


class PoisonedDataset(Dataset):
    def __init__(self, config, train=True, poison=False, asr_eval=False):
        self.config = config;
        self.train = train;
        self.poison = poison;
        self.asr_eval = asr_eval
        dataset_config = self.config['dataset'];
        attack_config = self.config['attack']
        self.image_size = dataset_config['image_size']

        try:
            interpolation = transforms.InterpolationMode.LANCZOS
        except AttributeError:
            interpolation = Image.LANCZOS

        self.transform_pre = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), interpolation=interpolation),
            transforms.ToTensor()
        ])
        self.transform_post = transforms.Normalize(dataset_config['mean'], dataset_config['std'])
        self.injector = None

        dataset_name = dataset_config['name'].lower()
        data_path = dataset_config['data_path']

        logging.info(f"--- [torchvision自动模式] 正在加载 {dataset_name.upper()} (train={train})... ---")

        if dataset_name == 'cifar10':
            self.clean_dataset = datasets.CIFAR10(root=data_path, train=self.train, download=True)
            # [!!!] CIFAR-10 有 .targets 属性
            self.targets = np.array(self.clean_dataset.targets)

        elif dataset_name == 'gtsrb':
            split = 'train' if self.train else 'test'
            self.clean_dataset = datasets.GTSRB(root=data_path, split=split, download=True)
            # [!!! 核心修复 !!!] GTSRB 没有 .targets, 我们从 ._samples 中提取
            self.targets = np.array([s[1] for s in self.clean_dataset._samples])

        elif dataset_name == 'tiny_imagenet':
            split = 'train' if train else 'val'
            data_dir = os.path.join(data_path, 'tiny-imagenet-200', split)
            if not os.path.exists(data_dir):
                raise FileNotFoundError(f"Tiny ImageNet not found at {data_dir}. Please download it manually.")
            self.clean_dataset = datasets.ImageFolder(data_dir)
            # [!!!] ImageFolder 也有 .targets 属性
            self.targets = np.array(self.clean_dataset.targets)

        else:
            raise ValueError(f"Dataset {dataset_name} not supported.")

        self.target_label = attack_config['target_label']
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

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if (self.poison or self.asr_eval) and self.injector is None:
            self.injector = get_injector_instance(self.config, self.image_size)

        original_idx = self.indices[idx]
        img, label = self.clean_dataset[original_idx]
        img = img.convert("RGB")

        img_tensor = self.transform_pre(img)
        final_label = label

        is_poison = (self.poison and original_idx in self.poison_indices) or self.asr_eval
        if is_poison:
            img_tensor = self.injector.inject(img_tensor)
            final_label = self.target_label

        return self.transform_post(img_tensor), final_label