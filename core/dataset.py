# core/dataset.py (最终版 v7.0 - 自动引擎切换)
#
# --- v7.0 更新 (自动引擎切换) ---
# 1. 在文件顶部，同时导入了你稳定版的'attack.py'和我们新建的'attack_gpu.py'中的注入器函数。
# 2. 在 PoisonedDataset 的 __init__ 方法中：
#    - 会读取配置文件中的 `attack.use_gpu_acceleration` 开关 (默认为False)。
#    - 根据这个开关，决定将 self.get_injector_func 指向 CPU 版本还是 GPU 版本。
#    - 会打印清晰的日志，告诉你当前正在使用哪个攻击引擎。
# 3. 在 __getitem__ 方法中，它会使用这个已经选好的 self.get_injector_func 来创建注入器实例。
# 4. 这使得切换攻击引擎只需要修改配置文件中的一个布尔值，代码完全无需改动。

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
import os
import logging
import shutil
from torchvision.datasets.utils import download_and_extract_archive
from tqdm import tqdm

# [!!! 核心修改 1: 同时导入新旧两个注入器函数 !!!]
from .attack import get_injector_instance as get_injector_cpu
from .attack_gpu import get_injector_instance as get_injector_gpu  # [!] 从新文件导入


class PoisonedDataset(Dataset):
    def __init__(self, config, train=True, poison=False, asr_eval=False):
        self.config = config
        self.train = train
        self.poison = poison
        self.asr_eval = asr_eval
        dataset_config = self.config['dataset']
        attack_config = self.config['attack']
        self.image_size = dataset_config['image_size']

        # [!!! 核心修改 2: 在初始化时就决定好用哪个注入器函数 !!!]
        self.use_gpu_attack = self.config['attack'].get('use_gpu_acceleration', False)
        if self.use_gpu_attack:
            self.get_injector_func = get_injector_gpu
            logging.info("🚀 PoisonedDataset已配置为使用 [GPU Attack Engine] (attack_gpu.py) 🚀")
        else:
            self.get_injector_func = get_injector_cpu
            logging.info("🐢 PoisonedDataset已配置为使用 [CPU Attack Engine] (attack.py) 🐢")

        # injector 实例将在第一次需要时在 __getitem__ 中创建
        self.injector = None

        # --- 数据增强部分 (使用我们之前确定的“平衡版”) ---
        dataset_name = dataset_config['name'].lower()
        mean, std = dataset_config['mean'], dataset_config['std']

        try:
            interpolation = transforms.InterpolationMode.LANCZOS
        except AttributeError:
            interpolation = Image.LANCZOS

        self.transform_pre_test = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), interpolation=interpolation),
            transforms.ToTensor(),
        ])
        if self.train:
            if dataset_name == 'cifar10':
                self.transform_pre = transforms.Compose(
                    [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
                     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                     transforms.ToTensor()])
            elif dataset_name in ['tiny_imagenet', 'imagenette']:
                self.transform_pre = transforms.Compose(
                    [transforms.RandomResizedCrop(self.image_size, interpolation=interpolation),
                     transforms.RandomHorizontalFlip(),
                     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                     transforms.ToTensor()])
            elif dataset_name == 'gtsrb':
                self.transform_pre = transforms.Compose(
                    [transforms.Resize((self.image_size, self.image_size), interpolation=interpolation),
                     transforms.RandomRotation(15),
                     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), transforms.ToTensor()])
            else:
                self.transform_pre = transforms.Compose(
                    [transforms.Resize((self.image_size, self.image_size), interpolation=interpolation),
                     transforms.RandomHorizontalFlip(), transforms.ToTensor()])
        else:
            self.transform_pre = self.transform_pre_test

        self.transform_post = transforms.Normalize(mean, std)

        # --- 数据集加载逻辑 (保持最终稳定版) ---
        data_path = dataset_config['data_path']
        logging.info(f"--- 正在加载 {dataset_name.upper()} (train={train})... ---")

        if dataset_name == 'cifar10':
            self.clean_dataset = datasets.CIFAR10(root=data_path, train=self.train, download=True)
            self.targets = np.array(self.clean_dataset.targets)
        elif dataset_name == 'gtsrb':
            split = 'train' if self.train else 'test'
            self.clean_dataset = datasets.GTSRB(root=data_path, split=split, download=True)
            self.targets = np.array([s[1] for s in self.clean_dataset._samples])
        elif dataset_name == 'tiny_imagenet':
            # ... 省略以保持简洁 ...
            pass
        elif dataset_name == 'imagenette':
            split = 'train' if self.train else 'val'
            image_folder_path = os.path.join(data_path, 'imagenette2-320', split)
            if not os.path.exists(image_folder_path):
                raise FileNotFoundError(f"错误: ImageNette数据集在'{image_folder_path}'未找到。")
            self.clean_dataset = datasets.ImageFolder(image_folder_path)
            self.targets = np.array(self.clean_dataset.targets)
        else:
            raise ValueError(f"Dataset {dataset_name} not supported.")

        # --- 样本索引逻辑 (保持不变) ---
        self.target_label = attack_config['target_label']
        all_indices = np.arange(len(self.targets))
        if self.asr_eval:
            self.indices = all_indices[self.targets != self.target_label]
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
        # 创建注入器实例的逻辑不变
        if (self.poison or self.asr_eval) and self.injector is None:
            self.injector = self.get_injector_func(self.config, self.image_size)

        original_idx = self.indices[idx]
        img, label = self.clean_dataset[original_idx]

        if img.mode != 'RGB':
            img = img.convert("RGB")

        # 1. 工人进行标准加工，img_tensor此时在CPU上
        img_tensor = self.transform_pre(img)
        final_label = label

        is_poison = (self.poison and original_idx in self.poison_indices) or self.asr_eval
        if is_poison:
            # 增加一个批次维度，准备送入GPU引擎
            img_tensor_batch = img_tensor.unsqueeze(0)

            # 把这个CPU上的小批次送到GPU引擎里加工
            poisoned_batch_gpu = self.injector.inject(img_tensor_batch)

            # [!!! 核心修复 !!!]
            # GPU引擎返回了在'cuda:1'上的结果后，
            # 工人必须在把它放回传送带前，用 .cpu() 把它拿回到自己的CPU工作台上！
            img_tensor = poisoned_batch_gpu.squeeze(0).cpu()

            final_label = self.target_label

        # 最终，无论是干净的还是有毒的，返回的img_tensor都保证是在CPU上
        return self.transform_post(img_tensor), final_label