# core/dataset.py (已修复：仅在需要时才 Resize)

import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from PIL import Image
import os  # (导入 os 以支持 tiny_imagenet)

# 导入工厂函数
from .attack import get_injector_instance


class PoisonedDataset(Dataset):
    def __init__(self, config, train=True, poison=False, asr_eval=False):
        self.config = config
        self.train = train
        self.poison = poison
        self.asr_eval = asr_eval

        self.dataset_config = config['dataset']
        self.attack_config = config['attack']

        self.data_path = self.dataset_config['data_path']
        self.target_label = self.attack_config['target_label']

        self.image_size = self.dataset_config.get('image_size', 32)

        # [!!! 核心修改 !!!]
        # pil_transform 现在是一个空列表，我们将动态填充它
        pil_transform_list = []

        # (我们只在 __getitem__ 中检查一次，而不是每次都检查)
        self.needs_resize = True

        self.tensor_transform = transforms.ToTensor()
        self.normalize_transform = transforms.Normalize(
            self.dataset_config['mean'],
            self.dataset_config['std']
        )

        dataset_name = self.dataset_config['name'].lower()
        if dataset_name == 'cifar10':
            self.clean_dataset = datasets.CIFAR10(
                root=self.data_path, train=self.train, download=True, transform=None
            )
            self.data = self.clean_dataset.data
            self.targets = np.array(self.clean_dataset.targets)
            self._get_item_logic = self._get_item_cifar
            # [!!! 核心修改 !!!] CIFAR-10 已经是 32x32
            # 如果目标尺寸也是 32x32，我们就不需要 Resize
            if self.image_size == 32:
                self.needs_resize = False

        elif dataset_name == 'gtsrb':
            split = 'train' if self.train else 'test'
            self.clean_dataset = datasets.GTSRB(
                root=self.data_path, split=split, download=True, transform=None
            )
            self.data = None
            self.targets = np.array([sample[1] for sample in self.clean_dataset])
            self._get_item_logic = self._get_item_gtsrb
            # [!!! 核心修改 !!!] GTSRB 尺寸不一，*总是*需要 Resize
            self.needs_resize = True

        elif dataset_name == 'tiny_imagenet':
            split = 'train' if self.train else 'val'
            data_dir = os.path.join(self.data_path, 'tiny-imagenet-200', split)
            if not os.path.exists(data_dir):
                raise FileNotFoundError(f"未找到 Tiny ImageNet 数据集: {data_dir}")

            self.clean_dataset = datasets.ImageFolder(
                root=data_dir, transform=None
            )
            self.data = None
            self.targets = np.array([s[1] for s in self.clean_dataset.samples])
            self._get_item_logic = self._get_item_imagefolder
            # [!!! 核心修改 !!!] Tiny ImageNet 尺寸不一，*总是*需要 Resize
            self.needs_resize = True

        else:
            raise ValueError(f"不支持的数据集: {dataset_name}")

        # [!!! 核心修改 !!!]
        # 现在我们根据 needs_resize 动态构建 pil_transform
        if self.needs_resize:
            print(f"INFO: 数据集 {dataset_name} 将被 Resize 到 {self.image_size}x{self.image_size}。")
            pil_transform_list.append(
                transforms.Resize((self.image_size, self.image_size),
                                  interpolation=transforms.InterpolationMode.LANCZOS)
            )
        else:
            print(f"INFO: 数据集 {dataset_name} (32x32) 与目标尺寸 (32x32) 匹配，跳过 Resize。")

        self.pil_transform = transforms.Compose(pil_transform_list)

        self.injector = None
        if self.poison or self.asr_eval:
            self.injector = get_injector_instance(config, self.image_size)

        self.poison_indices = []

        if self.poison:
            self._setup_train_poisoning()

        elif self.asr_eval:
            all_indices = np.arange(len(self.targets))
            non_target_mask = (self.targets != self.target_label)
            non_target_indices = all_indices[non_target_mask]

            self.targets = self.targets[non_target_indices]

            if self.data is None:
                self.filtered_indices = non_target_indices
            else:
                self.data = self.data[non_target_indices]

            self.poison_indices = np.arange(len(self.targets))

    def _setup_train_poisoning(self):
        poison_rate = self.attack_config['poison_rate']
        all_indices = np.arange(len(self.targets))
        non_target_mask = (self.targets != self.target_label)
        non_target_indices = all_indices[non_target_mask]
        num_to_poison = int(len(non_target_indices) * poison_rate)
        self.poison_indices = np.random.choice(non_target_indices, num_to_poison, replace=False)

    def __len__(self):
        return len(self.targets)

    def _get_item_cifar(self, idx):
        img_np, label = self.data[idx], self.targets[idx]
        img_pil = Image.fromarray(img_np)
        return img_pil, label

    def _get_item_gtsrb(self, idx):
        original_idx = idx
        if self.asr_eval:
            original_idx = self.filtered_indices[idx]
        img_pil, label = self.clean_dataset[original_idx]
        return img_pil, label

    def _get_item_imagefolder(self, idx):  # (用于 Tiny ImageNet)
        original_idx = idx
        if self.asr_eval:
            original_idx = self.filtered_indices[idx]
        img_pil, label = self.clean_dataset[original_idx]
        return img_pil.convert('RGB'), label  # (确保为 RGB)

    def __getitem__(self, idx):

        img_pil, label = self._get_item_logic(idx)

        # [!!! 核心修改 !!!]
        # pil_transform 现在是动态的：
        # - 对 GTSRB/TinyImageNet: 它会 Resize
        # - 对 CIFAR-10 (32x32): 它会什么都不做
        img_pil_resized = self.pil_transform(img_pil)

        img_tensor = self.tensor_transform(img_pil_resized)

        is_poison = (self.poison or self.asr_eval) and (idx in self.poison_indices)

        if is_poison:
            img_tensor = self.injector.inject(img_tensor)
            label = self.target_label

        img_tensor_normalized = self.normalize_transform(img_tensor)

        return img_tensor_normalized, int(label)