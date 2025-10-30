# core/dataset.py (最终方案 + 全局单例模式)

import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset

# --- [!!! 核心修改 !!!] ---
# 导入工厂函数，而不是注入器类本身
from .attack import get_injector_instance


class PoisonedCifar10(Dataset):
    def __init__(self, config, train=True, poison=False, asr_eval=False):
        self.config = config
        self.train = train
        self.poison = poison
        self.asr_eval = asr_eval

        self.data_path = config['dataset']['data_path']
        self.target_label = config['attack']['target_label']

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(config['dataset']['mean'], config['dataset']['std'])
        ])

        self.clean_dataset = datasets.CIFAR10(
            root=self.data_path,
            train=self.train,
            download=True
        )

        # --- [!!! 核心修改 !!!] ---
        # 不再自己创建injector，而是通过工厂函数获取全局实例。
        # 昂贵的初始化只会在第一个调用它的worker中执行一次。
        self.injector = None
        if self.poison or self.asr_eval:
            self.injector = get_injector_instance(config)

        self.poison_indices = []
        self.data = self.clean_dataset.data
        self.targets = np.array(self.clean_dataset.targets)

        if self.poison:
            self._setup_train_poisoning()
        elif self.asr_eval:
            all_indices = np.arange(len(self.targets))
            self.poison_indices = all_indices[self.targets != self.target_label]

    def _setup_train_poisoning(self):
        poison_rate = self.config['attack']['poison_rate']
        all_indices = np.arange(len(self.targets))
        non_target_mask = (self.targets != self.target_label)
        non_target_indices = all_indices[non_target_mask]
        num_to_poison = int(len(non_target_indices) * poison_rate)
        self.poison_indices = np.random.choice(non_target_indices, num_to_poison, replace=False)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx], self.targets[idx]
        img_tensor = transforms.functional.to_tensor(img)
        is_poison = (self.poison or self.asr_eval) and (idx in self.poison_indices)

        if is_poison:
            # 这里的self.injector就是那个全局唯一的实例
            img_tensor = self.injector.inject(img_tensor)
            label = self.target_label

        img_tensor_normalized = transforms.functional.normalize(
            img_tensor,
            mean=self.config['dataset']['mean'],
            std=self.config['dataset']['std']
        )
        return img_tensor_normalized, int(label)