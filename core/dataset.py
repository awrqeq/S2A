# core/dataset.py (已修改 ASR-Eval 逻辑)

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

        # [!!! 核心修改：ASR 评估逻辑 !!!]
        elif self.asr_eval:
            # 1. 找到所有非目标图像的索引
            all_indices = np.arange(len(self.targets))
            non_target_mask = (self.targets != self.target_label)
            non_target_indices = all_indices[non_target_mask]

            # 2. [!!!] 重新过滤 self.data 和 self.targets
            #    只保留那 9000 张非目标图像 (或训练集对应的非目标图像)
            self.data = self.data[non_target_indices]
            self.targets = self.targets[non_target_indices]

            # 3. 现在，poison_indices 应该包含 *所有* 剩余的图像 (0到8999)
            self.poison_indices = np.arange(len(self.targets))

            # (现在这个 dataset 的 __len__ 将返回 9000)

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

        # [!!!] 这个逻辑现在对于 asr_eval 会始终为 True
        # 因为 self.poison_indices 包含了所有索引 (0 到 len-1)
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