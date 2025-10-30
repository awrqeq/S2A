import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from .attack import S2A_Ultimate_Injector as S2A_Injector


class PoisonedCifar10(Dataset):
    def __init__(self, config, train=True, poison=False, asr_eval=False):
        self.config = config
        self.train = train
        self.poison = poison
        self.asr_eval = asr_eval

        self.data_path = config['dataset']['data_path']
        self.target_label = config['attack']['target_label']

        # 1. 定义数据变换
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # PIL -> Tensor [0, 1]
            transforms.Normalize(config['dataset']['mean'], config['dataset']['std'])
        ])

        # 2. 加载干净数据集
        self.clean_dataset = datasets.CIFAR10(
            root=self.data_path,
            train=self.train,
            download=True
        )

        self.injector = None
        self.poison_indices = []
        self.data = self.clean_dataset.data
        self.targets = np.array(self.clean_dataset.targets)

        # 3. 设置投毒
        if self.poison:
            # 训练投毒：只毒害 'poison_rate' 的非目标图像
            self.injector = S2A_Injector(config)
            self._setup_train_poisoning()

        elif self.asr_eval:
            # ASR 评估：毒害所有非目标图像
            self.injector = S2A_Injector(config)
            all_indices = np.arange(len(self.targets))
            self.poison_indices = all_indices[self.targets != self.target_label]

    def _setup_train_poisoning(self):
        """
        根据 config['attack']['poison_rate'] 选择要投毒的训练样本。
        只选择 "非目标标签" 的图像进行投毒。
        """
        poison_rate = self.config['attack']['poison_rate']
        all_indices = np.arange(len(self.targets))

        # 找到所有非目标标签的图像
        non_target_mask = (self.targets != self.target_label)
        non_target_indices = all_indices[non_target_mask]

        num_to_poison = int(len(non_target_indices) * poison_rate)

        # 随机选择
        self.poison_indices = np.random.choice(non_target_indices, num_to_poison, replace=False)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx], self.targets[idx]  # [H, W, C] numpy array

        # 转换为 [C, H, W] Tensor [0, 1]
        img_tensor = transforms.functional.to_tensor(img)

        is_poison = (self.poison or self.asr_eval) and (idx in self.poison_indices)

        if is_poison:
            # 执行 S2A 注入 (在 [0, 1] 张量上)
            img_tensor = self.injector.inject(img_tensor)
            label = self.target_label

        # 应用归一化
        img_tensor_normalized = transforms.functional.normalize(
            img_tensor,
            mean=self.config['dataset']['mean'],
            std=self.config['dataset']['std']
        )

        return img_tensor_normalized, int(label)