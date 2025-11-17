# core/dataset.py

import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, GTSRB
import torchvision.datasets
import torchvision.transforms as transforms
import numpy as np
import logging
import os
from tqdm import tqdm
from PIL import Image

from .attack_gpu import UniversalAttackInjector

_GLOBAL_INJECTOR_INSTANCE = None


def get_shared_injector(config, image_size):
    global _GLOBAL_INJECTOR_INSTANCE
    if _GLOBAL_INJECTOR_INSTANCE is None:
        logging.info("首次创建并共享 UniversalAttackInjector 实例...")
        _GLOBAL_INJECTOR_INSTANCE = UniversalAttackInjector(config, image_size)
    return _GLOBAL_INJECTOR_INSTANCE


class PoisonedDataset(Dataset):
    def __init__(self, config, train=True, poison=True, asr_eval=False):
        self.config = config
        self.dataset_config = config['dataset']
        self.attack_config = config['attack']
        self.dataset_name = self.dataset_config['name'].lower()
        self.data_path = self.dataset_config['data_path']
        self.image_size = self.dataset_config.get('image_size', 32)
        self.train = train
        self.poison = poison
        self.asr_eval = asr_eval

        logging.info(f"--- (通用版) 高性能数据集加载: {self.dataset_name.upper()} | Size: {self.image_size} ---")

        # --- 1. 加载原始数据并统一转为 Tensor ---
        if self.dataset_name == 'cifar10':
            raw_dataset = CIFAR10(root=self.data_path, train=self.train, download=True)
            original_data_np = raw_dataset.data
            original_targets = np.array(raw_dataset.targets)
            logging.info("正在处理 CIFAR-10 数据...")
            temp_tensor = torch.from_numpy(original_data_np.transpose(0, 3, 1, 2)).float().div(255)
            if self.image_size != 32:
                logging.info(f"将 CIFAR10 从 32x32 Resize 到 {self.image_size}x{self.image_size}")
                resize_op = transforms.Resize((self.image_size, self.image_size), antialias=True)
                temp_tensor = resize_op(temp_tensor)
            all_images_tensor = temp_tensor
            all_targets_tensor = torch.from_numpy(original_targets).long()
        elif self.dataset_name == 'gtsrb':
            split = 'train' if self.train else 'test'
            raw_dataset = GTSRB(root=self.data_path, split=split, download=True)
            logging.info(f"正在遍历加载 GTSRB ({split}) 并 Resize 到 {self.image_size}x{self.image_size}...")
            data_list, target_list = [], []
            pre_transform = transforms.Compose(
                [transforms.Resize((self.image_size, self.image_size)), transforms.ToTensor()])
            for img, label in tqdm(raw_dataset, desc="Loading GTSRB"):
                data_list.append(pre_transform(img))
                target_list.append(label)
            all_images_tensor = torch.stack(data_list)
            all_targets_tensor = torch.tensor(target_list).long()
        elif self.dataset_name == 'imagenette':
            split = 'train' if self.train else 'val'
            pre_transform = transforms.Compose(
                [transforms.Resize((self.image_size, self.image_size)), transforms.ToTensor()])
            folder_path = os.path.join(self.data_path, 'imagenette2-320', split)
            if os.path.exists(folder_path):
                raw_dataset = torchvision.datasets.ImageFolder(root=folder_path)
            else:
                raise FileNotFoundError(f"未找到 Imagenette 路径: {folder_path}")
            logging.info(f"正在遍历加载 Imagenette 并强制 Resize 到 {self.image_size}x{self.image_size}...")
            data_list, target_list = [], []
            for img, label in tqdm(raw_dataset, desc="Loading Imagenette"):
                if img.mode != 'RGB': img = img.convert('RGB')
                data_list.append(pre_transform(img))
                target_list.append(label)
            all_images_tensor = torch.stack(data_list)
            all_targets_tensor = torch.tensor(target_list).long()
        else:
            raise ValueError(f"不支持的数据集: {self.dataset_name}")

        # --- 2. 执行中毒逻辑 (全新、更鲁棒的逻辑) ---
        final_images = all_images_tensor.clone()
        final_targets = all_targets_tensor.clone()

        self.attack_method = self.attack_config.get('attack_method', 'none')

        if self.attack_method != 'none':
            injector = get_shared_injector(config, self.image_size)
            target_label = self.attack_config['target_label']
            np.random.seed(42)

            if not injector.triggers_forged and train and poison:
                logging.info("--- 准备锻造样本集 (包含目标类) ---")
                poison_ratio = float(self.attack_config.get('poison_ratio', 0.1))
                num_forging_samples = 5000

                target_indices = torch.where(all_targets_tensor == target_label)[0]
                num_target_forging = int(num_forging_samples * poison_ratio)
                perm_target = torch.from_numpy(np.random.permutation(len(target_indices)))
                forging_indices_target = target_indices[perm_target[:num_target_forging]]

                non_target_indices = torch.where(all_targets_tensor != target_label)[0]
                num_nontarget_forging = num_forging_samples - num_target_forging
                perm_nontarget = torch.from_numpy(np.random.permutation(len(non_target_indices)))
                forging_indices_nontarget = non_target_indices[perm_nontarget[:num_nontarget_forging]]

                forging_indices = torch.cat([forging_indices_target, forging_indices_nontarget])
                forging_images = all_images_tensor[forging_indices]

                logging.info(f"使用 {len(forging_images)} 张代表性图片 (含目标类) 锻造通用触发器...")
                injector.forge_triggers_and_inject(forging_images, forge_only=True)
                logging.info("--- 代表性触发器锻造完毕 ---")

            should_poison = (train and poison) or asr_eval
            if should_poison:
                if asr_eval:
                    indices_to_process = torch.where(all_targets_tensor != target_label)[0]
                    if len(indices_to_process) > 0:
                        images_to_inject = all_images_tensor[indices_to_process]
                        poisoned_images = injector.inject(images_to_inject)
                        final_images = poisoned_images
                        final_targets = torch.full_like(all_targets_tensor[indices_to_process], target_label)
                    else:
                        final_images = torch.empty(0, *all_images_tensor.shape[1:])
                        final_targets = torch.empty(0, dtype=torch.long)
                else:
                    non_target_indices = torch.where(all_targets_tensor != target_label)[0]
                    poison_ratio = float(self.attack_config.get('poison_ratio', 0.1))
                    num_to_poison = int(len(non_target_indices) * poison_ratio)

                    perm = torch.from_numpy(np.random.permutation(len(non_target_indices)))
                    indices_to_process = non_target_indices[perm[:num_to_poison]]

                    if len(indices_to_process) > 0:
                        images_to_inject = all_images_tensor[indices_to_process]
                        logging.info(f"选中 {len(images_to_inject)} 张非目标类图片进行注入...")
                        poisoned_images = injector.inject(images_to_inject)
                        final_images[indices_to_process] = poisoned_images
                        final_targets[indices_to_process] = target_label

        # --- 3. 标准化 ---
        logging.info("对所有Tensor进行一次性标准化...")
        if self.dataset_name == 'cifar10' and 'mean' not in self.dataset_config:
            mean_vals, std_vals = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
        else:
            mean_vals = self.dataset_config.get('mean', [0.485, 0.456, 0.406])
            std_vals = self.dataset_config.get('std', [0.229, 0.224, 0.225])

        mean = torch.tensor(mean_vals).view(1, 3, 1, 1)
        std = torch.tensor(std_vals).view(1, 3, 1, 1)

        if final_images.shape[0] > 0:
            self.data = final_images.sub_(mean).div_(std)
        else:
            self.data = final_images

        self.targets = final_targets
        logging.info(f"--- 数据集加载完毕 (Shape: {self.data.shape}) ---")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]