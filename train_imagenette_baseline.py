# train_imagenette_baseline.py
#
# 目的：使用与后门攻击实验完全相同的配置（模型/数据增强/训练策略），
#       为ImageNette数据集训练一个干净的ResNet18基线模型。
#       这个脚本跑出的最终BA，将是你论文表格中ImageNette对应的那个权威的“Clean”数据。
#
# --- 核心配置 (与你的最终YAML文件100%同步) ---
# 1. 模型: 你的“智能适配版”ResNet18 (已传入 dataset_name='imagenette')。
# 2. 数据增强: 你的“BA/ASR平衡型”增强 (RandomResizedCrop, Flip, ColorJitter)。
# 3. 优化器: SGD (lr=0.1, momentum=0.9, weight_decay=1e-4)。
# 4. 调度器: CosineAnnealingLR。
# 5. 训练周期: 200轮。

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np
from tqdm import tqdm
import random
import time
import os
import torch.nn.functional as F


# --- 1. 超参数与配置 (与你的最终imagenette.yaml完全一致) ---
class Config:
    EPOCHS = 200
    BATCH_SIZE = 32  # 使用你已验证过不会OOM的安全值

    # SGD + CosineAnnealingLR 配置
    LEARNING_RATE = 0.1
    MOMENTUM = 0.9
    WEIGHT_DECAY = 1.0e-4

    # 工作线程数
    NUM_WORKERS = 8  # 使用一个安全值

    # 其他关键配置
    SEED = 42
    IMAGE_SIZE = 224
    NUM_CLASSES = 10
    DATA_PATH = './data'
    DATASET_NAME = 'imagenette2-320'
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# --- 2. 模型定义 (从你的 core/models/resnet.py 完整复制) ---
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__();
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False);
        self.bn1 = nn.BatchNorm2d(planes);
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False);
        self.bn2 = nn.BatchNorm2d(planes);
        self.shortcut = nn.Sequential();
        if stride != 1 or in_planes != self.expansion * planes: self.shortcut = nn.Sequential(
            nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)));
        out = self.bn2(self.conv2(out));
        out += self.shortcut(x);
        out = F.relu(out);
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, dataset_name='cifar10'):
        super(ResNet, self).__init__();
        self.in_planes = 64
        if dataset_name.lower() in ['cifar10', 'gtsrb']:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False); self.maxpool = nn.Identity()
        elif dataset_name.lower() in ['tiny_imagenet', 'imagenette', 'imagenet10']:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False); self.maxpool = nn.MaxPool2d(
                kernel_size=3, stride=2, padding=1)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False); self.maxpool = nn.Identity()
        self.bn1 = nn.BatchNorm2d(64);
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1);
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2);
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2);
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2);
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1);
        layers = []
        for s in strides: layers.append(block(self.in_planes, planes, s)); self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)));
        out = self.maxpool(out);
        out = self.layer1(out);
        out = self.layer2(out);
        out = self.layer3(out);
        out = self.layer4(out);
        out = F.adaptive_avg_pool2d(out, (1, 1));
        out = out.view(out.size(0), -1);
        out = self.linear(out);
        return out


def ResNet18(num_classes=10, dataset_name='cifar10'):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, dataset_name=dataset_name)


# --- 3. 数据加载与数据增强 (与你的 core/dataset.py 中为ImageNette配置的逻辑完全一致) ---
def get_dataloaders(batch_size, num_workers, image_size, data_path, dataset_folder, mean, std):
    try:
        interpolation = transforms.InterpolationMode.LANCZOS
    except AttributeError:
        interpolation = Image.LANCZOS

    # [!!! 核心一致性保证 1 !!!]
    # 与你 core/dataset.py 中 imagenette 的 'BA/ASR平衡型' 增强完全相同
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(image_size, interpolation=interpolation),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(image_size, interpolation=interpolation),
        transforms.CenterCrop(image_size),  # 测试时通常用 CenterCrop 保证一致性
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_path = os.path.join(data_path, dataset_folder, 'train')
    val_path = os.path.join(data_path, dataset_folder, 'val')

    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError(
            f"错误: ImageNette数据集在'{train_path}'或'{val_path}'未找到。\n"
            f"请确保您已经手动下载'imagenette2-320.tgz'并将其解压到'{data_path}'目录下。"
        )

    trainset = ImageFolder(root=train_path, transform=transform_train)
    testset = ImageFolder(root=val_path, transform=transform_test)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                              pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                             pin_memory=True)

    return trainloader, testloader


# --- 4. 训练与测试循环 (标准流程) ---
def train(model, trainloader, optimizer, criterion, device):
    model.train()
    progress_bar = tqdm(trainloader, desc="Training")
    for inputs, targets in progress_bar:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad();
        loss.backward();
        optimizer.step()
        progress_bar.set_postfix(Loss=f"{loss.item():.3f}")


def test(model, testloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in tqdm(testloader, desc="Testing"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    return 100 * correct / total


# --- 5. 主函数 ---
def main():
    start_time = time.time()
    cfg = Config()
    set_seed(cfg.SEED)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"正在使用设备: {device}")

    try:
        trainloader, testloader = get_dataloaders(cfg.BATCH_SIZE, cfg.NUM_WORKERS, cfg.IMAGE_SIZE, cfg.DATA_PATH,
                                                  cfg.DATASET_NAME, cfg.MEAN, cfg.STD)
    except Exception as e:
        print(f"\n[错误] 数据加载失败。请检查路径或num_workers设置。错误: {e}")
        return

    # [!!! 核心一致性保证 2 !!!]
    # 创建模型时，正确传入类别数和数据集名称('imagenette')
    model = ResNet18(num_classes=cfg.NUM_CLASSES, dataset_name='imagenette').to(device)

    criterion = nn.CrossEntropyLoss()

    # [!!! 核心一致性保证 3 !!!]
    # 优化器和调度器配置与你的YAML文件完全相同
    optimizer = optim.SGD(model.parameters(), lr=cfg.LEARNING_RATE, momentum=cfg.MOMENTUM,
                          weight_decay=cfg.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS)

    print(f"--- 开始为 {cfg.DATASET_NAME} 训练干净的ResNet18基线 ---")
    print(f"    (Epochs: {cfg.EPOCHS}, BS: {cfg.BATCH_SIZE}, Optim: SGD, LR: {cfg.LEARNING_RATE}, Scheduler: Cosine)")

    best_acc = 0.0
    for epoch in range(cfg.EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{cfg.EPOCHS} ---")
        train(model, trainloader, optimizer, criterion, device)
        test_acc = test(model, testloader, device)
        scheduler.step()

        print(f"Epoch {epoch + 1}: Test Acc (BA): {test_acc:.2f}% | LR: {optimizer.param_groups[0]['lr']:.5f}")

        if test_acc > best_acc:
            best_acc = test_acc
            print(f"*** 发现新的最高准确率: {best_acc:.2f}%. 正在保存模型... ***")
            save_filename = f"baseline_{cfg.DATASET_NAME}_resnet18_ba{best_acc:.2f}.pth"
            torch.save(model.state_dict(), save_filename)

    end_time = time.time()
    print("\n--- 训练完成 ---")
    print(f"最高测试准确率 (BA): {best_acc:.2f}%")
    print(f"总耗时: {(end_time - start_time) / 3600:.2f} 小时")


if __name__ == '__main__':
    main()