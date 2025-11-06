# train_densenet121_cifar10_baseline.py
#
# 目的：使用与后门攻击实验完全相同的配置，为CIFAR-10数据集训练一个干净的DenseNet121基线模型。
#
# --- 核心技术栈 ---
# 1. 模型: [!!!] 针对CIFAR-10优化的DenseNet121。
# 2. 数据增强: “BA/ASR平衡型”增强 (RandomCrop, Flip, ColorJitter)。
# 3. 优化器: SGD + CosineAnnealingLR。
# 4. 训练周期: 200轮。
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models  # [!!!] 新增，用于加载官方DenseNet
import numpy as np
from tqdm import tqdm
import random
import time
import torch.nn.functional as F


# --- 1. 超参数与配置 ---
class Config:
    EPOCHS = 200
    # DenseNet比ResNet18略大，为安全起见，可以将batch_size设为96或保持128
    BATCH_SIZE = 128

    # 保持冠军组合：SGD + CosineAnnealingLR
    LEARNING_RATE = 0.1
    MOMENTUM = 0.9
    WEIGHT_DECAY = 5.0e-4

    # 工作线程数
    NUM_WORKERS = 8  # 使用安全值
    SEED = 42
    # 为了清晰，移除LABEL_SMOOTHING，使用默认的CrossEntropyLoss
    # LABEL_SMOOTHING = 0.0


# --- 随机种子设置函数 (完整) ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# --- 2. 模型定义 (完整) ---
def create_densenet121_cifar(num_classes=10):
    """
    加载官方DenseNet121，并应用针对CIFAR-10的优化。
    这个逻辑与你 core/models/resnet.py 中的实现完全一致。
    """
    model = models.densenet121(weights=None)

    # 修改最后的分类层以匹配我们的num_classes
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)

    # [关键] 对初始层进行CIFAR-10适配优化
    model.features.conv0 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # 移除初始的最大池化层，这对于32x32的小图像是标准操作
    model.features.pool0 = nn.Identity()

    return model


# --- 3. 数据加载 (完整) ---
def get_dataloaders(batch_size, num_workers):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    # “BA/ASR平衡型”数据增强
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                                              pin_memory=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                                             pin_memory=True)

    return trainloader, testloader


# --- 4. 训练与测试循环 (完整) ---
def train(model, trainloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    progress_bar = tqdm(trainloader, desc="Training")
    for inputs, targets in progress_bar:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar.set_postfix(Loss=f"{(running_loss / (total / inputs.size(0))):.3f}",
                                 Acc=f"{(100. * correct / total):.2f}%")
    return (100. * correct / total)


def test(model, testloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        progress_bar = tqdm(testloader, desc="Testing")
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            progress_bar.set_postfix(Acc=f"{(100. * correct / total):.2f}%")

    accuracy = 100 * correct / total
    return accuracy


# --- 5. 主函数 (完整) ---
def main():
    start_time = time.time()
    cfg = Config()
    set_seed(cfg.SEED)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"正在使用设备: {device}")

    try:
        trainloader, testloader = get_dataloaders(cfg.BATCH_SIZE, cfg.NUM_WORKERS)
    except Exception as e:
        print(f"数据加载失败: {e}")
        return

    print("正在创建针对CIFAR-10优化的DenseNet121模型...")
    model = create_densenet121_cifar(num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(model.parameters(), lr=cfg.LEARNING_RATE, momentum=cfg.MOMENTUM,
                          weight_decay=cfg.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS)

    print("--- 开始为 CIFAR-10 + DenseNet121 训练干净基线 ---")
    best_acc = 0.0
    best_model_path = ""

    for epoch in range(cfg.EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{cfg.EPOCHS} ---")
        train(model, trainloader, optimizer, criterion, device)
        test_acc = test(model, testloader, device)
        scheduler.step()

        print(f"Epoch {epoch + 1}: Test Acc (BA): {test_acc:.2f}% | LR: {optimizer.param_groups[0]['lr']:.5f}")

        if test_acc > best_acc:
            old_best_acc = best_acc
            best_acc = test_acc
            print(f"*** 发现新的最高准确率: {best_acc:.2f}%. 正在保存模型... ***")

            # 构造新的文件名
            new_model_path = f'baseline_densenet121_cifar10_ba{best_acc:.2f}.pth'

            # 保存新模型
            torch.save(model.state_dict(), new_model_path)

            # 删除旧的最佳模型文件（如果存在）
            if old_best_acc > 0:
                old_model_path = f'baseline_densenet121_cifar10_ba{old_best_acc:.2f}.pth'
                if os.path.exists(old_model_path):
                    os.remove(old_model_path)

            best_model_path = new_model_path

    end_time = time.time()
    print("\n--- 训练完成 ---")
    print(f"最高测试准确率 (BA): {best_acc:.2f}%")
    if best_model_path:
        print(f"最佳模型已保存在: {best_model_path}")
    print(f"总耗时: {(end_time - start_time) / 3600:.2f} 小时")


if __name__ == '__main__':
    main()