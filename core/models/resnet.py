# core/models/resnet.py (v3.0 - 终极智能适配版, 新增DenseNet)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ==============================================================================
# ResNet Family
# ==============================================================================
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
        super(ResNet, self).__init__()
        self.in_planes = 64

        # 智能适配初始层
        if dataset_name.lower() in ['cifar10', 'gtsrb']:
            # For 32x32, 64x64 images
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.maxpool = nn.Identity()
        elif dataset_name.lower() in ['tiny_imagenet', 'imagenette', 'imagenet10']:
            # For 224x224 (or similar) images
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            # Default fallback
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.maxpool = nn.Identity()

        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out);
        out = self.layer2(out);
        out = self.layer3(out);
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18(num_classes=10, dataset_name='cifar10'):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, dataset_name=dataset_name)


# ==============================================================================
# DenseNet Family (新增)
# ==============================================================================

def DenseNet121(num_classes=10, dataset_name='cifar10'):
    """
    一个包装好的、能够智能适应不同数据集尺寸的DenseNet-121模型。
    """
    # 1. 从torchvision.models加载一个标准的densenet121，不使用预训练权重
    model = models.densenet121(weights=None)

    # 2. 修改最后的分类层以匹配我们的num_classes
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, num_classes)

    # 3. [!!! 核心修改 !!!] 根据数据集名称，智能地修改初始卷积层
    #    DenseNet的初始层是一个名叫 'features' 的 Sequential 模块
    if dataset_name.lower() in ['cifar10']:
        # For CIFAR-10 (32x32), need small conv and remove pooling
        model.features.conv0 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.features.pool0 = nn.Identity()

    elif dataset_name.lower() in ['gtsrb', 'tiny_imagenet']:
        # For GTSRB/Tiny-ImageNet (64x64), a slightly larger initial conv is better
        # We can still remove the aggressive initial pooling
        model.features.conv0 = nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2, bias=False)
        model.features.pool0 = nn.Identity()

    # For ImageNette (224x224), the default torchvision initial layers are PERFECT.
    # So we don't need an 'elif' for it, no changes are made.

    return model