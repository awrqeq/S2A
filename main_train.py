# main_train.py
# [!!! 核心修改 !!!]
# 移除了 DDP (mp.spawn, setup_ddp, DistributedSampler, DDP)
# 新增了 'device' 配置，用于单 GPU 训练。
# 导入了 'PoisonedDataset' (来自修改后的 dataset.py)

import os
import torch
import torch.nn as nn
import torch.optim as optim
# import torch.multiprocessing as mp       # (DDP 已移除)
# import torch.distributed as dist          # (DDP 已移除)
from torch.utils.data import DataLoader
# from torch.utils.data.distributed import DistributedSampler # (DDP 已移除)
# from torch.nn.parallel import DistributedDataParallel as DDP # (DDP 已移除)

import logging
from tqdm import tqdm
import argparse

from core.utils import load_config, setup_logger, AverageMeter, accuracy
# [!!! 核心修改 !!!] 导入我们新的通用 Dataset 类
from core.dataset import PoisonedDataset
from core.models.resnet import ResNet18 as ModelToUse


def main():
    parser = argparse.ArgumentParser(description='S2A Backdoor Attack Training (Single GPU)')
    parser.add_argument('--config', default='./configs/cifar10_resnet18.yaml',
                        help='路径到 YAML 配置文件')
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logger()

    # [!!! 核心修改 !!!]
    # DDP 相关逻辑已移除

    # 从 config 文件中获取设备
    device_str = config.get('device', 'cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device_str)

    if not torch.cuda.is_available() and device_str.startswith('cuda'):
        logging.error("CUDA 不可用。此脚本需要 GPU。正在切换到 CPU...")
        device = torch.device('cpu')

    logging.info(f"正在使用设备: {device}")

    # 调用主训练器
    main_worker(device, config)


def main_worker(device, config):
    # [!!! 核心修改 !!!]
    # 'gpu' 和 DDP 参数已移除

    logging.info(f"使用模型: {ModelToUse.__name__}")

    # [!!! 核心修改 !!!] 使用 PoisonedDataset
    train_dataset = PoisonedDataset(config, train=True, poison=True)

    # (DistributedSampler 已移除)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=True,  # (DDP sampler 为 None, 恢复 shuffle=True)
        num_workers=16,
        pin_memory=True
    )

    # (验证集加载逻辑保持不变)
    val_clean_dataset = PoisonedDataset(config, train=False, poison=False)
    val_clean_loader = DataLoader(val_clean_dataset,
                                  batch_size=config['train']['batch_size'] * 2,
                                  shuffle=False, num_workers=16)

    # (使用我们修复后的 asr_eval 逻辑)
    val_asr_dataset = PoisonedDataset(config, train=False, asr_eval=True)
    val_asr_loader = DataLoader(val_asr_dataset,
                                batch_size=config['train']['batch_size'] * 2,
                                shuffle=False, num_workers=16)

    # [!!! 核心修改 !!!]
    # 模型直接加载到指定设备，移除 DDP 封装
    model = ModelToUse(num_classes=config['dataset']['num_classes']).to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(),
                          lr=config['train']['learning_rate'],
                          momentum=config['train']['momentum'],
                          weight_decay=config['train']['weight_decay'])

    # [!!! 核心修改 !!!]
    # 使用我们在 configs/cifar10_resnet18.yaml 中定义的 'scheduler'
    scheduler_type = config['train'].get('scheduler', 'cosine')
    if scheduler_type == 'step':
        logging.info("使用 MultiStepLR 调度器 (参考 FSBA)")
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                   milestones=[50, 100, 150, 200, 250],
                                                   gamma=0.1)
    else:  # 默认为 cosine
        logging.info("使用 CosineAnnealingLR 调度器")
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['train']['epochs'])

    logging.info("--- 开始训练 ---")
    best_c_acc, best_asr = 0.0, 0.0
    best_c_acc_at_best_asr, best_asr_at_best_c_acc = 0.0, 0.0
    best_c_acc_epoch, best_asr_epoch = 0, 0

    for epoch in range(config['train']['epochs']):

        # (DDP sampler.set_epoch 已移除)

        train_one_epoch(train_loader, model, criterion, optimizer, epoch, config, device)

        # (DDP 广播逻辑已移除)
        clean_acc = validate(val_clean_loader, model, criterion, config, device, "C-ACC")

        # (我们使用只包含 9000 张图片的严格 ASR 评估)
        asr = validate(val_asr_loader, model, criterion, config, device, "ASR")

        print()
        logging.info(f"--- Epoch {epoch + 1}/{config['train']['epochs']} --- "
                     f"C-ACC (BA): {clean_acc:.2f}% | "
                     f"ASR: {asr:.2f}%")

        # 更新最佳结果记录
        if clean_acc > best_c_acc:
            best_c_acc = clean_acc
            best_asr_at_best_c_acc = asr
            best_c_acc_epoch = epoch + 1

            model_to_save = model
            save_path = f'./checkpoint_best_c_acc.pth'
            torch.save({'epoch': epoch + 1, 'model_state_dict': model_to_save.state_dict()}, save_path)

        if asr > best_asr:
            best_asr = asr
            best_c_acc_at_best_asr = clean_acc
            best_asr_epoch = epoch + 1

        # (DDP barrier 已移除)
        scheduler.step()

    logging.info("\n" + "=" * 50)
    logging.info("--- 训练完成：最终评估总结 ---")
    logging.info("=" * 50)
    logging.info(f"最佳 C-ACC (BA) (在 Epoch {best_c_acc_epoch}): {best_c_acc:.2f}%")
    logging.info(f"    (此时 ASR): {best_asr_at_best_c_acc:.2f}%")
    logging.info(f"    (模型保存在: ./checkpoint_best_c_acc.pth)")
    logging.info("-" * 50)
    logging.info(f"最佳 ASR (在 Epoch {best_asr_epoch}): {best_asr:.2f}%")
    logging.info(f"    (此时 C-ACC): {best_c_acc_at_best_asr:.2f}%")
    logging.info("=" * 50)

    # (DDP cleanup 已移除)


def train_one_epoch(loader, model, criterion, optimizer, epoch, config, device):
    losses, top1 = AverageMeter(), AverageMeter()
    model.train()
    progress_bar = tqdm(loader, desc=f"训练 Epoch {epoch + 1}", leave=False)

    for i, (images, target) in enumerate(progress_bar):
        # [!!! 核心修改 !!!] 使用 device 变量
        images, target = images.to(device, non_blocking=True), target.to(device, non_blocking=True)

        output = model(images)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc1, _ = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))
        progress_bar.set_postfix(Loss=f"{losses.avg:.4f}", Acc=f"{top1.avg:.2f}%")


def validate(loader, model, criterion, config, device, eval_type="Eval"):
    losses, top1 = AverageMeter(), AverageMeter()
    model.eval()

    # [!!! 核心修改 !!!] 移除 model.module
    model_to_eval = model

    progress_bar = tqdm(loader, desc=f"评估 {eval_type}", leave=False)

    with torch.no_grad():
        for (images, target) in progress_bar:
            # [!!! 核心修改 !!!] 使用 device 变量
            images, target = images.to(device, non_blocking=True), target.to(device, non_blocking=True)

            output = model_to_eval(images)
            loss = criterion(output, target)
            acc1, _ = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))
            progress_bar.set_postfix(Loss=f"{losses.avg:.4f}", Acc=f"{top1.avg:.2f}%")

    return top1.avg


if __name__ == '__main__':
    main()