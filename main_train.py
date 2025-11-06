# main_train.py (最终版 v7.2 - 修复TypeError)

import os
import torch
import multiprocessing as mp
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
import argparse

from core.utils import load_config, setup_experiment, save_config_to_experiment_dir, AverageMeter, accuracy
from core.dataset import PoisonedDataset
import core.models.resnet as models


def main():
    parser = argparse.ArgumentParser(description='S2A Backdoor Attack Training (Online Poisoning)')
    parser.add_argument('--config', default='./configs/cifar10_resnet18.yaml', help='Path to YAML config')
    args = parser.parse_args()
    config = load_config(args.config)

    device_str = config.get('device', 'cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device_str)

    if not torch.cuda.is_available() and device_str.startswith('cuda'):
        print("错误: CUDA 不可用。正在切换到 CPU...")
        device = torch.device('cpu')

    main_worker(device, config, args)


def main_worker(device, config, args):
    experiment_dir = setup_experiment(config)
    save_config_to_experiment_dir(args.config, experiment_dir)
    logging.info(f"本次实验的所有产物将被保存在: {experiment_dir}")
    logging.info(f"正在使用设备: {device}")

    data_path = config['dataset']['data_path']
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    logging.info("加载在线中毒训练集...")
    train_dataset = PoisonedDataset(config, train=True, poison=True)
    train_loader = DataLoader(
        train_dataset, batch_size=config['train']['batch_size'], shuffle=True,
        num_workers=config['train'].get('num_workers', 0), pin_memory=True
    )

    logging.info("加载验证集 (C-ACC 和 ASR)...")
    val_clean_dataset = PoisonedDataset(config, train=False, poison=False)
    val_clean_loader = DataLoader(val_clean_dataset, batch_size=config['train']['batch_size'] * 2,
                                  shuffle=False, num_workers=config['train'].get('num_workers', 0))
    val_asr_dataset = PoisonedDataset(config, train=False, asr_eval=True)
    val_asr_loader = DataLoader(val_asr_dataset, batch_size=config['train']['batch_size'] * 2,
                                shuffle=False, num_workers=config['train'].get('num_workers', 0))

    dataset_name = config['dataset']['name']
    model_name_str = config.get('model_name', 'ResNet18')

    logging.info(f"从配置文件读取到模型: {model_name_str}")
    try:
        ModelToUse = getattr(models, model_name_str)
    except AttributeError:
        raise ValueError(f"错误: 在 core/models/resnet.py 中未找到名为 '{model_name_str}' 的模型函数。")

    model = ModelToUse(
        num_classes=config['dataset']['num_classes'],
        dataset_name=dataset_name
    ).to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    train_config = config['train']

    optimizer_name = train_config['optimizer'].lower()
    if optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=train_config['learning_rate'], momentum=train_config['momentum'],
                              weight_decay=train_config['weight_decay'])
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=train_config['learning_rate'],
                                weight_decay=train_config['weight_decay'])
    else:
        raise ValueError(f"不支持的优化器: {optimizer_name}")

    scheduler_name = train_config['scheduler'].lower()
    num_epochs = train_config['epochs']
    if scheduler_name == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif scheduler_name == 'multistep':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=train_config['milestones'], gamma=0.1)
    else:
        raise ValueError(f"不支持的调度器: {scheduler_name}")

    logging.info("--- 开始训练 ---")
    best_ba_under_high_asr, asr_at_best_ba, best_epoch, best_model_save_path = 0.0, 0.0, 0, ""

    for epoch in range(num_epochs):
        train_one_epoch(train_loader, model, criterion, optimizer, epoch, device)
        clean_acc = validate(val_clean_loader, model, criterion, device, "C-ACC")
        asr, current_epoch = -1.0, epoch + 1

        if current_epoch > 50 or current_epoch == num_epochs:
            asr = validate(val_asr_loader, model, criterion, device, "ASR")

        print()
        asr_log_str = f"{asr:.2f}%" if asr != -1.0 else " (跳过)"
        logging.info(f"--- Epoch {current_epoch}/{num_epochs} --- "
                     f"C-ACC (BA): {clean_acc:.2f}% | ASR: {asr_log_str} | "
                     f"LR: {scheduler.get_last_lr()[0]:.5f}")

        if asr > 99.0:
            if clean_acc > best_ba_under_high_asr:
                best_ba_under_high_asr, asr_at_best_ba, best_epoch = clean_acc, asr, current_epoch
                model_filename = (f'checkpoint_{dataset_name}_{model_name_str}_asr{asr:.2f}_ba{clean_acc:.2f}.pth')
                new_save_path = os.path.join(experiment_dir, model_filename)
                logging.info(
                    f"🏆 新的冠军模型诞生 (ASR>99%): BA: {clean_acc:.2f}%, ASR: {asr:.2f}%. 保存至该实验文件夹内 🏆")
                torch.save({'epoch': current_epoch, 'model_state_dict': model.state_dict()}, new_save_path)
                if best_model_save_path and os.path.exists(best_model_save_path): os.remove(best_model_save_path)
                best_model_save_path = new_save_path
        scheduler.step()

    # Final summary ...


def train_one_epoch(loader, model, criterion, optimizer, epoch, device):
    # [!!! 核心修复 !!!]
    # 我们需要为 losses 和 top1 创建两个独立的 AverageMeter 实例
    losses, top1 = AverageMeter(), AverageMeter()
    model.train()
    progress_bar = tqdm(loader, desc=f"训练 Epoch {epoch + 1}", leave=False)
    for i, (images, target) in enumerate(progress_bar):
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


def validate(loader, model, criterion, device, eval_type="Eval"):
    # [!!! 核心修复 !!!]
    # validate 函数里也需要同样的修复
    losses, top1 = AverageMeter(), AverageMeter()
    model.eval()
    progress_bar = tqdm(loader, desc=f"评估 {eval_type}", leave=False)
    with torch.no_grad():
        for (images, target) in progress_bar:
            images, target = images.to(device, non_blocking=True), target.to(device, non_blocking=True)
            output = model(images)
            loss = criterion(output, target)
            acc1, _ = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))
            progress_bar.set_postfix(Loss=f"{losses.avg:.4f}", Acc=f"{top1.avg:.2f}%")
    return top1.avg


if __name__ == '__main__':
    main()