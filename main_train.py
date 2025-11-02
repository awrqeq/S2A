# main_train.py (最终版：在线中毒 + 按频率评估 + 数据本地化)
#
# --- 最终修改 ---
# - 移除了所有将数据集重定向到 /data1 的逻辑。
# - 脚本现在会严格遵守你在 .yaml 配置文件中为 'data_path' 指定的路径（例如 './data'）。
# - 保留了在线中毒和按 1:5 频率评估 BA/ASR 的核心功能。

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import logging
from tqdm import tqdm
import argparse

from core.utils import load_config, setup_logger, AverageMeter, accuracy
from core.dataset import PoisonedDataset
from core.models.resnet import ResNet18 as ModelToUse


def main():
    parser = argparse.ArgumentParser(description='S2A Backdoor Attack Training (Online Poisoning)')
    parser.add_argument('--config', default='./configs/gtsrb_64x64_random.yaml', help='路径到 YAML 配置文件')
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logger()

    device_str = config.get('device', 'cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device_str)

    if not torch.cuda.is_available() and device_str.startswith('cuda'):
        logging.error("CUDA 不可用。此脚本需要 GPU。正在切换到 CPU...")
        device = torch.device('cpu')

    logging.info(f"正在使用设备: {device}")

    main_worker(device, config)


def main_worker(device, config):
    logging.info(f"使用模型: {ModelToUse.__name__}")

    # [!!!] 现在，脚本将直接使用你在 .yaml 文件中为 data_path 指定的路径
    # 请确保你的主目录分区有足够的空间来存放数据集
    data_path = config['dataset']['data_path']
    logging.info(f"所有原始数据集将被下载到 .yaml 文件指定的路径: {data_path}")
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # [!!!] 在线中毒模式
    logging.info("使用在线中毒模式加载训练集...")
    train_dataset = PoisonedDataset(config, train=True, poison=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=True,
        # 建议保持为 0 以避免潜在的环境问题
        num_workers=config['train'].get('num_workers', 0),
        pin_memory=True
    )

    logging.info("加载验证集 (C-ACC 和 ASR)...")
    val_clean_dataset = PoisonedDataset(config, train=False, poison=False)
    val_clean_loader = DataLoader(val_clean_dataset,
                                  batch_size=config['train']['batch_size'] * 2,
                                  shuffle=False, num_workers=config['train'].get('num_workers', 0))

    val_asr_dataset = PoisonedDataset(config, train=False, asr_eval=True)
    val_asr_loader = DataLoader(val_asr_dataset,
                                batch_size=config['train']['batch_size'] * 2,
                                shuffle=False, num_workers=config['train'].get('num_workers', 0))

    # --- 模型和优化器设置 (保持不变) ---
    model = ModelToUse(num_classes=config['dataset']['num_classes']).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(),
                          lr=config['train']['learning_rate'],
                          momentum=config['train']['momentum'],
                          weight_decay=config['train']['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['train']['epochs'])

    # --- 训练循环 (按频率评估) ---
    logging.info("--- 开始训练 ---")
    best_c_acc, best_asr = 0.0, 0.0
    best_c_acc_at_best_asr, best_asr_at_best_c_acc = 0.0, 0.0
    best_c_acc_epoch, best_asr_epoch = 0, 0

    num_epochs = config['train']['epochs']

    for epoch in range(num_epochs):
        train_one_epoch(train_loader, model, criterion, optimizer, epoch, device)

        clean_acc = validate(val_clean_loader, model, criterion, device, "C-ACC")
        asr = -1.0

        if (epoch + 1) % 5 == 0 or (epoch + 1) == num_epochs:
            logging.info(f"Epoch {epoch + 1} 是 5 的倍数或最后一个 epoch，开始评估 ASR...")
            asr = validate(val_asr_loader, model, criterion, device, "ASR")

        print()
        asr_log_str = f"{asr:.2f}%" if asr != -1.0 else " (跳过)"
        logging.info(f"--- Epoch {epoch + 1}/{num_epochs} --- "
                     f"C-ACC (BA): {clean_acc:.2f}% | "
                     f"ASR: {asr_log_str} | "
                     f"LR: {scheduler.get_last_lr()[0]:.5f}")

        if clean_acc > best_c_acc:
            best_c_acc = clean_acc
            best_asr_at_best_c_acc = asr if asr != -1.0 else -1.0
            best_c_acc_epoch = epoch + 1
            save_path = f'./checkpoint_best_c_acc.pth'
            torch.save({'epoch': epoch + 1, 'model_state_dict': model.state_dict()}, save_path)

        if asr != -1.0 and asr > best_asr:
            best_asr = asr
            best_c_acc_at_best_asr = clean_acc
            best_asr_epoch = epoch + 1

        scheduler.step()

    logging.info("\n" + "=" * 50)
    logging.info("--- 训练完成：最终评估总结 ---")
    logging.info("=" * 50)
    logging.info(f"最佳 C-ACC (BA) (在 Epoch {best_c_acc_epoch}): {best_c_acc:.2f}%")
    asr_summary_str = f"{best_asr_at_best_c_acc:.2f}%" if best_asr_at_best_c_acc != -1.0 else "(未在最佳BA轮次评估)"
    logging.info(f"    (此时 ASR): {asr_summary_str}")
    logging.info(f"    (模型保存在: ./checkpoint_best_c_acc.pth)")
    logging.info("-" * 50)
    logging.info(f"最佳 ASR (在 Epoch {best_asr_epoch}): {best_asr:.2f}%")
    logging.info(f"    (此时 C-ACC): {best_c_acc_at_best_asr:.2f}%")
    logging.info("=" * 50)


def train_one_epoch(loader, model, criterion, optimizer, epoch, device):
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