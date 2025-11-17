# main_train_gpu.py

import os
import sys

import torch
import scipy.signal

import multiprocessing as mp
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import logging
from tqdm import tqdm
import argparse

from torch.cuda.amp import GradScaler, autocast

from core.utils import load_config, setup_experiment, save_config_to_experiment_dir, AverageMeter, accuracy
from core.dataset import PoisonedDataset
import core.models.resnet as models


def main():
    parser = argparse.ArgumentParser(description='S2A Backdoor Attack Training (Multi-GPU Aware)')
    parser.add_argument('--config', default='./configs/cifar10_resnet18.yaml', help='Path to YAML config')
    parser.add_argument('--device', default=None, help='覆盖配置文件中的设备设置 (e.g., "cuda:0", "cuda:1", "cpu")')

    args = parser.parse_args()
    config = load_config(args.config)

    if args.device:
        device_str = args.device
    else:
        device_str = config.get('device', 'cuda:0' if torch.cuda.is_available() else 'cpu')

    if not torch.cuda.is_available() and device_str.startswith('cuda'):
        logging.error(f"错误: CUDA 不可用，但请求的设备是 '{device_str}'。")
        exit()

    if device_str.startswith('cuda'):
        try:
            torch.cuda.set_device(device_str)
        except (ValueError, AssertionError) as e:
            logging.error(
                f"错误: 无效的CUDA设备 '{device_str}'. 请确保GPU ID存在。可用GPU数量: {torch.cuda.device_count()}")
            exit()

    device = torch.device(device_str)
    main_worker(device, config, args)


def main_worker(device, config, args):
    torch.backends.cudnn.benchmark = True

    config['device'] = str(device)
    attack_method = config['attack'].get('attack_method', 'none')
    logging.info(f"从配置文件加载的攻击方法: '{attack_method}'")

    experiment_dir = setup_experiment(config)
    save_config_to_experiment_dir(args.config, experiment_dir)
    logging.info(f"本次实验的所有产物将被保存在: {experiment_dir}")
    logging.info(f"正在使用设备: {device}")

    data_path = config['dataset']['data_path']
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    logging.info("加载训练集 (执行一次性离线中毒)...")
    train_dataset = PoisonedDataset(config, train=True, poison=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=True,
        num_workers=config['train'].get('num_workers', 4),
        pin_memory=True,
        persistent_workers=True if config['train'].get('num_workers', 0) > 0 else False
    )

    logging.info("加载验证集...")
    val_clean_dataset = PoisonedDataset(config, train=False, poison=False, asr_eval=False)
    val_clean_loader = DataLoader(val_clean_dataset, batch_size=config['train']['batch_size'] * 2,
                                  shuffle=False, num_workers=config['train'].get('num_workers', 4),
                                  pin_memory=True,
                                  persistent_workers=True if config['train'].get('num_workers', 0) > 0 else False)

    val_asr_dataset = PoisonedDataset(config, train=False, poison=False, asr_eval=True)
    val_asr_loader = DataLoader(val_asr_dataset, batch_size=config['train']['batch_size'] * 2,
                                shuffle=False, num_workers=config['train'].get('num_workers', 4),
                                pin_memory=True,
                                persistent_workers=True if config['train'].get('num_workers', 0) > 0 else False)

    dataset_name = config['dataset']['name']
    model_name_str = config.get('model_name', 'ResNet18')
    ModelToUse = getattr(models, model_name_str)
    model = ModelToUse(num_classes=config['dataset']['num_classes'], dataset_name=dataset_name).to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    train_config = config['train']
    optimizer = optim.SGD(model.parameters(), lr=train_config['learning_rate'], momentum=train_config['momentum'],
                          weight_decay=train_config['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=train_config['epochs'])

    scaler = GradScaler()

    # --- 新增：用于追踪最佳模型的变量 ---
    best_clean_acc = 0.0
    # 删除 asr_threshold_reached 标志位，因为逻辑变更为严格的单次判断
    asr_threshold = 99.5  # ASR 阈值

    logging.info("--- 开始正式训练 ---")

    for epoch in range(train_config['epochs']):
        train_one_epoch(train_loader, model, criterion, optimizer, epoch, device, scaler)

        clean_acc = validate(val_clean_loader, model, criterion, device, "C-ACC")
        asr = validate(val_asr_loader, model, criterion, device, "ASR")

        logging.info(
            f"--- Epoch {epoch + 1}/{train_config['epochs']} --- C-ACC (BA): {clean_acc:.2f}% | ASR: {asr:.2f}% | LR: {scheduler.get_last_lr()[0]:.5f}")

        # --- 更新后的模型保存逻辑 (修正版) ---

        # 逻辑：必须同时满足 ASR > 99.5 且 BA > 历史最佳BA 才会保存
        if asr > asr_threshold:
            if clean_acc > best_clean_acc:
                best_clean_acc = clean_acc

                # 移除旧的最佳模型（如果有的话），避免文件混乱
                for f in os.listdir(experiment_dir):
                    if f.startswith('checkpoint_best_model'):
                        os.remove(os.path.join(experiment_dir, f))

                # 保存新的最佳模型
                model_filename = f'checkpoint_best_model_asr_{asr:.2f}_ba_{clean_acc:.2f}.pth'
                save_path = os.path.join(experiment_dir, model_filename)

                logging.info(f"🏆 新的最佳模型！(ASR > {asr_threshold}%) BA: {clean_acc:.2f}%, ASR: {asr:.2f}%. 保存至: {save_path} 🏆")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'best_clean_acc': best_clean_acc,
                    'current_asr': asr,
                }, save_path)

        scheduler.step()

    logging.info("--- 训练结束 ---")
    if best_clean_acc > 0:
        logging.info(f"最终保存的最佳模型 BA 为: {best_clean_acc:.2f}%")
    else:
        logging.warning(f"警告：在整个训练过程中，没有模型同时满足 ASR > {asr_threshold}% 且 BA 创新高，因此没有模型被保存。")


def train_one_epoch(loader, model, criterion, optimizer, epoch, device, scaler):
    model.train()
    progress_bar = tqdm(loader, desc=f"训练 Epoch {epoch + 1}", leave=True)

    for i, (images, target) in enumerate(progress_bar):
        images, target = images.to(device, non_blocking=True), target.to(device, non_blocking=True)

        with autocast():
            output = model(images)
            loss = criterion(output, target)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if i % 20 == 0:
            progress_bar.set_postfix(Loss=f"{loss.item():.4f}")


def validate(loader, model, criterion, device, eval_type="Eval"):
    top1 = AverageMeter()
    model.eval()
    progress_bar = tqdm(loader, desc=f"评估 {eval_type}", leave=False)
    with torch.no_grad():
        for (images, target) in progress_bar:
            images, target = images.to(device, non_blocking=True), target.to(device, non_blocking=True)
            with autocast():
                output = model(images)

            acc1, = accuracy(output, target, topk=(1,))
            top1.update(acc1.item(), images.size(0))

            progress_bar.set_postfix(Acc=f"{top1.avg:.2f}%")
    return top1.avg


if __name__ == '__main__':
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()