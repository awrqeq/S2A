# main_train.py (最终版 - v5.0 集成智能实验档案系统)
#
# --- v5.0 更新 (档案系统集成) ---
# 1. 在 main 函数的开头，我们不再调用旧的 setup_logger。
# 2. 在 main_worker 的开头，我们现在调用新的 setup_experiment 函数：
#    - 这会立即创建本次实验的专属文件夹（例如 'experiments/20251104-...')。
#    - 同时会自动配置好日志，使其同时输出到控制台和该文件夹下的 'training_log.txt'。
#    - 我们还会立即将你使用的 .yaml 配置文件复制一份存档到该文件夹。
# 3. 在模型保存逻辑中，所有路径现在都使用 setup_experiment 返回的 experiment_dir 作为根目录。
# 4. 这确保了每一次运行，所有的产物（模型、日志、配置）都被完美地、原子化地保存在同一个地方。

import os
import torch
import multiprocessing as mp

mp.set_start_method('spawn', force=True)
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import logging
from tqdm import tqdm
import argparse

# [!!!] 导入我们新的工具函数
from core.utils import load_config, setup_experiment, save_config_to_experiment_dir, AverageMeter, accuracy
from core.dataset import PoisonedDataset
from core.models.resnet import ResNet18 as ModelToUse


def main():
    parser = argparse.ArgumentParser(description='S2A Backdoor Attack Training (Online Poisoning)')
    parser.add_argument('--config', default='./configs/gtsrb_64x64_random.yaml', help='路径到 YAML 配置文件')
    args = parser.parse_args()

    config = load_config(args.config)
    # [!!!] setup_logger() 已被移除，新的设置将在 main_worker 中进行

    device_str = config.get('device', 'cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device(device_str)

    if not torch.cuda.is_available() and device_str.startswith('cuda'):
        # logging 在 setup_experiment 中配置，但我们可以在这之前用 print 应急
        print("错误: CUDA 不可用，但设备被设置为CUDA。")
        device = torch.device('cpu')

    # 将 args 传递下去，我们需要原始的 config 路径来存档
    main_worker(device, config, args)


def main_worker(device, config, args):
    # [!!! 核心修改 1: 创建实验文件夹并设置日志 !!!]
    experiment_dir = setup_experiment(config)

    # [!!! 核心修改 2: 存档本次实验使用的配置文件 !!!]
    save_config_to_experiment_dir(args.config, experiment_dir)

    logging.info(f"本次实验的所有产物将被保存在: {experiment_dir}")
    logging.info(f"正在使用设备: {device}")

    # --- 后续代码几乎不变，除了保存路径 ---

    logging.info(f"使用模型: {ModelToUse.__name__}")
    data_path = config['dataset']['data_path']
    logging.info(f"所有原始数据集将被下载到 .yaml 文件指定的路径: {data_path}")
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    logging.info("使用在线中毒模式加载训练集...")
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
    model = ModelToUse(num_classes=config['dataset']['num_classes'], dataset_name=dataset_name).to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    train_config = config['train']
    optimizer_name = train_config['optimizer'].lower()
    logging.info(f"从配置文件读取到优化器: {optimizer_name}")
    if optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=train_config['learning_rate'], momentum=train_config['momentum'],
                              weight_decay=train_config['weight_decay'])
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=train_config['learning_rate'],
                                weight_decay=train_config['weight_decay'])
    else:
        raise ValueError(f"不支持的优化器: {optimizer_name}. 请在 'sgd' 或 'adamw' 中选择。")

    scheduler_name = train_config['scheduler'].lower()
    num_epochs = train_config['epochs']
    logging.info(f"从配置文件读取到调度器: {scheduler_name}")
    if scheduler_name == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif scheduler_name == 'multistep':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=train_config['milestones'], gamma=0.1)
    else:
        raise ValueError(f"不支持的调度器: {scheduler_name}. 请在 'cosine' 或 'multistep' 中选择。")

    logging.info("--- 开始训练 ---")
    best_ba_under_high_asr = 0.0
    asr_at_best_ba = 0.0
    best_epoch = 0
    best_model_save_path = ""

    for epoch in range(num_epochs):
        train_one_epoch(train_loader, model, criterion, optimizer, epoch, device)

        clean_acc = validate(val_clean_loader, model, criterion, device, "C-ACC")
        asr = -1.0

        current_epoch = epoch + 1
        if current_epoch > 50 or current_epoch == num_epochs:
            asr = validate(val_asr_loader, model, criterion, device, "ASR")

        print()
        asr_log_str = f"{asr:.2f}%" if asr != -1.0 else " (跳过)"
        logging.info(f"--- Epoch {current_epoch}/{num_epochs} --- "
                     f"C-ACC (BA): {clean_acc:.2f}% | ASR: {asr_log_str} | "
                     f"LR: {scheduler.get_last_lr()[0]:.5f}")

        if asr > 99.0:
            if clean_acc > best_ba_under_high_asr:
                best_ba_under_high_asr = clean_acc
                asr_at_best_ba = asr
                best_epoch = current_epoch

                # [!!! 核心修改 3: 保存路径使用 experiment_dir !!!]
                model_filename = (f'checkpoint_{dataset_name}_{ModelToUse.__name__}'
                                  f'_asr{asr:.2f}_ba{clean_acc:.2f}.pth')
                new_save_path = os.path.join(experiment_dir, model_filename)

                logging.info(
                    f"🏆 新的冠军模型诞生 (ASR>99%): BA: {clean_acc:.2f}%, ASR: {asr:.2f}%. 保存至该实验文件夹内 🏆")
                torch.save({'epoch': current_epoch, 'model_state_dict': model.state_dict()}, new_save_path)

                if best_model_save_path and os.path.exists(best_model_save_path):
                    os.remove(best_model_save_path)

                best_model_save_path = new_save_path

        scheduler.step()

    logging.info("\n" + "=" * 50)
    logging.info("--- 训练完成：最终评估总结 ---")
    logging.info("=" * 50)
    if best_epoch > 0:
        logging.info(f"🏆 最终冠军模型 (ASR > 99% 且 BA 最高):")
        logging.info(f"   - 在 Epoch {best_epoch} 获得")
        logging.info(f"   - 最佳 BA: {best_ba_under_high_asr:.2f}%")
        logging.info(f"   - 对应 ASR: {asr_at_best_ba:.2f}%")
        logging.info(f"   - 模型和日志已保存在: {experiment_dir}")
    else:
        logging.warning("⚠️ 警告: 在整个训练过程中，ASR未能达到99%的保存标准。")
        logging.warning(f"   - 没有保存任何模型。日志和配置已保存在: {experiment_dir}")
    logging.info("=" * 50)


# train_one_epoch 和 validate 函数保持不变
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