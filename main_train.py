# main_train.py
# [最终修复] 通过添加 dist.barrier() 解决DDP死锁问题，确保评估和日志打印能够正常执行。
# [新增] 通过 broadcast 让所有 GPU 进程也能打印评估结果。

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import logging
from tqdm import tqdm
import argparse

from core.utils import load_config, setup_logger, setup_ddp, cleanup_ddp, AverageMeter, accuracy
from core.dataset import PoisonedCifar10
from core.models.resnet import ResNet18 as ModelToUse


def main():
    parser = argparse.ArgumentParser(description='S2A Backdoor Attack Training')
    parser.add_argument('--config', default='./configs/cifar10_resnet18.yaml',
                        help='路径到 YAML 配置文件')
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logger()

    if not torch.cuda.is_available():
        logging.error("CUDA 不可用。此脚本需要 GPU。")
        return

    ngpus_per_node = torch.cuda.device_count()
    if ngpus_per_node == 0:
        logging.error("未检测到 GPU。")
        return

    config['ddp']['world_size'] = ngpus_per_node

    if config['ddp']['multiprocessing_distributed'] and ngpus_per_node > 1:
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, config))
    else:
        main_worker(0, ngpus_per_node, config)


def main_worker(gpu, ngpus_per_node, config):
    config['ddp']['rank'] = gpu
    is_main_process = (gpu == 0)
    use_ddp = (ngpus_per_node > 1 and config['ddp']['multiprocessing_distributed'])

    if use_ddp:
        setup_ddp(config, config['ddp']['rank'], config['ddp']['world_size'])

    if is_main_process:
        logging.info(f"正在使用 {ngpus_per_node} 个 GPU。分布式训练: {'启用' if use_ddp else '禁用'}。")
        logging.info(f"使用模型: {ModelToUse.__name__}")

    train_dataset = PoisonedCifar10(config, train=True, poison=True)
    train_sampler = DistributedSampler(train_dataset, num_replicas=config['ddp']['world_size'],
                                       rank=config['ddp']['rank']) if use_ddp else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['train']['batch_size'],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=16,
        pin_memory=True
    )

    val_clean_loader, val_asr_loader = None, None
    if is_main_process:
        val_clean_dataset = PoisonedCifar10(config, train=False, poison=False)
        val_clean_loader = DataLoader(val_clean_dataset,
                                      batch_size=config['train']['batch_size'] * 2,
                                      shuffle=False, num_workers=16)

        val_asr_dataset = PoisonedCifar10(config, train=False, asr_eval=True)
        val_asr_loader = DataLoader(val_asr_dataset,
                                    batch_size=config['train']['batch_size'] * 2,
                                    shuffle=False, num_workers=16)

    model = ModelToUse(num_classes=config['dataset']['num_classes']).cuda(gpu)
    if use_ddp:
        # 移除 find_unused_parameters=True 以优化性能
        model = DDP(model, device_ids=[gpu])

    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = optim.SGD(model.parameters(),
                          lr=config['train']['learning_rate'],
                          momentum=config['train']['momentum'],
                          weight_decay=config['train']['weight_decay'])

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['train']['epochs'])

    if is_main_process:
        logging.info("--- 开始训练 ---")
        best_c_acc, best_asr = 0.0, 0.0
        best_c_acc_at_best_asr, best_asr_at_best_c_acc = 0.0, 0.0
        best_c_acc_epoch, best_asr_epoch = 0, 0

    for epoch in range(config['train']['epochs']):
        if use_ddp:
            train_sampler.set_epoch(epoch)

        train_one_epoch(train_loader, model, criterion, optimizer, epoch, config, gpu)

        # --- 修改开始 ---
        # 在所有进程中初始化变量，以便接收广播的值
        local_clean_acc_tensor = torch.zeros(1).cuda(gpu)
        local_asr_tensor = torch.zeros(1).cuda(gpu)

        if is_main_process:
            # 主进程执行验证
            clean_acc = validate(val_clean_loader, model, criterion, config, gpu, "C-ACC")
            asr = validate(val_asr_loader, model, criterion, config, gpu, "ASR")

            # 将验证结果转换为 tensor
            local_clean_acc_tensor.fill_(clean_acc)
            local_asr_tensor.fill_(asr)

            print() # 主进程先打印换行
            logging.info(f"--- Epoch {epoch + 1}/{config['train']['epochs']} --- "
                         f"C-ACC (BA): {clean_acc:.2f}% | "
                         f"ASR: {asr:.2f}%")

            # 更新最佳结果记录
            if clean_acc > best_c_acc:
                best_c_acc = clean_acc
                best_asr_at_best_c_acc = asr
                best_c_acc_epoch = epoch + 1

                model_to_save = model.module if use_ddp else model
                save_path = f'./checkpoint_best_c_acc.pth'
                torch.save({'epoch': epoch + 1, 'model_state_dict': model_to_save.state_dict()}, save_path)

            if asr > best_asr:
                best_asr = asr
                best_c_acc_at_best_asr = clean_acc
                best_asr_epoch = epoch + 1

        # 广播验证结果 (主进程 -> 所有其他进程)
        # 从 rank 0 (主进程) 广播 tensor
        dist.broadcast(local_clean_acc_tensor, src=0)
        dist.broadcast(local_asr_tensor, src=0)

        # 所有进程都可以获取到验证结果
        received_clean_acc = local_clean_acc_tensor.item()
        received_asr = local_asr_tensor.item()

        # 所有进程都打印结果到自己的控制台
        # 注意：这里使用 print，不是 logging，因为 logging 通常只在主进程配置
        if not is_main_process: # 主进程已经在上面打印了，其他进程才需要打印
            print(f"GPU {gpu}: --- Epoch {epoch + 1}/{config['train']['epochs']} --- "
                  f"C-ACC (BA): {received_clean_acc:.2f}% | "
                  f"ASR: {received_asr:.2f}%")
        # --- 修改结束 ---

        # [!!!!!! 核心的、最终的修复 !!!!!!]
        # 添加一个同步屏障，解决DDP死锁问题。
        if use_ddp:
            dist.barrier()

        scheduler.step()

    if is_main_process:
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

    if use_ddp:
        cleanup_ddp()


def train_one_epoch(loader, model, criterion, optimizer, epoch, config, rank):
    losses, top1 = AverageMeter(), AverageMeter()
    model.train()
    is_main_process = (rank == 0)
    progress_bar = tqdm(loader, desc=f"训练 Epoch {epoch + 1}", disable=not is_main_process, leave=False)

    for i, (images, target) in enumerate(progress_bar):
        images, target = images.cuda(rank, non_blocking=True), target.cuda(rank, non_blocking=True)
        output = model(images)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc1, _ = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))
        if is_main_process:
            progress_bar.set_postfix(Loss=f"{losses.avg:.4f}", Acc=f"{top1.avg:.2f}%")


def validate(loader, model, criterion, config, rank, eval_type="Eval"):
    losses, top1 = AverageMeter(), AverageMeter()
    model.eval()
    is_main_process = (rank == 0)
    model_to_eval = model.module if isinstance(model, DDP) else model
    progress_bar = tqdm(loader, desc=f"评估 {eval_type}", disable=not is_main_process, leave=False)

    with torch.no_grad():
        for (images, target) in progress_bar:
            images, target = images.cuda(rank, non_blocking=True), target.cuda(rank, non_blocking=True)
            output = model_to_eval(images)
            loss = criterion(output, target)
            acc1, _ = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))
            if is_main_process:
                progress_bar.set_postfix(Loss=f"{losses.avg:.4f}", Acc=f"{top1.avg:.2f}%")

    return top1.avg


if __name__ == '__main__':
    main()