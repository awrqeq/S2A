# core/utils.py
# [已确认] 这是正确的“仅控制台”日志版本，无需修改。

import os
import torch
import logging
import sys
import yaml
import torch.distributed as dist

def load_config(config_path):
    """加载 YAML 配置文件"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_logger():
    """设置一个干净的 logger, 只输出到控制台"""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
        stream=sys.stdout,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.getLogger("torchvision").setLevel(logging.WARNING)

def setup_ddp(config, rank, world_size):
    """初始化 DDP (DistributedDataParallel)"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = config['ddp']['dist_url'].split(':')[-1]

    dist.init_process_group(
        backend=config['ddp']['dist_backend'],
        init_method=config['ddp']['dist_url'],
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(rank)

def cleanup_ddp():
    """清理 DDP"""
    dist.destroy_process_group()

class AverageMeter:
    """计算并存储平均值和当前值"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """计算 top-k 准确率"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res