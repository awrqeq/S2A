# core/utils.py (最终版 - 智能实验档案系统)
#
# --- v2.0 更新 (实验档案系统) ---
# 1. [新增] 导入了 time 模块。
# 2. [重写] 重写了 setup_logger 函数，现在它更名为 setup_experiment。
# 3. setup_experiment 函数现在执行以下核心操作：
#    a. 根据当前时间、数据集、模型名称，创建一个唯一的实验文件夹路径。
#       例如：'./experiments/20251104-093000_cifar10_ResNet18'
#    b. 自动创建这个文件夹。
#    c. 配置一个“双通道”logger：
#       - 一路像以前一样，实时输出到你的控制台。
#       - 另一路同时将所有日志信息，一字不差地写入到该实验文件夹下的 `training_log.txt` 文件中。
#    d. 返回创建好的实验文件夹路径，供 main_train.py 使用。
# 4. [新增] shutil 模块，用于复制配置文件。

import os
import torch
import logging
import sys
import yaml
import torch.distributed as dist
import time  # [新增]
import shutil  # [新增]


def load_config(config_path):
    """加载 YAML 配置文件"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def setup_experiment(config):
    """
    为一次实验创建专属文件夹，并设置一个同时输出到控制台和文件的logger。
    """
    # 1. 从配置中获取关键信息
    dataset_name = config['dataset']['name']
    # 假设模型名称可以在config中指定，或使用默认值
    model_name = config.get('model_name', 'ResNet18')

    # 2. 创建唯一的、信息丰富的文件夹名称
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    experiment_name = f"{timestamp}_{dataset_name}_{model_name}"

    # 3. 定义并创建实验文件夹
    # 所有实验都将保存在项目根目录下的 'experiments' 文件夹中
    experiments_base_dir = config.get('experiments_base_dir', './experiments')
    experiment_dir = os.path.join(experiments_base_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)

    # 4. 配置 "双通道" 日志系统
    log_file_path = os.path.join(experiment_dir, 'training_log.txt')

    # 获取根logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 清除之前可能存在的任何handler，防止日志重复打印
    if logger.hasHandlers():
        logger.handlers.clear()

    # 创建一个通用的格式化器
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 创建控制台handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 创建文件handler
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 屏蔽 torchvision 的冗长日志
    logging.getLogger("torchvision").setLevel(logging.WARNING)

    # 5. 返回创建好的实验文件夹路径
    return experiment_dir


def save_config_to_experiment_dir(config_path, experiment_dir):
    """将原始配置文件复制到实验文件夹中进行存档。"""
    if os.path.exists(config_path):
        shutil.copy(config_path, os.path.join(experiment_dir, 'config.yaml'))
    else:
        logging.warning(f"无法找到原始配置文件 at {config_path}，跳过存档。")


# --- 以下函数保持不变 ---

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