# measure_noise_norm.py (修复版)

import torch
import numpy as np
import logging
from tqdm import tqdm
import argparse
import yaml
import pywt  # 将 import pywt 移到顶部，是个好习惯

# [!!! 关键修复 !!!]
# 我们现在从 utils 导入新的“实验设置”函数
from core.utils import load_config, setup_experiment
from core.dataset import PoisonedDataset
from core.attack import get_injector_instance, S2A_Final_Injector


def main():
    parser = argparse.ArgumentParser(description='Measure Noise Norm Distribution for a Dataset')
    parser.add_argument('--config', default='./configs/imagenette_resnet18.yaml', help='Path to the YAML config file')
    args = parser.parse_args()

    config = load_config(args.config)

    # [!!! 关键修复 !!!]
    # 调用新的 setup_experiment 函数。
    # 它会创建日志文件等，但对于测量脚本来说，我们主要用它来初始化日志系统。
    # 我们不需要用到它返回的 experiment_dir。
    setup_experiment(config)

    logging.info(f"Loading dataset '{config['dataset']['name']}' to measure noise norm...")

    dataset = PoisonedDataset(config, train=True, poison=False)

    # 获取S2A注入器实例，我们只是为了借用它的内部方法
    # (注意: 这里我们假设在measure_noise_norm.py运行时，attack.py里的全局变量_GLOBAL_INJECTOR会被创建)
    s2a_injector = get_injector_instance(config, image_size=config['dataset']['image_size'])
    if not isinstance(s2a_injector, S2A_Final_Injector):
        raise TypeError("This script requires the DWT-based S2A_Final_Injector for measurement.")

    noise_norms = []
    num_samples_to_measure = min(1000, len(dataset))

    logging.info(f"Iterating through {num_samples_to_measure} samples to collect noise norms...")

    # (之前关于如何获取pre-transform张量的逻辑有点复杂，我们简化并修正一下)
    for i in tqdm(range(num_samples_to_measure)):
        # 直接从 clean_dataset 获取原始PIL Image，这是最可靠的方式
        original_idx = dataset.indices[i]
        img, _ = dataset.clean_dataset[original_idx]

        if img.mode != 'RGB':
            img = img.convert("RGB")

        # 手动应用注入前的所有变换 (transform_pre)
        img_tensor_pre = dataset.transform_pre(img)

        for c in range(img_tensor_pre.shape[0]):
            img_np = img_tensor_pre[c].numpy().astype(np.float64)

            # 使用与攻击代码完全相同的DWT分解
            coeffs = pywt.wavedec2(img_np, s2a_injector.wavelet, level=1)
            _, (HL, LH, HH) = coeffs
            sb_map_clean = {'hl': HL, 'lh': LH, 'hh': HH}

            for key in s2a_injector.subband_keys_to_attack:
                subband_data = sb_map_clean.get(key.lower())
                if subband_data is not None:
                    _, noise_clean = s2a_injector._decompose_subband(subband_data, key.lower())
                    norm = np.linalg.norm(noise_clean)
                    noise_norms.append(norm)

    noise_norms = np.array(noise_norms)

    logging.info("\n" + "=" * 50)
    logging.info("--- Noise Norm Measurement Results ---")
    logging.info("=" * 50)
    logging.info(f"Total norms measured: {len(noise_norms)}")
    logging.info(f"Mean (平均值):     {np.mean(noise_norms):.6f}")
    logging.info(f"Median (中位数):   {np.median(noise_norms):.6f}  <-- [!!!] 推荐作为 midpoint_norm 的值")
    logging.info(f"Std Dev (标准差):  {np.std(noise_norms):.6f}")
    logging.info(f"Min (最小值):      {np.min(noise_norms):.6f}")
    logging.info(f"Max (最大值):      {np.max(noise_norms):.6f}")
    logging.info(f"25th Percentile:  {np.percentile(noise_norms, 25):.6f}")
    logging.info(f"75th Percentile:  {np.percentile(noise_norms, 75):.6f}")
    logging.info("=" * 50)


if __name__ == '__main__':
    main()