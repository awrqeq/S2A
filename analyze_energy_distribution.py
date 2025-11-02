# analyze_energy_distribution.py
#
# --- 目的 ---
# 1. 动态加载并分析一个数据集 (CIFAR-10, GTSRB, etc.)。
# 2. 对数据集中所有图像进行 DWT 分解。
# 3. 对指定的子带 (hl, lh, hh) 进行 SSA 分离。
# 4. 计算并输出以下关键指标：
#    - 每个子带的“平均总能量 (范数的平方)”
#    - 每个子带的“平均噪声能量 (范数的平方)”
#
# --- 如何运行 ---
# 1. 修改 CONFIG_PATH 指向你的配置文件。
# 2. 运行脚本: python analyze_energy_distribution.py

import numpy as np
import yaml
from tqdm import tqdm

# (直接复用我们之前完善的、健壮的 analyze_dataset_wide_noise_floor.py 的代码)
from analyze_dataset_wide_noise_floor import S2A_Final_Injector, main as analyze_main, load_config
from torchvision import datasets, transforms
from PIL import Image
import os
import pywt


def main():
    # --- 配置 ---
    # [!!! 核心 !!!] 修改这里，指向你想要分析的数据集的配置文件
    # 首先，分析 CIFAR-10
    #CONFIG_PATH = './configs/cifar10_resnet18.yaml'
    # 稍后，再切换到 GTSRB 进行分析
    CONFIG_PATH = 'configs/gtsrb_64x64_random.yaml'

    print(f"===========================================================")
    print(f"======  开始分析配置文件: {CONFIG_PATH}  ======")
    print(f"===========================================================")

    # 1. 加载配置和S2A注入器 (仅用于算法)
    config = load_config(CONFIG_PATH)
    injector = S2A_Final_Injector(config)

    # 2. 动态加载数据集 (复用之前的逻辑)
    dataset_config = config['dataset']
    dataset_name = dataset_config['name'].lower()
    image_size = dataset_config.get('image_size', 32)
    data_path = dataset_config.get('data_path', './data')

    print(f"正在加载 {dataset_name.upper()} 训练集...")
    if dataset_name == 'cifar10':
        raw_dataset = datasets.CIFAR10(root=data_path, train=True, download=True, transform=None)
        image_source_list = raw_dataset.data
        data_loader = lambda x: Image.fromarray(x)
    elif dataset_name == 'gtsrb':
        raw_dataset = datasets.GTSRB(root=data_path, split='train', download=True, transform=None)
        image_source_list = raw_dataset
        data_loader = lambda x: x[0]
    # (可以按需加入 tiny_imagenet 的逻辑)
    else:
        raise ValueError(f"不支持的数据集: {dataset_name}")

    analysis_transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.Lambda(lambda img: img.convert('RGB'))
    ])

    # 3. 循环分析，并累加能量
    hl_total_energies, lh_total_energies, hh_total_energies = [], [], []
    hl_noise_energies, lh_noise_energies, hh_noise_energies = [], [], []

    print(f"正在分析 {len(image_source_list)} 张图像的能量分布...")
    for raw_img_data in tqdm(image_source_list):
        pil_img = data_loader(raw_img_data)
        resized_pil = analysis_transform(pil_img)
        img_np_rgb = np.array(resized_pil).astype(np.float64) / 255.0

        for c in range(3):
            img_np_channel = img_np_rgb[:, :, c]
            try:
                coeffs = pywt.wavedec2(img_np_channel, injector.wavelet, level=1)
                _, (HL, LH, HH) = coeffs
            except ValueError:
                continue

            # 存储总能量 (范数的平方)
            hl_total_energies.append(np.linalg.norm(HL) ** 2)
            lh_total_energies.append(np.linalg.norm(LH) ** 2)
            hh_total_energies.append(np.linalg.norm(HH) ** 2)

            # 分离并存储噪声能量
            _, hl_noise = injector._decompose_subband(HL, 'hl')
            _, lh_noise = injector._decompose_subband(LH, 'lh')
            _, hh_noise = injector._decompose_subband(HH, 'hh')

            hl_noise_energies.append(np.linalg.norm(hl_noise) ** 2)
            lh_noise_energies.append(np.linalg.norm(lh_noise) ** 2)
            hh_noise_energies.append(np.linalg.norm(hh_noise) ** 2)

    # 4. 计算平均值并输出
    print("\n" + "=" * 50)
    print(f"数据集 [{dataset_name.upper()}] 能量分布分析完成！")
    print("=" * 50)
    print("平均总能量 (Avg_Energy_Total):")
    print(f"  HL: {np.mean(hl_total_energies):.6f}")
    print(f"  LH: {np.mean(lh_total_energies):.6f}")
    print(f"  HH: {np.mean(hh_total_energies):.6f}")
    print("-" * 50)
    print("平均噪声能量 (Avg_Energy_Noise):")
    print(f"  HL: {np.mean(hl_noise_energies):.6f}")
    print(f"  LH: {np.mean(lh_noise_energies):.6f}")
    print(f"  HH: {np.mean(hh_noise_energies):.6f}")
    print("=" * 50)


if __name__ == "__main__":
    main()