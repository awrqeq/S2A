# visualize.py (最终、无 matplotlib、绝对稳定版)
#
# --- 终极修复 ---
# 1. 彻底移除了对 matplotlib.pyplot 的所有依赖和导入。
# 2. 使用纯 Pillow (PIL) 库来手动创建画布、绘制文字和拼接图片。
# 3. 这个版本只依赖核心计算库和图像处理库，稳定性达到最高。

import torch
import torchvision
from torchvision import transforms
import argparse
import yaml
from PIL import Image, ImageDraw, ImageFont  # <--- [!!!] 导入 Pillow 的绘图工具
import os
import logging

from core.utils import setup_logger
from core.attack import get_injector_instance


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def add_text_to_image(image, text):
    """一个辅助函数，用于在图片上方添加标题文字。"""
    draw = ImageDraw.Draw(image)
    try:
        # 尝试加载一个字体，如果失败则使用默认字体
        font = ImageFont.truetype("arial.ttf", size=24)
    except IOError:
        font = ImageFont.load_default()

    # 获取文字尺寸来居中
    text_width, text_height = draw.textsize(text, font=font)
    position = ((image.width - text_width) // 2, 5)  # 顶部居中

    # 绘制文字（带简单的描边效果，使其更清晰）
    shadow_color = "black"
    text_color = "white"
    draw.text((position[0] - 1, position[1] - 1), text, font=font, fill=shadow_color)
    draw.text((position[0] + 1, position[1] - 1), text, font=font, fill=shadow_color)
    draw.text((position[0] - 1, position[1] + 1), text, font=font, fill=shadow_color)
    draw.text((position[0] + 1, position[1] + 1), text, font=font, fill=shadow_color)
    draw.text(position, text, font=font, fill=text_color)

    return image


def main_visualize(args):
    setup_logger()
    logging.info(f"--- [No-Matplotlib] 加载配置文件: {args.config} ---")
    config = load_config(args.config)

    dataset_config = config['dataset']
    dataset_name = dataset_config.get('name').lower()
    image_size = dataset_config.get('image_size')
    data_path = dataset_config.get('data_path')
    logging.info(f"--- [No-Matplotlib] 配置检测到: Dataset={dataset_name}, ImageSize={image_size}x{image_size} ---")

    injector = get_injector_instance(config, image_size)

    logging.info("--- 加载数据集 ---")
    try:
        dataset_raw = torchvision.datasets.GTSRB(root=data_path, split='train', download=True, transform=None)
    except Exception as e:
        logging.error(f"加载数据集失败: {e}");
        return

    pil_raw, _ = dataset_raw[args.idx]

    resize = transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.LANCZOS)
    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()

    clean_pil = resize(pil_raw.convert("RGB"))
    clean_tensor = to_tensor(clean_pil)

    logging.info(f"--- 已加载图像 #{args.idx} ---")
    poisoned_tensor = injector.inject(clean_tensor.clone())

    diff_tensor = poisoned_tensor - clean_tensor
    if diff_tensor.abs().max() > 1e-6:
        amplified_diff = (diff_tensor - diff_tensor.min()) / (diff_tensor.max() - diff_tensor.min())
    else:
        amplified_diff = torch.zeros_like(diff_tensor)

    poisoned_pil = to_pil(poisoned_tensor)
    diff_pil = to_pil(amplified_diff)

    logging.info("--- 数据处理完毕，开始用 Pillow 拼接图像... ---")

    # [!!!] 使用 Pillow 进行图像拼接
    # 添加标题
    img1 = add_text_to_image(clean_pil, "Original Clean")
    img2 = add_text_to_image(poisoned_pil, "Poisoned Image")
    img3 = add_text_to_image(diff_pil, "Amplified Residual")

    # 创建一个足够大的新画布
    # 我们在图片之间留出 10 像素的白边
    gap = 10
    total_width = img1.width * 3 + gap * 2
    total_height = img1.height

    combined_image = Image.new('RGB', (total_width, total_height), color='white')

    # 将三张图粘贴到画布上
    combined_image.paste(img1, (0, 0))
    combined_image.paste(img2, (img1.width + gap, 0))
    combined_image.paste(img3, (img1.width * 2 + gap * 2, 0))

    # 保存最终的拼接图
    save_path = f"{dataset_name}_visualization_final_idx_{args.idx}.png"
    combined_image.save(save_path)

    logging.info("\n" + "=" * 50)
    logging.info(f"成功！可视化结果已保存到: {save_path}")
    logging.info("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 我们可以用更灵活的默认配置
    parser.add_argument('--config', default='./configs/cifar10_resnet18.yaml')
    parser.add_argument('--idx', type=int, default=100)
    args = parser.parse_args()

    main_visualize(args)