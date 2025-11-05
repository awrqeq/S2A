# prepare_tiny_imagenet_val.py
import os
import shutil

print("开始整理 Tiny ImageNet 验证集...")

# 定义你的 Tiny ImageNet 验证集路径
val_dir = './data/tiny-imagenet-200/val'
val_img_dir = os.path.join(val_dir, 'images')
annotations_file = os.path.join(val_dir, 'val_annotations.txt')

if not os.path.exists(val_img_dir):
    print(f"错误：找不到验证集图片目录 '{val_img_dir}'。请确保数据集已正确解压。")
else:
    # 1. 读取标注文件并存储标签
    img_to_class = {}
    with open(annotations_file, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            img_name, class_id = parts[0], parts[1]
            img_to_class[img_name] = class_id

    # 2. 创建类别子文件夹
    for class_id in set(img_to_class.values()):
        class_path = os.path.join(val_img_dir, class_id)
        os.makedirs(class_path, exist_ok=True)

    # 3. 移动图片到对应的子文件夹
    moved_count = 0
    for img_name, class_id in img_to_class.items():
        old_path = os.path.join(val_img_dir, img_name)
        new_path = os.path.join(val_img_dir, class_id, img_name)
        if os.path.exists(old_path):
            shutil.move(old_path, new_path)
            moved_count += 1

    print(f"成功移动 {moved_count} 张验证图片到对应的类别子文件夹中。")
    print("Tiny ImageNet 验证集已成功重构！")