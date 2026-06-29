# -*- coding: utf-8 -*-
"""
实例标签转语义标签
输入：实例灰度图 + class_map.json
输出：语义灰度图
"""
import os
import json
import numpy as np
from PIL import Image


def convert_instance_to_semantic(instance_mask_path, class_map, output_path):
    """将实例掩码转换为语义掩码"""
    # 读取实例掩码
    instance_mask = np.array(Image.open(instance_mask_path).convert("L"))

    # 创建语义掩码（默认背景=255）
    semantic_mask = np.full_like(instance_mask, 255)

    # 获取图片文件名
    img_name = os.path.basename(instance_mask_path)

    # 获取该图片的类别映射
    if img_name not in class_map:
        print(f"Warning: {img_name} not found in class_map.json")
        return False

    img_class_map = class_map[img_name]

    # 遍历每个实例，替换为对应的类别ID
    for instance_id, class_id in img_class_map.items():
        instance_id = int(instance_id)
        semantic_mask[instance_mask == instance_id] = class_id

    # 保存语义掩码
    Image.fromarray(semantic_mask.astype(np.uint8)).save(output_path)
    return True


def main():
    # 输入文件夹：包含实例掩码PNG和class_map.json
    input_dir = r"F:\B-研生学习\A-科研ing\D-无锡\A-特征识别\test-instance-mask-0608\masks_val"
    output_dir = os.path.join(input_dir, "semantic_masks_val")
    class_map_path = os.path.join(input_dir, "class_map_val.json")

    # 检查class_map.json是否存在
    if not os.path.exists(class_map_path):
        print(f"Error: class_map.json not found in {input_dir}")
        return

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 读取类别映射
    with open(class_map_path, 'r', encoding='utf-8') as f:
        class_map = json.load(f)

    print(f"Loaded class_map.json with {len(class_map)} images")

    # 获取所有实例掩码文件（排除semantic_masks子文件夹里的）
    mask_files = sorted([
        f for f in os.listdir(input_dir)
        if f.endswith('.png') and os.path.isfile(os.path.join(input_dir, f))
    ])
    print(f"Found {len(mask_files)} instance masks")

    # 转换
    success_count = 0
    for mask_file in mask_files:
        instance_path = os.path.join(input_dir, mask_file)
        out_path = os.path.join(output_dir, mask_file)

        if convert_instance_to_semantic(instance_path, class_map, out_path):
            success_count += 1
            print(f"  Converted: {mask_file}")
        else:
            print(f"  Skipped: {mask_file}")

    print(f"\nDone! Converted {success_count}/{len(mask_files)} masks")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
