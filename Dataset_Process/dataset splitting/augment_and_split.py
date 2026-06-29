# -*- coding: utf-8 -*-
"""
数据集增强 + 划分脚本
适配: test-instance-mask-linghting-0626

功能:
  1. 图像增强 (7×): 原图 + 水平翻转 + 垂直翻转 + 45°旋转 + 90°顺时针 + 135°旋转 + 180°旋转
  2. 对应的 masks 用最近邻插值保持像素值
  3. JSON 标签同步扩充
  4. train/val 划分 (按模型分组, 80%/20%)
"""
import os
import sys
import json
import shutil
import random
import cv2
import numpy as np

# ==================== 配置 ====================
DATASET_DIR = r"E:\aaaa-WUT\lw\ASCCAD\test_step\test_datasets\test-instance-mask-linghting-0626"
OUTPUT_DIR = os.path.join(DATASET_DIR, "split_dataset")

# 增强的图像文件夹
AUGMENT_FOLDERS = ["encoded_views", "masks"]
# 需要扩充的 JSON 文件 (name: input_key)
JSON_FILES = {
    "class_map": "class_map.json",
    "camera_views": "camera_views.json",
    "face_encoding_map": "face_encoding_map.json",
}

TRAIN_RATIO = 0.8
RANDOM_SEED = 42
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".bmp")


# ==================== 工具函数 ====================
def imread_unicode(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite_unicode(path, img):
    ext = os.path.splitext(path)[1]
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(path)
    return ok


def rotate_image(img, angle, is_mask=False):
    """旋转图像，mask 用最近邻插值，普通图用白色填充"""
    h, w = img.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    if is_mask:
        # mask: 最近邻插值 + 边界填0（背景）
        if len(img.shape) == 2:
            return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
        else:
            return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=(0, 0, 0))
    else:
        # RGB: 确保3通道，双线性插值 + 边界填白
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))


def augment_image(img, aug_idx, is_mask=False):
    """根据 aug_idx 返回增强后的图像 (0~6, 共7种)"""
    if aug_idx == 0:
        return img.copy()
    elif aug_idx == 1:
        return cv2.flip(img, 1)   # 水平翻转
    elif aug_idx == 2:
        return cv2.flip(img, 0)   # 垂直翻转
    elif aug_idx == 3:
        return rotate_image(img, 45, is_mask)     # 45°
    elif aug_idx == 4:
        if is_mask:
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif aug_idx == 5:
        return rotate_image(img, 135, is_mask)    # 135°
    elif aug_idx == 6:
        return cv2.rotate(img, cv2.ROTATE_180)
    return img.copy()


# ==================== Step 1: 图像增强 ====================
def step1_augment_images():
    print("=" * 60)
    print("Step 1: 图像数据增强 (7×)")
    print("=" * 60)

    aug_base = os.path.join(DATASET_DIR, "_augmented")
    os.makedirs(aug_base, exist_ok=True)

    # 确定参考文件集: 以 masks 为准（因为 class_map 只覆盖 masks 中的文件）
    # encoded_views 可能比 masks 多（如 180 vs 84），只处理配对的文件
    masks_dir = os.path.join(DATASET_DIR, "masks")
    if os.path.exists(masks_dir):
        paired_names = set(sorted([
            f for f in os.listdir(masks_dir) if f.lower().endswith(IMAGE_EXT)
        ]))
        print(f"  参考文件集 (masks): {len(paired_names)} 个文件")
    else:
        paired_names = None

    for folder_name in AUGMENT_FOLDERS:
        src_dir = os.path.join(DATASET_DIR, folder_name)
        dst_dir = os.path.join(aug_base, folder_name)
        os.makedirs(dst_dir, exist_ok=True)

        if not os.path.exists(src_dir):
            print(f"  [跳过] 目录不存在: {src_dir}")
            continue

        files = sorted([f for f in os.listdir(src_dir) if f.lower().endswith(IMAGE_EXT)])

        # 如果有参考文件集，只处理配对的文件
        if paired_names is not None and folder_name != "masks":
            files = [f for f in files if f in paired_names]

        is_mask = (folder_name == "masks")
        print(f"\n  处理: {folder_name} ({len(files)} 张原图)")

        for fi, fname in enumerate(files):
            src_path = os.path.join(src_dir, fname)
            img = imread_unicode(src_path)
            if img is None:
                print(f"    [警告] 无法读取: {fname}")
                continue

            name, ext = os.path.splitext(fname)
            for aug_idx in range(7):
                aug_img = augment_image(img, aug_idx, is_mask=is_mask)
                out_name = f"{name}-{aug_idx}{ext}"
                imwrite_unicode(os.path.join(dst_dir, out_name), aug_img)

            if (fi + 1) % 20 == 0 or fi + 1 == len(files):
                print(f"    进度: {fi + 1}/{len(files)}")

    print(f"\n  增强图像保存在: {aug_base}")


# ==================== Step 2: JSON 扩充 ====================
def step2_augment_json():
    print("\n" + "=" * 60)
    print("Step 2: JSON 标签扩充 (7×)")
    print("=" * 60)

    aug_dir = os.path.join(DATASET_DIR, "_augmented")

    # 确定需要保留的图片名集合（以 masks 中的文件名为准）
    masks_dir = os.path.join(DATASET_DIR, "masks")
    if os.path.exists(masks_dir):
        paired_set = set([
            f for f in os.listdir(masks_dir) if f.lower().endswith(IMAGE_EXT)
        ])
    else:
        paired_set = None

    for json_name, json_file in JSON_FILES.items():
        src_path = os.path.join(DATASET_DIR, json_file)
        if not os.path.exists(src_path):
            print(f"  [跳过] {json_file} 不存在")
            continue

        with open(src_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 跳过 encoding_rule 等非图片键
        if json_name == "face_encoding_map":
            # face_encoding_map 包含 models 嵌套结构，需要特殊处理
            new_data = _augment_face_encoding_map(data)
        else:
            # 过滤: 只处理与 masks 配对的条目
            items = data.items()
            if paired_set is not None and json_name == "camera_views":
                items = [(k, v) for k, v in items if k in paired_set]

            new_data = {}
            for img_name, info in items:
                name, ext = os.path.splitext(img_name)
                for i in range(7):
                    new_data[f"{name}-{i}{ext}"] = info

        out_path = os.path.join(aug_dir, f"{json_name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(new_data, f, indent=2, ensure_ascii=False)

        print(f"  {json_name}: {len(data)} -> {len(new_data)} 条")

    print(f"\n  扩充 JSON 保存在: {aug_dir}")


def _augment_face_encoding_map(data):
    """face_encoding_map 有嵌套结构: encoding_rule + models, images 键需要扩充"""
    new_data = {}
    for key, value in data.items():
        if key == "encoding_rule":
            new_data[key] = value
        elif key == "models":
            new_models = {}
            for model_name, model_info in value.items():
                new_model = dict(model_info)
                # 扩充 images 列表
                if "images" in new_model:
                    new_images = []
                    for img in new_model["images"]:
                        name, ext = os.path.splitext(img)
                        for i in range(7):
                            new_images.append(f"{name}-{i}{ext}")
                    new_model["images"] = new_images
                new_models[model_name] = new_model
            new_data[key] = new_models
        else:
            new_data[key] = value
    return new_data


# ==================== Step 3: train/val 划分 ====================
def step3_split():
    print("\n" + "=" * 60)
    print("Step 3: train/val 划分")
    print("=" * 60)

    aug_dir = os.path.join(DATASET_DIR, "_augmented")

    # 增强后的图像文件夹
    image_folders = {}
    for name in AUGMENT_FOLDERS:
        p = os.path.join(aug_dir, name)
        if os.path.exists(p):
            image_folders[name] = p

    # 增强后的 JSON 文件
    json_files = {}
    for json_name in JSON_FILES:
        p = os.path.join(aug_dir, f"{json_name}.json")
        if os.path.exists(p):
            json_files[json_name] = p

    if not image_folders:
        print("  [错误] 没有找到增强后的图像文件夹")
        return

    # 创建输出目录
    train_dir = os.path.join(OUTPUT_DIR, "train")
    val_dir = os.path.join(OUTPUT_DIR, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # 创建图像子目录
    for name in image_folders:
        os.makedirs(os.path.join(train_dir, name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, name), exist_ok=True)

    # 读取所有 JSON 数据
    all_json_data = {}
    for json_name, path in json_files.items():
        with open(path, "r", encoding="utf-8") as f:
            all_json_data[json_name] = json.load(f)

    train_json = {k: {} for k in json_files}
    val_json = {k: {} for k in json_files}

    # 获取参考文件列表（以 masks 为主，如果没有 masks 就用 encoded_views）
    ref_key = "masks" if "masks" in image_folders else list(image_folders.keys())[0]
    ref_dir = image_folders[ref_key]
    all_files = sorted([f for f in os.listdir(ref_dir) if f.lower().endswith(IMAGE_EXT)])

    if not all_files:
        print("  [错误] 没有找到图片文件")
        return

    # 分组: 去掉增强后缀 (-0 ~ -6) 得到原始文件名，按原始文件名分组
    groups = {}
    for fname in all_files:
        name, ext = os.path.splitext(fname)
        # 去掉 -0 ~ -6 后缀
        if name.endswith(("-0", "-1", "-2", "-3", "-4", "-5", "-6")):
            base_name = name[:-2]
        else:
            base_name = name
        if base_name not in groups:
            groups[base_name] = []
        groups[base_name].append(fname)

    group_keys = sorted(groups.keys())
    num_groups = len(group_keys)
    print(f"  共 {num_groups} 个原始样本 (每个含 {len(groups[group_keys[0]])} 张增强图)")

    # 划分: 每个原始样本的 7 张图全部放同一侧
    random.seed(RANDOM_SEED)
    random.shuffle(group_keys)

    n_train = max(1, int(num_groups * TRAIN_RATIO))
    train_keys = sorted(group_keys[:n_train])
    val_keys = sorted(group_keys[n_train:])

    print(f"  训练集: {len(train_keys)} 个样本 ({len(train_keys) * 7} 张图)")
    print(f"  验证集: {len(val_keys)} 个样本 ({len(val_keys) * 7} 张图)")

    # 复制文件 + 提取 JSON
    def copy_split(keys, dest_dir, json_target):
        for base_name in keys:
            for fname in groups[base_name]:
                # 复制所有图像文件夹中的文件
                for folder_name, folder_path in image_folders.items():
                    src = os.path.join(folder_path, fname)
                    dst = os.path.join(dest_dir, folder_name, fname)
                    if os.path.exists(src):
                        shutil.copy2(src, dst)

                # 提取 JSON
                for json_name in json_files:
                    if fname in all_json_data.get(json_name, {}):
                        json_target[json_name][fname] = all_json_data[json_name][fname]

    copy_split(train_keys, train_dir, train_json)
    copy_split(val_keys, val_dir, val_json)

    # 保存 JSON
    for json_name in json_files:
        with open(os.path.join(train_dir, f"{json_name}.json"), "w", encoding="utf-8") as f:
            json.dump(train_json[json_name], f, indent=2, ensure_ascii=False)
        with open(os.path.join(val_dir, f"{json_name}.json"), "w", encoding="utf-8") as f:
            json.dump(val_json[json_name], f, indent=2, ensure_ascii=False)

    # 统计
    print(f"\n  ====== 划分完成 ======")
    for name in image_folders:
        n_train_files = len(os.listdir(os.path.join(train_dir, name)))
        n_val_files = len(os.listdir(os.path.join(val_dir, name)))
        print(f"  {name}: train={n_train_files}, val={n_val_files}")
    for json_name in json_files:
        print(f"  {json_name}.json: train={len(train_json[json_name])}, val={len(val_json[json_name])}")

    print(f"\n  输出目录: {OUTPUT_DIR}")
    print(f"    train/: {os.path.join(OUTPUT_DIR, 'train')}")
    print(f"    val/:   {os.path.join(OUTPUT_DIR, 'val')}")


# ==================== 主函数 ====================
def main():
    print("数据集增强 + 划分工具")
    print(f"数据集路径: {DATASET_DIR}")
    print(f"输出路径:   {OUTPUT_DIR}")
    print()

    step1_augment_images()
    step2_augment_json()
    step3_split()

    print("\n" + "=" * 60)
    print("全部完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
