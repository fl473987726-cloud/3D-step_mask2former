import json
import os
import random
import shutil
from collections import defaultdict
from PIL import Image, ImageOps


# 输入数据目录
DATASET_DIR = r"e:\aaaa-WUT\lw\ASCCAD\test_step\test_datasets\test-instance-mask-0608"
SEMANTIC_DIR = os.path.join(DATASET_DIR, "semantic_views")
MASK_DIR = os.path.join(DATASET_DIR, "masks")
CLASS_MAP_PATH = os.path.join(DATASET_DIR, "class_map.json")

# 输出目录
OUTPUT_DIR = os.path.join(DATASET_DIR, "split_augmented_dataset")

# 数据组织规则：原始数据每 12 张图对应一个模型
VIEWS_PER_MODEL = 12
TEST_VIEWS_PER_MODEL = 3
RANDOM_SEED = 42

# 原图 + 6 种增强 = 7 个版本
AUGMENTATIONS = [
    ("original", None),
    ("flip_ud", "flip_ud"),
    ("flip_lr", "flip_lr"),
    ("rot45", 45),
    ("rot90", 90),
    ("rot135", 135),
    ("rot180", 180),
]


def list_png_names(directory):
    return {
        name for name in os.listdir(directory)
        if name.lower().endswith(".png")
    }


def sort_by_number(names):
    return sorted(names, key=lambda x: int(os.path.splitext(x)[0]))


def get_resample_nearest():
    if hasattr(Image, "Resampling"):
        return Image.Resampling.NEAREST
    return Image.NEAREST


def apply_augmentation(img, aug_op, fillcolor):
    if aug_op is None:
        return img.copy()
    if aug_op == "flip_ud":
        return ImageOps.flip(img)
    if aug_op == "flip_lr":
        return ImageOps.mirror(img)

    return img.rotate(
        aug_op,
        resample=get_resample_nearest(),
        expand=False,
        fillcolor=fillcolor,
    )


def ensure_clean_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def prepare_output_dirs():
    ensure_clean_dir(OUTPUT_DIR)
    for split in ["train", "test"]:
        os.makedirs(os.path.join(OUTPUT_DIR, split, "semantic_views"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_DIR, split, "masks"), exist_ok=True)


def load_valid_image_names():
    with open(CLASS_MAP_PATH, "r", encoding="utf-8") as f:
        class_map = json.load(f)

    semantic_names = list_png_names(SEMANTIC_DIR)
    mask_names = list_png_names(MASK_DIR)
    class_map_names = set(class_map.keys())

    valid_names = sort_by_number(semantic_names & mask_names & class_map_names)

    print(f"semantic_views 数量: {len(semantic_names)}")
    print(f"masks 数量: {len(mask_names)}")
    print(f"class_map 条目数: {len(class_map_names)}")
    print(f"三者都存在的有效图片: {len(valid_names)}")

    missing_semantic = sort_by_number((mask_names | class_map_names) - semantic_names)
    missing_mask = sort_by_number((semantic_names | class_map_names) - mask_names)
    missing_class_map = sort_by_number((semantic_names | mask_names) - class_map_names)

    if missing_semantic:
        print(f"警告: {len(missing_semantic)} 张图片缺少 semantic_views，将跳过，例如: {missing_semantic[:5]}")
    if missing_mask:
        print(f"警告: {len(missing_mask)} 张图片缺少 masks，将跳过，例如: {missing_mask[:5]}")
    if missing_class_map:
        print(f"警告: {len(missing_class_map)} 张图片缺少 class_map，将跳过，例如: {missing_class_map[:5]}")

    return valid_names, class_map


def group_by_model(valid_names):
    groups = defaultdict(list)
    for name in valid_names:
        image_no = int(os.path.splitext(name)[0])
        model_idx = (image_no - 1) // VIEWS_PER_MODEL
        view_idx = (image_no - 1) % VIEWS_PER_MODEL
        groups[model_idx].append((view_idx, name))

    model_groups = []
    for model_idx in sorted(groups.keys()):
        views = sorted(groups[model_idx], key=lambda x: x[0])
        if len(views) != VIEWS_PER_MODEL:
            print(f"警告: 模型 {model_idx + 1} 只有 {len(views)} 个有效视角，不足 12 个，仍按现有视角划分")
        model_groups.append((model_idx, views))

    return model_groups


def get_fillcolor(img, is_mask):
    if is_mask:
        return 255
    if img.mode == "RGBA":
        return (255, 255, 255, 255)
    if img.mode == "RGB":
        return (255, 255, 255)
    return 255


def save_augmented_pair(split_name, src_name, class_map, global_counter, split_counters, split_class_maps, split_records):
    semantic_path = os.path.join(SEMANTIC_DIR, src_name)
    mask_path = os.path.join(MASK_DIR, src_name)

    with Image.open(semantic_path) as img:
        semantic_img = img.convert("RGB")
        semantic_img.load()

    with Image.open(mask_path) as img:
        mask_img = img.convert("L")
        mask_img.load()

    for aug_name, aug_op in AUGMENTATIONS:
        global_counter["value"] += 1
        split_counters[split_name] += 1
        new_name = f"{global_counter['value']:06d}.png"

        semantic_aug = apply_augmentation(
            semantic_img,
            aug_op,
            fillcolor=get_fillcolor(semantic_img, is_mask=False),
        )
        mask_aug = apply_augmentation(
            mask_img,
            aug_op,
            fillcolor=get_fillcolor(mask_img, is_mask=True),
        )

        semantic_out = os.path.join(OUTPUT_DIR, split_name, "semantic_views", new_name)
        mask_out = os.path.join(OUTPUT_DIR, split_name, "masks", new_name)

        semantic_aug.save(semantic_out)
        mask_aug.save(mask_out)
        semantic_aug.close()
        mask_aug.close()

        split_class_maps[split_name][new_name] = class_map[src_name]
        split_records[split_name].append({
            "new_name": new_name,
            "source_name": src_name,
            "augmentation": aug_name,
        })

    semantic_img.close()
    mask_img.close()


def main():
    random.seed(RANDOM_SEED)
    valid_names, class_map = load_valid_image_names()
    model_groups = group_by_model(valid_names)

    prepare_output_dirs()

    global_counter = {"value": 0}
    split_counters = {"train": 0, "test": 0}
    split_class_maps = {"train": {}, "test": {}}
    split_records = {"train": [], "test": []}
    split_model_views = {}

    print(f"模型数量: {len(model_groups)}")
    print(f"每个原始视角输出版本数: {len(AUGMENTATIONS)}")
    print(f"随机种子: {RANDOM_SEED}")
    print("\n开始增强并划分...")

    for model_idx, views in model_groups:
        view_indices = [view_idx for view_idx, _ in views]
        test_count = min(TEST_VIEWS_PER_MODEL, len(view_indices))
        test_view_set = set(random.sample(view_indices, test_count))

        split_model_views[f"model_{model_idx + 1:04d}"] = {
            "test_original_views_0_based": sorted(test_view_set),
            "test_original_views_1_based": [v + 1 for v in sorted(test_view_set)],
        }

        for view_idx, src_name in views:
            split_name = "test" if view_idx in test_view_set else "train"
            save_augmented_pair(
                split_name,
                src_name,
                class_map,
                global_counter,
                split_counters,
                split_class_maps,
                split_records,
            )

    for split in ["train", "test"]:
        class_map_out = os.path.join(OUTPUT_DIR, split, "class_map.json")
        with open(class_map_out, "w", encoding="utf-8") as f:
            json.dump(split_class_maps[split], f, ensure_ascii=False, indent=2)

    split_info = {
        "dataset_dir": DATASET_DIR,
        "views_per_model": VIEWS_PER_MODEL,
        "test_views_per_model_before_augmentation": TEST_VIEWS_PER_MODEL,
        "random_seed": RANDOM_SEED,
        "augmentations": [name for name, _ in AUGMENTATIONS],
        "model_views": split_model_views,
        "records": split_records,
        "counts": {
            "valid_original_images": len(valid_names),
            "models": len(model_groups),
            "train_images": split_counters["train"],
            "test_images": split_counters["test"],
            "total_augmented_images": split_counters["train"] + split_counters["test"],
        },
    }
    with open(os.path.join(OUTPUT_DIR, "split_info.json"), "w", encoding="utf-8") as f:
        json.dump(split_info, f, ensure_ascii=False, indent=2)

    print("\n完成")
    print(f"训练集: {split_counters['train']} 张 semantic_views + {split_counters['train']} 张 masks")
    print(f"测试集: {split_counters['test']} 张 semantic_views + {split_counters['test']} 张 masks")
    print(f"输出目录: {OUTPUT_DIR}")
    print("说明: train/test 内 semantic_views 与 masks 使用相同编号，class_map.json 的键也同步为增强后的新编号。")


if __name__ == "__main__":
    main()
