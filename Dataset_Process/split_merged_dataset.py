# 数据集划分
# 
import json, os, shutil, random
from collections import defaultdict
dataset_dir = r"e:\aaaa-WUT\lw\ASCCAD\test_step\test_datasets\test-instance-0603_augmented"
output_dir = os.path.join(dataset_dir, "split_dataset")
images_per_model = 84  # 12 views * 7 variants
train_count = 17
test_count = 2

ann_path = os.path.join(dataset_dir, "annotations.json")
images_dir = os.path.join(dataset_dir, "images")

with open(ann_path, "r", encoding="utf-8") as f:
    coco = json.load(f)

all_images = sorted(coco["images"], key=lambda x: x["id"])
total_models = len(all_images) // images_per_model
print(f"原始数据: {len(all_images)} 张图片, {total_models} 个模型")

# group images by model (consecutive 84)
model_groups = []
for m in range(total_models):
    start = m * images_per_model
    end = start + images_per_model
    model_groups.append(all_images[start:end])

# shuffle models
random.seed(42)
random.shuffle(model_groups)

train_models = model_groups[:train_count]
test_models = model_groups[train_count:train_count + test_count]

def build_split(split_name, model_list):
    split_dir = os.path.join(output_dir, split_name, "images")
    os.makedirs(split_dir, exist_ok=True)

    split_img_ids = set()
    new_images = []
    for model_imgs in model_list:
        for img_info in model_imgs:
            split_img_ids.add(img_info["id"])
            new_images.append(img_info)
            src = os.path.join(images_dir, img_info["file_name"])
            dst = os.path.join(split_dir, img_info["file_name"])
            shutil.copy2(src, dst)

    new_annotations = []
    for ann in coco["annotations"]:
        if ann["image_id"] in split_img_ids:
            new_annotations.append(ann)

    new_coco = {
        "images": new_images,
        "annotations": new_annotations,
        "categories": coco["categories"]
    }

    ann_out_dir = os.path.join(output_dir, split_name, "annotations")
    os.makedirs(ann_out_dir, exist_ok=True)
    out_path = os.path.join(ann_out_dir, f"instances_{split_name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(new_coco, f, ensure_ascii=False, indent=2)

    print(f"{split_name}: {len(new_images)} 图片, {len(new_annotations)} 标注")
    return len(new_images), len(new_annotations)

print("\n划分结果:")
total_train_img, total_train_ann = build_split("train", train_models)
total_test_img, total_test_ann = build_split("test", test_models)
print(f"\n总计: 训练 {total_train_img} 图片 / {total_train_ann} 标注, 测试 {total_test_img} 图片 / {total_test_ann} 标注")
print(f"输出目录: {output_dir}")
