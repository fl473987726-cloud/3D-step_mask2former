import os
import json
import shutil

SRC_DIR = r"e:\aaaa-WUT\lw\ASCCAD\test_step\label\train"
JSON_PATH = r"e:\aaaa-WUT\lw\ASCCAD\test_step\label\annotations\coco_dataset.json"
TOTAL_COPIES = 50  # 每张图复制 50 份 → 共 100 张

# 读取原 JSON
with open(JSON_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

original_images = data["images"]
original_annotations = data["annotations"]
categories = data["categories"]

# 建立映射：原始 image_id → {annotations, file_path}
image_annotations_map = {1: [], 2: []}
for ann in original_annotations:
    img_id = ann["image_id"]
    if img_id in image_annotations_map:
        image_annotations_map[img_id].append(ann)

# 建立映射：原始 image_id → 源文件名
src_file_map = {
    1: os.path.join(SRC_DIR, "001.png"),
    2: os.path.join(SRC_DIR, "002.png"),
}

new_images = []
new_annotations = []
next_ann_id = 1

for i in range(1, TOTAL_COPIES * 2 + 1):
    file_name = f"{i:03d}.png"
    new_images.append({
        "id": i,
        "file_name": file_name,
        "width": 1024,
        "height": 768,
    })

    # 确定来源：奇数 → 001.png(原image_id=1)，偶数 → 002.png(原image_id=2)
    if i % 2 == 1:
        src_img_id = 1
    else:
        src_img_id = 2

    # 复制图片（001 和 002 保留原样）
    dst_path = os.path.join(SRC_DIR, file_name)
    if not os.path.exists(dst_path):
        shutil.copy2(src_file_map[src_img_id], dst_path)

    # 复制标注
    for ann in image_annotations_map[src_img_id]:
        new_ann = dict(ann)  # 浅拷贝
        new_ann["id"] = next_ann_id
        new_ann["image_id"] = i
        new_annotations.append(new_ann)
        next_ann_id += 1

# 输出新 JSON
output_data = {
    "images": new_images,
    "annotations": new_annotations,
    "categories": categories,
}

with open(JSON_PATH, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)

print(f"完成：{len(new_images)} 张图片, {len(new_annotations)} 条标注")
print(f"JSON 已更新: {JSON_PATH}")
print(f"图片目录: {SRC_DIR}")
