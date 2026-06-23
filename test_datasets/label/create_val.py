import os
import json
import shutil

LABEL_DIR = r"e:\aaaa-WUT\lw\ASCCAD\test_step\label"
VAL_DIR = os.path.join(LABEL_DIR, "val")
TRAIN_DIR = os.path.join(LABEL_DIR, "train")
ANNO_DIR = os.path.join(LABEL_DIR, "annotations")
SRC_JSON = os.path.join(ANNO_DIR, "coco_dataset.json")

os.makedirs(VAL_DIR, exist_ok=True)

# 复制全部 100 张图片到 val
from pathlib import Path
for png in Path(TRAIN_DIR).glob("*.png"):
    dst = os.path.join(VAL_DIR, png.name)
    shutil.copy2(str(png), dst)

# 拷贝 coco 全部结构到 val_dataset.json
with open(SRC_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

val_json_path = os.path.join(ANNO_DIR, "val_dataset.json")
with open(val_json_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"val/ 目录: {len(list(Path(VAL_DIR).glob('*.png')))} 张图片")
print(f"val_dataset.json: {len(data['images'])} images, {len(data['annotations'])} annotations")
print("结构与 coco_dataset.json 完全一致")
