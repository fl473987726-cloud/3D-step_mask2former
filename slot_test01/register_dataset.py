import os
import json
from detectron2.data import DatasetCatalog, MetadataCatalog

def get_mask2former_instances(dataset_dir):
    coco_json = os.path.join(dataset_dir, "coco_instances.json")
    images_dir = os.path.join(dataset_dir, "images")
    sem_masks_dir = os.path.join(dataset_dir, "semantic_masks")

    with open(coco_json, "r") as f:
        coco = json.load(f)

    categories = {c["id"]: c["name"] for c in coco["categories"]}
    images = {img["id"]: img for img in coco["images"]}

    annotations_by_image = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        if img_id not in annotations_by_image:
            annotations_by_image[img_id] = []
        annotations_by_image[img_id].append(ann)

    dataset_dicts = []
    for img_id, img_info in images.items():
        record = {
            "file_name": os.path.join(images_dir, img_info["file_name"]),
            "image_id": img_id,
            "height": img_info["height"],
            "width": img_info["width"],
            "sem_seg_file_name": os.path.join(sem_masks_dir, img_info["file_name"]),
        }

        objs = []
        for ann in annotations_by_image.get(img_id, []):
            obj = {
                "bbox": ann["bbox"],
                "bbox_mode": 0,
                "category_id": ann["category_id"],
                "segmentation": ann["segmentation"],
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts

DATASET_NAME = "mask2former_instances_train"
DATASET_DIR = r"E:/aaaa-WUT/lw/ASCCAD/test_step/slot_test01"

DatasetCatalog.register(DATASET_NAME, lambda: get_mask2former_instances(DATASET_DIR))
MetadataCatalog.get(DATASET_NAME).set(
    thing_classes=list(categories.values()),
    thing_colors=[tuple(c["color"]) if "color" in c else (128, 128, 128) for c in [{'id': 1, 'name': 'Wide Slot', 'display_name': '宽体槽', 'color': (255, 0, 0)}, {'id': 2, 'name': 'Closed Slot', 'display_name': '封闭槽', 'color': (255, 255, 0)}, {'id': 3, 'name': 'Open Slot', 'display_name': '开放槽', 'color': (0, 0, 255)}, {'id': 4, 'name': 'Hole', 'display_name': '孔', 'color': (0, 255, 0)}]],
    evaluator_type="coco",
    image_root=DATASET_DIR,
    json_file=r"E:/aaaa-WUT/lw/ASCCAD/test_step/slot_test01\coco_instances.json",
)
