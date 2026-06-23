import json
import os
import random
import shutil

def split_coco_dataset(dataset_dir):
    annotations_file = os.path.join(dataset_dir, 'annotations.json')
    images_dir = os.path.join(dataset_dir, 'images')

    with open(annotations_file, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    # Each model originally has 12 views (IDs 1-300 for 25 models)
    # Augmented images start at ID 301, with 7 variants per original image
    # So for merged dataset: model_id = (orig_id - 1) // 12 for all images
    AUG_START_ID = 301
    VIEWS_PER_MODEL = 12
    AUG_PER_IMAGE = 6

    def get_model_id(img_id):
        if img_id < AUG_START_ID:
            return (img_id - 1) // VIEWS_PER_MODEL
        else:
            orig_id = ((img_id - AUG_START_ID) // AUG_PER_IMAGE) + 1
            return (orig_id - 1) // VIEWS_PER_MODEL

    model_groups = {}
    for img in coco_data['images']:
        img_id = img['id']
        model_id = get_model_id(img_id)
        if model_id not in model_groups:
            model_groups[model_id] = []
        model_groups[model_id].append(img_id)

    model_ids = list(model_groups.keys())
    print(f"Total models found: {len(model_ids)}")

    random.seed(42)
    random.shuffle(model_ids)

    train_models = model_ids[:20]
    val_models = model_ids[20:23]
    test_models = model_ids[23:]

    print(f"Train models: {len(train_models)}")
    print(f"Val models: {len(val_models)}")
    print(f"Test models: {len(test_models)}")

    splits = {
        'train': train_models,
        'val': val_models,
        'test': test_models
    }

    output_dir = os.path.join(dataset_dir, 'split_dataset')
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    annotations_dir = os.path.join(output_dir, 'annotations')
    os.makedirs(annotations_dir, exist_ok=True)

    for split_name, split_model_ids in splits.items():
        split_images_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_images_dir, exist_ok=True)

        split_image_ids = set()
        for mid in split_model_ids:
            split_image_ids.update(model_groups[mid])

        split_images = [img for img in coco_data['images'] if img['id'] in split_image_ids]
        split_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] in split_image_ids]

        split_json = {
            'images': split_images,
            'annotations': split_annotations,
            'categories': coco_data['categories']
        }

        json_path = os.path.join(annotations_dir, f'instances_{split_name}.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(split_json, f, ensure_ascii=False, indent=2)

        for img in split_images:
            src_img_path = os.path.join(images_dir, img['file_name'])
            dst_img_path = os.path.join(split_images_dir, img['file_name'])
            if os.path.exists(src_img_path):
                shutil.copy2(src_img_path, dst_img_path)

        print(f"{split_name}: {len(split_images)} images, {len(split_annotations)} annotations")

    print("Dataset successfully split!")

if __name__ == '__main__':
    dataset_dir = r"e:\aaaa-WUT\lw\ASCCAD\test_step\label\datatest01_0527"
    split_coco_dataset(dataset_dir)
