import json
import os
import shutil

def merge_datasets(original_dir, aug_dir):
    images_dir = os.path.join(original_dir, 'images')
    orig_json_path = os.path.join(original_dir, 'annotations.json')
    aug_json_path = os.path.join(aug_dir, 'annotations.json')
    aug_images_dir = os.path.join(aug_dir, 'images')

    with open(orig_json_path, 'r', encoding='utf-8') as f:
        orig_data = json.load(f)

    with open(aug_json_path, 'r', encoding='utf-8') as f:
        aug_data = json.load(f)

    existing_files = set(os.listdir(images_dir))

    print(f"Original images: {len(orig_data['images'])}")
    print(f"Original annotations: {len(orig_data['annotations'])}")
    print(f"Augmented images: {len(aug_data['images'])}")
    print(f"Augmented annotations: {len(aug_data['annotations'])}")

    copied_count = 0
    for img in aug_data['images']:
        fname = img['file_name']
        if fname not in existing_files:
            src = os.path.join(aug_images_dir, fname)
            dst = os.path.join(images_dir, fname)
            if os.path.exists(src):
                shutil.copy2(src, dst)
                copied_count += 1

    print(f"Copied {copied_count} new images")

    max_ann_id = max(a['id'] for a in orig_data['annotations']) if orig_data['annotations'] else 0
    for ann in aug_data['annotations']:
        ann['id'] += max_ann_id

    merged_images = orig_data['images'] + aug_data['images']
    merged_annotations = orig_data['annotations'] + aug_data['annotations']

    merged_data = {
        'images': merged_images,
        'annotations': merged_annotations,
        'categories': orig_data['categories']
    }

    backup_path = os.path.join(original_dir, 'annotations_backup.json')
    shutil.copy2(orig_json_path, backup_path)
    print(f"Backup saved to {backup_path}")

    with open(orig_json_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)

    print(f"Merged annotations.json: {len(merged_images)} images, {len(merged_annotations)} annotations")
    print("Done!")

if __name__ == '__main__':
    original = r'e:\aaaa-WUT\lw\ASCCAD\test_step\label\datatest01_0527'
    augmented = r'e:\aaaa-WUT\lw\ASCCAD\test_step\label\datatest01_0527_aug'
    merge_datasets(original, augmented)
