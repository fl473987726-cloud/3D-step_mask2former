import json
import cv2
import os
import numpy as np
import copy
import argparse

def augment_coco(dataset_dir, out_dir, start_img_id=1):
    annotations_file = os.path.join(dataset_dir, 'annotations.json')
    images_dir = os.path.join(dataset_dir, 'images')
    
    if not os.path.exists(annotations_file):
        print(f"Annotations not found in {dataset_dir}")
        return
        
    with open(annotations_file, 'r', encoding='utf-8') as f:
        coco = json.load(f)
        
    out_images_dir = os.path.join(out_dir, 'images')
    os.makedirs(out_images_dir, exist_ok=True)
    
    new_coco = {
        'images': [],
        'annotations': [],
        'categories': coco['categories']
    }
    
    img_id_counter = start_img_id
    ann_id_counter = 1
    
    img_to_anns = {}
    for ann in coco['annotations']:
        img_to_anns.setdefault(ann['image_id'], []).append(ann)
        
    operations = [
        ('orig', None),
        ('flip_h', lambda img: cv2.flip(img, 1)),
        ('flip_v', lambda img: cv2.flip(img, 0)),
        ('rot_45', 45),
        ('rot_90', 90),
        ('rot_135', 135),
        ('rot_180', 180)
    ]
    
    for img_info in coco['images']:
        orig_img_path = os.path.join(images_dir, img_info['file_name'])
        if not os.path.exists(orig_img_path):
            continue
            
        img = cv2.imdecode(np.fromfile(orig_img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
            
        h, w = img.shape[:2]
        center = (w / 2.0, h / 2.0)
        
        anns = img_to_anns.get(img_info['id'], [])
        
        for op_name, op_val in operations:
            # Generate new 6-digit image name
            aug_img_name = f"{img_id_counter:06d}.png"
            aug_img_path = os.path.join(out_images_dir, aug_img_name)
            
            aug_img = None
            M = None
            is_flip_h = False
            is_flip_v = False
            
            if op_name == 'orig':
                aug_img = img.copy()
            elif op_name == 'flip_h':
                aug_img = op_val(img)
                is_flip_h = True
            elif op_name == 'flip_v':
                aug_img = op_val(img)
                is_flip_v = True
            elif op_name.startswith('rot_'):
                angle = op_val
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                borderValue = (255, 255, 255)
                aug_img = cv2.warpAffine(img, M, (w, h), borderValue=borderValue)
                
            cv2.imencode('.png', aug_img)[1].tofile(aug_img_path)
            
            new_img_info = {
                'id': img_id_counter,
                'file_name': aug_img_name,
                'width': w,
                'height': h
            }
            new_coco['images'].append(new_img_info)
            
            for ann in anns:
                new_ann = copy.deepcopy(ann)
                new_ann['id'] = ann_id_counter
                new_ann['image_id'] = img_id_counter
                
                new_segmentation = []
                for poly in ann['segmentation']:
                    pts = np.array(poly).reshape(-1, 2)
                    new_pts = np.zeros_like(pts, dtype=np.float32)
                    
                    if op_name == 'orig':
                        new_pts = pts.copy()
                    elif is_flip_h:
                        new_pts[:, 0] = w - pts[:, 0]
                        new_pts[:, 1] = pts[:, 1]
                    elif is_flip_v:
                        new_pts[:, 0] = pts[:, 0]
                        new_pts[:, 1] = h - pts[:, 1]
                    elif M is not None:
                        pts_ones = np.hstack([pts, np.ones((pts.shape[0], 1))])
                        new_pts = pts_ones.dot(M.T)
                        
                    new_segmentation.append(new_pts.flatten().tolist())
                    
                new_ann['segmentation'] = new_segmentation
                
                all_pts = np.vstack([np.array(p).reshape(-1, 2) for p in new_segmentation])
                x_min, y_min = np.min(all_pts, axis=0)
                x_max, y_max = np.max(all_pts, axis=0)
                
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(w, x_max)
                y_max = min(h, y_max)
                
                new_ann['bbox'] = [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]
                
                total_area = 0.0
                for poly in new_segmentation:
                    contour = np.array(poly).reshape(-1, 1, 2).astype(np.float32)
                    total_area += cv2.contourArea(contour)
                new_ann['area'] = float(total_area)
                
                new_coco['annotations'].append(new_ann)
                ann_id_counter += 1
                
            img_id_counter += 1
            
        if img_info['id'] % 10 == 0:
            print(f"Processed {img_info['id']} / {len(coco['images'])} original images...")

    out_json = os.path.join(out_dir, 'annotations.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(new_coco, f, ensure_ascii=False, indent=2)
    print(f"Augmentation completed! Output saved to {out_dir}")
    print(f"Original images: {len(coco['images'])}, Augmented images: {len(new_coco['images'])}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--start_id', type=int, default=1, help='Starting image ID and filename number')
    args = parser.parse_args()
    augment_coco(args.input, args.output, start_img_id=args.start_id)
