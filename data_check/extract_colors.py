# 从png文件中提取所有颜色的坐标
# 输入png文件，输出所有颜色的坐标
import os
import json
import glob
import numpy as np
from PIL import Image
from collections import deque

def label_connected_components(mask):
    h, w = mask.shape
    labeled = np.zeros_like(mask, dtype=np.int32)
    comp_sizes = []
    comp_coords = []
    label = 0

    for y in range(h):
        for x in range(w):
            if mask[y, x] and labeled[y, x] == 0:
                label += 1
                q = deque([(y, x)])
                labeled[y, x] = label
                coords = []
                size = 0
                while q:
                    cy, cx = q.popleft()
                    coords.append([int(cx), int(cy)])
                    size += 1
                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and labeled[ny, nx] == 0:
                            labeled[ny, nx] = label
                            q.append((ny, nx))
                comp_sizes.append(size)
                comp_coords.append(coords)
    return labeled, label, comp_sizes, comp_coords


def process_image(img_path, output_dir, min_component_size=2):
    basename = os.path.splitext(os.path.basename(img_path))[0]
    out_sub_dir = os.path.join(output_dir, basename)
    if not os.path.exists(out_sub_dir):
        os.makedirs(out_sub_dir)

    img = Image.open(img_path).convert('RGB')
    img_arr = np.array(img)

    pixels = img_arr.reshape(-1, 3)
    unique_colors, counts = np.unique(pixels, axis=0, return_counts=True)

    results = {}
    color_idx = 0
    skipped_colors = []

    for color, count in zip(unique_colors, counts):
        if np.all(color == [255, 255, 255]):
            continue

        mask = np.all(img_arr == color, axis=-1)
        labeled, n_components, comp_sizes, comp_coords = label_connected_components(mask)

        # 只保留连通块 >= min_component_size 的坐标
        large_components = [(sz, coords) for sz, coords in zip(comp_sizes, comp_coords) if sz >= min_component_size]

        if not large_components:
            skipped_colors.append([int(color[0]), int(color[1]), int(color[2])])
            continue

        all_coords = []
        for _sz, coords in large_components:
            all_coords.extend(coords)

        color_hex = f"{color[0]:02x}{color[1]:02x}{color[2]:02x}"

        results[f"color_{color_hex}"] = {
            "rgb": [int(color[0]), int(color[1]), int(color[2])],
            "pixel_count": int(len(all_coords)),
            "total_pixels_in_image": int(count),
            "connected_components": len(large_components),
            "coordinates": all_coords
        }

        out_img_arr = np.zeros_like(img_arr)
        for _sz, coords in large_components:
            for x, y in coords:
                out_img_arr[y, x] = color

        out_img = Image.fromarray(out_img_arr)
        out_img.save(os.path.join(out_sub_dir, f"mask_{color_hex}.png"))
        color_idx += 1

    with open(os.path.join(out_sub_dir, "coordinates.json"), 'w') as f:
        json.dump(results, f, indent=2)

    # 保存被跳过的噪点颜色
    if skipped_colors:
        with open(os.path.join(out_sub_dir, "skipped_noise_colors.json"), 'w') as f:
            json.dump(skipped_colors, f, indent=2)

    print(f"Processed {basename}: extracted {color_idx} color regions, skipped {len(skipped_colors)} noise colors.")


if __name__ == "__main__":
    input_dir = r"e:\aaaa-WUT\lw\ASCCAD\test_step\test\step_colored_renders"
    output_dir = r"e:\aaaa-WUT\lw\ASCCAD\test_step\test\extracted_colors"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = glob.glob(os.path.join(input_dir, "*.png"))
    for img_path in images:
        process_image(img_path, output_dir, min_component_size=50)
