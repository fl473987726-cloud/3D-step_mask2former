import os
import json
import glob
import numpy as np
from PIL import Image

def process_image(img_path, colors_json_dir, output_dir):
    basename = os.path.splitext(os.path.basename(img_path))[0]
    out_sub_dir = os.path.join(output_dir, basename)
    if not os.path.exists(out_sub_dir):
        os.makedirs(out_sub_dir)

    img = Image.open(img_path).convert('RGB')
    img_arr = np.array(img)

    # 读取原始染色 JSON
    json_basename = basename.replace("_ortho_front", "") + "_colors.json"
    json_path = os.path.join(colors_json_dir, json_basename)
    original_rgbs = set()
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            orig_data = json.load(f)
            original_rgbs = set(tuple(item['rgb']) for item in orig_data)

    h, w, _ = img_arr.shape

    # 输出矩阵: 每行 [x, y, r, g, b, is_valid]
    matrix = []

    belongs_arr = img_arr.copy()
    not_belongs_arr = img_arr.copy()

    total_valid = 0
    total_invalid = 0

    for y in range(h):
        for x in range(w):
            r, g, b = img_arr[y, x]
            # 跳过白色背景
            if r == 255 and g == 255 and b == 255:
                continue

            rgb_tuple = (int(r), int(g), int(b))
            is_valid = rgb_tuple in original_rgbs

            matrix.append([x, y, int(r), int(g), int(b), is_valid])

            if is_valid:
                total_valid += 1
            else:
                belongs_arr[y, x] = [255, 255, 255]
                total_invalid += 1

    # 第二遍：把 valid 的像素从 not_belongs 中抹白
    for y in range(h):
        for x in range(w):
            r, g, b = img_arr[y, x]
            if r == 255 and g == 255 and b == 255:
                continue
            rgb_tuple = (int(r), int(g), int(b))
            if rgb_tuple in original_rgbs:
                not_belongs_arr[y, x] = [255, 255, 255]

    # 保存矩阵 JSON（每个数组一行）
    matrix_json_path = os.path.join(out_sub_dir, "pixel_matrix.json")
    with open(matrix_json_path, 'w') as f:
        f.write("[\n")
        for i, row in enumerate(matrix):
            comma = "," if i < len(matrix) - 1 else ""
            f.write(f"  {json.dumps(row)}{comma}\n")
        f.write("]\n")

    # 保存属于染色 RGB 的区域图
    belongs_img = Image.fromarray(belongs_arr)
    belongs_img.save(os.path.join(out_sub_dir, "belongs.png"))

    # 保存不属于染色 RGB 的区域图
    not_belongs_img = Image.fromarray(not_belongs_arr)
    not_belongs_img.save(os.path.join(out_sub_dir, "not_belongs.png"))

    # 统计：图片中出现了哪些颜色，哪些在 JSON 中、哪些不在
    img_rgbs = {}
    for row in matrix:
        rgb = (row[2], row[3], row[4])
        is_valid = row[5]
        if rgb not in img_rgbs:
            img_rgbs[rgb] = {"valid": 0, "invalid": 0}
        if is_valid:
            img_rgbs[rgb]["valid"] += 1
        else:
            img_rgbs[rgb]["invalid"] += 1

    stats = {
        "total_valid_pixels": total_valid,
        "total_invalid_pixels": total_invalid,
        "color_count_in_json": 0,
        "color_count_not_in_json": 0,
        "colors_in_json": [],
        "colors_not_in_json": []
    }

    for rgb, counts in img_rgbs.items():
        entry = {"rgb": list(rgb), "pixel_count": counts["valid"] + counts["invalid"]}
        if counts["invalid"] > 0:
            stats["colors_not_in_json"].append(entry)
        else:
            stats["colors_in_json"].append(entry)

    stats["color_count_in_json"] = len(stats["colors_in_json"])
    stats["color_count_not_in_json"] = len(stats["colors_not_in_json"])

    stats_path = os.path.join(out_sub_dir, "stats.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"Processed {basename}: valid={total_valid}, invalid={total_invalid}")

if __name__ == "__main__":
    input_dir = r"e:\aaaa-WUT\lw\ASCCAD\test_step\test\step_colored_renders"
    colors_json_dir = r"e:\aaaa-WUT\lw\ASCCAD\test_step\test\step_colored"
    output_dir = r"e:\aaaa-WUT\lw\ASCCAD\test_step\test\pixel_valid_check"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    images = glob.glob(os.path.join(input_dir, "*.png"))
    for img_path in images:
        process_image(img_path, colors_json_dir, output_dir)
