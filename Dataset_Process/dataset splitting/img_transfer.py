import os
import json
import cv2
import numpy as np

# ================= 配置路径 =================
base_path = r"e:\aaaa-WUT\lw\ASCCAD\test_step\test_datasets\test-instance-mask-linghting-0626"

# 1. 原始图像文件夹列表
src_folders = [
    os.path.join(base_path, "encoded_views"),
    os.path.join(base_path, "masks"),
]

# 2. JSON 标签文件路径 (严格一一对应：一个输入，一个输出)
json_tasks = [
    (
        os.path.join(base_path, "class_map.json"),
        os.path.join(base_path, "class_map_augmented.json")
    ),
    (
        os.path.join(base_path, "camera_views.json"),
        os.path.join(base_path, "camera_views_augmented.json")
    ),
    # face_encoding_map.json 是模型级全局编码配置，不是逐图标签，不做7倍扩充
]

# 支持的图像格式后缀
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')


# ================= 图像处理辅助函数 =================
def imread_unicode(path):
    """支持中文路径的图片读取"""
    try:
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"读取错误: {path}, 原因: {e}")
        return None


def imwrite_unicode(path, img):
    """支持中文路径的图片保存"""
    try:
        ext = os.path.splitext(path)[1]
        result, encoded_img = cv2.imencode(ext, img)
        if result:
            encoded_img.tofile(path)
            return True
    except Exception as e:
        print(f"保存错误: {path}, 原因: {e}")
    return False


def rotate_image_white_bg(image, angle):
    """顺时针旋转图像，保持原图尺寸，空白处补白"""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    rotated = cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return rotated


# ================= 核心任务函数 =================
def process_image_augmentation():
    """任务一：图像数据增强"""
    print("====== 开始执行：任务一（图像数据增强） ======")
    for src_path in src_folders:
        if not os.path.exists(src_path):
            print(f"路径不存在，跳过图像处理: {src_path}")
            continue

        dst_path = src_path + "_augmented"
        os.makedirs(dst_path, exist_ok=True)
        print(f"\n正在处理文件夹: {src_path}")
        print(f"增强后的图像将保存至: {dst_path}")

        file_list = sorted(os.listdir(src_path))
        total_files = len([f for f in file_list if f.lower().endswith(IMAGE_EXTENSIONS)])
        print(f"查找到有效图片共计: {total_files} 张")

        processed_count = 0
        for file_name in file_list:
            if not file_name.lower().endswith(IMAGE_EXTENSIONS):
                continue

            full_file_path = os.path.join(src_path, file_name)
            img = imread_unicode(full_file_path)

            if img is None:
                print(f"【警告】无法读取或图片受损，跳过文件: {file_name}")
                continue

            name, ext = os.path.splitext(file_name)

            try:
                imwrite_unicode(os.path.join(dst_path, f"{name}-0{ext}"), img)
                img_lr = cv2.flip(img, 1)
                imwrite_unicode(os.path.join(dst_path, f"{name}-1{ext}"), img_lr)
                img_ud = cv2.flip(img, 0)
                imwrite_unicode(os.path.join(dst_path, f"{name}-2{ext}"), img_ud)
                img_45 = rotate_image_white_bg(img, 45)
                imwrite_unicode(os.path.join(dst_path, f"{name}-3{ext}"), img_45)
                img_90 = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                imwrite_unicode(os.path.join(dst_path, f"{name}-4{ext}"), img_90)
                img_135 = rotate_image_white_bg(img, 135)
                imwrite_unicode(os.path.join(dst_path, f"{name}-5{ext}"), img_135)
                img_180 = cv2.rotate(img, cv2.ROTATE_180)
                imwrite_unicode(os.path.join(dst_path, f"{name}-6{ext}"), img_180)
                processed_count += 1
            except Exception as e:
                print(f"【错误】处理图片 {file_name} 时发生未知异常: {e}")

        print(f"文件夹 {os.path.basename(src_path)} 处理完成！成功转换 {processed_count}/{total_files} 张图片。")


def process_json_augmentation():
    """任务二：JSON 标签扩充"""
    print("\n====== 开始执行：任务二（JSON 标签扩充） ======")

    # 遍历任务列表，分别获取对应的输入路径和输出路径
    for json_path, output_json_path in json_tasks:
        print(f"\n>> 正在处理 JSON 文件: {os.path.basename(json_path)}")

        if not os.path.exists(json_path):
            print(f"找不到 JSON 文件，跳过此文件处理: {json_path}")
            continue

        # 1. 读取原始 JSON 数据
        with open(json_path, 'r', encoding='utf-8') as f:
            try:
                old_data = json.load(f)
            except json.JSONDecodeError:
                print(f"JSON 文件格式有误，解析失败: {json_path}")
                continue

        new_data = {}

        # 2. 遍历原始数据，进行 7 倍扩充
        for img_name, label_info in old_data.items():
            name, ext = os.path.splitext(img_name)
            for i in range(7):
                new_img_name = f"{name}-{i}{ext}"
                new_data[new_img_name] = label_info

        # 3. 将扩充后的新数据写入新 JSON 文件
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, indent=2, ensure_ascii=False)

        print(f"JSON 标签文件扩充成功！")
        print(f"原图标签数量: {len(old_data)} | 增强后标签数量: {len(new_data)}")
        print(f"新标签文件已保存至: {output_json_path}")


# ================= 主程序入口 =================
if __name__ == "__main__":
    # 1. 先跑图像增强
    process_image_augmentation()

    # 2. 接着跑 JSON 标签扩充
    process_json_augmentation()

    print("\n 所有图像增强与 JSON 标签扩充任务已完美对接并全部完成！")