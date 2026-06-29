import os
import shutil
import random
import json

# ================= 配置路径 =================
base_path = r"F:\B-研生学习\A-科研ing\D-无锡\A-特征识别\test_instance_0624"

# 1. 源文件夹与源 JSON 配置
src_folders = {
    "encoded": os.path.join(base_path, "encoded_views_augmented"),
    "masks": os.path.join(base_path, "masks_augmented"),
    "semantic": os.path.join(base_path, "semantic_views_augmented"),
    "unique": os.path.join(base_path, "unique_views_augmented")
}

src_jsons = {
    "camera": os.path.join(base_path, "camera_views_augmented.json"),
    "class": os.path.join(base_path, "class_map_augmented.json"),
    "manifest": os.path.join(base_path, "export_manifest_augmented.json"),
    "face_encoding": os.path.join(base_path, "face_encoding_map_augmented.json"),  # 新增
    "image_index": os.path.join(base_path, "image_index_map_augmented.json")  # 新增
}

# 2. 目标文件夹与目标 JSON 配置
dst_dirs = {
    "encoded_train": os.path.join(base_path, "encoded_train"),
    "encoded_val": os.path.join(base_path, "encoded_val"),
    "masks_train": os.path.join(base_path, "masks_train"),
    "masks_val": os.path.join(base_path, "masks_val"),
    "semantic_train": os.path.join(base_path, "semantic_views_train"),
    "semantic_val": os.path.join(base_path, "semantic_views_val"),
    "unique_train": os.path.join(base_path, "unique_views_train"),
    "unique_val": os.path.join(base_path, "unique_views_val")
}

dst_jsons = {
    "camera_train": os.path.join(base_path, "camera_views_train.json"),
    "camera_val": os.path.join(base_path, "camera_views_val.json"),
    "class_train": os.path.join(base_path, "class_map_train.json"),
    "class_val": os.path.join(base_path, "class_map_val.json"),
    "manifest_train": os.path.join(base_path, "export_manifest_train.json"),
    "manifest_val": os.path.join(base_path, "export_manifest_val.json"),
    "face_encoding_train": os.path.join(base_path, "face_encoding_map_train.json"),  # 新增
    "face_encoding_val": os.path.join(base_path, "face_encoding_map_val.json"),  # 新增
    "image_index_train": os.path.join(base_path, "image_index_map_train.json"),  # 新增
    "image_index_val": os.path.join(base_path, "image_index_map_val.json")  # 新增
}

# 支持的图像格式后缀
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')


def split_dataset_with_json():
    # 1. 检查源文件和目录是否存在
    for key, path in src_folders.items():
        if not os.path.exists(path):
            print(f"错误：找不到源图像文件夹：{path}")
            return

    for key, path in src_jsons.items():
        if not os.path.exists(path):
            print(f"错误：找不到源 JSON 文件：{path}")
            return

    # 2. 创建所有目标文件夹
    for path in dst_dirs.values():
        os.makedirs(path, exist_ok=True)

    # 3. 读取所有源 JSON 标签数据
    full_jsons_data = {}
    for key, path in src_jsons.items():
        with open(path, 'r', encoding='utf-8') as f:
            try:
                full_jsons_data[key] = json.load(f)
            except json.JSONDecodeError:
                print(f"错误：JSON 文件格式有误，解析失败：{path}")
                return

    # 初始化训练集和验证集的 JSON 字典
    train_jsons_data = {key: {} for key in src_jsons.keys()}
    val_jsons_data = {key: {} for key in src_jsons.keys()}

    # 4. 获取并排序 masks 文件夹下的所有图片
    all_files = sorted([f for f in os.listdir(src_folders["masks"]) if f.lower().endswith(IMAGE_EXTENSIONS)])
    total_files = len(all_files)

    if total_files == 0:
        print("未在源文件夹中找到有效图片！")
        return

    if total_files % 84 != 0:
        print(f"警告：图片总数 ({total_files}) 不是 84 的整数倍，将按现有模型数尽可能切分。")

    # 5. 按每 84 张图（一个模型）进行遍历
    num_models = total_files // 84
    print(f"检测到共有 {num_models} 个模型的数据（每模型 84 张图）。开始并行抽选图像与标签...")

    # 设置随机种子确保结果可复现
    random.seed(42)

    for m_idx in range(num_models):
        # 取出当前模型的 84 张图片
        model_files = all_files[m_idx * 84: (m_idx + 1) * 84]

        # 将 84 张图按原名分组（7张一组，共12组）
        groups = {}
        for file_name in model_files:
            name, ext = os.path.splitext(file_name)
            base_name = name.rsplit('-', 1)[0]
            if base_name not in groups:
                groups[base_name] = []
            groups[base_name].append(file_name)

        group_keys = list(groups.keys())  # 12个基本样本

        # 6. 随机抽取 2 组作为验证集，剩下 10 组作为训练集
        val_group_keys = random.sample(group_keys, 2)
        train_group_keys = [k for k in group_keys if k not in val_group_keys]

        # 7. 开始复制文件并提取 JSON 标签
        # 分配验证集 (4类图像 + 5类JSON)
        for k in val_group_keys:
            for file_name in groups[k]:
                # 复制四类图像文件
                shutil.copy2(os.path.join(src_folders["encoded"], file_name),
                             os.path.join(dst_dirs["encoded_val"], file_name))
                shutil.copy2(os.path.join(src_folders["masks"], file_name),
                             os.path.join(dst_dirs["masks_val"], file_name))
                shutil.copy2(os.path.join(src_folders["semantic"], file_name),
                             os.path.join(dst_dirs["semantic_val"], file_name))
                shutil.copy2(os.path.join(src_folders["unique"], file_name),
                             os.path.join(dst_dirs["unique_val"], file_name))

                # 提取五类对应的 JSON 标签
                for key in src_jsons.keys():
                    if file_name in full_jsons_data[key]:
                        val_jsons_data[key][file_name] = full_jsons_data[key][file_name]

        # 分配训练集 (4类图像 + 5类JSON)
        for k in train_group_keys:
            for file_name in groups[k]:
                # 复制四类图像文件
                shutil.copy2(os.path.join(src_folders["encoded"], file_name),
                             os.path.join(dst_dirs["encoded_train"], file_name))
                shutil.copy2(os.path.join(src_folders["masks"], file_name),
                             os.path.join(dst_dirs["masks_train"], file_name))
                shutil.copy2(os.path.join(src_folders["semantic"], file_name),
                             os.path.join(dst_dirs["semantic_train"], file_name))
                shutil.copy2(os.path.join(src_folders["unique"], file_name),
                             os.path.join(dst_dirs["unique_train"], file_name))

                # 提取五类对应的 JSON 标签
                for key in src_jsons.keys():
                    if file_name in full_jsons_data[key]:
                        train_jsons_data[key][file_name] = full_jsons_data[key][file_name]

        print(f"-> 模型 {m_idx + 1}/{num_models} 划分完成。")

    # 8. 将切分后的 JSON 数据分别写入对应的新文件
    for key in src_jsons.keys():
        with open(dst_jsons[f"{key}_train"], 'w', encoding='utf-8') as f:
            json.dump(train_jsons_data[key], f, indent=2, ensure_ascii=False)
        with open(dst_jsons[f"{key}_val"], 'w', encoding='utf-8') as f:
            json.dump(val_jsons_data[key], f, indent=2, ensure_ascii=False)

    # 9. 打印最终统计结果
    print("\n====== 数据集与标签切分全部完成！ ======")
    print(f"处理的模型总数: {num_models}")
    print(f"训练集各个 JSON 标签数:")
    for key in src_jsons.keys():
        print(f" -> {key}_train: {len(train_jsons_data[key])}")
    print(f"验证集各个 JSON 标签数:")
    for key in src_jsons.keys():
        print(f" -> {key}_val: {len(val_jsons_data[key])}")


if __name__ == "__main__":
    split_dataset_with_json()