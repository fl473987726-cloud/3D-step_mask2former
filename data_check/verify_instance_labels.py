# -*- coding: utf-8 -*-
"""
验证实例标签正确性
输入：实例掩码图 + class_map.json
输出：实例可视化图 + 语义可视化图
"""
import os
import json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import font_manager

# 设置中文字体
import matplotlib.font_manager as fm
# 查找系统中文字体
chinese_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong', 'STSong']
available_font = None
for font_name in chinese_fonts:
    font_path = fm.findfont(fm.FontProperties(family=font_name))
    if font_path and 'LastResort' not in font_path:
        available_font = font_name
        break

if available_font:
    plt.rcParams['font.sans-serif'] = [available_font]
else:
    # 尝试使用系统默认字体
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 语义颜色定义
SEMANTIC_COLORS = {
    1: [255, 165, 0],    # 宽体槽 - 橙色
    2: [128, 0, 128],    # 封闭槽 - 紫色
    3: [0, 255, 255],    # 开放槽 - 青色
    4: [255, 0, 0],      # 孔 - 红色
    0: [255, 255, 255] # 背景 - 白色
}

SEMANTIC_NAMES = {
    1: "宽体槽",
    2: "封闭槽",
    3: "开放槽",
    4: "孔",
    0: "背景"
}


def generate_instance_colors(num_instances):
    """为每个实例生成不同颜色"""
    colors = []
    golden_ratio = (1 + np.sqrt(5)) / 2
    for i in range(num_instances):
        hue = (i * golden_ratio) % 1.0
        h = hue * 6
        c = 0.85
        x = c * (1 - abs(h % 2 - 1))
        m = 0.1
        if h < 1:
            r, g, b = c, x, 0
        elif h < 2:
            r, g, b = x, c, 0
        elif h < 3:
            r, g, b = 0, c, x
        elif h < 4:
            r, g, b = 0, x, c
        elif h < 5:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        colors.append([int((r + m) * 255), int((g + m) * 255), int((b + m) * 255)])
    return colors


def visualize_instance_labels(instance_mask_path, class_map_path, output_dir):
    """可视化实例标签"""
    os.makedirs(output_dir, exist_ok=True)

    # 读取实例掩码
    instance_mask = np.array(Image.open(instance_mask_path).convert("L"))
    img_name = os.path.basename(instance_mask_path)

    # 尝试读取class_map
    has_class_map = False
    img_class_map = {}
    if class_map_path and os.path.exists(class_map_path) and os.path.getsize(class_map_path) > 0:
        try:
            with open(class_map_path, 'r', encoding='utf-8') as f:
                class_map = json.load(f)
            if img_name in class_map:
                img_class_map = class_map[img_name]
                has_class_map = True
        except:
            pass

    # 获取所有实例ID（0-254都是特征，255是背景）
    instance_ids = sorted(np.unique(instance_mask))
    instance_ids = [int(x) for x in instance_ids if x < 255]
    print(f"Image: {img_name}")
    print(f"Instance IDs: {instance_ids}")

    # 生成实例可视化
    instance_colors = generate_instance_colors(len(instance_ids))
    instance_id_to_color = {inst_id: instance_colors[i] for i, inst_id in enumerate(instance_ids)}

    instance_vis = np.full((*instance_mask.shape, 3), 255, dtype=np.uint8)
    for inst_id in instance_ids:
        instance_vis[instance_mask == inst_id] = instance_id_to_color[inst_id]

    # 根据是否有class_map决定布局
    if has_class_map:
        print(f"Class mapping: {img_class_map}")

        # 生成语义可视化
        semantic_vis = np.full((*instance_mask.shape, 3), 255, dtype=np.uint8)
        for inst_id, class_id in img_class_map.items():
            inst_id = int(inst_id)
            class_id = int(class_id)
            if class_id in SEMANTIC_COLORS:
                semantic_vis[instance_mask == inst_id] = SEMANTIC_COLORS[class_id]

        # 5图布局
        fig = plt.figure(figsize=(22, 8))
        gs = fig.add_gridspec(1, 5, width_ratios=[3, 1.5, 3, 1.5, 3])

        # 1. 实例可视化
        ax1 = fig.add_subplot(gs[0])
        ax1.imshow(instance_vis)
        ax1.set_title("Instance Visualization", fontsize=12, fontweight='bold')
        ax1.axis("off")

        # 2. 实例图例
        ax2 = fig.add_subplot(gs[1])
        ax2.axis("off")
        legend_instance = []
        for inst_id in instance_ids:
            color_norm = [c / 255 for c in instance_id_to_color[inst_id]]
            class_id = int(img_class_map[str(inst_id)])
            class_name = SEMANTIC_NAMES.get(class_id, "未知")
            legend_instance.append(mpatches.Patch(color=color_norm, label=f"{inst_id}: {class_name}"))
        ax2.legend(handles=legend_instance, loc='center left', fontsize=9,
                   title="Instance ID -> Class", title_fontsize=10)

        # 3. 语义可视化
        ax3 = fig.add_subplot(gs[2])
        ax3.imshow(semantic_vis)
        ax3.set_title("Semantic Visualization", fontsize=12, fontweight='bold')
        ax3.axis("off")

        # 4. 语义图例
        ax4 = fig.add_subplot(gs[3])
        ax4.axis("off")
        legend_semantic = []
        for class_id in sorted(SEMANTIC_NAMES.keys()):
            if class_id < 255:
                color_norm = [c / 255 for c in SEMANTIC_COLORS[class_id]]
                legend_semantic.append(mpatches.Patch(color=color_norm, label=SEMANTIC_NAMES[class_id]))
        ax4.legend(handles=legend_semantic, loc='center left', fontsize=9,
                   title="Semantic Class", title_fontsize=10)

        # 5. 原始掩码图
        ax5 = fig.add_subplot(gs[4])
        ax5.set_title("Original Mask", fontsize=11, pad=8)
        ax5.axis("off")
        ax5.imshow(instance_mask, cmap="gray", vmin=0, vmax=255, interpolation="nearest")

    else:
        # 只有实例可视化
        print("No class_map found, only generating instance visualization")
        fig = plt.figure(figsize=(10, 5))
        gs = fig.add_gridspec(1, 2, width_ratios=[3, 1.5])

        # 1. 实例可视化
        ax1 = fig.add_subplot(gs[0])
        ax1.imshow(instance_vis)
        ax1.set_title("Instance Visualization", fontsize=12, fontweight='bold')
        ax1.axis("off")

        # 2. 图例
        ax2 = fig.add_subplot(gs[1])
        ax2.axis("off")
        legend_instance = []
        for i, inst_id in enumerate(instance_ids):
            color_norm = [c / 255 for c in instance_id_to_color[inst_id]]
            legend_instance.append(mpatches.Patch(color=color_norm, label=f"Instance {inst_id}"))
        ax2.legend(handles=legend_instance, loc='center left', fontsize=9,
                   title="Instance IDs", title_fontsize=10)

    plt.suptitle(f"Instance Label Verification: {img_name}", fontsize=14)
    plt.tight_layout()

    output_path = os.path.join(output_dir, f"verify_{img_name}.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    # 配置路径
    instance_mask_dir = r"E:\soft\code\Mask2former\temp\masks"
    class_map_path = r"E:\soft\code\Mask2former\temp\masks\class_map.json"
    # class_map_path = None
    output_dir = r"E:\soft\code\Mask2former\results\verify_instance_labels"

    # 获取所有实例掩码
    if os.path.exists(instance_mask_dir):
        mask_files = sorted([f for f in os.listdir(instance_mask_dir) if f.endswith('.png')])
        print(f"Found {len(mask_files)} instance masks")

        for mask_file in mask_files:
            instance_mask_path = os.path.join(instance_mask_dir, mask_file)
            visualize_instance_labels(instance_mask_path, class_map_path, output_dir)
    else:
        print(f"Directory not found: {instance_mask_dir}")
