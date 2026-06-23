# -*- coding: utf-8 -*-
"""
RGB颜色分析脚本
分析图片中的RGB像素颜色分布
"""
import os
import sys
import numpy as np
from collections import Counter
from PIL import Image


# 预期的颜色定义
EXPECTED_COLORS = {
    "平面": (255, 165, 0),
    "圆柱面": (128, 0, 128),
    "圆锥面": (0, 255, 255),
    "其他": (255, 192, 203),
    "背景": (255, 255, 255),
    "边": (0, 0, 0),
}

# 容差（允许的RGB偏差）
TOLERANCE = 10


def color_distance(c1, c2):
    """计算两个颜色的欧氏距离"""
    return sum((a - b) ** 2 for a, b in zip(c1, c2)) ** 0.5


def find_closest_color(rgb, tolerance=TOLERANCE):
    """找到最接近的预定义颜色"""
    min_dist = float('inf')
    closest = "未知"
    for name, expected in EXPECTED_COLORS.items():
        dist = color_distance(rgb, expected)
        if dist < min_dist:
            min_dist = dist
            closest = name
    if min_dist <= tolerance:
        return closest
    return f"未知({rgb})"


def analyze_image(image_path):
    """分析图片的颜色分布"""
    print(f"\n分析图片: {image_path}")

    # 读取图片
    img = Image.open(image_path).convert("RGB")
    pixels = np.array(img)

    # 统计每个RGB值出现的次数
    pixels_tuple = pixels.reshape(-1, 3)
    color_counter = Counter(map(tuple, pixels_tuple))

    # 按出现次数排序
    sorted_colors = color_counter.most_common()

    print(f"\n图片尺寸: {img.size[0]} x {img.size[1]}")
    print(f"总像素数: {len(pixels_tuple)}")
    print(f"不同颜色数: {len(sorted_colors)}")

    # 统计匹配预定义颜色的像素
    matched_count = 0
    unmatched_count = 0
    color_groups = {}

    for rgb, count in sorted_colors:
        name = find_closest_color(rgb)
        if name.startswith("未知"):
            unmatched_count += count
        else:
            matched_count += count
            if name not in color_groups:
                color_groups[name] = {"count": 0, "rgb": rgb}
            color_groups[name]["count"] += count

    print(f"\n--- 颜色统计 ---")
    print(f"匹配预定义颜色: {matched_count} 像素 ({matched_count/len(pixels_tuple)*100:.1f}%)")
    print(f"未匹配颜色: {unmatched_count} 像素 ({unmatched_count/len(pixels_tuple)*100:.1f}%)")

    # 显示匹配的颜色分布
    print(f"\n--- 匹配的颜色分布 ---")
    for name, info in sorted(color_groups.items(), key=lambda x: -x[1]["count"]):
        rgb = info["rgb"]
        count = info["count"]
        pct = count / len(pixels_tuple) * 100
        print(f"  {name}: RGB{rgb} - {count} 像素 ({pct:.1f}%)")

    # 显示所有不同的RGB值（前20个）
    print(f"\n--- 所有RGB值 (前20) ---")
    for rgb, count in sorted_colors[:20]:
        name = find_closest_color(rgb)
        pct = count / len(pixels_tuple) * 100
        print(f"  RGB{rgb} ({name}): {count} 像素 ({pct:.1f}%)")

    # 如果有未匹配的颜色，显示它们
    if unmatched_count > 0:
        print(f"\n--- 未匹配的RGB值 ---")
        unmatched_colors = [(rgb, count) for rgb, count in sorted_colors 
                          if find_closest_color(rgb).startswith("未知")]
        for rgb, count in unmatched_colors[:20]:
            print(f"  RGB{rgb}: {count} 像素")

    return sorted_colors


def compare_images(image1_path, image2_path):
    """对比两张图片的颜色差异"""
    print(f"\n对比两张图片:")
    print(f"  图片1: {image1_path}")
    print(f"  图片2: {image2_path}")

    img1 = np.array(Image.open(image1_path).convert("RGB"))
    img2 = np.array(Image.open(image2_path).convert("RGB"))

    if img1.shape != img2.shape:
        print("错误: 图片尺寸不同")
        return

    # 计算像素差异
    diff = np.abs(img1.astype(int) - img2.astype(int))
    diff_sum = diff.sum(axis=2)

    # 统计差异
    identical = (diff_sum == 0).sum()
    total = diff_sum.size
    different = total - identical

    print(f"\n--- 对比结果 ---")
    print(f"相同像素: {identical} ({identical/total*100:.1f}%)")
    print(f"不同像素: {different} ({different/total*100:.1f}%)")
    print(f"平均差异: {diff_sum.mean():.2f}")
    print(f"最大差异: {diff_sum.max()}")


if __name__ == "__main__":
    # 默认分析的图片
    default_image = r"E:\aaaa-WUT\lw\ASCCAD\test_step\test_datasets\test-instance-mask-0608\semantic_views\000049.png"

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = default_image

    if not os.path.exists(image_path):
        print(f"文件不存在: {image_path}")
        sys.exit(1)

    analyze_image(image_path)

    # 如果提供了两张图片，进行对比
    if len(sys.argv) > 2:
        image2_path = sys.argv[2]
        if os.path.exists(image2_path):
            compare_images(image_path, image2_path)
