# -*- coding: utf-8 -*-
"""
自适应面ID颜色映射测试（含面积编码）

R 通道编码：类型 + 面积
  R = 类型基值 + 归一化面积偏移
  每种类型间隙=51，面积51级精度

GB 通道编码：面ID（自适应网格，可选打乱）

用法：
    python test_color_mapping.py --faces 50
    python test_color_mapping.py --faces 50 --shuffle
    python test_color_mapping.py --faces 50 --area 0.5
    python test_color_mapping.py --faces 50 --area 0.0 --area 1.0
    python test_color_mapping.py --faces 50 --show_all_types
"""
import argparse
import math
import os
import numpy as np
import matplotlib.pyplot as plt

from color_encoder import FaceColorEncoder, TYPE_GAP, TYPE_R_BASE, TYPE_NAMES

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

AREA_LEVELS = TYPE_GAP  # 51级面积精度（0~50）


def compute_color_distance(encoder, type_id=0, area_ratio=0.0):
    """计算相邻ID之间的最小颜色距离（RGB全通道）"""
    r = encoder.encode(1, type_id, area_ratio)[0]
    rgb_mapping = {
        fid: (r, g, b) for fid, (g, b) in encoder.gb_mapping.items()
    }
    ids = sorted(rgb_mapping.keys())
    min_dist = float('inf')
    min_pair = (0, 0)

    for i in range(len(ids) - 1):
        id_a, id_b = ids[i], ids[i + 1]
        r_a, g_a, b_a = rgb_mapping[id_a]
        r_b, g_b, b_b = rgb_mapping[id_b]
        dist = math.sqrt((r_a - r_b) ** 2 + (g_a - g_b) ** 2 + (b_a - b_b) ** 2)
        if dist < min_dist:
            min_dist = dist
            min_pair = (id_a, id_b)

    return min_dist, min_pair


def visualize_gb_grid(encoder, type_id, area_ratio, output_path, shuffle=False, seed=42):
    """可视化 GB 平面网格 + 颜色预览"""
    K = math.ceil(math.sqrt(encoder.num_faces))
    step = 254 / K
    r = encoder.encode(1, type_id, area_ratio)[0]
    type_name = TYPE_NAMES[type_id]
    min_dist, min_pair = compute_color_distance(encoder, type_id, area_ratio)

    title_shuffle = f" [shuffle seed={seed}]" if shuffle else ""
    area_pct = f"{area_ratio*100:.0f}%"
    num_faces = encoder.num_faces

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # === 左图：GB 平面分布 ===
    ax1 = axes[0]
    for face_id, (g, b) in encoder.gb_mapping.items():
        ax1.scatter(b, g, c=[(r/255, g/255, b/255)], s=80, edgecolors="black", linewidths=0.5)
        if num_faces <= 50:
            ax1.annotate(str(face_id), (b, g), fontsize=7,
                         ha="center", va="bottom", xytext=(0, 5),
                         textcoords="offset points")

    ax1.set_xlim(-10, 265)
    ax1.set_ylim(-10, 265)
    ax1.set_xlabel("B 通道", fontsize=12)
    ax1.set_ylabel("G 通道", fontsize=12)
    ax1.set_title(f"GB 平面分布 (K={K}, step={step:.1f})", fontsize=13)
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)

    # === 右图：颜色条预览 ===
    ax2 = axes[1]
    cols = min(20, num_faces)
    rows = math.ceil(num_faces / cols)
    preview = np.zeros((rows * 40, cols * 40, 3), dtype=np.uint8)

    for face_id, (g, b) in encoder.gb_mapping.items():
        idx = face_id - 1
        r_row = idx // cols
        r_col = idx % cols
        y0, x0 = r_row * 40, r_col * 40
        preview[y0:y0 + 40, x0:x0 + 40] = [r, g, b]

    ax2.imshow(preview.astype(np.float32) / 255.0)
    ax2.set_title(f"颜色预览 ({type_name}, 面积={area_pct}, R={r})", fontsize=13)

    for face_id in range(1, num_faces + 1):
        idx = face_id - 1
        r_row = idx // cols
        r_col = idx % cols
        x = r_col * 40 + 20
        y = r_row * 40 + 20
        g_val, b_val = encoder.gb_mapping[face_id]
        luminance = 0.299 * r + 0.587 * g_val + 0.114 * b_val
        text_color = "white" if luminance < 128 else "black"
        ax2.text(x, y, str(face_id), ha="center", va="center",
                 fontsize=6, color=text_color, fontweight="bold")

    ax2.axis("off")

    fig.suptitle(
        f"自适应颜色映射{title_shuffle} — {type_name} 共 {num_faces} 个面, 面积={area_pct}\n"
        f"R={r} (基值{TYPE_R_BASE[type_id]}+偏移{r-TYPE_R_BASE[type_id]}), "
        f"相邻ID最小RGB距离: {min_dist:.1f} (ID {min_pair[0]}<->{min_pair[1]})",
        fontsize=12, y=0.98
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"可视化已保存: {output_path}")


def visualize_all_types(encoder, output_path, shuffle=False, seed=42):
    """可视化所有类型在不同面积下的颜色变化"""
    num_faces = encoder.num_faces
    K = math.ceil(math.sqrt(num_faces))
    fig, axes = plt.subplots(5, 6, figsize=(22, 18))
    area_ratios = [0.0, 0.25, 0.5, 0.75, 1.0]

    for type_id in range(5):
        type_name = TYPE_NAMES[type_id]
        base = TYPE_R_BASE[type_id]

        for col_idx, area_ratio in enumerate(area_ratios):
            ax = axes[type_id, col_idx]
            r = encoder.encode(1, type_id, area_ratio)[0]

            preview_size = min(16, num_faces)
            cols = 4
            rows_p = math.ceil(preview_size / 4)
            preview = np.zeros((rows_p * 30, cols * 30, 3), dtype=np.uint8)

            for i in range(preview_size):
                face_id = i + 1
                g, b = encoder.gb_mapping[face_id]
                pr = i // cols
                pc = i % cols
                preview[pr*30:(pr+1)*30, pc*30:(pc+1)*30] = [r, g, b]

            ax.imshow(preview.astype(np.float32) / 255.0)
            area_pct = f"{area_ratio*100:.0f}%"
            ax.set_title(f"R={r}\n面积={area_pct}", fontsize=8)
            ax.axis("off")

        ax_bar = axes[type_id, 5]
        bar = np.zeros((30, 250, 3), dtype=np.uint8)
        r_max = base + TYPE_GAP - 1
        for px in range(250):
            r_val = min(base + int(px * (TYPE_GAP - 1) / 250), r_max)
            bar[:, px] = [r_val, 128, 128]
        ax_bar.imshow(bar.astype(np.float32) / 255.0)
        ax_bar.set_title(f"{type_name}\nR=[{base},{r_max}]", fontsize=9)
        ax_bar.set_xlabel("R值", fontsize=8)
        ax_bar.set_xticks([0, 125, 250])
        ax_bar.set_xticklabels([str(base), str(base + TYPE_GAP // 2), str(r_max)], fontsize=7)

    fig.suptitle(
        f"所有类型 x 面积编码可视化 ({num_faces}个面, K={K})\n"
        f"R = 类型基值 + 面积偏移, GB = 面ID自适应网格",
        fontsize=14, y=0.99
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"全类型可视化已保存: {output_path}")


def print_mapping_table(encoder, type_id, area_ratio, shuffle=False):
    """打印映射表"""
    type_name = TYPE_NAMES[type_id]
    K = math.ceil(math.sqrt(encoder.num_faces))
    min_dist, min_pair = compute_color_distance(encoder, type_id, area_ratio)

    shuffle_tag = " (已打乱)" if shuffle else ""
    area_pct = f"{area_ratio*100:.0f}%"

    print(f"\n{'='*70}")
    print(f"面ID颜色映射表 — {type_name}{shuffle_tag}, 面积={area_pct}")
    print(f"面数: {len(encoder.gb_mapping)}, K={K}")
    print(f"相邻ID最小RGB距离: {min_dist:.1f} (ID {min_pair[0]}<->{min_pair[1]})")
    print(f"{'='*70}")
    print(f"{'面ID':<8}{'G':<8}{'B':<8}{'R':<8}{'RGB':<24}{'十六进制'}")
    print(f"{'-'*70}")

    for face_id, (g, b) in encoder.gb_mapping.items():
        r = encoder.encode(face_id, type_id, area_ratio)[0]
        hex_color = f"#{r:02X}{g:02X}{b:02X}"
        print(f"{face_id:<8}{g:<8}{b:<8}{r:<8}({r:>3},{g:>3},{b:>3})       {hex_color}")

    print(f"\n最大支持面数: {K * K}")
    print(f"面积精度: {AREA_LEVELS}级 (每级={100/TYPE_GAP:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="自适应面ID颜色映射测试（含面积编码）")
    parser.add_argument("--faces", type=int, required=True, help="面数量 N")
    parser.add_argument("--type_id", type=int, default=0,
                        choices=[0, 1, 2, 3, 4],
                        help="面类型 (0=平面, 1=圆柱, 2=圆锥, 3=球面, 4=其他)")
    parser.add_argument("--area", type=float, nargs="+", default=[0.0],
                        help="归一化面积 (0.0~1.0)，可传多个值同时对比")
    parser.add_argument("--shuffle", action="store_true",
                        help="打乱ID分配顺序")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--show_all_types", action="store_true",
                        help="可视化所有类型 x 面积的编码矩阵")
    parser.add_argument("--output_dir", type=str,
                        default=r"E:\soft\code\Mask2former\results\color_mapping",
                        help="输出目录")
    args = parser.parse_args()

    num_faces = args.faces
    type_id = args.type_id

    os.makedirs(args.output_dir, exist_ok=True)

    encoder = FaceColorEncoder(num_faces, shuffle=args.shuffle, seed=args.seed)

    for area_ratio in args.area:
        area_ratio = max(0.0, min(1.0, area_ratio))
        print_mapping_table(encoder, type_id, area_ratio, shuffle=args.shuffle)

        suffix = f"_T{type_id}_A{int(area_ratio*100)}"
        if args.shuffle:
            suffix += f"_S{args.seed}"
        output_path = os.path.join(args.output_dir, f"color_map_N{num_faces}{suffix}.png")
        visualize_gb_grid(encoder, type_id, area_ratio,
                          output_path, shuffle=args.shuffle, seed=args.seed)

    if args.show_all_types:
        all_types_path = os.path.join(args.output_dir, f"all_types_N{num_faces}.png")
        visualize_all_types(encoder, all_types_path,
                            shuffle=args.shuffle, seed=args.seed)


if __name__ == "__main__":
    main()
