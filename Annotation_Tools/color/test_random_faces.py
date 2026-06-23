# -*- coding: utf-8 -*-
"""
随机生成50个面的类型和面积，生成彩色编码图 + mapping.json

用于验证编码方案的可行性
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import math

from color_encoder import FaceColorEncoder, TYPE_NAMES

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

NUM_FACES = 50
SHUFFLE_SEED = 42
OUTPUT_DIR = r"E:\soft\code\Mask2former\results\color_mapping\test_random"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rng = np.random.default_rng(123)

    encoder = FaceColorEncoder(NUM_FACES, shuffle=True, seed=SHUFFLE_SEED)

    # 随机生成每个面的类型和面积
    faces = {}
    for face_id in range(1, NUM_FACES + 1):
        type_id = int(rng.integers(0, 5))
        area_ratio = round(float(rng.uniform(0.0, 1.0)), 4)
        r, g, b = encoder.encode(face_id, type_id, area_ratio)
        faces[str(face_id)] = {
            "type_id": type_id,
            "type_name": TYPE_NAMES[type_id],
            "area_ratio": area_ratio,
            "R": r, "G": g, "B": b,
            "hex": f"#{r:02X}{g:02X}{b:02X}",
        }

    # 打印映射表
    print(f"{'面ID':<6}{'类型':<8}{'面积':<10}{'R':<6}{'G':<6}{'B':<6}{'十六进制'}")
    print("-" * 60)
    for fid, info in faces.items():
        print(f"{fid:<6}{info['type_name']:<8}{info['area_ratio']:<10.2%}"
              f"{info['R']:<6}{info['G']:<6}{info['B']:<6}{info['hex']}")

    type_counts = {}
    for info in faces.values():
        t = info["type_name"]
        type_counts[t] = type_counts.get(t, 0) + 1
    print(f"\n类型分布: {type_counts}")

    # 生成彩色编码图
    cols = 10
    rows = math.ceil(NUM_FACES / cols)
    block = 50
    img = np.zeros((rows * block, cols * block, 3), dtype=np.uint8)

    fig, ax = plt.subplots(figsize=(14, 8))
    for face_id, info in faces.items():
        idx = int(face_id) - 1
        r_row = idx // cols
        r_col = idx % cols
        y0, x0 = r_row * block, r_col * block
        img[y0:y0+block, x0:x0+block] = [info["R"], info["G"], info["B"]]

    ax.imshow(img.astype(np.float32) / 255.0)
    for face_id, info in faces.items():
        idx = int(face_id) - 1
        r_row = idx // cols
        r_col = idx % cols
        x = r_col * block + block // 2
        y = r_row * block + block // 2
        lum = 0.299 * info["R"] + 0.587 * info["G"] + 0.114 * info["B"]
        tc = "white" if lum < 128 else "black"
        ax.text(x, y - 8, str(face_id), ha="center", va="center",
                fontsize=7, color=tc, fontweight="bold")
        ax.text(x, y + 8, f"{info['area_ratio']:.0%}", ha="center", va="center",
                fontsize=5, color=tc)

    K = math.ceil(math.sqrt(NUM_FACES))
    ax.set_title(f"随机50面编码测试 (seed={SHUFFLE_SEED}, K={K})\n"
                 f"色块上: 面ID / 面积比例", fontsize=12)
    ax.axis("off")
    plt.tight_layout()

    img_path = os.path.join(OUTPUT_DIR, "random_50faces_color.png")
    plt.savefig(img_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n彩色图已保存: {img_path}")

    # 保存 mapping.json
    encoder.save_mapping(
        os.path.join(OUTPUT_DIR, "mapping.json"),
        extra_config={"faces": faces}
    )
    print(f"映射表已保存: {os.path.join(OUTPUT_DIR, 'mapping.json')}")


if __name__ == "__main__":
    main()
