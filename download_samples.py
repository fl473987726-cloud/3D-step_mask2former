"""
从 Zero-to-CAD-100k 数据集下载前3个样本，保存到本地供检查。
使用 pandas 直接从 HF 镜像读取 Parquet 文件，避免 symlink 问题。
"""
import os
import sys
import json
import base64
import io

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ZeroToCAD_samples")
NUM_SAMPLES = 3

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("正在从 HF 镜像下载 Zero-to-CAD-100k (前3条)...")
    print("=" * 60)

    # 直接用 pandas 从 Parquet URL 读取
    import pandas as pd
    import requests

    # 数据集 parquet 文件名（从 HuggingFace tree 获取，每个约115MB，含~8k行）
    parquet_name = "94_ffef3c87e14b4916a1b595bea8229662_000000_000000-0.parquet"
    parquet_url = f"https://hf-mirror.com/datasets/ADSKAILab/Zero-To-CAD-100k/resolve/main/data/train/{parquet_name}"
    print(f"下载 Parquet 文件: {parquet_url}")
    print("文件约 115MB，请稍候...")

    # 先下载到本地再读取（避免大文件内存问题）
    local_parquet = os.path.join(OUTPUT_DIR, "_temp.parquet")
    with requests.get(parquet_url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(local_parquet, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                pct = downloaded / total * 100 if total else 0
                print(f"\r  下载进度: {downloaded/1024/1024:.1f}MB / {total/1024/1024:.1f}MB ({pct:.0f}%)", end="", flush=True)
    print()

    df = pd.read_parquet(local_parquet)
    os.remove(local_parquet)
    print(f"下载完成! 共 {len(df)} 行, 取前 {NUM_SAMPLES} 条")

    columns = list(df.columns)
    print(f"列名: {columns}")

    for idx in range(min(NUM_SAMPLES, len(df))):
        sample = df.iloc[idx]
        sample_dir = os.path.join(OUTPUT_DIR, f"sample_{idx}")
        os.makedirs(sample_dir, exist_ok=True)

        print(f"\n--- 样本 {idx} ---")

        # 1. 元数据
        uuid = str(sample.get("uuid", "unknown"))
        num_faces = int(sample.get("num_faces", 0))
        ops_count = int(sample.get("cadquery_ops_count", 0))
        num_renders = int(sample.get("num_renders", 0))
        face_latency = float(sample.get("face_latency_ms", 0))
        ops_latency = float(sample.get("ops_latency_ms", 0))
        print(f"  UUID: {uuid}")
        print(f"  面数: {num_faces}, 操作数: {ops_count}, 渲染图: {num_renders}")

        # 2. CadQuery 代码
        cq_code = sample.get("cadquery_file")
        if cq_code is not None:
            if isinstance(cq_code, bytes):
                cq_code = cq_code.decode("utf-8", errors="replace")
            code_path = os.path.join(sample_dir, "code.py")
            with open(code_path, "w", encoding="utf-8") as f:
                f.write(cq_code)
            print(f"  CadQuery代码: {len(cq_code)} 字符")

        # 3. 操作序列 JSON
        ops_json_str = sample.get("cadquery_ops_json")
        if ops_json_str is not None:
            if isinstance(ops_json_str, str):
                try:
                    ops_data = json.loads(ops_json_str)
                except json.JSONDecodeError:
                    ops_data = ops_json_str
            else:
                ops_data = ops_json_str
            ops_path = os.path.join(sample_dir, "ops.json")
            with open(ops_path, "w", encoding="utf-8") as f:
                json.dump(ops_data, f, ensure_ascii=False, indent=2)
            if isinstance(ops_data, list):
                op_names = [op.get("function", op.get("op_name", "?")) for op in ops_data]
                print(f"  操作序列: {len(ops_data)} 个操作 -> {op_names}")

        # 4. STEP 文件
        step_data = sample.get("step_file")
        if step_data is not None:
            step_bytes = _to_bytes(step_data)
            if step_bytes:
                step_path = os.path.join(sample_dir, "model.step")
                with open(step_path, "wb") as f:
                    f.write(step_bytes)
                print(f"  STEP文件: {len(step_bytes)} 字节")

        # 5. STL 文件
        stl_data = sample.get("stl_file")
        if stl_data is not None:
            stl_bytes = _to_bytes(stl_data)
            if stl_bytes:
                stl_path = os.path.join(sample_dir, "model.stl")
                with open(stl_path, "wb") as f:
                    f.write(stl_bytes)
                print(f"  STL文件: {len(stl_bytes)} 字节")

        # 6. 8张2D渲染图
        for view_idx in range(8):
            img_data = sample.get(f"image_{view_idx}")
            if img_data is not None:
                img_bytes = _to_bytes(img_data)
                if img_bytes:
                    img_path = os.path.join(sample_dir, f"render_{view_idx}.png")
                    with open(img_path, "wb") as f:
                        f.write(img_bytes)

        print(f"  渲染图: 8张 render_0.png ~ render_7.png")

        # 7. 写元数据摘要
        info_path = os.path.join(sample_dir, "info.txt")
        with open(info_path, "w", encoding="utf-8") as f:
            f.write(f"UUID: {uuid}\n")
            f.write(f"面数: {num_faces}\n")
            f.write(f"操作数: {ops_count}\n")
            f.write(f"渲染图数: {num_renders}\n")
            f.write(f"面延迟: {face_latency:.2f} ms\n")
            f.write(f"操作延迟: {ops_latency:.2f} ms\n")
            if isinstance(ops_data, list):
                f.write(f"操作列表: {op_names}\n")

    print("\n" + "=" * 60)
    print(f"下载完成! 共 {NUM_SAMPLES} 个样本保存在: {OUTPUT_DIR}")
    print("=" * 60)

    # 打印目录结构
    for root, dirs, files in os.walk(OUTPUT_DIR):
        level = root.replace(OUTPUT_DIR, "").count(os.sep)
        indent = "  " * level
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = "  " * (level + 1)
        for file in sorted(files):
            fpath = os.path.join(root, file)
            size = os.path.getsize(fpath)
            if size > 1024 * 1024:
                size_str = f"{size / 1024 / 1024:.1f} MB"
            elif size > 1024:
                size_str = f"{size / 1024:.1f} KB"
            else:
                size_str = f"{size} B"
            print(f"{sub_indent}{file} ({size_str})")


def _to_bytes(data):
    """将 base64 字符串或 bytes 统一转换为 bytes"""
    if isinstance(data, bytes):
        return data
    if isinstance(data, str):
        try:
            return base64.b64decode(data)
        except Exception:
            return data.encode("utf-8") if data else None
    return None


if __name__ == "__main__":
    main()
