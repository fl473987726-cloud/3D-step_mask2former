# -*- coding: utf-8 -*-
"""
3D → 2D → 3D 完整流水线：STEP模型 → 多视角渲染 → Mask2Former推理 → 回传3D面标签

流水线：
  STEP模型
      ↓ Step 1: 生成多视角图
      ├─ semantic_views/   → 输入Mask2Former做2D推理
      └─ unique_views/     → 每个面固定RGB颜色，用于回传face_id
      ↓ Step 2: Mask2Former推理
      └─ pred_masks/       → 每个视角的逐像素类别预测
      ↓ Step 3: 回传3D
      ├─ face_label.json   → 每个3D面的预测类别
      └─ colored_step.step → 按类别染色的STEP模型

用法：
  python back_project_to_step.py --step_file model.step --output_dir output/
  python back_project_to_step.py --step_dir model_folder/ --output_dir output/
"""
import argparse
import json
import os
import sys
import math
import shutil
from collections import defaultdict

import numpy as np
import pyvista as pv
from PIL import Image

# ---- OCP imports ----
from OCP.STEPControl import STEPControl_Reader, STEPControl_AsIs
from OCP.STEPCAFControl import STEPCAFControl_Writer
from OCP.XCAFApp import XCAFApp_Application
from OCP.TDocStd import TDocStd_Document
from OCP.XCAFDoc import XCAFDoc_DocumentTool, XCAFDoc_ColorType
from OCP.Quantity import Quantity_Color, Quantity_TOC_RGB
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
from OCP.TCollection import TCollection_ExtendedString
from OCP.TopoDS import TopoDS
from OCP.TopLoc import TopLoc_Location
from OCP.BRep import BRep_Tool
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.GeomAbs import GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Sphere


# ==================== 配置 ====================

CLASS_NAMES = {
    0: "Background",
    1: "宽体槽",
    2: "封闭槽",
    3: "开放槽",
    4: "孔",
    5: "开放型腔",
    6: "封闭型腔",
    7: "复合型腔",
}

CLASS_COLORS_3D = {
    1: (255, 0, 0),
    2: (255, 255, 0),
    3: (0, 0, 255),
    4: (0, 255, 0),
    5: (255, 128, 0),
    6: (128, 0, 255),
    7: (0, 200, 200),
}

FACE_TYPE_R_MAP = {
    "Plane": 20,
    "Cylinder": 70,
    "Cone": 120,
    "Sphere": 170,
    "Other": 220,
}

OUTPUT_WIDTH = 1024
OUTPUT_HEIGHT = 1024


# ==================== 视角工具 (from generate_inference_views.py) ====================

def get_dodecahedron_view_directions():
    phi = (1 + math.sqrt(5)) / 2
    length = math.sqrt(1 + phi ** 2)
    return [
        (0, 1 / length, phi / length),
        (0, -1 / length, phi / length),
        (0, 1 / length, -phi / length),
        (0, -1 / length, -phi / length),
        (1 / length, phi / length, 0),
        (-1 / length, phi / length, 0),
        (1 / length, -phi / length, 0),
        (-1 / length, -phi / length, 0),
        (phi / length, 0, 1 / length),
        (-phi / length, 0, 1 / length),
        (phi / length, 0, -1 / length),
        (-phi / length, 0, -1 / length),
    ]


def get_viewup(direction):
    if abs(direction[2]) > 0.99:
        return (0, 1, 0)
    dot = direction[2]
    vx = -dot * direction[0]
    vy = -dot * direction[1]
    vz = 1 - dot * direction[2]
    length = math.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
    return (vx / length, vy / length, vz / length)


def get_parallel_scale(bounds, center, direction, viewup, aspect_ratio=1.0, margin=1.10):
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    corners = [
        (xmin, ymin, zmin), (xmin, ymin, zmax),
        (xmin, ymax, zmin), (xmin, ymax, zmax),
        (xmax, ymin, zmin), (xmax, ymin, zmax),
        (xmax, ymax, zmin), (xmax, ymax, zmax),
    ]
    right = (
        direction[1] * viewup[2] - direction[2] * viewup[1],
        direction[2] * viewup[0] - direction[0] * viewup[2],
        direction[0] * viewup[1] - direction[1] * viewup[0],
    )
    right_len = math.sqrt(right[0] ** 2 + right[1] ** 2 + right[2] ** 2)
    right = (right[0] / right_len, right[1] / right_len, right[2] / right_len)

    max_u = max_v = 0.0
    for corner in corners:
        rel = (corner[0] - center[0], corner[1] - center[1], corner[2] - center[2])
        u = abs(rel[0] * right[0] + rel[1] * right[1] + rel[2] * right[2])
        v = abs(rel[0] * viewup[0] + rel[1] * viewup[1] + rel[2] * viewup[2])
        max_u = max(max_u, u)
        max_v = max(max_v, v)

    return max(max_v, max_u / max(aspect_ratio, 1e-6)) * margin


def rgb_to_float(rgb):
    return tuple(c / 255.0 for c in rgb)


def hsv_to_rgb(h, s, v):
    h = h % 360
    c = v * s
    x = c * (1 - abs((h / 60) % 2 - 1))
    m = v - c
    if h < 60:    r, g, b = c, x, 0
    elif h < 120: r, g, b = x, c, 0
    elif h < 180: r, g, b = 0, c, x
    elif h < 240: r, g, b = 0, x, c
    elif h < 300: r, g, b = x, 0, c
    else:         r, g, b = c, 0, x
    return (int((r + m) * 255), int((g + m) * 255), int((b + m) * 255))


# ==================== STEP处理 ====================

def get_face_type(face):
    surf = BRepAdaptor_Surface(face)
    stype = surf.GetType()
    if stype == GeomAbs_Plane:    return "Plane"
    if stype == GeomAbs_Cylinder: return "Cylinder"
    if stype == GeomAbs_Cone:     return "Cone"
    if stype == GeomAbs_Sphere:   return "Sphere"
    return "Other"


def face_to_pyvista_mesh(face):
    location = TopLoc_Location()
    triangulation = BRep_Tool.Triangulation_s(face, location)
    if triangulation is None or triangulation.NbTriangles() == 0:
        return None

    transform = location.Transformation()
    points = []
    for i in range(1, triangulation.NbNodes() + 1):
        point = triangulation.Node(i)
        try:
            point = point.Transformed(transform)
        except Exception:
            pass
        points.append([point.X(), point.Y(), point.Z()])

    faces_pv = []
    is_reversed = face.Orientation() == TopAbs_REVERSED
    for i in range(1, triangulation.NbTriangles() + 1):
        tri = triangulation.Triangle(i)
        n1 = tri.Value(1) - 1
        n2 = tri.Value(2) - 1
        n3 = tri.Value(3) - 1
        if is_reversed:
            n2, n3 = n3, n2
        faces_pv.append([3, n1, n2, n3])

    return pv.PolyData(np.array(points), np.array(faces_pv, dtype=np.int64))


def load_step_faces(step_path):
    """读取STEP文件，提取所有面的网格和信息"""
    reader = STEPControl_Reader()
    if reader.ReadFile(step_path) != 1:
        print(f"  Error reading STEP: {step_path}")
        return None

    reader.TransferRoots()
    shape = reader.OneShape()
    BRepMesh_IncrementalMesh(shape, 0.1)

    faces = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    face_id = 1
    while exp.More():
        face = TopoDS.Face_s(exp.Current())
        face_type = get_face_type(face)
        mesh = face_to_pyvista_mesh(face)
        if mesh is not None:
            faces.append({
                "face_id": face_id,
                "face_type": face_type,
                "mesh": mesh,
            })
        face_id += 1
        exp.Next()

    bounds = [1e9, -1e9, 1e9, -1e9, 1e9, -1e9]
    for item in faces:
        fb = item["mesh"].bounds
        bounds[0] = min(bounds[0], fb[0])
        bounds[1] = max(bounds[1], fb[1])
        bounds[2] = min(bounds[2], fb[2])
        bounds[3] = max(bounds[3], fb[3])
        bounds[4] = min(bounds[4], fb[4])
        bounds[5] = max(bounds[5], fb[5])

    return {"faces": faces, "bounds": bounds, "shape": shape}


# ==================== Step 1: 生成多视角图 ====================

def generate_unique_colors(num_faces):
    """为每个面生成视觉上明显不同的颜色（HSV均匀色相+抖动），用于unique图渲染

    返回: (colors, color_to_face_id)
      - colors: list of (R, G, B)，每个面一个颜色
      - color_to_face_id: {(R,G,B): face_id} 映射表，用于从unique图回传face_id
    背景色 = (0, 0, 0)
    """
    colors = []
    color_to_face_id = {}
    golden_angle = 137.508
    for i in range(num_faces):
        face_id = i + 1
        hue = (i * golden_angle) % 360
        # 交替使用不同饱和度和亮度，增加颜色区分度
        saturation = 0.85 if i % 2 == 0 else 1.0
        value = 1.0 if i % 3 != 2 else 0.85
        r, g, b = hsv_to_rgb(hue, saturation, value)
        colors.append((r, g, b))
        color_to_face_id[(r, g, b)] = face_id
    return colors, color_to_face_id


def generate_views(step_data, output_dir, directions):
    """生成语义染色图和固定颜色unique图（无hue offset）"""
    semantic_dir = os.path.join(output_dir, "semantic_views")
    unique_dir = os.path.join(output_dir, "unique_views")
    os.makedirs(semantic_dir, exist_ok=True)
    os.makedirs(unique_dir, exist_ok=True)

    faces = step_data["faces"]
    bounds = step_data["bounds"]
    num_faces = len(faces)
    num_views = len(directions)

    # 语义颜色
    SEMANTIC_FACE_COLOR = {
        "Plane": (255, 165, 0),
        "Cylinder": (128, 0, 128),
        "Cone": (0, 255, 255),
        "Sphere": (255, 255, 0),
        "Other": (255, 192, 203),
    }
    EDGE_COLOR = (0, 0, 0)

    # 固定 unique 颜色（每面不同色，视觉可区分）
    unique_colors, color_to_face_id = generate_unique_colors(num_faces)

    aspect_ratio = OUTPUT_WIDTH / OUTPUT_HEIGHT
    cx = (bounds[0] + bounds[1]) / 2
    cy = (bounds[2] + bounds[3]) / 2
    cz = (bounds[4] + bounds[5]) / 2
    max_dim = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
    if max_dim < 0.001:
        max_dim = 1.0
    center = (cx, cy, cz)
    dist = max_dim * 3

    # --- 渲染语义图（所有视角共享同一套颜色） ---
    print(f"  渲染 {num_views} 个视角的语义图...")
    plotter_s = pv.Plotter(off_screen=True, window_size=[OUTPUT_WIDTH, OUTPUT_HEIGHT])
    plotter_s.disable_anti_aliasing()
    plotter_s.set_background("white")
    plotter_s.camera.SetParallelProjection(True)

    for item in faces:
        color = SEMANTIC_FACE_COLOR.get(item["face_type"], SEMANTIC_FACE_COLOR["Other"])
        plotter_s.add_mesh(item["mesh"], color=rgb_to_float(color),
                           lighting=False, smooth_shading=False)

    for idx, direction in enumerate(directions):
        viewup = get_viewup(direction)
        cam_pos = (center[0] + direction[0] * dist,
                   center[1] + direction[1] * dist,
                   center[2] + direction[2] * dist)
        plotter_s.camera_position = [cam_pos, center, viewup]
        ps = max(get_parallel_scale(bounds, center, direction, viewup,
                                    aspect_ratio=aspect_ratio, margin=1.10), 0.01)
        plotter_s.camera.SetParallelScale(ps)
        plotter_s.render()
        img_name = f"{idx + 1:06d}.png"
        plotter_s.screenshot(os.path.join(semantic_dir, img_name))

    plotter_s.close()

    # --- 渲染 unique 图（每个视角使用相同的固定颜色） ---
    print(f"  渲染 {num_views} 个视角的unique图（固定颜色）...")
    for view_idx, direction in enumerate(directions):
        plotter_u = pv.Plotter(off_screen=True, window_size=[OUTPUT_WIDTH, OUTPUT_HEIGHT])
        plotter_u.disable_anti_aliasing()
        plotter_u.set_background("black")
        plotter_u.camera.SetParallelProjection(True)

        for face_idx, item in enumerate(faces):
            plotter_u.add_mesh(item["mesh"], color=rgb_to_float(unique_colors[face_idx]),
                               lighting=False, smooth_shading=False)

        viewup = get_viewup(direction)
        cam_pos = (center[0] + direction[0] * dist,
                   center[1] + direction[1] * dist,
                   center[2] + direction[2] * dist)
        plotter_u.camera_position = [cam_pos, center, viewup]
        ps = max(get_parallel_scale(bounds, center, direction, viewup,
                                    aspect_ratio=aspect_ratio, margin=1.10), 0.01)
        plotter_u.camera.SetParallelScale(ps)
        plotter_u.render()

        img_name = f"{view_idx + 1:06d}.png"
        plotter_u.screenshot(os.path.join(unique_dir, img_name))
        plotter_u.close()

    print(f"  语义图: {semantic_dir}")
    print(f"  Unique图: {unique_dir}")

    # 保存 color → face_id 映射表
    mapping_path = os.path.join(output_dir, "color_face_id_map.json")
    serializable_map = {f"{r},{g},{b}": fid for (r, g, b), fid in color_to_face_id.items()}
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(serializable_map, f, indent=2)
    print(f"  颜色映射: {mapping_path}")

    return semantic_dir, unique_dir


# ==================== Step 2: Mask2Former推理 ====================

def run_inference_batch(image_dir, model_dir, output_dir, device="cuda"):
    """对image_dir中所有PNG运行Mask2Former推理，结果保存到output_dir"""
    import subprocess
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"  启动子进程加载模型并推理: {model_dir}")
    
    # 将推理逻辑写入临时脚本，使用 mask2former 环境执行
    script_content = f"""
import os
import json
import torch
import numpy as np
from PIL import Image
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

def main():
    model_dir = r'{model_dir}'
    image_dir = r'{image_dir}'
    output_dir = r'{output_dir}'
    device_str = '{device}'

    processor = Mask2FormerImageProcessor.from_pretrained(model_dir)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_dir)
    model.eval()

    dev = torch.device(device_str if torch.cuda.is_available() else "cpu")
    model = model.to(dev)

    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])

    for img_name in image_files:
        img_path = os.path.join(image_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        inputs = processor(images=image, return_tensors="pt", padding=True)
        inputs = {{k: v.to(dev) for k, v in inputs.items()}}

        with torch.no_grad():
            outputs = model(**inputs)

        result = processor.post_process_instance_segmentation(
            outputs,
            target_sizes=[image.size[::-1]],
            threshold=0.5,
            mask_threshold=0.5,
        )[0]

        segmentation = result["segmentation"]
        if isinstance(segmentation, torch.Tensor):
            segmentation = segmentation.cpu().numpy()
        segments_info = result["segments_info"]

        # 保存为numpy格式（更高效）
        np.save(os.path.join(output_dir, f"{{img_name}}_seg.npy"),
                np.array(segmentation, dtype=np.uint16))
        
        info_data = [
            {{"id": int(s["id"]), "label_id": int(s["label_id"]), "score": float(s.get("score", 0.0))}}
            for s in segments_info
        ]
        with open(os.path.join(output_dir, f"{{img_name}}_info.json"), "w") as f:
            json.dump(info_data, f)

if __name__ == "__main__":
    main()
"""
    temp_script = os.path.join(output_dir, "_temp_inference.py")
    with open(temp_script, "w", encoding="utf-8") as f:
        f.write(script_content)

    python_exe = r"D:\ProgramData\Miniconda3\envs\vlm_afr\python.exe"
    if not os.path.exists(python_exe):
        python_exe = sys.executable  # 降级使用当前python
        
    try:
        result = subprocess.run(
            [python_exe, temp_script],
            capture_output=True, text=True, check=True
        )
        if result.stdout.strip():
            print(result.stdout.strip())
        print(f"  推理完成: {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"  推理过程出错 (exit code {e.returncode}):")
        if e.stdout:
            print(f"  stdout: {e.stdout[:2000]}")
        if e.stderr:
            print(f"  stderr: {e.stderr[:2000]}")
    finally:
        if os.path.exists(temp_script):
            os.remove(temp_script)


# ==================== Step 3: 回传3D ====================

def extract_face_id_from_unique(unique_img_path, color_to_face_id):
    """从unique图解码每个像素的face_id（查表法，O(1) 每像素）

    使用 color_to_face_id 映射表查找，背景像素(0,0,0) → face_id = 0
    加速：从逐通道逐mask扫描 → 建256³查找表后一次性查表
    """
    img = np.array(Image.open(unique_img_path).convert("RGB"))
    h, w = img.shape[:2]

    # 建256³查表：RGB三维一次查 face_id
    lut = np.zeros((256, 256, 256), dtype=np.int32)
    for (r, g, b), fid in color_to_face_id.items():
        lut[r, g, b] = fid

    flat = img.reshape(-1, 3)
    face_id_map = lut[flat[:, 0], flat[:, 1], flat[:, 2]].reshape(h, w)
    return face_id_map


def back_project_single_view(seg_path, info_path, unique_img_path, face_votes, color_to_face_id):
    """将单个视角的预测结果回传到3D面"""
    # 读取分割结果
    seg = np.array(Image.open(seg_path.replace("_seg.npy", ".png")).convert("L")) if False else (
        np.load(seg_path).astype(np.int32)
    ) if seg_path.endswith(".npy") else np.load(seg_path).astype(np.int32)
    # seg 是 .npy 文件
    seg = np.load(seg_path).astype(np.int32)
    with open(info_path, "r") as f:
        segments_info = json.load(f)

    # 读取unique图解码face_id
    face_id_map = extract_face_id_from_unique(unique_img_path, color_to_face_id)

    # 建立 segment_id → predicted_class 的映射
    seg_to_class = {}
    for s in segments_info:
        if s["label_id"] != 0:  # 跳过背景
            seg_to_class[s["id"]] = s["label_id"]  # 只保留label_id，加速

    # 向量化投票：一次统计所有非背景像素
    # face_id_map.shape == seg.shape == (H, W)
    # 只处理非背景 seg 像素 (seg != 0) 且非背景 face_id (face_id != 0)
    non_bg = (seg != 0) & (face_id_map != 0)
    if not non_bg.any():
        return

    valid_fids = face_id_map[non_bg]
    valid_segs = seg[non_bg]

    # 一次成对统计 (face_id, pred_label) → 计数
    pairs = np.stack([valid_fids, valid_segs], axis=-1)
    unique_pairs, counts = np.unique(pairs, axis=0, return_counts=True)

    # 按 face_id 聚合
    for (fid, seg_id), cnt in zip(unique_pairs, counts):
        seg_id_int = int(seg_id)
        if seg_id_int not in seg_to_class:
            continue
        label_id = seg_to_class[seg_id_int]
        fid_int = int(fid)
        if fid_int not in face_votes:
            face_votes[fid_int] = defaultdict(lambda: {"count": 0, "total_score": 0.0, "total_pixels": 0})
        face_votes[fid_int][label_id]["count"] += 1
        face_votes[fid_int][label_id]["total_pixels"] += int(cnt)


def aggregate_votes(face_votes, num_faces):
    """多视角投票，确定每个面的最终类别"""
    results = {}
    for face_id in range(1, num_faces + 1):
        if face_id not in face_votes:
            results[face_id] = {
                "class_id": 0,
                "class_name": "未检测到",
                "confidence": 0.0,
                "votes": {},
            }
            continue

        votes = face_votes[face_id]
        # 按投票次数排序
        sorted_classes = sorted(votes.items(), key=lambda x: x[1]["count"], reverse=True)
        best_class, best_info = sorted_classes[0]

        total_votes = sum(v["count"] for v in votes.values())
        confidence = best_info["count"] / total_votes if total_votes > 0 else 0.0

        vote_summary = {}
        for cls, info in votes.items():
            vote_summary[str(cls)] = {
                "class_name": CLASS_NAMES.get(cls, f"class_{cls}"),
                "view_count": info["count"],
                "avg_score": info["total_score"] / info["count"] if info["count"] > 0 else 0,
            }

        results[face_id] = {
            "class_id": best_class,
            "class_name": CLASS_NAMES.get(best_class, f"class_{best_class}"),
            "confidence": round(confidence, 4),
            "votes": vote_summary,
        }

    return results


# ==================== Step 4: 生成带颜色的STEP ====================

def colorize_step(step_path, face_labels, output_path):
    """按预测类别给STEP模型的每个面染色"""
    reader = STEPControl_Reader()
    if reader.ReadFile(step_path) != 1:
        print(f"  Error reading STEP: {step_path}")
        return

    reader.TransferRoots()
    shape = reader.OneShape()

    app = XCAFApp_Application.GetApplication_s()
    doc = TDocStd_Document(TCollection_ExtendedString("MDTV-XCAF"))
    app.NewDocument(TCollection_ExtendedString("MDTV-XCAF"), doc)

    shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    color_tool = XCAFDoc_DocumentTool.ColorTool_s(doc.Main())
    shape_label = shape_tool.AddShape(shape)

    face_idx = 0
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face_idx += 1
        face = TopoDS.Face_s(exp.Current())

        label_info = face_labels.get(face_idx, {})
        class_id = label_info.get("class_id", 0)

        if class_id > 0 and class_id in CLASS_COLORS_3D:
            r, g, b = CLASS_COLORS_3D[class_id]
        else:
            r, g, b = 200, 200, 200  # 未检测到的面用灰色

        q_color = Quantity_Color(r / 255.0, g / 255.0, b / 255.0, Quantity_TOC_RGB)
        face_label = shape_tool.AddSubShape(shape_label, face)
        if not face_label.IsNull():
            color_tool.SetColor(face_label, q_color, XCAFDoc_ColorType.XCAFDoc_ColorSurf)
            color_tool.SetColor(face_label, q_color, XCAFDoc_ColorType.XCAFDoc_ColorGen)

        exp.Next()

    writer = STEPCAFControl_Writer()
    writer.Transfer(doc, STEPControl_AsIs)
    writer.Write(output_path)
    print(f"  彩色STEP: {output_path}")


# ==================== 主流程 ====================

def process_single_step(step_path, output_base, model_dir, device="cuda"):
    """处理单个STEP文件的完整流水线"""
    basename = os.path.splitext(os.path.basename(step_path))[0]
    output_dir = os.path.join(output_base, basename)

    print(f"\n{'='*60}")
    print(f"处理: {basename}")
    print(f"输出: {output_dir}")
    print(f"{'='*60}")

    # Step 1: 读取STEP + 生成多视角图
    print("\n[Step 1] 读取STEP，生成多视角图...")
    step_data = load_step_faces(step_path)
    if step_data is None:
        print(f"  跳过: 无法读取STEP文件")
        return None

    num_faces = len(step_data["faces"])
    print(f"  面数: {num_faces}")

    directions = get_dodecahedron_view_directions()
    semantic_dir, unique_dir = generate_views(step_data, output_dir, directions)

    # Step 2: Mask2Former推理
    print("\n[Step 2] Mask2Former推理...")
    pred_dir = os.path.join(output_dir, "pred_masks")
    run_inference_batch(semantic_dir, model_dir, pred_dir, device=device)

    # Step 3: 回传3D
    print("\n[Step 3] 回传3D面标签...")
    face_votes = {}
    num_views = len(directions)

    # 加载颜色→face_id映射
    mapping_path = os.path.join(output_dir, "color_face_id_map.json")
    with open(mapping_path, "r", encoding="utf-8") as f:
        raw_map = json.load(f)
    color_to_face_id = {}
    for key, fid in raw_map.items():
        r, g, b = [int(x) for x in key.split(",")]
        color_to_face_id[(r, g, b)] = fid

    for view_idx in range(1, num_views + 1):
        img_name = f"{view_idx:06d}.png"
        seg_path = os.path.join(pred_dir, f"{img_name}_seg.npy")
        info_path = os.path.join(pred_dir, f"{img_name}_info.json")
        unique_path = os.path.join(unique_dir, img_name)

        if not all(os.path.exists(p) for p in [seg_path, info_path, unique_path]):
            continue

        back_project_single_view(seg_path, info_path, unique_path, face_votes, color_to_face_id)

    face_labels = aggregate_votes(face_votes, num_faces)

    # 保存 face_label.json
    label_path = os.path.join(output_dir, "face_label.json")
    with open(label_path, "w", encoding="utf-8") as f:
        json.dump(face_labels, f, ensure_ascii=False, indent=2)
    print(f"  面标签: {label_path}")

    # 统计
    class_counts = defaultdict(int)
    for info in face_labels.values():
        class_counts[info["class_id"]] += 1
    print(f"  预测统计:")
    for cls_id in sorted(class_counts.keys()):
        cls_name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
        print(f"    {cls_name}: {class_counts[cls_id]} 个面")

    # Step 4: 生成彩色STEP
    print("\n[Step 4] 生成彩色STEP...")
    colored_step_path = os.path.join(output_dir, f"{basename}_colored.step")
    colorize_step(step_path, face_labels, colored_step_path)

    print(f"\n完成! 结果在: {output_dir}")
    return face_labels


def main():
    parser = argparse.ArgumentParser(description="3D→2D→3D流水线：STEP → 多视角 → Mask2Former推理 → 回传3D面标签")
    parser.add_argument("--step_file", type=str, help="单个STEP文件路径")
    parser.add_argument("--step_dir", type=str, help="STEP文件夹路径（批量处理）")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--model_dir", type=str,
                        default=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "mask2former_syb",
                                             "models", "finetuned_instance_model_v610")),
                        help="Mask2Former权重目录")
    parser.add_argument("--device", type=str, default="cuda", help="推理设备 (cuda/cpu)")
    args = parser.parse_args()

    if not args.step_file and not args.step_dir:
        parser.error("请指定 --step_file 或 --step_dir")

    os.makedirs(args.output_dir, exist_ok=True)

    step_files = []
    if args.step_file:
        step_files.append(args.step_file)
    if args.step_dir:
        for ext in ("*.step", "*.stp", "*.STEP", "*.STP"):
            import glob
            step_files.extend(glob.glob(os.path.join(args.step_dir, ext)))
        step_files.sort()

    if not step_files:
        print("未找到STEP文件")
        return

    print(f"共 {len(step_files)} 个STEP文件待处理")
    print(f"模型权重: {args.model_dir}")
    print(f"设备: {args.device}")

    for step_path in step_files:
        process_single_step(step_path, args.output_dir, args.model_dir, args.device)

    print(f"\n全部完成!")


if __name__ == "__main__":
    main()
