# -*- coding: utf-8 -*-
"""
生成RGB编码视图数据集：
输入STEP文件夹，输出12视角encoded_views、face_encoding_map.json、camera_views.json。

RGB编码规则：
- 背景: (255, 255, 255)
- R: 面类型编码 Plane/Cylinder/Cone/Sphere/Other
- G: face_id低8位，G = face_id & 255
- B: face_id高8位，B = (face_id >> 8) & 255

说明：R保留面类型+面积信息，G/B共同编码face_id的自适应网格位置。
"""
import os
import sys
import glob
import json
import math

import numpy as np
import cv2
import pyvista as pv

from OCP.STEPControl import STEPControl_Reader
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
from OCP.TopoDS import TopoDS
from OCP.TopLoc import TopLoc_Location
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRep import BRep_Tool
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.GeomAbs import GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone, GeomAbs_Sphere
from OCP.GProp import GProp_GProps
from OCP.BRepGProp import BRepGProp

COLOR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "color")
if COLOR_DIR not in sys.path:
    sys.path.insert(0, COLOR_DIR)
from color_encoder import FaceColorEncoder, NAME_TO_TYPE_ID, TYPE_GAP, TYPE_R_BASE, TYPE_NAMES


# ==================== 配置 ====================

INPUT_DIR = r"E:\aaaa-WUT\lw\ASCCAD\test_step\slot_test01"
OUTPUT_DIR = r"E:\aaaa-WUT\lw\ASCCAD\test_step\encoded_inference_dataset"

OUTPUT_WIDTH = 1024
OUTPUT_HEIGHT = 1024
BACKGROUND_RGB = (255, 255, 255)

FACE_TYPE_R = {
    "Plane": 20,
    "Cylinder": 70,
    "Cone": 120,
    "Sphere": 170,
    "Other": 220,
}


# ==================== 视角与相机工具 ====================

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

    max_u = 0.0
    max_v = 0.0
    for corner in corners:
        rel = (
            corner[0] - center[0],
            corner[1] - center[1],
            corner[2] - center[2],
        )
        u = abs(rel[0] * right[0] + rel[1] * right[1] + rel[2] * right[2])
        v = abs(rel[0] * viewup[0] + rel[1] * viewup[1] + rel[2] * viewup[2])
        max_u = max(max_u, u)
        max_v = max(max_v, v)

    return max(max_v, max_u / max(aspect_ratio, 1e-6)) * margin


def rgb_to_float(rgb):
    return tuple(c / 255.0 for c in rgb)


# ==================== STEP信息提取 ====================

def get_face_type(face):
    surface = BRepAdaptor_Surface(face)
    stype = surface.GetType()
    if stype == GeomAbs_Plane:
        return "Plane"
    if stype == GeomAbs_Cylinder:
        return "Cylinder"
    if stype == GeomAbs_Cone:
        return "Cone"
    if stype == GeomAbs_Sphere:
        return "Sphere"
    return "Other"


def get_face_area(face):
    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(face, props)
    return float(props.Mass())


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


def encode_face_rgb(face_type, area, max_area, face_id, encoder):
    type_id = NAME_TO_TYPE_ID.get(face_type, NAME_TO_TYPE_ID["Other"])
    area_ratio = 0.0 if max_area <= 0 else area / max_area
    return encoder.encode(face_id, type_id, area_ratio), type_id, area_ratio


def load_step_and_encode_faces(step_path):
    reader = STEPControl_Reader()
    if reader.ReadFile(step_path) != 1:
        print(f"  Error reading STEP: {step_path}")
        return None

    reader.TransferRoots()
    shape = reader.OneShape()
    BRepMesh_IncrementalMesh(shape, 0.1)

    raw_faces = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    face_id = 1
    while exp.More():
        face = TopoDS.Face_s(exp.Current())
        face_type = get_face_type(face)
        area = get_face_area(face)
        mesh = face_to_pyvista_mesh(face)
        if mesh is not None:
            raw_faces.append({
                "face_id": face_id,
                "face_type": face_type,
                "area": area,
                "mesh": mesh,
            })
        face_id += 1
        exp.Next()

    if not raw_faces:
        return None

    max_area = max(item["area"] for item in raw_faces)
    max_face_id = max(item["face_id"] for item in raw_faces)
    encoder = FaceColorEncoder(max_face_id, shuffle=False)
    bounds = [1e9, -1e9, 1e9, -1e9, 1e9, -1e9]
    encoded_faces = []

    for item in raw_faces:
        rgb, type_id, area_ratio = encode_face_rgb(
            item["face_type"],
            item["area"],
            max_area,
            item["face_id"],
            encoder,
        )
        mesh = item["mesh"]
        fb = mesh.bounds
        bounds[0] = min(bounds[0], fb[0])
        bounds[1] = max(bounds[1], fb[1])
        bounds[2] = min(bounds[2], fb[2])
        bounds[3] = max(bounds[3], fb[3])
        bounds[4] = min(bounds[4], fb[4])
        bounds[5] = max(bounds[5], fb[5])
        encoded_faces.append({
            "face_id": item["face_id"],
            "face_type": item["face_type"],
            "area": item["area"],
            "area_ratio": area_ratio,
            "type_id": type_id,
            "encoded_rgb": rgb,
            "mesh": mesh,
        })

    return {
        "faces": encoded_faces,
        "bounds": bounds,
        "max_area": max_area,
        "max_face_id": max_face_id,
        "face_count": len(encoded_faces),
        "encoder_config": {
            "num_faces": encoder.num_faces,
            "K": encoder.K,
            "shuffle": encoder.shuffle,
            "seed": encoder.seed,
            "TYPE_GAP": TYPE_GAP,
            "TYPE_R_BASE": TYPE_R_BASE,
            "TYPE_NAMES": TYPE_NAMES,
        },
        "gb_mapping": {str(fid): {"G": g, "B": b} for fid, (g, b) in encoder.gb_mapping.items()},
    }


# ==================== 渲染 ====================

def setup_plotter(bounds):
    cx = (bounds[0] + bounds[1]) / 2
    cy = (bounds[2] + bounds[3]) / 2
    cz = (bounds[4] + bounds[5]) / 2
    max_dim = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
    if max_dim < 0.001:
        max_dim = 1.0

    center = (cx, cy, cz)
    dist = max_dim * 3
    plotter = pv.Plotter(off_screen=True, window_size=[OUTPUT_WIDTH, OUTPUT_HEIGHT])
    plotter.disable_anti_aliasing()
    plotter.set_background("white")
    plotter.camera.SetParallelProjection(True)
    return plotter, center, dist


def render_encoded_views(face_items, bounds, image_paths, view_directions=None):
    """渲染GB编码视图（带光照效果）

    两趟渲染合成：
      Pass 1 — 白色面 + 光照 → 提取光照强度图 (H×W×3 float)
      Pass 2 — 编码色面 + 无光照 → 提取纯色图   (H×W×3 uint8)
    合成：对每个非背景像素，用光照因子调制编码色，
    使最终图既有正确的 RGB 编码，又有光照立体感。

    Args:
        face_items: 面列表
        bounds: 包围盒
        image_paths: 输出图片路径列表（长度=视角数）
        view_directions: 可选，自定义视角方向列表。默认使用 get_dodecahedron_view_directions()
    """
    if not face_items:
        return []

    if view_directions is None:
        view_directions = get_dodecahedron_view_directions()

    aspect_ratio = OUTPUT_WIDTH / OUTPUT_HEIGHT
    plotter, center, dist = setup_plotter(bounds)

    # --- 添加所有面（先不设颜色，后面逐趟改） ---
    actors = []
    for item in face_items:
        actor = plotter.add_mesh(
            item["mesh"],
            color=(1.0, 1.0, 1.0),   # 临时白色，Pass 1 用
            lighting=True,
            smooth_shading=True,
            ambient=0.3,
            diffuse=0.7,
            specular=0.15,
            specular_power=20,
        )
        actors.append(actor)

    camera_records = []
    for view_index, direction in enumerate(view_directions, start=1):
        viewup = get_viewup(direction)
        cam_pos = (
            center[0] + direction[0] * dist,
            center[1] + direction[1] * dist,
            center[2] + direction[2] * dist,
        )
        parallel_scale = max(
            get_parallel_scale(bounds, center, direction, viewup, aspect_ratio=aspect_ratio, margin=1.10),
            0.01,
        )
        plotter.camera_position = [cam_pos, center, viewup]
        plotter.camera.SetParallelScale(parallel_scale)

        # ==========================================================
        # Pass 1 — 白色面 + 光照 → 光照强度图
        # 所有面保持白色，靠光照产生明暗变化
        # ==========================================================
        for actor in actors:
            prop = actor.GetProperty()
            prop.SetColor(1.0, 1.0, 1.0)
            prop.LightingOn()
            prop.SetAmbient(0.3)
            prop.SetDiffuse(0.7)
            prop.SetSpecular(0.15)
            prop.SetSpecularPower(20)
        plotter.enable_lightkit()
        plotter.render()
        light_img = plotter.screenshot(None)  # H×W×3 uint8 (BGR)

        # ==========================================================
        # Pass 2 — 编码色面 + 无光照 → 纯色图
        # ==========================================================
        for actor, item in zip(actors, face_items):
            prop = actor.GetProperty()
            prop.SetColor(*rgb_to_float(item["encoded_rgb"]))
            prop.LightingOff()
            prop.SetAmbient(1.0)
            prop.SetDiffuse(0.0)
            prop.SetSpecular(0.0)
        plotter.render()
        color_img = plotter.screenshot(None)  # H×W×3 uint8 (BGR)

        # ==========================================================
        # 合成：光照因子 × 编码色
        # ==========================================================
        light_rgb = light_img[:, :, ::-1].astype(np.float32)  # BGR→RGB
        color_rgb = color_img[:, :, ::-1].astype(np.float32)  # BGR→RGB

        # 非背景掩码：纯色图中 RGB 不全为 255 的像素
        is_fg = np.any(color_rgb < 250, axis=2)  # (H, W) bool

        # 光照因子：白色渲染的亮度 (0~1)
        light_gray = np.mean(light_rgb, axis=2) / 255.0  # (H, W)

        # 限制光照因子下界，防止暗面编码值丢失
        light_gray = np.clip(light_gray, 0.35, 1.0)  # (H, W)

        # 合成：仅对前景像素调制
        combined = np.where(
            is_fg[:, :, np.newaxis],
            np.clip(color_rgb * light_gray[:, :, np.newaxis], 0, 255),
            color_rgb,  # 背景保持白色
        ).astype(np.uint8)

        # 保存合成图
        combined_bgr = combined[:, :, ::-1]
        image_path = image_paths[view_index - 1]
        cv2.imencode(".png", combined_bgr)[1].tofile(image_path)

        camera_records.append({
            "view_index": view_index,
            "direction": list(direction),
            "camera_position": list(cam_pos),
            "focal_point": list(center),
            "viewup": list(viewup),
            "parallel_scale": float(parallel_scale),
        })

    plotter.close()
    return camera_records


# ==================== 主流程 ====================

def process_step_folder(input_dir, output_dir):
    encoded_dir = os.path.join(output_dir, "encoded_views")
    os.makedirs(encoded_dir, exist_ok=True)

    step_files = glob.glob(os.path.join(input_dir, "*.step"))
    step_files.extend(glob.glob(os.path.join(input_dir, "*.stp")))
    step_files.sort()

    face_encoding_map = {
        "encoding_rule": {
            "background_rgb": list(BACKGROUND_RGB),
            "R": "face_type: Plane=20, Cylinder=70, Cone=120, Sphere=170, Other=220",
            "G": "face_id low 8 bits: face_id & 255",
            "B": "face_id high 8 bits: (face_id >> 8) & 255",
            "decode_face_id": "face_id = G + (B << 8)",
            "max_supported_face_id": 65535,
        },
        "models": {},
    }
    camera_views = {}

    if not step_files:
        print(f"Warning: No .step or .stp files found in {input_dir}")
        return face_encoding_map, camera_views

    print(f"Found {len(step_files)} STEP files.")
    global_img_id = 1

    for model_idx, step_path in enumerate(step_files, start=1):
        basename = os.path.splitext(os.path.basename(step_path))[0]
        model_id = f"model_{model_idx:04d}"
        print(f"[{model_idx}/{len(step_files)}] Processing: {basename}")

        data = load_step_and_encode_faces(step_path)
        if data is None:
            print(f"  Skip: no valid faces")
            continue

        image_paths = []
        image_names = []
        for _ in range(12):
            image_name = f"{global_img_id:06d}.png"
            image_names.append(image_name)
            image_paths.append(os.path.join(encoded_dir, image_name))
            global_img_id += 1

        camera_records = render_encoded_views(data["faces"], data["bounds"], image_paths)

        faces_json = {}
        for item in data["faces"]:
            faces_json[str(item["face_id"])] = {
                "face_type": item["face_type"],
                "area": item["area"],
                "encoded_rgb": list(item["encoded_rgb"]),
                "r_type": item["encoded_rgb"][0],
                "g_face_id_low": item["encoded_rgb"][1],
                "b_face_id_high": item["encoded_rgb"][2],
                "decoded_face_id": item["encoded_rgb"][1] + (item["encoded_rgb"][2] << 8),
            }

        face_encoding_map["models"][model_id] = {
            "model_name": basename,
            "step_file": step_path,
            "max_area": data["max_area"],
            "max_face_id": data["max_face_id"],
            "face_count": data["face_count"],
            "images": image_names,
            "faces": faces_json,
        }

        for image_name, camera_record in zip(image_names, camera_records):
            camera_views[image_name] = {
                "model_id": model_id,
                "model_name": basename,
                "step_file": step_path,
                **camera_record,
            }

        print(f"  Saved encoded views: {image_names[0]} ~ {image_names[-1]}")

    face_map_path = os.path.join(output_dir, "face_encoding_map.json")
    camera_path = os.path.join(output_dir, "camera_views.json")
    with open(face_map_path, "w", encoding="utf-8") as f:
        json.dump(face_encoding_map, f, ensure_ascii=False, indent=2)
    with open(camera_path, "w", encoding="utf-8") as f:
        json.dump(camera_views, f, ensure_ascii=False, indent=2)

    print("\nDone.")
    print(f"Encoded views: {encoded_dir}")
    print(f"Face encoding map: {face_map_path}")
    print(f"Camera views: {camera_path}")
    return face_encoding_map, camera_views


if __name__ == "__main__":
    process_step_folder(INPUT_DIR, OUTPUT_DIR)
