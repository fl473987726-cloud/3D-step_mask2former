# 生成推理数据集：给定STEP文件，生成12视角的语义（4种）染色图 + 12视角的独立着色图
# 每个模型输出24张图片（12对），编号一一对应
#   semantic_view_XX.png  - 按编码字典染色（面类型+边凹凸性），带边界增强
#   unique_view_XX.png   - 每个面独立着色，相邻视角色系差异明显（HSV色相偏移）
import os
import glob
import json
import math
import colorsys

import numpy as np
import pyvista as pv

from OCP.STEPControl import STEPControl_Reader, STEPControl_AsIs
from OCP.STEPCAFControl import STEPCAFControl_Writer, STEPCAFControl_Reader
from OCP.XCAFApp import XCAFApp_Application
from OCP.TDocStd import TDocStd_Document
from OCP.XCAFDoc import XCAFDoc_DocumentTool, XCAFDoc_ColorType
from OCP.Quantity import Quantity_Color, Quantity_TOC_RGB
from OCP.TopExp import TopExp_Explorer, TopExp
from OCP.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_REVERSED
from OCP.TCollection import TCollection_ExtendedString
from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape

from OCP.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve2d, BRepAdaptor_Curve
from OCP.GeomAbs import GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone
from OCP.GeomAbs import GeomAbs_G1, GeomAbs_G2, GeomAbs_C1, GeomAbs_C2
from OCP.BRepLProp import BRepLProp_SLProps
from OCP.BRep import BRep_Tool
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.gp import gp_Pnt, gp_Vec, gp_Dir
from OCP.TopoDS import TopoDS
from OCP.TopLoc import TopLoc_Location


# ==================== 配置 ====================

# 语义染色 - 面类型
SEMANTIC_FACE_COLOR = {
    "Plane":    (255, 165, 0),     # 橙色
    "Cylinder": (128, 0, 128),     # 紫色
    "Cone":     (0, 255, 255),     # 青色
    "Other":    (255, 192, 203),   # 粉色
}

# 边统一黑色加粗渲染（边界增强）
EDGE_COLOR = (0, 0, 0)

OUTPUT_WIDTH = 1024
OUTPUT_HEIGHT = 1024

# 色相偏移步长（度），使12个视角色系差异明显
HUE_OFFSET_STEP = 30.0


# ==================== 工具函数 ====================

def get_color_obj(r, g, b):
    return Quantity_Color(r / 255.0, g / 255.0, b / 255.0, Quantity_TOC_RGB)


def rgb_to_hex(r, g, b):
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}"


def rgb_to_float(rgb):
    return tuple(c / 255.0 for c in rgb)


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


def hsv_to_rgb(h, s, v):
    """HSV转RGB，h ∈ [0, 360)，s, v ∈ [0, 1]"""
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


def compute_unique_colors(num_faces, view_index):
    """为每个面计算独立颜色，通过HSV色相偏移保证相邻视角色系差异明显

    Args:
        num_faces: 面的数量
        view_index: 视角编号(1-12)，用于计算色相偏移
    Returns:
        list of (r, g, b) 元组，长度为num_faces
    """
    if num_faces == 0:
        return []
    colors = []
    hue_offset = (view_index - 1) * HUE_OFFSET_STEP
    # 使用黄金角分布使色相间隔最大化
    golden_angle = 137.508
    for i in range(num_faces):
        hue = (hue_offset + i * golden_angle) % 360
        colors.append(hsv_to_rgb(hue, 0.85, 0.95))
    return colors


def compute_all_views_colors(num_faces):
    """为12个视角分别计算独立着色方案

    Returns:
        list of 12个颜色列表，每个列表长度为num_faces
    """
    return [compute_unique_colors(num_faces, v) for v in range(1, 13)]


# ==================== 几何分析 ====================

def get_face_type(face):
    surf = BRepAdaptor_Surface(face)
    stype = surf.GetType()
    if stype == GeomAbs_Plane:    return "Plane"
    if stype == GeomAbs_Cylinder: return "Cylinder"
    if stype == GeomAbs_Cone:     return "Cone"
    return "Other"


def get_edge_type(edge, edge_face_map):
    faces_list = edge_face_map.FindFromKey(edge)
    faces = [TopoDS.Face_s(f) for f in faces_list]

    if len(faces) != 2:
        return "Other"

    f1, f2 = faces[0], faces[1]

    continuity = BRep_Tool.Continuity_s(edge, f1, f2)
    if continuity in [GeomAbs_G1, GeomAbs_G2, GeomAbs_C1, GeomAbs_C2]:
        return "Smooth"

    try:
        adp_edge = BRepAdaptor_Curve(edge)
        mid_param = (adp_edge.FirstParameter() + adp_edge.LastParameter()) / 2.0

        pnt = gp_Pnt()
        vec = gp_Vec()
        adp_edge.D1(mid_param, pnt, vec)
        if vec.Magnitude() < 1e-7:
            return "Other"
        tangent = gp_Dir(vec)
        if edge.Orientation() == TopAbs_REVERSED:
            tangent.Reverse()

        adp_c2d_1 = BRepAdaptor_Curve2d(edge, f1)
        uv1 = adp_c2d_1.Value(mid_param)
        prop1 = BRepLProp_SLProps(BRepAdaptor_Surface(f1), uv1.X(), uv1.Y(), 1, 1e-6)
        if not prop1.IsNormalDefined():
            return "Other"
        n1 = prop1.Normal()
        if f1.Orientation() == TopAbs_REVERSED:
            n1.Reverse()

        adp_c2d_2 = BRepAdaptor_Curve2d(edge, f2)
        uv2 = adp_c2d_2.Value(mid_param)
        prop2 = BRepLProp_SLProps(BRepAdaptor_Surface(f2), uv2.X(), uv2.Y(), 1, 1e-6)
        if not prop2.IsNormalDefined():
            return "Other"
        n2 = prop2.Normal()
        if f2.Orientation() == TopAbs_REVERSED:
            n2.Reverse()

        cross = n1.Crossed(n2)
        if cross.Magnitude() < 1e-5:
            return "Smooth"

        dot = cross.Dot(tangent)
        if dot > 1e-5:
            return "Convex"
        elif dot < -1e-5:
            return "Concave"
        else:
            return "Smooth"
    except Exception:
        return "Other"


# ==================== 网格转换 ====================

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


def edge_to_pyvista_polyline(edge, sample_count=100):
    try:
        adaptor = BRepAdaptor_Curve(edge)
        first = adaptor.FirstParameter()
        last = adaptor.LastParameter()
        if not math.isfinite(first) or not math.isfinite(last):
            return None
        if abs(last - first) < 1e-9:
            return None

        points = []
        for i in range(sample_count):
            t = first + (last - first) * (i / max(sample_count - 1, 1))
            pnt = adaptor.Value(t)
            points.append([pnt.X(), pnt.Y(), pnt.Z()])

        if len(points) < 2:
            return None

        lines = np.concatenate(([len(points)], np.arange(len(points), dtype=np.int64)))
        return pv.PolyData(np.array(points), lines=lines)
    except Exception:
        return None


# ==================== STEP处理 ====================

def load_step_and_compute_data(input_file):
    """读取STEP文件，计算面/边的语义数据，返回网格列表（unique颜色在渲染时按视角计算）"""
    reader = STEPControl_Reader()
    if reader.ReadFile(input_file) != 1:
        print(f"  Error reading {input_file}")
        return None

    reader.TransferRoots()
    shape = reader.OneShape()
    BRepMesh_IncrementalMesh(shape, 0.1)

    # 提取所有面
    faces = []
    exp_face = TopExp_Explorer(shape, TopAbs_FACE)
    while exp_face.More():
        faces.append(TopoDS.Face_s(exp_face.Current()))
        exp_face.Next()

    # 提取所有边 + 边-面邻接映射
    edges = []
    exp_edge = TopExp_Explorer(shape, TopAbs_EDGE)
    while exp_edge.More():
        edges.append(TopoDS.Edge_s(exp_edge.Current()))
        exp_edge.Next()

    edge_face_map = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(shape, TopAbs_EDGE, TopAbs_FACE, edge_face_map)

    # ---- 处理面 ----
    face_render_items = []
    annotation_info = []
    bounds = [1e9, -1e9, 1e9, -1e9, 1e9, -1e9]

    for i, face in enumerate(faces):
        f_type = get_face_type(face)
        sr, sg, sb = SEMANTIC_FACE_COLOR[f_type]

        face_mesh = face_to_pyvista_mesh(face)
        if face_mesh is None:
            continue

        face_render_items.append({
            "mesh": face_mesh,
            "semantic_color": (sr, sg, sb),
        })

        fb = face_mesh.bounds
        bounds[0] = min(bounds[0], fb[0])
        bounds[1] = max(bounds[1], fb[1])
        bounds[2] = min(bounds[2], fb[2])
        bounds[3] = max(bounds[3], fb[3])
        bounds[4] = min(bounds[4], fb[4])
        bounds[5] = max(bounds[5], fb[5])

        annotation_info.append({
            "entity": "Face", "index": i + 1, "type": f_type,
            "semantic_rgb": [sr, sg, sb], "semantic_hex": rgb_to_hex(sr, sg, sb),
        })

    # ---- 处理边 ----
    edge_render_items = []
    for i, edge in enumerate(edges):
        e_type = get_edge_type(edge, edge_face_map)
        edge_mesh = edge_to_pyvista_polyline(edge)
        if edge_mesh is not None:
            edge_render_items.append({
                "mesh": edge_mesh,
                "type": e_type,
            })
        annotation_info.append({
            "entity": "Edge", "index": i + 1, "type": e_type,
        })

    return {
        "faces": face_render_items,
        "edges": edge_render_items,
        "bounds": bounds,
        "annotation": annotation_info,
        "face_count": len(face_render_items),
        "edge_count": len(edge_render_items),
    }


# ==================== 渲染 ====================

def setup_plotter(bounds, aspect_ratio):
    """创建离屏渲染画布并设置平行投影"""
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


def render_12_views(face_items, edge_items, bounds, semantic_paths, unique_paths, views_colors, view_directions=None):
    """渲染多视角的语义图和独立着色图，编号一一对应

    Args:
        semantic_paths: 语义图文件路径的列表
        unique_paths: 独立着色图文件路径的列表
        views_colors: 颜色列表的列表，每个列表长度=len(face_items)，
                      views_colors[view_idx][face_idx] = (r, g, b)
        view_directions: 可选，自定义视角方向列表。默认使用 get_dodecahedron_view_directions()
    """
    if view_directions is None:
        view_directions = get_dodecahedron_view_directions()
    if not face_items:
        return

    aspect_ratio = OUTPUT_WIDTH / OUTPUT_HEIGHT
    edge_width = max(3.0, OUTPUT_WIDTH / 300.0)

    # --- 语义图渲染（12个视角共享同一套语义颜色） ---
    plotter_s, center, dist = setup_plotter(bounds, aspect_ratio)

    for item in face_items:
        plotter_s.add_mesh(item["mesh"], color=rgb_to_float(item["semantic_color"]),
                           lighting=False, smooth_shading=False)

    for item in edge_items:
        plotter_s.add_mesh(item["mesh"], color=rgb_to_float(EDGE_COLOR),
                           line_width=edge_width, lighting=False,
                           render_lines_as_tubes=True)

    for idx, direction in enumerate(view_directions):
        viewup = get_viewup(direction)
        cam_pos = (center[0] + direction[0] * dist,
                   center[1] + direction[1] * dist,
                   center[2] + direction[2] * dist)
        plotter_s.camera_position = [cam_pos, center, viewup]
        ps = max(get_parallel_scale(bounds, center, direction, viewup,
                                    aspect_ratio=aspect_ratio, margin=1.10), 0.01)
        plotter_s.camera.SetParallelScale(ps)
        plotter_s.render()
        plotter_s.screenshot(semantic_paths[idx])

    plotter_s.close()

    # --- 独立着色图渲染（复用plotter，只改颜色） ---
    plotter_u, center, dist = setup_plotter(bounds, aspect_ratio)

    face_actors = []
    for face_idx, item in enumerate(face_items):
        actor = plotter_u.add_mesh(item["mesh"], color=rgb_to_float(views_colors[0][face_idx]),
                                   lighting=False, smooth_shading=False)
        face_actors.append(actor)

    for item in edge_items:
        plotter_u.add_mesh(item["mesh"], color=rgb_to_float(EDGE_COLOR),
                           line_width=edge_width, lighting=False,
                           render_lines_as_tubes=True)

    for view_idx, direction in enumerate(view_directions):
        current_colors = views_colors[view_idx]

        # 只改颜色，不重建场景
        for face_idx, actor in enumerate(face_actors):
            actor.GetProperty().SetColor(*rgb_to_float(current_colors[face_idx]))

        viewup = get_viewup(direction)
        cam_pos = (center[0] + direction[0] * dist,
                   center[1] + direction[1] * dist,
                   center[2] + direction[2] * dist)
        plotter_u.camera_position = [cam_pos, center, viewup]
        ps = max(get_parallel_scale(bounds, center, direction, viewup,
                                    aspect_ratio=aspect_ratio, margin=1.10), 0.01)
        plotter_u.camera.SetParallelScale(ps)
        plotter_u.render()
        plotter_u.screenshot(unique_paths[view_idx])

    plotter_u.close()


# ==================== 导出语义STEP ====================

def export_colored_step(input_file, output_file, annotation_info):
    """导出带语义颜色的STEP文件"""
    reader = STEPControl_Reader()
    if reader.ReadFile(input_file) != 1:
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
    edge_idx = 0
    for ann in annotation_info:
        if ann["entity"] == "Face":
            rgb = ann["semantic_rgb"]
            q_color = get_color_obj(*rgb)
            face = None
            exp = TopExp_Explorer(shape, TopAbs_FACE)
            for _ in range(ann["index"]):
                face = TopoDS.Face_s(exp.Current())
                exp.Next()
            if face is not None:
                label = shape_tool.AddSubShape(shape_label, face)
                if not label.IsNull():
                    color_tool.SetColor(label, q_color, XCAFDoc_ColorType.XCAFDoc_ColorSurf)
                    color_tool.SetColor(label, q_color, XCAFDoc_ColorType.XCAFDoc_ColorGen)

        elif ann["entity"] == "Edge":
            exp = TopExp_Explorer(shape, TopAbs_EDGE)
            edge = None
            for _ in range(ann["index"]):
                edge = TopoDS.Edge_s(exp.Current())
                exp.Next()
            if edge is not None:
                q_color = get_color_obj(*EDGE_COLOR)
                label = shape_tool.AddSubShape(shape_label, edge)
                if not label.IsNull():
                    color_tool.SetColor(label, q_color, XCAFDoc_ColorType.XCAFDoc_ColorCurv)
                    color_tool.SetColor(label, q_color, XCAFDoc_ColorType.XCAFDoc_ColorGen)

    writer = STEPCAFControl_Writer()
    writer.Transfer(doc, STEPControl_AsIs)
    writer.Write(output_file)


def export_unique_colored_step(input_file, output_file, unique_colors):
    """将每个面染上独立颜色后导出STEP文件

    Args:
        input_file: 原始STEP文件路径
        output_file: 输出带颜色的STEP文件路径
        unique_colors: 长度=面数的 (r,g,b) 列表，每个面唯一颜色(0-255)
    """
    reader = STEPControl_Reader()
    if reader.ReadFile(input_file) != 1:
        print(f"  Error reading {input_file}")
        return
    reader.TransferRoots()
    shape = reader.OneShape()

    app = XCAFApp_Application.GetApplication_s()
    doc = TDocStd_Document(TCollection_ExtendedString("MDTV-XCAF"))
    app.NewDocument(TCollection_ExtendedString("MDTV-XCAF"), doc)

    shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    color_tool = XCAFDoc_DocumentTool.ColorTool_s(doc.Main())
    shape_label = shape_tool.AddShape(shape)

    # 为每个面设置独立颜色
    face_idx = 0
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face_s(exp.Current())
        if face_idx < len(unique_colors):
            r, g, b = unique_colors[face_idx]
            q_color = get_color_obj(r, g, b)
            face_label = shape_tool.AddSubShape(shape_label, face)
            if not face_label.IsNull():
                color_tool.SetColor(face_label, q_color, XCAFDoc_ColorType.XCAFDoc_ColorSurf)
                color_tool.SetColor(face_label, q_color, XCAFDoc_ColorType.XCAFDoc_ColorGen)
        face_idx += 1
        exp.Next()

    # 为所有边设置黑色
    exp_edge = TopExp_Explorer(shape, TopAbs_EDGE)
    while exp_edge.More():
        edge = TopoDS.Edge_s(exp_edge.Current())
        q_color = get_color_obj(*EDGE_COLOR)
        edge_label = shape_tool.AddSubShape(shape_label, edge)
        if not edge_label.IsNull():
            color_tool.SetColor(edge_label, q_color, XCAFDoc_ColorType.XCAFDoc_ColorCurv)
            color_tool.SetColor(edge_label, q_color, XCAFDoc_ColorType.XCAFDoc_ColorGen)
        exp_edge.Next()

    writer = STEPCAFControl_Writer()
    writer.Transfer(doc, STEPControl_AsIs)
    writer.Write(output_file)
    print(f"  Saved unique-colored STEP: {output_file}")


def read_colored_step(input_file):
    """从已染色的STEP文件中读取每个面的颜色

    Returns:
        list of (r,g,b) 元组，与面的遍历顺序一一对应
    """
    reader = STEPControl_Reader()
    if reader.ReadFile(input_file) != 1:
        print(f"  Error reading colored STEP: {input_file}")
        return []

    reader.TransferRoots()
    shape = reader.OneShape()

    app = XCAFApp_Application.GetApplication_s()
    doc = TDocStd_Document(TCollection_ExtendedString("MDTV-XCAF"))
    app.NewDocument(TCollection_ExtendedString("MDTV-XCAF"), doc)

    caf_reader = STEPCAFControl_Reader()
    caf_reader.ReadFile(input_file)
    caf_reader.Transfer(doc)

    shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    color_tool = XCAFDoc_DocumentTool.ColorTool_s(doc.Main())

    # 获取所有自由形状的标签
    from OCP.TDF import TDF_LabelSequence
    labels = TDF_LabelSequence()
    shape_tool.GetFreeShapes(labels)

    # 遍历面，按顺序读取颜色
    colors = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        face = TopoDS.Face_s(exp.Current())
        q_color = Quantity_Color()
        has_color = color_tool.GetColor_s(face, q_color, XCAFDoc_ColorType.XCAFDoc_ColorSurf)
        if has_color:
            r = int(q_color.Red() * 255)
            g = int(q_color.Green() * 255)
            b = int(q_color.Blue() * 255)
            colors.append((r, g, b))
        else:
            colors.append((255, 255, 255))  # 无颜色默认白色
        exp.Next()

    return colors


# ==================== 主入口 ====================

if __name__ == "__main__":
    # -------- 配置路径 --------
    input_dir = r"E:\aaaa-WUT\lw\ASCCAD\test_step\slot_test01"

    output_dir = r"E:\aaaa-WUT\lw\ASCCAD\test_step\inference_dataset\colored_step"
    semantic_dir = r"E:\aaaa-WUT\lw\ASCCAD\test_step\inference_dataset\semantic_views"
    unique_dir = r"E:\aaaa-WUT\lw\ASCCAD\test_step\inference_dataset\unique_views"

    for d in [output_dir, semantic_dir, unique_dir]:
        os.makedirs(d, exist_ok=True)

    # -------- 查找STEP文件 --------
    step_files = glob.glob(os.path.join(input_dir, "*.step"))
    step_files.extend(glob.glob(os.path.join(input_dir, "*.stp")))
    step_files.sort()

    if not step_files:
        print(f"Warning: No .step or .stp files found in {input_dir}")
    else:
        print(f"Found {len(step_files)} STEP files to process.\n")

    # 全局图片编号，从000001开始
    global_img_id = 1
    results = []

    for step_idx, step_file in enumerate(step_files, start=1):
        basename = os.path.splitext(os.path.basename(step_file))[0]
        model_name = f"model_{step_idx:04d}_{basename}"

        print(f"[{step_idx}/{len(step_files)}] Processing: {basename}")

        # ---- 第1步: 读取原始STEP，提取面/边几何信息 ----
        data = load_step_and_compute_data(step_file)
        if data is None:
            results.append({"file": basename, "success": False})
            continue

        print(f"  Faces: {data['face_count']}, Edges: {data['edge_count']}")

        # ---- 第2步: 计算每个面的独立颜色(黄金角HSV)，先导出染色STEP ----
        unique_colors = compute_unique_colors(data["face_count"], view_index=1)
        colored_step_path = os.path.join(output_dir, f"{model_name}_unique_colored.step")
        export_unique_colored_step(step_file, colored_step_path, unique_colors)

        # ---- 第3步: 使用写入染色STEP的颜色作为渲染颜色 ----
        # 注意：前一步已经先生成 unique_colored.step，这里使用同一组颜色保证渲染与染色STEP一致
        step_colors = unique_colors

        # ---- 第4步: 用STEP中的颜色渲染12视角 ----
        # 为12个视角分别计算着色方案（基于STEP中的颜色，每个视角做色相偏移）
        views_colors = []
        for v in range(1, 13):
            hue_offset = (v - 1) * HUE_OFFSET_STEP
            view_palette = []
            for r, g, b in step_colors:
                h, s, val = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
                h = (h + hue_offset / 360.0) % 1.0
                nr, ng, nb = colorsys.hsv_to_rgb(h, s, val)
                view_palette.append((int(nr * 255), int(ng * 255), int(nb * 255)))
            views_colors.append(view_palette)

        # ---- 第5步: 生成24张图片的文件路径（全局编号，语义/独立各12张） ----
        semantic_paths = []
        unique_paths = []
        for view_idx in range(12):
            semantic_paths.append(os.path.join(semantic_dir, f"{global_img_id:06d}.png"))
            unique_paths.append(os.path.join(unique_dir, f"{global_img_id:06d}.png"))
            global_img_id += 1

        # 渲染24张图片（12对，编号一一对应）
        render_12_views(
            data["faces"], data["edges"], data["bounds"],
            semantic_paths, unique_paths, views_colors
        )

        # 导出带语义颜色的STEP（按面类型）
        export_colored_step(
            step_file,
            os.path.join(output_dir, f"{model_name}_semantic_colored.step"),
            data["annotation"]
        )

        print(f"  Saved: semantic {os.path.basename(semantic_paths[0])}~{os.path.basename(semantic_paths[-1])}, "
              f"unique {os.path.basename(unique_paths[0])}~{os.path.basename(unique_paths[-1])}")
        results.append({
            "file": basename,
            "success": True,
            "faces": data["face_count"],
            "edges": data["edge_count"],
        })

    # -------- 汇总 --------
    print(f"\n{'='*60}")
    print(f"Done! Processed {len(results)} files.")
    ok = sum(1 for r in results if r.get("success"))
    fail = len(results) - ok
    print(f"  Success: {ok}, Failed: {fail}")
    print(f"  Total images: {ok * 24} ({ok * 12} semantic + {ok * 12} unique)")
    print(f"\nOutput directories:")
    print(f"  Semantic views: {semantic_dir}")
    print(f"  Unique views:   {unique_dir}")
    print(f"  Colored STEP:   {output_dir}")
