# 为STEP文件的每个面和边根据几何属性分配指定的颜色，并输出24视角渲染图
# 输入step文件（没有染色的），输出step文件（有染色的）和24视角渲染图的png文件
# 24视角方向来自正二十四面体（四方三八面体 / tetrakis hexahedron）的24个面法向
import os
import glob
import json
import math

import numpy as np
import pyvista as pv

from OCP.STEPControl import STEPControl_Reader, STEPControl_AsIs
from OCP.STEPCAFControl import STEPCAFControl_Writer
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

# --- 颜色编码字典 ---
COLOR_MAP = {
    # 面的类型
    "Plane":    (255, 165, 0),    # 橙色
    "Cylinder": (128, 0, 128),    # 紫色
    "Cone":     (0, 255, 255),    # 青色
    "Face_Other":(255, 192, 203), # 粉色

    # 边的凹凸性（渲染时统一黑色加粗）
    "Convex":   (0, 0, 0),      # 黑色
    "Concave":  (0, 0, 0),      # 黑色
    "Smooth":   (0, 0, 0),      # 黑色
    "Edge_Other":(0, 0, 0)      # 黑色
}

OUTPUT_WIDTH = 1024
OUTPUT_HEIGHT = 1024

def get_color_obj(r, g, b):
    return Quantity_Color(r/255.0, g/255.0, b/255.0, Quantity_TOC_RGB)

def rgb_to_hex(r, g, b):
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}"


def get_icositetrahedron_face_normal_directions():
    """正二十四面体（四方三八面体）的24个面法向方向，用于24视角渲染"""
    # 四方三八面体的对偶体是截角八面体。
    # 因此它的24个面法向方向可用截角八面体的24个顶点方向表示：
    # (0, ±1, ±2) 的所有坐标排列，共 3 个零轴位置 * 2 个非零轴顺序 * 4 个符号组合 = 24 个方向。
    dirs = []

    for zero_axis in range(3):
        nonzero_axes = [axis for axis in range(3) if axis != zero_axis]
        for one_axis, two_axis in [nonzero_axes, nonzero_axes[::-1]]:
            for s1 in (-1, 1):
                for s2 in (-1, 1):
                    vec = [0.0, 0.0, 0.0]
                    vec[one_axis] = s1 * 1.0
                    vec[two_axis] = s2 * 2.0
                    x, y, z = vec
                    length = math.sqrt(x * x + y * y + z * z)
                    dirs.append((x / length, y / length, z / length))

    return dirs


def get_viewup(direction):
    if abs(direction[2]) > 0.99:
        return (0, 1, 0)
    dot = direction[2]
    vx = -dot * direction[0]
    vy = -dot * direction[1]
    vz = 1 - dot * direction[2]
    length = math.sqrt(vx**2 + vy**2 + vz**2)
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
    right_len = math.sqrt(right[0]**2 + right[1]**2 + right[2]**2)
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
    return tuple(channel / 255.0 for channel in rgb)


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


def render_24_views(face_render_items, edge_render_items, bounds, image_output_dir):
    if not face_render_items and not edge_render_items:
        return

    os.makedirs(image_output_dir, exist_ok=True)

    cx = (bounds[0] + bounds[1]) / 2
    cy = (bounds[2] + bounds[3]) / 2
    cz = (bounds[4] + bounds[5]) / 2
    max_dim = max(
        bounds[1] - bounds[0],
        bounds[3] - bounds[2],
        bounds[5] - bounds[4],
    )
    if max_dim < 0.001:
        max_dim = 1.0

    center = (cx, cy, cz)
    dist = max_dim * 3
    aspect_ratio = OUTPUT_WIDTH / OUTPUT_HEIGHT

    plotter = pv.Plotter(off_screen=True, window_size=[OUTPUT_WIDTH, OUTPUT_HEIGHT])
    plotter.disable_anti_aliasing()
    plotter.set_background("white")
    plotter.camera.SetParallelProjection(True)

    for mesh, rgb in face_render_items:
        plotter.add_mesh(mesh, color=rgb_to_float(rgb), lighting=False, smooth_shading=False)

    edge_width = max(3.0, OUTPUT_WIDTH / 300.0)
    for mesh, rgb in edge_render_items:
        plotter.add_mesh(
            mesh,
            color=rgb_to_float(rgb),
            line_width=edge_width,
            lighting=False,
            render_lines_as_tubes=True  # 将线渲染为管状，彻底解决Z-fighting（面遮挡线）问题
        )

    for image_index, direction in enumerate(get_icositetrahedron_face_normal_directions(), start=1):
        viewup = get_viewup(direction)
        cam_pos = (
            center[0] + direction[0] * dist,
            center[1] + direction[1] * dist,
            center[2] + direction[2] * dist,
        )
        plotter.camera_position = [cam_pos, center, viewup]
        parallel_scale = max(
            get_parallel_scale(
                bounds,
                center,
                direction,
                viewup,
                aspect_ratio=aspect_ratio,
                margin=1.10,
            ),
            0.01,
        )
        plotter.camera.SetParallelScale(parallel_scale)
        plotter.render()

        image_name = f"face_normal_view_{image_index:02d}.png"
        image_path = os.path.join(image_output_dir, image_name)
        plotter.screenshot(image_path)
        print(f"Saved render to {image_path}")

    plotter.close()

def get_face_type(face):
    """提取面的几何类型"""
    surf = BRepAdaptor_Surface(face)
    stype = surf.GetType()
    if stype == GeomAbs_Plane: return "Plane"
    if stype == GeomAbs_Cylinder: return "Cylinder"
    if stype == GeomAbs_Cone: return "Cone"
    return "Face_Other"

def get_edge_type(edge, edge_face_map):
    """计算边的凹凸性"""
    faces_list = edge_face_map.FindFromKey(edge)
    faces = []
    for face in faces_list:
        faces.append(TopoDS.Face_s(face))

    if len(faces) != 2:
        return "Edge_Other"  # 边界边或非流形边

    f1, f2 = faces[0], faces[1]

    # 1. 检查拓扑连续性 (G1/C1及以上为平滑)
    continuity = BRep_Tool.Continuity_s(edge, f1, f2)
    if continuity in [GeomAbs_G1, GeomAbs_G2, GeomAbs_C1, GeomAbs_C2]:
        return "Smooth"

    # 2. 几何计算凹凸性 (计算法线和切线)
    try:
        adp_edge = BRepAdaptor_Curve(edge)
        mid_param = (adp_edge.FirstParameter() + adp_edge.LastParameter()) / 2.0

        # 获取边的切线
        pnt = gp_Pnt()
        vec = gp_Vec()
        adp_edge.D1(mid_param, pnt, vec)
        if vec.Magnitude() < 1e-7: return "Edge_Other"
        tangent = gp_Dir(vec)
        if edge.Orientation() == TopAbs_REVERSED:
            tangent.Reverse()

        # 获取面1法线
        adp_c2d_1 = BRepAdaptor_Curve2d(edge, f1)
        uv1 = adp_c2d_1.Value(mid_param)
        prop1 = BRepLProp_SLProps(BRepAdaptor_Surface(f1), uv1.X(), uv1.Y(), 1, 1e-6)
        if not prop1.IsNormalDefined(): return "Edge_Other"
        n1 = prop1.Normal()
        if f1.Orientation() == TopAbs_REVERSED: n1.Reverse()

        # 获取面2法线
        adp_c2d_2 = BRepAdaptor_Curve2d(edge, f2)
        uv2 = adp_c2d_2.Value(mid_param)
        prop2 = BRepLProp_SLProps(BRepAdaptor_Surface(f2), uv2.X(), uv2.Y(), 1, 1e-6)
        if not prop2.IsNormalDefined(): return "Edge_Other"
        n2 = prop2.Normal()
        if f2.Orientation() == TopAbs_REVERSED: n2.Reverse()

        # 计算叉积与点积
        cross = n1.Crossed(n2)
        if cross.Magnitude() < 1e-5:
            return "Smooth" # 法线平行，近似平滑

        dot = cross.Dot(tangent)
        if dot > 1e-5:
            return "Convex"
        elif dot < -1e-5:
            return "Concave"
        else:
            return "Smooth"
    except Exception:
        return "Edge_Other"


def process_step(input_file, output_file, render_output_dir=None):
    print(f"\nProcessing {input_file} ...")

    # 1. 读取 STEP
    reader = STEPControl_Reader()
    if reader.ReadFile(input_file) != 1:
        print(f"Error reading {input_file}")
        return

    reader.TransferRoots()
    shape = reader.OneShape()
    BRepMesh_IncrementalMesh(shape, 0.1)

    # 2. 提取面和边，并建立 Edge -> Faces 映射
    faces = []
    exp_face = TopExp_Explorer(shape, TopAbs_FACE)
    while exp_face.More():
        faces.append(TopoDS.Face_s(exp_face.Current()))
        exp_face.Next()

    edges = []
    exp_edge = TopExp_Explorer(shape, TopAbs_EDGE)
    while exp_edge.More():
        edges.append(TopoDS.Edge_s(exp_edge.Current()))
        exp_edge.Next()

    edge_face_map = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(shape, TopAbs_EDGE, TopAbs_FACE, edge_face_map)

    print(f"Found {len(faces)} faces and {len(edges)} edges.")

    # 3. 创建用于颜色导出的文档
    app = XCAFApp_Application.GetApplication_s()
    doc = TDocStd_Document(TCollection_ExtendedString("MDTV-XCAF"))
    app.NewDocument(TCollection_ExtendedString("MDTV-XCAF"), doc)

    shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    color_tool = XCAFDoc_DocumentTool.ColorTool_s(doc.Main())

    shape_label = shape_tool.AddShape(shape)
    annotation_info = []
    face_render_items = []
    edge_render_items = []
    bounds = [1e9, -1e9, 1e9, -1e9, 1e9, -1e9]

    # === 面染色 (Face Coloring) ===
    for i, face in enumerate(faces):
        f_type = get_face_type(face)
        r, g, b = COLOR_MAP[f_type]
        q_color = get_color_obj(r, g, b)

        face_label = shape_tool.AddSubShape(shape_label, face)
        if face_label.IsNull(): continue

        color_tool.SetColor(face_label, q_color, XCAFDoc_ColorType.XCAFDoc_ColorSurf)
        color_tool.SetColor(face_label, q_color, XCAFDoc_ColorType.XCAFDoc_ColorGen)
        annotation_info.append({
            "entity": "Face", "index": i + 1, "type": f_type,
            "rgb": [r, g, b], "hex": rgb_to_hex(r, g, b)
        })
        face_mesh = face_to_pyvista_mesh(face)
        if face_mesh is not None:
            face_render_items.append((face_mesh, (r, g, b)))
            fb = face_mesh.bounds
            bounds[0] = min(bounds[0], fb[0])
            bounds[1] = max(bounds[1], fb[1])
            bounds[2] = min(bounds[2], fb[2])
            bounds[3] = max(bounds[3], fb[3])
            bounds[4] = min(bounds[4], fb[4])
            bounds[5] = max(bounds[5], fb[5])

    # === 边染色 (Edge Coloring) ===
    for i, edge in enumerate(edges):
        e_type = get_edge_type(edge, edge_face_map)
        r, g, b = COLOR_MAP[e_type]
        q_color = get_color_obj(r, g, b)

        edge_label = shape_tool.AddSubShape(shape_label, edge)
        if edge_label.IsNull(): continue

        color_tool.SetColor(edge_label, q_color, XCAFDoc_ColorType.XCAFDoc_ColorCurv)
        color_tool.SetColor(edge_label, q_color, XCAFDoc_ColorType.XCAFDoc_ColorGen)
        annotation_info.append({
            "entity": "Edge", "index": i + 1, "type": e_type,
            "rgb": [r, g, b], "hex": rgb_to_hex(r, g, b)
        })
        edge_mesh = edge_to_pyvista_polyline(edge)
        if edge_mesh is not None:
            edge_render_items.append((edge_mesh, (r, g, b)))

    # 4. 导出 STEP
    writer = STEPCAFControl_Writer()
    writer.Transfer(doc, STEPControl_AsIs)
    writer.Write(output_file)
    print(f"Saved Colored STEP to {output_file}")

    # 5. 保存 JSON 标签数据
    json_output = os.path.splitext(output_file)[0] + "_labels.json"
    with open(json_output, 'w', encoding='utf-8') as f:
        json.dump(annotation_info, f, indent=2, ensure_ascii=False)
    print(f"Saved semantic mapping to {json_output}")

    if render_output_dir is not None and face_render_items:
        render_24_views(face_render_items, edge_render_items, bounds, render_output_dir)

if __name__ == "__main__":
    # --- 替换为你要求的路径 ---
    input_dir = r"E:\aaaa-WUT\lw\ASCCAD\test_step\slot_test01"

    output_dir = r"E:\aaaa-WUT\lw\ASCCAD\test_step\slot_test01_face_edge_colored_24"
    render_dir = r"E:\aaaa-WUT\lw\ASCCAD\test_step\slot_test01_face_edge_colored_24_renders"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(render_dir):
        os.makedirs(render_dir)

    step_files = glob.glob(os.path.join(input_dir, "*.step"))
    step_files.extend(glob.glob(os.path.join(input_dir, "*.stp")))
    step_files.sort()

    if not step_files:
        print(f"Warning: No .step or .stp files found in {input_dir}")

    for f in step_files:
        basename = os.path.basename(f)
        out_f = os.path.join(output_dir, basename)
        image_dir = os.path.join(render_dir, os.path.splitext(basename)[0])
        process_step(f, out_f, image_dir)
