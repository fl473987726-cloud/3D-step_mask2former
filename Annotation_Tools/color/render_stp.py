# -*- coding: utf-8 -*-
"""
STP文件渲染脚本 - 使用pyvista
输入：STP/STEP文件
输出：12视角渲染图片（面按类型染色，边为黑色）
"""
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# 绕过 vtkRenderingMatplotlib 缺失问题
import types
if 'vtkmodules.vtkRenderingMatplotlib' not in sys.modules:
    sys.modules['vtkmodules.vtkRenderingMatplotlib'] = types.ModuleType('vtkRenderingMatplotlib')

import numpy as np
import pyvista as pv

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.GeomAbs import GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.BRep import BRep_Tool

# 颜色定义
COLORS = {
    "Plane":    (255, 165, 0),    # 橙色 - 平面
    "Cylinder": (128, 0, 128),    # 紫色 - 圆柱面
    "Cone":     (0, 255, 255),    # 青色 - 圆锥面
    "Other":    (255, 192, 203),  # 粉色 - 其他面
}
EDGE_COLOR = (0, 0, 0)  # 黑色 - 边

OUTPUT_SIZE = 1024


def get_face_type(face):
    """获取面的几何类型"""
    try:
        surf = BRepAdaptor_Surface(face)
        stype = surf.GetType()
        if stype == GeomAbs_Plane:
            return "Plane"
        elif stype == GeomAbs_Cylinder:
            return "Cylinder"
        elif stype == GeomAbs_Cone:
            return "Cone"
        else:
            return "Other"
    except:
        return "Other"


def face_to_mesh(face):
    """将OCC面转换为pyvista网格"""
    location = TopLoc_Location()
    triangulation = BRep_Tool.Triangulation(face, location)
    if triangulation is None or triangulation.NbTriangles() == 0:
        return None

    transform = location.Transformation()
    points = []
    for i in range(1, triangulation.NbNodes() + 1):
        point = triangulation.Node(i)
        point = point.Transformed(transform)
        points.append([point.X(), point.Y(), point.Z()])

    faces_list = []
    is_reversed = face.Orientation() == 1  # TopAbs_REVERSED
    for i in range(1, triangulation.NbTriangles() + 1):
        tri = triangulation.Triangle(i)
        n1, n2, n3 = tri.Get()
        n1, n2, n3 = n1 - 1, n2 - 1, n3 - 1
        if is_reversed:
            n2, n3 = n3, n2
        faces_list.append([3, n1, n2, n3])

    return pv.PolyData(np.array(points), np.array(faces_list, dtype=np.int64))


def edge_to_polyline(edge, sample_count=50):
    """将OCC边转换为pyvista折线"""
    try:
        from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
        from OCC.Core.gp import gp_Pnt

        adaptor = BRepAdaptor_Curve(edge)
        first = adaptor.FirstParameter()
        last = adaptor.LastParameter()

        if not np.isfinite(first) or not np.isfinite(last):
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
    except:
        return None


def get_dodecahedron_views():
    """生成正十二面体的12个视角"""
    phi = (1 + np.sqrt(5)) / 2
    length = np.sqrt(1 + phi**2)
    return [
        (0, 1/length, phi/length),
        (0, -1/length, phi/length),
        (0, 1/length, -phi/length),
        (0, -1/length, -phi/length),
        (1/length, phi/length, 0),
        (-1/length, phi/length, 0),
        (1/length, -phi/length, 0),
        (-1/length, -phi/length, 0),
        (phi/length, 0, 1/length),
        (-phi/length, 0, 1/length),
        (phi/length, 0, -1/length),
        (-phi/length, 0, -1/length),
    ]


def get_viewup(direction):
    """计算Up向量"""
    if abs(direction[2]) > 0.99:
        return (0, 1, 0)
    dot = direction[2]
    vx = -dot * direction[0]
    vy = -dot * direction[1]
    vz = 1 - dot * direction[2]
    length = np.sqrt(vx**2 + vy**2 + vz**2)
    return (vx/length, vy/length, vz/length)


def render_views(plotter, face_meshes, face_colors, edge_meshes, bounds, 
                 output_dir, use_lighting=False):
    """渲染12视角"""
    max_dim = max(bounds[1]-bounds[0], bounds[3]-bounds[2], bounds[5]-bounds[4])
    if max_dim < 0.001:
        max_dim = 1.0
    dist = max_dim * 3
    center = (
        (bounds[0] + bounds[1]) / 2,
        (bounds[2] + bounds[3]) / 2,
        (bounds[4] + bounds[5]) / 2,
    )

    os.makedirs(output_dir, exist_ok=True)

    # 添加面
    for mesh, color in zip(face_meshes, face_colors):
        if use_lighting:
            # 有光照模式  ambient自发光比例, diffuse漫反射比例, specular镜面反射比例
            plotter.add_mesh(mesh, color=color, lighting=True,
                           ambient=0.4, diffuse=0.7, specular=0.2)
        else:
            # 无光照模式（纯色，使用 color 参数）
            plotter.add_mesh(mesh, color=color, lighting=False)

    # 添加边
    edge_width = max(3.0, OUTPUT_SIZE / 300.0)
    for mesh in edge_meshes:
        plotter.add_mesh(mesh, color=EDGE_COLOR, line_width=edge_width,
                         lighting=use_lighting, render_lines_as_tubes=True)

    # 渲染12视角
    for idx, direction in enumerate(get_dodecahedron_views(), start=1):
        viewup = get_viewup(direction)
        cam_pos = (
            center[0] + direction[0] * dist,
            center[1] + direction[1] * dist,
            center[2] + direction[2] * dist,
        )
        plotter.camera_position = [cam_pos, center, viewup]
        plotter.camera.SetParallelScale(max_dim * 0.6)
        plotter.render()

        image_path = os.path.join(output_dir, f"view_{idx:02d}.png")
        plotter.screenshot(image_path)

    plotter.close()
    print(f"  Saved 12 views to {output_dir}")


def process_stp(input_file, output_base):
    """处理单个STP文件"""
    print(f"\nProcessing: {input_file}")

    # 读取STEP文件
    reader = STEPControl_Reader()
    if reader.ReadFile(input_file) != 1:
        print(f"Error reading {input_file}")
        return

    reader.TransferRoots()
    shape = reader.OneShape()

    # 网格化
    BRepMesh_IncrementalMesh(shape, 0.1)

    # 提取面
    faces = []
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        faces.append(exp.Current())
        exp.Next()
    print(f"Found {len(faces)} faces")

    # 提取边
    edges = []
    exp = TopExp_Explorer(shape, TopAbs_EDGE)
    while exp.More():
        edges.append(exp.Current())
        exp.Next()
    print(f"Found {len(edges)} edges")

    # 转换为pyvista网格
    face_meshes = []
    face_colors = []
    for face in faces:
        f_type = get_face_type(face)
        color = COLORS[f_type]
        mesh = face_to_mesh(face)
        if mesh is not None:
            face_meshes.append(mesh)
            face_colors.append(tuple(c/255.0 for c in color))

    edge_meshes = []
    for edge in edges:
        mesh = edge_to_polyline(edge)
        if mesh is not None:
            edge_meshes.append(mesh)

    print(f"Got {len(face_meshes)} face meshes, {len(edge_meshes)} edge meshes")

    if not face_meshes:
        print("No meshes to render")
        return

    # 计算包围盒
    all_meshes = face_meshes + edge_meshes if edge_meshes else face_meshes
    bounds = pv.merge(all_meshes).bounds

    # 渲染无光照版本
    print("Rendering without lighting...")
    plotter1 = pv.Plotter(off_screen=True, window_size=[OUTPUT_SIZE, OUTPUT_SIZE])
    plotter1.set_background("white")
    plotter1.camera.SetParallelProjection(True)
    plotter1.disable_anti_aliasing()  # 关闭抗锯齿
    render_views(plotter1, face_meshes, face_colors, edge_meshes, bounds,
                 os.path.join(output_base, "no_lighting"), use_lighting=False)

    # 渲染有光照版本
    print("Rendering with lighting...")
    plotter2 = pv.Plotter(off_screen=True, window_size=[OUTPUT_SIZE, OUTPUT_SIZE])
    plotter2.set_background("white")
    plotter2.camera.SetParallelProjection(True)
    plotter2.disable_anti_aliasing()  # 关闭抗锯齿
    render_views(plotter2, face_meshes, face_colors, edge_meshes, bounds,
                 os.path.join(output_base, "with_lighting"), use_lighting=True)

    print("Done!")


if __name__ == "__main__":
    # 配置路径
    input_dir = r"E:\soft\code\Mask2former\stp"
    output_base = r"E:\soft\code\Mask2former\stp\rendered"

    # 获取所有STP文件
    step_files = []
    for ext in ["*.stp", "*.step"]:
        step_files.extend([
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if f.lower().endswith(ext[1:])
        ])
    step_files.sort()

    if not step_files:
        print(f"No STP files found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(step_files)} STP files")

    for f in step_files:
        basename = os.path.splitext(os.path.basename(f))[0]
        output_dir = os.path.join(output_base, basename)
        process_stp(f, output_dir)
