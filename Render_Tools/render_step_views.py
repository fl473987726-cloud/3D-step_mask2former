import os
import glob
import math
import json
import numpy as np
import pyvista as pv

from OCP.STEPControl import STEPControl_Reader
from OCP.TopExp import TopExp_Explorer
from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
from OCP.BRep import BRep_Tool
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS

OUTPUT_WIDTH = 1024
OUTPUT_HEIGHT = 1024


def get_dodecahedron_view_directions():
    """返回正十二面体的12个视角方向（与 label_tool_instance.py 一致）"""
    phi = (1 + math.sqrt(5)) / 2
    length = math.sqrt(1 + phi**2)
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
    """计算与视角方向正交的 viewup 向量"""
    if abs(direction[2]) > 0.99:
        return (0, 1, 0)
    dot = direction[2]
    vx = -dot * direction[0]
    vy = -dot * direction[1]
    vz = 1 - dot * direction[2]
    length = math.sqrt(vx**2 + vy**2 + vz**2)
    return (vx / length, vy / length, vz / length)


def get_parallel_scale(bounds, center, direction, viewup, aspect_ratio=1.0, margin=1.10):
    """计算正交投影的 parallel scale，使模型完整可见"""
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


def face_to_pyvista_mesh(face):
    """将 OCC 面转换为 PyVista PolyData 网格"""
    location = TopLoc_Location()
    triangulation = BRep_Tool.Triangulation_s(face, location)
    if triangulation is None or triangulation.NbTriangles() == 0:
        return None

    transform = location.Transformation()

    points = []
    for i in range(1, triangulation.NbNodes() + 1):
        pt = triangulation.Node(i)
        try:
            pt = pt.Transformed(transform)
        except Exception:
            pass
        points.append([pt.X(), pt.Y(), pt.Z()])

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

    return pv.PolyData(np.array(points), np.array(faces_pv))


def load_step_meshes(step_path, colors_json=None):
    """读取 STEP 文件，返回 (mesh列表, bounds) 元组"""
    reader = STEPControl_Reader()
    status = reader.ReadFile(step_path)
    if status != 1:
        print(f"  读取失败: {step_path}")
        return [], None

    reader.TransferRoots()
    shape = reader.OneShape()
    BRepMesh_IncrementalMesh(shape, 0.1)

    colors = []
    if colors_json and os.path.exists(colors_json):
        with open(colors_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
            colors = [item['rgb'] for item in data]

    meshes = []
    bounds = [1e9, -1e9, 1e9, -1e9, 1e9, -1e9]
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    face_index = 0

    while explorer.More():
        face = TopoDS.Face_s(explorer.Current())
        mesh = face_to_pyvista_mesh(face)
        if mesh is not None:
            b = mesh.bounds
            bounds[0] = min(bounds[0], b[0])
            bounds[1] = max(bounds[1], b[1])
            bounds[2] = min(bounds[2], b[2])
            bounds[3] = max(bounds[3], b[3])
            bounds[4] = min(bounds[4], b[4])
            bounds[5] = max(bounds[5], b[5])

            if face_index < len(colors):
                r, g, b_color = colors[face_index]
                meshes.append((mesh, (r / 255.0, g / 255.0, b_color / 255.0)))
            else:
                meshes.append((mesh, (0.5, 0.5, 0.5)))
            face_index += 1
        explorer.Next()

    return meshes, bounds


def process_directory(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    step_files = glob.glob(os.path.join(input_dir, "*.step"))
    step_files.extend(glob.glob(os.path.join(input_dir, "*.stp")))
    step_files.sort()

    global_image_id = 1
    coco_images = []
    coco_annotations = []
    coco_categories = []
    annotation_id = 1

    # COCO categories（颜色分类信息，用于参考）
    cat_id_set = set()

    for step_file in step_files:
        basename = os.path.splitext(os.path.basename(step_file))[0]
        colors_json = step_file.replace(".step", "_colors.json").replace(".stp", "_colors.json")

        print(f"\n处理 {step_file} ...")
        meshes, bounds = load_step_meshes(step_file, colors_json)
        if not meshes:
            continue

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
        dist = max_dim * 3

        plotter = pv.Plotter(off_screen=True, window_size=[OUTPUT_WIDTH, OUTPUT_HEIGHT])
        plotter.disable_anti_aliasing()

        for mesh, color in meshes:
            plotter.add_mesh(mesh, color=color, lighting=False, smooth_shading=False)

        plotter.set_background('white')
        plotter.camera.SetParallelProjection(True)

        aspect_ratio = OUTPUT_WIDTH / OUTPUT_HEIGHT
        directions = get_dodecahedron_view_directions()
        center = (cx, cy, cz)

        for vi, direction in enumerate(directions, start=1):
            viewup = get_viewup(direction)
            cam_pos = (
                center[0] + direction[0] * dist,
                center[1] + direction[1] * dist,
                center[2] + direction[2] * dist,
            )
            plotter.camera_position = [cam_pos, center, viewup]

            parallel_scale = max(
                get_parallel_scale(
                    bounds, center, direction, viewup,
                    aspect_ratio=aspect_ratio, margin=1.10,
                ),
                0.01,
            )
            plotter.camera.SetParallelScale(parallel_scale)

            plotter.render()
            file_name = f"{global_image_id:06d}.png"
            file_path = os.path.join(output_dir, file_name)
            plotter.screenshot(file_path)

            coco_images.append({
                "id": global_image_id,
                "file_name": file_name,
                "width": OUTPUT_WIDTH,
                "height": OUTPUT_HEIGHT,
            })

            print(f"  Saved {file_name} (model={basename}, view={vi})")
            global_image_id += 1

        plotter.close()

    # 写出 COCO JSON
    coco_output = {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": coco_categories,
    }
    coco_path = os.path.join(output_dir, "annotations.json")
    with open(coco_path, 'w', encoding='utf-8') as f:
        json.dump(coco_output, f, indent=2, ensure_ascii=False)
    print(f"\nCOCO JSON saved to {coco_path}")
    print(f"Total images: {len(coco_images)}")


if __name__ == "__main__":
    input_dir = r"e:\aaaa-WUT\lw\ASCCAD\test_step\slot_test01_colored"
    output_dir = r"e:\aaaa-WUT\lw\ASCCAD\test_step\slot_test01_colored_renders"

    process_directory(input_dir, output_dir)
    print("\n全部完成！")
