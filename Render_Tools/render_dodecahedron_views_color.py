import os
import glob
import math
import json
import tempfile
import pyvista as pv

from OCP.STEPCAFControl import STEPCAFControl_Reader
from OCP.RWGltf import RWGltf_CafWriter
from OCP.TDocStd import TDocStd_Document
from OCP.XCAFApp import XCAFApp_Application
from OCP.TCollection import TCollection_ExtendedString, TCollection_AsciiString
from OCP.TColStd import TColStd_IndexedDataMapOfStringString
from OCP.Message import Message_ProgressRange
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.XCAFDoc import XCAFDoc_DocumentTool
from OCP.TDF import TDF_LabelSequence

OUTPUT_WIDTH = 1024
OUTPUT_HEIGHT = 1024


def get_dodecahedron_view_directions():
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


def convert_step_to_gltf(step_file, gltf_file):
    app = XCAFApp_Application.GetApplication_s()
    doc = TDocStd_Document(TCollection_ExtendedString("MDTV-XCAF"))
    app.NewDocument(TCollection_ExtendedString("MDTV-XCAF"), doc)

    reader = STEPCAFControl_Reader()
    status = reader.ReadFile(step_file)
    if status != 1:
        print(f"Failed to read {step_file}")
        return False
    reader.Transfer(doc)

    shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    labels = TDF_LabelSequence()
    shape_tool.GetFreeShapes(labels)
    for i in range(1, labels.Length() + 1):
        shape = shape_tool.GetShape_s(labels.Value(i))
        BRepMesh_IncrementalMesh(shape, 0.1)

    writer = RWGltf_CafWriter(TCollection_AsciiString(gltf_file), True)
    metadata = TColStd_IndexedDataMapOfStringString()
    writer.Perform(doc, metadata, Message_ProgressRange())
    return True


def compute_bounds(block):
    bounds = [1e9, -1e9, 1e9, -1e9, 1e9, -1e9]

    def walk(node):
        if isinstance(node, pv.MultiBlock):
            for i in range(node.n_blocks):
                child = node[i]
                if child is not None:
                    walk(child)
        elif isinstance(node, pv.PolyData):
            b = node.bounds
            bounds[0] = min(bounds[0], b[0])
            bounds[1] = max(bounds[1], b[1])
            bounds[2] = min(bounds[2], b[2])
            bounds[3] = max(bounds[3], b[3])
            bounds[4] = min(bounds[4], b[4])
            bounds[5] = max(bounds[5], b[5])

    walk(block)
    return bounds


def flatten_polydata(block):
    meshes = []

    def walk(node):
        if isinstance(node, pv.MultiBlock):
            for i in range(node.n_blocks):
                child = node[i]
                if child is not None:
                    walk(child)
        elif isinstance(node, pv.PolyData):
            meshes.append(node)

    walk(block)
    return meshes


def load_face_colors(colors_json):
    colors = []
    if colors_json and os.path.exists(colors_json):
        with open(colors_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        colors = [item["rgb"] for item in data]
    return colors


def render_12_views(gltf_file, colors_json, out_dir):
    block = pv.read(gltf_file)
    bounds = compute_bounds(block)
    cx = (bounds[0] + bounds[1]) / 2
    cy = (bounds[2] + bounds[3]) / 2
    cz = (bounds[4] + bounds[5]) / 2
    max_dim = max(bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
    if max_dim < 0.001:
        max_dim = 1.0

    colors = load_face_colors(colors_json)
    meshes = flatten_polydata(block)

    plotter = pv.Plotter(off_screen=True, window_size=[OUTPUT_WIDTH, OUTPUT_HEIGHT])
    plotter.disable_anti_aliasing()
    plotter.set_background("white")

    for index, mesh in enumerate(meshes):
        if index < len(colors):
            r, g, b = [c / 255.0 for c in colors[index]]
        else:
            r, g, b = 0.7, 0.7, 0.7
        plotter.add_mesh(mesh, color=(r, g, b), lighting=False, smooth_shading=False)

    plotter.camera.SetParallelProjection(True)
    focal = (cx, cy, cz)
    dist = max_dim * 3
    directions = get_dodecahedron_view_directions()

    print(f"  Using {len(directions)} dodecahedron view directions:")
    for i, direction in enumerate(directions):
        print(f"    View {i + 1:02d}: ({direction[0]:.4f}, {direction[1]:.4f}, {direction[2]:.4f})")

    os.makedirs(out_dir, exist_ok=True)
    for i, direction in enumerate(directions):
        cam_pos = (
            cx + direction[0] * dist,
            cy + direction[1] * dist,
            cz + direction[2] * dist,
        )
        viewup = get_viewup(direction)
        plotter.camera_position = [cam_pos, focal, viewup]
        parallel_scale = get_parallel_scale(
            bounds,
            focal,
            direction,
            viewup,
            aspect_ratio=OUTPUT_WIDTH / OUTPUT_HEIGHT,
            margin=1.10,
        )
        plotter.camera.SetParallelScale(max(parallel_scale, 0.01))
        plotter.render()

        output_png = os.path.join(out_dir, f"dodecahedron_view_{i + 1:02d}.png")
        plotter.screenshot(output_png)
        print(f"  - Saved {output_png}")

    plotter.close()


def process_step(step_file, output_dir):
    print(f"Processing {step_file} ...")
    basename = os.path.splitext(os.path.basename(step_file))[0]
    colors_json = os.path.splitext(step_file)[0] + "_colors.json"
    model_output_dir = os.path.join(output_dir, basename)

    fd, temp_gltf = tempfile.mkstemp(suffix=".gltf")
    os.close(fd)
    try:
        if convert_step_to_gltf(step_file, temp_gltf):
            render_12_views(temp_gltf, colors_json, model_output_dir)
    finally:
        if os.path.exists(temp_gltf):
            os.remove(temp_gltf)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=r"e:\aaaa-WUT\lw\ASCCAD\test_step\slot_test01_colored")
    parser.add_argument("--output_dir", type=str, default=r"e:\aaaa-WUT\lw\ASCCAD\test_step\slot_test01_colored_views")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    step_files = glob.glob(os.path.join(args.input_dir, "*.step"))
    step_files.extend(glob.glob(os.path.join(args.input_dir, "*.stp")))

    for step_file in step_files:
        process_step(step_file, args.output_dir)

    print("\nAll done!")
