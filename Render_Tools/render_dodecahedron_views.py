# 生成模型的12个视图（灰度图）
import os
import glob
import math
import tempfile
import cv2
import numpy as np
import pyvista as pv

from OCP.STEPControl import STEPControl_Reader
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.StlAPI import StlAPI_Writer

OUTPUT_WIDTH = 1024
OUTPUT_HEIGHT = 1024

def get_dodecahedron_view_directions():
    phi = (1 + math.sqrt(5)) / 2
    length = math.sqrt(1 + phi**2)
    
    # 12 face centers of a regular dodecahedron (exactly 12 directions)
    dirs = [
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
        (-phi/length, 0, -1/length)
    ]
    return dirs

def get_viewup(d):
    if abs(d[2]) > 0.99:
        return (0, 1, 0)
    else:
        dot = d[2]
        vx = -dot * d[0]
        vy = -dot * d[1]
        vz = 1 - dot * d[2]
        l = math.sqrt(vx**2 + vy**2 + vz**2)
        return (vx/l, vy/l, vz/l)

def get_parallel_scale(bounds, center, direction, viewup, aspect_ratio=1.0, margin=1.08):
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

def enhance_gray_image(gray):
    # Improve local contrast first so subtle face transitions are easier to detect.
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # A mild unsharp mask makes shading boundaries a bit crisper.
    blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.2)
    sharpened = cv2.addWeighted(enhanced, 1.35, blurred, -0.35, 0)

    # Extract visible intensity transitions and darken them into thin contour lines.
    grad_x = cv2.Sobel(sharpened, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(sharpened, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    edges = cv2.threshold(magnitude, 28, 255, cv2.THRESH_BINARY)[1]
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))

    result = sharpened.copy()
    edge_mask = edges > 0
    result[edge_mask] = np.clip(result[edge_mask] * 0.45, 0, 255).astype(np.uint8)
    return result

def process_step(input_file, output_dir):
    print(f"Processing {input_file} ...")
    
    # 1. Read STEP
    reader = STEPControl_Reader()
    status = reader.ReadFile(input_file)
    if status != 1:
        print(f"Error reading {input_file}")
        return
    
    reader.TransferRoots()
    shape = reader.OneShape()
    
    # 2. Convert to STL
    BRepMesh_IncrementalMesh(shape, 0.1)
    
    fd, temp_stl = tempfile.mkstemp(suffix=".stl")
    os.close(fd)
    
    writer = StlAPI_Writer()
    writer.Write(shape, temp_stl)
    
    # 3. Load into PyVista
    mesh = pv.read(temp_stl)
    
    bb = mesh.bounds
    cx = (bb[0] + bb[1]) / 2
    cy = (bb[2] + bb[3]) / 2
    cz = (bb[4] + bb[5]) / 2
    max_dim = max(bb[1] - bb[0], bb[3] - bb[2], bb[5] - bb[4])
    if max_dim < 0.001:
        max_dim = 1.0
        
    plotter = pv.Plotter(off_screen=True, window_size=[OUTPUT_WIDTH, OUTPUT_HEIGHT])
    plotter.disable_anti_aliasing()
    
    # White background, gray mesh with lighting
    plotter.set_background('white')
    plotter.add_mesh(mesh, color='lightgray', smooth_shading=False, lighting=True)
    feature_edges = mesh.extract_feature_edges(
        boundary_edges=True,
        feature_edges=True,
        manifold_edges=False,
        non_manifold_edges=True,
        feature_angle=25,
    )
    if feature_edges.n_points > 0:
        plotter.add_mesh(
            feature_edges,
            color='black',
            line_width=3.0,
            lighting=False,
            render_lines_as_tubes=True,
        )
    plotter.enable_lightkit()
    
    plotter.camera.SetParallelProjection(True)
    
    dist = max_dim * 3
    focal = (cx, cy, cz)
    
    dirs = get_dodecahedron_view_directions()
    print(f"  Using {len(dirs)} dodecahedron view directions:")
    for i, d in enumerate(dirs):
        print(f"    View {i+1:02d}: ({d[0]:.4f}, {d[1]:.4f}, {d[2]:.4f})")
    
    basename = os.path.splitext(os.path.basename(input_file))[0]
    out_prefix = os.path.join(output_dir, basename)
    if not os.path.exists(out_prefix):
        os.makedirs(out_prefix)
        
    for i, d in enumerate(dirs):
        cam_pos = (cx + d[0] * dist, cy + d[1] * dist, cz + d[2] * dist)
        viewup = get_viewup(d)
        plotter.camera_position = [cam_pos, focal, viewup]
        parallel_scale = get_parallel_scale(
            bb,
            focal,
            d,
            viewup,
            aspect_ratio=OUTPUT_WIDTH / OUTPUT_HEIGHT,
            margin=1.10,
        )
        plotter.camera.SetParallelScale(max(parallel_scale, 0.01))
        
        # Render
        plotter.render()
        
        # Save screenshot
        temp_png = os.path.join(out_prefix, f"temp_{i+1:02d}.png")
        plotter.screenshot(temp_png)
        
        # Convert to 8-bit grayscale using cv2 with numpy to support Chinese paths
        img = cv2.imdecode(np.fromfile(temp_png, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = enhance_gray_image(gray)
            final_png = os.path.join(out_prefix, f"dodecahedron_view_{i+1:02d}.png")
            cv2.imencode('.png', gray)[1].tofile(final_png)
            
            os.remove(temp_png)
            print(f"  - Saved {final_png}")
        else:
            os.remove(temp_png)
            print(f"  - Failed to read or save {temp_png}")
        
    plotter.close()
    if os.path.exists(temp_stl):
        os.remove(temp_stl)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=r"e:\aaaa-WUT\lw\ASCCAD\test_step\test\step")
    parser.add_argument("--output_dir", type=str, default=r"e:\aaaa-WUT\lw\ASCCAD\test_step\test\dodecahedron_views")
    args = parser.parse_args()
    
    input_dir = args.input_dir
    output_dir = args.output_dir
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    step_files = glob.glob(os.path.join(input_dir, "*.step"))
    step_files.extend(glob.glob(os.path.join(input_dir, "*.stp")))
    
    for f in step_files:
        process_step(f, output_dir)
    print("\nAll done!")
