import sys
import os
import math
import time
import cv2
import numpy as np
import tempfile
import pyvista as pv

# Temporarily remove local directory from sys.path to safely import standard json
_original_sys_path = list(sys.path)
_local_dir = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p) != _local_dir and p != ""]
import json as std_json
sys.path = _original_sys_path

from PyQt5.QtCore import Qt, QEvent
from PyQt5.QtGui import QColor, QBrush
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

import vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.util.numpy_support import vtk_to_numpy

from OCP.BRep import BRep_Tool
from OCP.BRepAdaptor import BRepAdaptor_Surface
from OCP.BRepGProp import BRepGProp
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.GeomAbs import GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone
from OCP.GeomAbs import GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_BezierSurface, GeomAbs_BSplineSurface
from OCP.GeomAbs import GeomAbs_SurfaceOfExtrusion, GeomAbs_SurfaceOfRevolution
from OCP.GProp import GProp_GProps
from OCP.STEPControl import STEPControl_Reader
from OCP.StlAPI import StlAPI_Writer
from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED, TopAbs_EDGE
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS
from OCP.TopLoc import TopLoc_Location

OUTPUT_WIDTH = 1024
OUTPUT_HEIGHT = 1024
DEFAULT_FACE_COLOR = (210, 210, 210)


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


def get_24_face_normal_directions():
    """正二十四面体（四方三八面体 / tetrakis hexahedron）的24个面法向方向"""
    # 对偶体为截角八面体，24个面法向 = (0, ±1, ±2) 的所有坐标排列
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


def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def normalize_json_path(path):
    return os.path.abspath(path).replace("\\", "/")


def capture_render_window_rgb(render_window):
    image_filter = vtk.vtkWindowToImageFilter()
    image_filter.SetInput(render_window)
    image_filter.SetScale(1)
    image_filter.ReadFrontBufferOff()

    last_array = None
    for _ in range(3):
        # The first off-screen capture can occasionally read an uninitialized
        # back buffer. Rendering twice and forcing the image filter to refresh
        # makes the first exported view stable.
        render_window.Render()
        render_window.Render()
        image_filter.Modified()
        image_filter.Update()

        vtk_image = image_filter.GetOutput()
        width, height, _ = vtk_image.GetDimensions()
        scalars = vtk_image.GetPointData().GetScalars()
        if scalars is None:
            continue
        array = vtk_to_numpy(scalars).reshape(height, width, -1)
        array = np.flipud(array)
        if array.shape[2] > 3:
            array = array[:, :, :3]
        last_array = array.copy()

        # Uniform white/black frames indicate the render buffer was not ready.
        if float(np.std(last_array)) > 1.0:
            return last_array

    if last_array is None:
        raise RuntimeError("无法从渲染窗口捕获图像。")
    return last_array


def write_png_rgb(file_path, image_rgb):
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imencode(".png", image_bgr)[1].tofile(file_path)


def write_png_gray(file_path, image_gray):
    cv2.imencode(".png", image_gray)[1].tofile(file_path)


def enhance_gray_image(gray):
    # Keep the same post-process steps as render_dodecahedron_views.py so
    # grayscale exports remain visually consistent with the standalone script.
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    blurred = cv2.GaussianBlur(enhanced, (0, 0), 1.2)
    sharpened = cv2.addWeighted(enhanced, 1.35, blurred, -0.35, 0)

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


def build_coco_annotation(binary_mask, image_id, category_id, annotation_id):
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    segmentation_polygons = []
    valid_contours = []
    for contour in contours:
        if cv2.contourArea(contour) < 4.0:
            continue
        flattened = contour.reshape(-1, 2)
        polygon = []
        for point in flattened:
            polygon.append(float(point[0]))
            polygon.append(float(point[1]))
        if len(polygon) >= 6:
            segmentation_polygons.append(polygon)
            valid_contours.append(contour)

    if not segmentation_polygons:
        return None

    all_points = np.vstack(valid_contours)
    bx, by, bw, bh = cv2.boundingRect(all_points)
    total_area = float(sum(cv2.contourArea(contour) for contour in valid_contours))

    return {
        "id": int(annotation_id),
        "image_id": int(image_id),
        "category_id": int(category_id),
        "bbox": [int(bx), int(by), int(bw), int(bh)],
        "segmentation": segmentation_polygons,
        "area": total_area,
        "iscrowd": 0,
    }


def face_to_polydata(face):
    location = TopLoc_Location()
    triangulation = BRep_Tool.Triangulation_s(face, location)
    if triangulation is None or triangulation.NbTriangles() == 0:
        return None

    transform = location.Transformation()
    points = vtk.vtkPoints()
    for i in range(1, triangulation.NbNodes() + 1):
        point = triangulation.Node(i)
        try:
            point = point.Transformed(transform)
        except Exception:
            pass
        points.InsertNextPoint(point.X(), point.Y(), point.Z())

    cells = vtk.vtkCellArray()
    is_reversed = face.Orientation() == TopAbs_REVERSED
    for i in range(1, triangulation.NbTriangles() + 1):
        tri = triangulation.Triangle(i)
        n1 = tri.Value(1) - 1
        n2 = tri.Value(2) - 1
        n3 = tri.Value(3) - 1
        if is_reversed:
            n2, n3 = n3, n2
        vtk_tri = vtk.vtkTriangle()
        vtk_tri.GetPointIds().SetId(0, n1)
        vtk_tri.GetPointIds().SetId(1, n2)
        vtk_tri.GetPointIds().SetId(2, n3)
        cells.InsertNextCell(vtk_tri)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(cells)

    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(polydata)
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.SplittingOff()
    normals.Update()

    output = vtk.vtkPolyData()
    output.DeepCopy(normals.GetOutput())
    return output


def get_face_center_and_area(face):
    props = GProp_GProps()
    BRepGProp.SurfaceProperties_s(face, props)
    center = props.CentreOfMass()
    return (center.X(), center.Y(), center.Z()), props.Mass()


_SURFACE_TYPE_NAMES = {
    GeomAbs_Plane: "plane",
    GeomAbs_Cylinder: "cylinder",
    GeomAbs_Cone: "cone",
    GeomAbs_Sphere: "sphere",
    GeomAbs_Torus: "torus",
    GeomAbs_BezierSurface: "bezier",
    GeomAbs_BSplineSurface: "bspline",
    GeomAbs_SurfaceOfExtrusion: "extrusion",
    GeomAbs_SurfaceOfRevolution: "revolution",
}


def _get_face_surface_type(face):
    try:
        adaptor = BRepAdaptor_Surface(face, True)
        return adaptor.GetType()
    except Exception:
        return None


def _get_face_perimeter(face):
    props = GProp_GProps()
    BRepGProp.LinearProperties_s(face, props)
    return props.Mass()


def _count_face_edges(face):
    count = 0
    explorer = TopExp_Explorer(face, TopAbs_EDGE)
    while explorer.More():
        count += 1
        explorer.Next()
    return count


def build_face_adjacency_graph(face_records):
    adjacency = {r["face_id"]: set() for r in face_records}
    n = len(face_records)

    for i in range(n):
        edges_i = set()
        explorer_i = TopExp_Explorer(face_records[i]["face"], TopAbs_EDGE)
        while explorer_i.More():
            edges_i.add(explorer_i.Current().TShape())
            explorer_i.Next()

        for j in range(i + 1, n):
            explorer_j = TopExp_Explorer(face_records[j]["face"], TopAbs_EDGE)
            shared = False
            while explorer_j.More():
                if explorer_j.Current().TShape() in edges_i:
                    shared = True
                    break
                explorer_j.Next()
            if shared:
                fid_i = face_records[i]["face_id"]
                fid_j = face_records[j]["face_id"]
                adjacency[fid_i].add(fid_j)
                adjacency[fid_j].add(fid_i)

    return adjacency


def compute_topo_fingerprint(face_id, adjacency, face_records_map, hops=2):
    surface_types = {}
    for record in face_records_map.values():
        st = _get_face_surface_type(record["face"])
        type_name = _SURFACE_TYPE_NAMES.get(st, "unknown")
        surface_types[record["face_id"]] = type_name

    visited = set()
    current_frontier = {face_id}
    visited.add(face_id)

    # Per-hop breakdown: hop → {type_name: count}
    hop_breakdown = {0: {}}
    template_type = surface_types.get(face_id, "unknown")
    hop_breakdown[0][template_type] = 1

    for hop in range(1, hops + 1):
        hop_breakdown[hop] = {}
        next_frontier = set()
        for fid in current_frontier:
            for neighbor in adjacency.get(fid, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    next_frontier.add(neighbor)
                    nt = surface_types.get(neighbor, "unknown")
                    hop_breakdown[hop][nt] = hop_breakdown[hop].get(nt, 0) + 1
        current_frontier = next_frontier

    # Flatten for comparison: aggregated by surface type across all hops
    total_by_type = {}
    face_count_by_hop = {}
    for hop in range(hops + 1):
        face_count_by_hop[hop] = sum(hop_breakdown[hop].values())
        for st, cnt in hop_breakdown[hop].items():
            total_by_type[st] = total_by_type.get(st, 0) + cnt

    # Degree (number of direct neighbors)
    degree = len(adjacency.get(face_id, set()))
    # Degree distribution of direct neighbors
    neighbor_degrees = sorted([len(adjacency.get(n, set())) for n in adjacency.get(face_id, set())])

    return {
        "face_id": face_id,
        "self_type": template_type,
        "degree": degree,
        "neighbor_degrees": neighbor_degrees,
        "hop_breakdown": hop_breakdown,
        "face_count_by_hop": face_count_by_hop,
        "total_by_type": total_by_type,
        "total_neighbor_faces": sum(face_count_by_hop[h] for h in range(1, hops + 1)),
    }


def _topo_fingerprint_similarity(fp_a, fp_b):
    if fp_a["self_type"] != fp_b["self_type"]:
        return 0.0

    deg_a, deg_b = fp_a["degree"], fp_b["degree"]
    deg_score = 1.0 - abs(deg_a - deg_b) / max(deg_a, deg_b, 1)

    # Compare neighbor degree distributions (sorted lists)
    nd_a, nd_b = fp_a["neighbor_degrees"], fp_b["neighbor_degrees"]
    nd_score = 1.0
    if nd_a and nd_b:
        max_len = max(len(nd_a), len(nd_b))
        nd_score = 1.0 - abs(len(nd_a) - len(nd_b)) / max_len
        for i in range(min(len(nd_a), len(nd_b))):
            if max(nd_a[i], nd_b[i], 1) > 0:
                nd_score += 1.0 - abs(nd_a[i] - nd_b[i]) / max(nd_a[i], nd_b[i], 1)
        nd_score /= (1 + min(len(nd_a), len(nd_b)))

    # Compare type distributions (Jaccard-like)
    types_a = set(fp_a["total_by_type"].keys())
    types_b = set(fp_b["total_by_type"].keys())
    if types_a or types_b:
        intersection = types_a & types_b
        union = types_a | types_b
        type_score = len(intersection) / len(union)
    else:
        type_score = 1.0

    # Compare face counts per hop
    hop_score = 0.0
    max_hops = max(len(fp_a["face_count_by_hop"]), len(fp_b["face_count_by_hop"]))
    for h in range(max_hops):
        ca = fp_a["face_count_by_hop"].get(h, 0)
        cb = fp_b["face_count_by_hop"].get(h, 0)
        if max(ca, cb, 1) > 0:
            hop_score += 1.0 - abs(ca - cb) / max(ca, cb, 1)
    hop_score /= max(max_hops, 1)

    return deg_score * 0.25 + nd_score * 0.30 + type_score * 0.20 + hop_score * 0.25


def find_similar_faces_by_topology(template_record, face_records, face_lookup, similarity_threshold=0.70):
    adjacency = build_face_adjacency_graph(face_records)
    face_records_map = {r["face_id"]: r for r in face_records}

    template_fp = compute_topo_fingerprint(
        template_record["face_id"], adjacency, face_records_map, hops=2
    )

    similarities = []
    for record in face_records:
        if record["face_id"] == template_record["face_id"]:
            continue
        fp = compute_topo_fingerprint(record["face_id"], adjacency, face_records_map, hops=2)
        sim = _topo_fingerprint_similarity(template_fp, fp)
        if sim >= similarity_threshold:
            similarities.append((sim, record["face_id"]))

    similarities.sort(key=lambda x: x[0], reverse=True)
    return similarities


def find_similar_feature_groups(template_face_ids, face_records, face_lookup, similarity_threshold=0.70):
    """
    Finds instances of the entire subgraph formed by template_face_ids.
    Returns a list of dictionaries mapping template face_id to matched face_id.
    """
    adjacency = build_face_adjacency_graph(face_records)
    face_records_map = {r["face_id"]: r for r in face_records}

    template_nodes = list(template_face_ids)
    if not template_nodes:
        return []

    fps = {}
    for r in face_records:
        fps[r["face_id"]] = compute_topo_fingerprint(r["face_id"], adjacency, face_records_map, hops=2)

    # Sort template nodes to pick a seed with the highest degree, to constrain search early
    template_nodes.sort(key=lambda fid: len(adjacency.get(fid, set())), reverse=True)
    seed_t = template_nodes[0]

    candidates = []
    for r in face_records:
        if r["face_id"] in template_face_ids:
            continue
        sim = _topo_fingerprint_similarity(fps[seed_t], fps[r["face_id"]])
        if sim >= similarity_threshold:
            candidates.append(r["face_id"])

    matches = []

    def backtrack(t_idx, current_mapping):
        if t_idx == len(template_nodes):
            return [dict(current_mapping)]

        t = template_nodes[t_idx]
        t_neighbors = adjacency.get(t, set()) & set(template_nodes[:t_idx])

        if t_neighbors:
            first_neighbor = list(t_neighbors)[0]
            possible_c = adjacency.get(current_mapping[first_neighbor], set()).copy()
            for n in list(t_neighbors)[1:]:
                possible_c &= adjacency.get(current_mapping[n], set())
        else:
            possible_c = set(r["face_id"] for r in face_records if r["face_id"] not in current_mapping.values())

        valid_matches = []
        for c in possible_c:
            if c in current_mapping.values() or c in template_face_ids:
                continue
            sim = _topo_fingerprint_similarity(fps[t], fps[c])
            if sim >= similarity_threshold:
                current_mapping[t] = c
                valid_matches.extend(backtrack(t_idx + 1, current_mapping))
                del current_mapping[t]

        return valid_matches

    for c_seed in candidates:
        mapping = {seed_t: c_seed}
        found_mappings = backtrack(1, mapping)
        for m in found_mappings:
            matches.append(m)

    unique_groups = []
    seen_sets = set()
    for m in matches:
        val_set = frozenset(m.values())
        if val_set not in seen_sets:
            seen_sets.add(val_set)
            unique_groups.append(m)

    return unique_groups


def build_feature_edge_actor(face_records):
    append_filter = vtk.vtkAppendPolyData()
    for record in face_records:
        append_filter.AddInputData(record["polydata"])
    append_filter.Update()

    feature_edges = vtk.vtkFeatureEdges()
    feature_edges.SetInputConnection(append_filter.GetOutputPort())
    feature_edges.BoundaryEdgesOn()
    feature_edges.FeatureEdgesOn()
    feature_edges.ManifoldEdgesOff()
    feature_edges.NonManifoldEdgesOn()
    feature_edges.SetFeatureAngle(25.0)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(feature_edges.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0.0, 0.0, 0.0)
    actor.GetProperty().LightingOff()
    actor.GetProperty().SetLineWidth(3.0)
    return actor


def load_step_faces(step_path, linear_deflection=0.1):
    reader = STEPControl_Reader()
    status = reader.ReadFile(step_path)
    if status != 1:
        raise RuntimeError(f"无法读取 STEP 文件: {step_path}")

    reader.TransferRoots()
    shape = reader.OneShape()
    BRepMesh_IncrementalMesh(shape, linear_deflection)

    records = []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    face_id = 1
    while explorer.More():
        face = TopoDS.Face_s(explorer.Current())
        polydata = face_to_polydata(face)
        if polydata is not None and polydata.GetNumberOfCells() > 0:
            center, area = get_face_center_and_area(face)
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputData(polydata)
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetInterpolationToPhong()

            records.append(
                {
                    "face_id": face_id,
                    "face": face,
                    "polydata": polydata,
                    "actor": actor,
                    "center": center,
                    "area": area,
                    "bounds": list(polydata.GetBounds()),
                    "label_id": None,
                }
            )
            face_id += 1
        explorer.Next()

    return shape, records


class LabelTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("STEP零件面自动化标注工具")
        self.setGeometry(50, 50, 1680, 960)

        self.categories = [
            {"id": 1, "name": "Wide Slot", "display_name": "宽体槽", "color": (255, 0, 0)},
            {"id": 2, "name": "Closed Slot", "display_name": "封闭槽", "color": (255, 255, 0)},
            {"id": 3, "name": "Open Slot", "display_name": "开放槽", "color": (0, 0, 255)},
            {"id": 4, "name": "Hole", "display_name": "孔", "color": (0, 255, 0)},
            {"id": 5, "name": "Open Cavity", "display_name": "开放型腔", "color": (255, 128, 0)},
            {"id": 6, "name": "Closed Cavity", "display_name": "封闭型腔", "color": (128, 0, 255)},
            {"id": 7, "name": "Compound Cavity", "display_name": "复合型腔", "color": (0, 200, 200)},
        ]
        self.category_by_id = {item["id"]: item for item in self.categories}

        self.model_dataset = {}
        self.step_paths = []
        self.step_path = ""
        self.shape = None
        self.face_records = []
        self.face_lookup = {}
        self.actor_lookup = {}
        self.undo_stack = []
        self.hover_face_id = None
        self.selected_face_id = None
        self.current_category_id = self.categories[0]["id"]
        self.model_bounds = None
        self.left_press_pos = None
        self.drag_threshold = 6
        self.similarity_threshold = 0.70
        self.template_face_ids = set()
        self.current_instance_id = None

        self._build_ui()
        self._init_vtk()

    def _build_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # ==================== 左侧面板：模型/标注/学习控制 ====================
        left_panel = QVBoxLayout()

        left_panel.addWidget(QLabel("已导入模型列表:"))
        self.model_list = QListWidget()
        self.model_list.currentRowChanged.connect(self.on_model_switched)
        left_panel.addWidget(self.model_list)

        self.btn_load_step = QPushButton("1. 导入多个 STEP 模型")
        self.btn_load_step.clicked.connect(self.open_step_dialog)
        left_panel.addWidget(self.btn_load_step)

        self.btn_delete_model = QPushButton("删除当前模型")
        self.btn_delete_model.clicked.connect(self.delete_current_model)
        left_panel.addWidget(self.btn_delete_model)

        left_panel.addWidget(QLabel("当前模型:"))
        self.model_label = QLabel("未加载")
        self.model_label.setWordWrap(True)
        left_panel.addWidget(self.model_label)

        left_panel.addWidget(QLabel("预设类型 (左键单击面即可标注当前类型):"))
        self.category_list = QListWidget()
        self.category_list.currentRowChanged.connect(self.on_category_changed)
        left_panel.addWidget(self.category_list)

        self.btn_new_instance = QPushButton("创建当前类别新实例")
        self.btn_new_instance.clicked.connect(self.create_current_instance)
        left_panel.addWidget(self.btn_new_instance)

        self.btn_delete_instance = QPushButton("删除当前实例")
        self.btn_delete_instance.clicked.connect(self.delete_current_instance)
        left_panel.addWidget(self.btn_delete_instance)

        left_panel.addWidget(QLabel("当前实例:"))
        self.current_instance_label = QLabel("未选择")
        self.current_instance_label.setWordWrap(True)
        left_panel.addWidget(self.current_instance_label)

        left_panel.addWidget(QLabel("实例列表:"))
        self.instance_list = QListWidget()
        self.instance_list.currentRowChanged.connect(self.on_instance_changed)
        left_panel.addWidget(self.instance_list)
        self._refresh_category_list()

        self.btn_clear_face = QPushButton("清除当前面的标注")
        self.btn_clear_face.clicked.connect(self.clear_selected_face_label)
        left_panel.addWidget(self.btn_clear_face)

        self.btn_undo = QPushButton("撤回上一次修改")
        self.btn_undo.clicked.connect(self.undo_last_action)
        left_panel.addWidget(self.btn_undo)

        self.btn_clear_all = QPushButton("清空全部标注")
        self.btn_clear_all.clicked.connect(self.clear_all_labels)
        left_panel.addWidget(self.btn_clear_all)

        left_panel.addWidget(QLabel("── 特征学习 ──"))
        left_panel.addWidget(QLabel("1. 左键单击选中已标注面"))
        self.btn_add_template = QPushButton("➕ 加入模板面列表")
        self.btn_add_template.clicked.connect(self.add_to_template)
        left_panel.addWidget(self.btn_add_template)

        self.btn_remove_template = QPushButton("➖ 从模板列表移除选中面")
        self.btn_remove_template.clicked.connect(self.remove_from_template)
        left_panel.addWidget(self.btn_remove_template)

        left_panel.addWidget(QLabel("模板面列表:"))
        self.template_list = QListWidget()
        self.template_list.itemClicked.connect(self._on_template_item_clicked)
        left_panel.addWidget(self.template_list)

        self.btn_learn_features = QPushButton("🔍 特征学习：按模板标注相似面")
        self.btn_learn_features.setStyleSheet("background-color: #FF9800; color: white; font-weight: bold; padding: 6px;")
        self.btn_learn_features.clicked.connect(self.learn_and_label_similar)
        left_panel.addWidget(self.btn_learn_features)

        left_panel.addWidget(QLabel("相似度阈值 (越高越严格):"))
        self.slider_similarity = QSlider(Qt.Horizontal)
        self.slider_similarity.setMinimum(50)
        self.slider_similarity.setMaximum(99)
        self.slider_similarity.setValue(70)
        self.slider_similarity.setTickPosition(QSlider.TicksBelow)
        self.slider_similarity.setTickInterval(10)
        self.slider_similarity.valueChanged.connect(self._on_similarity_changed)
        left_panel.addWidget(self.slider_similarity)

        self.label_similarity = QLabel("当前阈值: 0.70")
        left_panel.addWidget(self.label_similarity)

        self.status_label = QLabel(
            "操作说明:\n"
            "- 左键拖拽: 旋转模型\n"
            "- 滚轮: 缩放\n"
            "- 中键拖拽: 平移\n"
            "- 右键拖拽: 缩放\n"
            "- 左键单击: 选中并标注面\n"
            "- 鼠标移动: 实时高亮预选面"
        )
        self.status_label.setWordWrap(True)
        left_panel.addWidget(self.status_label)
        left_panel.addStretch(1)

        # ==================== 中间：3D视图 ====================
        center_panel = QVBoxLayout()
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        self.vtk_widget.setMouseTracking(True)
        self.vtk_widget.installEventFilter(self)
        center_panel.addWidget(self.vtk_widget)

        # ==================== 右侧面板：导出/预识别/面列表 ====================
        right_panel = QVBoxLayout()

        right_panel.addWidget(QLabel("── 数据导出 ──"))

        self.btn_load_json = QPushButton("2. 导出GB编码染色图")
        self.btn_load_json.clicked.connect(self.export_gb_encoded_views)
        right_panel.addWidget(self.btn_load_json)

        self.btn_export = QPushButton("3. 生成语义染色图 + 独立着色图")
        self.btn_export.clicked.connect(self.export_inference_views)
        right_panel.addWidget(self.btn_export)

        self.btn_export_gray = QPushButton("4. 导出12视角灰度图")
        self.btn_export_gray.clicked.connect(self.export_gray_package)
        right_panel.addWidget(self.btn_export_gray)

        # 视角数量选择
        hbox_view_count = QHBoxLayout()
        hbox_view_count.addWidget(QLabel("视角数量:"))
        self.combo_num_views = QComboBox()
        self.combo_num_views.addItems(["12", "24"])
        self.combo_num_views.setCurrentIndex(0)
        self.combo_num_views.currentTextChanged.connect(self._on_num_views_changed)
        hbox_view_count.addWidget(self.combo_num_views)
        right_panel.addLayout(hbox_view_count)

        # COCO 灰度数据集导出
        hbox_export_gray_coco = QHBoxLayout()
        hbox_export_gray_coco.addWidget(QLabel("起始图片编号:"))
        self.spin_start_id = QSpinBox()
        self.spin_start_id.setRange(1, 999999)
        self.spin_start_id.setValue(1)
        hbox_export_gray_coco.addWidget(self.spin_start_id)
        right_panel.addLayout(hbox_export_gray_coco)

        self.btn_export_gray_coco = QPushButton("5. 导出全局 COCO 灰度数据集")
        self.btn_export_gray_coco.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.btn_export_gray_coco.clicked.connect(self.export_global_gray_coco)
        right_panel.addWidget(self.btn_export_gray_coco)

        self.btn_export_coco_mask = QPushButton("6. 导出多灰度值掩码图 + class_map")
        self.btn_export_coco_mask.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
        self.btn_export_coco_mask.clicked.connect(self.export_coco_mask_labels)
        right_panel.addWidget(self.btn_export_coco_mask)

        right_panel.addWidget(QLabel("── 特征预识别 ──"))

        self.btn_pre_recognize = QPushButton("🤖 特征预识别 (Mask2Former)")
        self.btn_pre_recognize.setStyleSheet(
            "QPushButton { background-color: #9C27B0; color: white; font-weight: bold; padding: 6px; }"
            "QPushButton:disabled { background-color: #cccccc; }"
        )
        self.btn_pre_recognize.clicked.connect(self._run_feature_pre_recognition)
        right_panel.addWidget(self.btn_pre_recognize)

        self.btn_apply_predictions = QPushButton("✅ 应用预识别结果")
        self.btn_apply_predictions.setStyleSheet(
            "QPushButton { background-color: #009688; color: white; font-weight: bold; padding: 6px; }"
            "QPushButton:disabled { background-color: #cccccc; }"
        )
        self.btn_apply_predictions.setEnabled(False)
        self.btn_apply_predictions.clicked.connect(self._apply_pre_recognition_results)
        right_panel.addWidget(self.btn_apply_predictions)

        self.pre_recog_progress = QProgressBar()
        self.pre_recog_progress.setValue(0)
        self.pre_recog_progress.setFormat("等待开始...")
        right_panel.addWidget(self.pre_recog_progress)

        self.pre_recog_status = QLabel("未执行预识别")
        self.pre_recog_status.setWordWrap(True)
        right_panel.addWidget(self.pre_recog_status)

        right_panel.addWidget(QLabel("── 3D视图控制 ──"))

        self.btn_reset_camera = QPushButton("重置三维视角")
        self.btn_reset_camera.clicked.connect(self.reset_camera)
        right_panel.addWidget(self.btn_reset_camera)

        right_panel.addWidget(QLabel("── 面统计与列表 ──"))

        self.summary_label = QLabel("面统计: 未加载模型")
        self.summary_label.setWordWrap(True)
        right_panel.addWidget(self.summary_label)

        right_panel.addWidget(QLabel("已标注面列表:"))
        self.face_list = QListWidget()
        self.face_list.currentItemChanged.connect(self.on_face_item_changed)
        right_panel.addWidget(self.face_list)

        self.current_face_label = QLabel("当前面: 无")
        self.current_face_label.setWordWrap(True)
        right_panel.addWidget(self.current_face_label)

        main_layout.addLayout(left_panel, 2)
        main_layout.addLayout(center_panel, 7)
        main_layout.addLayout(right_panel, 3)

    def _init_vtk(self):
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(1.0, 1.0, 1.0)
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        style = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(style)
        self.picker = vtk.vtkCellPicker()
        self.picker.SetTolerance(0.0005)
        self.interactor.Initialize()

    def _refresh_category_list(self):
        self.category_list.clear()
        for category in self.categories:
            text = f"[{category['id']}] {category['display_name']} / {category['name']}"
            item = QListWidgetItem(text)
            color = QColor(*category["color"])
            item.setBackground(QBrush(color))
            item.setForeground(QBrush(QColor(0, 0, 0)))
            item.setData(Qt.UserRole, category["id"])
            self.category_list.addItem(item)
        self.category_list.setCurrentRow(0)

    def on_category_changed(self, row):
        if row < 0:
            return
        item = self.category_list.item(row)
        if item is not None:
            self.current_category_id = item.data(Qt.UserRole)
            self._update_current_instance_label()

    def _get_current_entry(self):
        return self.model_dataset.get(self.step_path)

    def _get_instance_lookup(self):
        entry = self._get_current_entry()
        if entry is None:
            return {}
        return {item["instance_id"]: item for item in entry.get("instances", [])}

    def _get_instance_by_id(self, instance_id):
        return self._get_instance_lookup().get(instance_id)

    def _ensure_current_instance(self, category_id=None):
        if category_id is None:
            category_id = self.current_category_id
        instance = self._get_instance_by_id(self.current_instance_id)
        if instance is not None and instance["category_id"] == category_id:
            return instance["instance_id"]
        return self._create_instance(category_id)

    def _create_instance(self, category_id, auto_select=True):
        entry = self._get_current_entry()
        if entry is None:
            return None
        instances = entry.setdefault("instances", [])
        next_instance_id = 1
        if instances:
            next_instance_id = max(item["instance_id"] for item in instances) + 1

        category = self.category_by_id[category_id]
        next_index = sum(1 for item in instances if item["category_id"] == category_id) + 1
        instance = {
            "instance_id": next_instance_id,
            "category_id": category_id,
            "instance_name": f"{category['display_name']}{next_index}",
        }
        instances.append(instance)
        self._renumber_instances(category_id)
        if auto_select:
            self.current_instance_id = next_instance_id
        self._refresh_instance_list()
        self._update_current_instance_label()
        return next_instance_id

    def _renumber_instances(self, category_id=None):
        entry = self._get_current_entry()
        if entry is None:
            return

        instances = entry.setdefault("instances", [])
        counters = {}
        for instance in sorted(instances, key=lambda item: (item["category_id"], item["instance_id"])):
            current_category_id = instance["category_id"]
            counters[current_category_id] = counters.get(current_category_id, 0) + 1
            if category_id is None or current_category_id == category_id:
                category = self.category_by_id.get(current_category_id)
                if category is not None:
                    instance["instance_name"] = f"{category['display_name']}{counters[current_category_id]}"

        entry["instance_counters"] = {str(key): value for key, value in counters.items()}

    def create_current_instance(self):
        if not self.step_path:
            QMessageBox.information(self, "提示", "请先加载 STEP 模型。")
            return
        instance_id = self._create_instance(self.current_category_id, auto_select=True)
        if instance_id is not None:
            self.status_label.setText(f"已创建实例：{self._get_instance_by_id(instance_id)['instance_name']}")

    def delete_current_instance(self):
        if not self.step_path:
            QMessageBox.information(self, "提示", "请先加载 STEP 模型。")
            return

        instance = self._get_instance_by_id(self.current_instance_id)
        if instance is None:
            QMessageBox.information(self, "提示", "请先在实例列表中选择一个实例。")
            return

        entry = self._get_current_entry()
        if entry is None:
            return

        target_face_ids = sorted(
            face_id
            for face_id, instance_id in entry.get("face_to_instance", {}).items()
            if instance_id == instance["instance_id"]
        )
        reply = QMessageBox.question(
            self,
            "确认删除实例",
            f"确定删除实例 {instance['instance_name']} 吗？\n\n"
            f"该实例包含 {len(target_face_ids)} 个面，删除后这些面会变成未标注状态，"
            f"并且同类别实例会自动重新编号。",
        )
        if reply != QMessageBox.Yes:
            return

        for face_id in target_face_ids:
            entry["face_to_instance"].pop(face_id, None)
            record = self.face_lookup.get(face_id)
            if record is not None:
                record["instance_id"] = None
                record["label_id"] = None
                self._apply_actor_style(face_id)

        entry["instances"] = [
            item for item in entry.get("instances", []) if item["instance_id"] != instance["instance_id"]
        ]
        entry["undo_stack"] = []
        self.undo_stack = []
        self.current_instance_id = None
        self.template_face_ids = {
            face_id
            for face_id in self.template_face_ids
            if self.face_lookup.get(face_id) is not None
            and self.face_lookup[face_id].get("instance_id") is not None
        }
        if self.selected_face_id in target_face_ids:
            self.selected_face_id = None
        if self.hover_face_id in target_face_ids:
            self.hover_face_id = None

        self._renumber_instances(instance["category_id"])
        self._refresh_template_list()
        self._refresh_instance_list()
        self._refresh_face_list()
        self._update_summary()
        self._update_current_instance_label()
        self._update_current_face_label()
        self.vtk_widget.GetRenderWindow().Render()
        self.status_label.setText(f"已删除实例：{instance['instance_name']}")

    def _refresh_instance_list(self):
        previous_instance_id = self.current_instance_id
        self.instance_list.clear()
        entry = self._get_current_entry()
        if entry is None:
            self._update_current_instance_label()
            return
        for instance in entry.get("instances", []):
            category = self.category_by_id.get(instance["category_id"], {})
            face_count = sum(
                1 for iid in entry.get("face_to_instance", {}).values() if iid == instance["instance_id"]
            )
            text = f"{instance['instance_name']} | {category.get('display_name', '?')} | 面数:{face_count}"
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, instance["instance_id"])
            if category:
                item.setBackground(QBrush(QColor(*category["color"])))
            self.instance_list.addItem(item)
        if previous_instance_id is not None:
            for row in range(self.instance_list.count()):
                item = self.instance_list.item(row)
                if item.data(Qt.UserRole) == previous_instance_id:
                    self.instance_list.setCurrentRow(row)
                    break
        self._update_current_instance_label()

    def on_instance_changed(self, row):
        if row < 0:
            self.current_instance_id = None
            self._update_current_instance_label()
            return
        item = self.instance_list.item(row)
        if item is None:
            return
        self.current_instance_id = item.data(Qt.UserRole)
        instance = self._get_instance_by_id(self.current_instance_id)
        if instance is not None:
            self.current_category_id = instance["category_id"]
            for row_idx in range(self.category_list.count()):
                category_item = self.category_list.item(row_idx)
                if category_item.data(Qt.UserRole) == self.current_category_id:
                    self.category_list.setCurrentRow(row_idx)
                    break
        self._update_current_instance_label()

    def _update_current_instance_label(self):
        instance = self._get_instance_by_id(self.current_instance_id)
        if instance is None:
            category = self.category_by_id.get(self.current_category_id, {})
            self.current_instance_label.setText(
                f"未选择实例\n当前类别: {category.get('display_name', '未知')}"
            )
            return
        category = self.category_by_id.get(instance["category_id"], {})
        self.current_instance_label.setText(
            f"{instance['instance_name']} / {category.get('display_name', '?')}"
        )

    def open_step_dialog(self):
        step_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "选择一个或多个 STEP 模型",
            "",
            "STEP Files (*.step *.stp)",
        )
        if step_paths:
            self.add_step_files(step_paths)

    def _ensure_model_entry(self, step_path):
        if step_path not in self.model_dataset:
            self.model_dataset[step_path] = {
                "labels": {},
                "face_to_instance": {},
                "instances": [],
                "instance_counters": {},
                "undo_stack": [],
            }

    def add_step_files(self, step_paths):
        new_paths = []
        for step_path in step_paths:
            if not step_path:
                continue
            self._ensure_model_entry(step_path)
            if step_path not in self.step_paths:
                self.step_paths.append(step_path)
                self.model_list.addItem(os.path.basename(step_path))
                new_paths.append(step_path)
        if self.model_list.count() > 0 and self.model_list.currentRow() < 0:
            self.model_list.setCurrentRow(0)
        elif new_paths:
            self.model_list.setCurrentRow(self.step_paths.index(new_paths[0]))

    def on_model_switched(self, row):
        if row < 0 or row >= len(self.step_paths):
            return
        step_path = self.step_paths[row]
        if os.path.normcase(step_path) == os.path.normcase(self.step_path):
            return
        self.load_step_file(step_path)

    def _reset_loaded_model_state(self):
        self._clear_renderer()
        self.shape = None
        self.step_path = ""
        self.face_records = []
        self.face_lookup = {}
        self.actor_lookup = {}
        self.undo_stack = []
        self.hover_face_id = None
        self.selected_face_id = None
        self.template_face_ids = set()
        self.current_instance_id = None
        self.model_bounds = None
        self.model_label.setText("未加载")
        self._refresh_template_list()
        self._refresh_instance_list()
        self._refresh_face_list()
        self._update_summary()
        self._update_current_instance_label()
        self._update_current_face_label()
        self.setWindowTitle("STEP零件面自动化标注工具")

    def delete_current_model(self):
        row = self.model_list.currentRow()
        if row < 0 or row >= len(self.step_paths):
            QMessageBox.information(self, "提示", "当前没有可删除的模型。")
            return

        step_path = self.step_paths[row]
        reply = QMessageBox.question(
            self,
            "确认删除",
            f"确定删除当前模型？\n\n{step_path}\n\n删除后会移除该模型的当前标注缓存。",
        )
        if reply != QMessageBox.Yes:
            return

        was_current = bool(self.step_path) and os.path.normcase(step_path) == os.path.normcase(self.step_path)
        self.step_paths.pop(row)
        self.model_list.blockSignals(True)
        item = self.model_list.takeItem(row)
        del item
        self.model_list.blockSignals(False)
        self.model_dataset.pop(step_path, None)

        if not self.step_paths:
            self._reset_loaded_model_state()
            self.model_list.clearSelection()
            return

        next_row = min(row, len(self.step_paths) - 1)
        if was_current:
            self.model_list.setCurrentRow(next_row)
            if os.path.normcase(self.step_paths[next_row]) == os.path.normcase(self.step_path):
                self.load_step_file(self.step_paths[next_row])
        else:
            current_row = self.model_list.currentRow()
            if current_row < 0:
                self.model_list.setCurrentRow(min(next_row, len(self.step_paths) - 1))

    def load_step_file(self, step_path):
        try:
            self.setWindowTitle("STEP零件面自动化标注工具 - 加载中...")
            QApplication.processEvents()
            shape, face_records = load_step_faces(step_path)
        except Exception as exc:
            QMessageBox.critical(self, "加载失败", str(exc))
            self.setWindowTitle("STEP零件面自动化标注工具")
            return

        self._ensure_model_entry(step_path)
        entry = self.model_dataset[step_path]
        entry.setdefault("face_to_instance", {})
        entry.setdefault("instances", [])
        entry.setdefault("instance_counters", {})

        self._clear_renderer()
        self.shape = shape
        self.step_path = step_path
        self.face_records = face_records
        self.face_lookup = {record["face_id"]: record for record in face_records}
        self.actor_lookup = {}
        self.undo_stack = list(entry["undo_stack"])
        self.hover_face_id = None
        self.selected_face_id = None
        self.template_face_ids = set()
        self.current_instance_id = None
        self._refresh_template_list()

        for record in self.face_records:
            actor_key = record["actor"].GetAddressAsString("")
            self.actor_lookup[actor_key] = record["face_id"]
            record["actor"].PickableOn()
            record["instance_id"] = entry["face_to_instance"].get(record["face_id"])
            instance = self._get_instance_by_id(record["instance_id"])
            record["label_id"] = instance["category_id"] if instance is not None else entry["labels"].get(record["face_id"])
            self.renderer.AddActor(record["actor"])
            self._apply_actor_style(record["face_id"])

        self.renderer.ResetCamera()
        self.model_bounds = self.renderer.ComputeVisiblePropBounds()
        self.vtk_widget.GetRenderWindow().Render()

        self.model_label.setText(step_path)
        self._refresh_instance_list()
        self._refresh_face_list()
        self._update_summary()
        self._update_current_instance_label()
        self._update_current_face_label()
        self.setWindowTitle(f"STEP零件面自动化标注工具 - {os.path.basename(step_path)}")

    def _clear_renderer(self):
        self.renderer.RemoveAllViewProps()
        self.vtk_widget.GetRenderWindow().Render()

    def eventFilter(self, watched, event):
        if watched is self.vtk_widget and self.face_records:
            if event.type() == QEvent.MouseMove:
                self._update_hover_face(event.pos())
            elif event.type() == QEvent.MouseButtonPress:
                if event.button() == Qt.LeftButton:
                    self.left_press_pos = event.pos()
            elif event.type() == QEvent.Leave:
                self._set_hover_face(None)
            elif event.type() == QEvent.MouseButtonRelease:
                if event.button() == Qt.LeftButton and self.left_press_pos is not None:
                    dx = event.pos().x() - self.left_press_pos.x()
                    dy = event.pos().y() - self.left_press_pos.y()
                    if dx * dx + dy * dy <= self.drag_threshold * self.drag_threshold:
                        self._assign_face_from_click(event.pos())
                    self.left_press_pos = None
        return super().eventFilter(watched, event)

    def _pick_face_id(self, qt_pos):
        window = self.vtk_widget.GetRenderWindow()
        x = int(qt_pos.x())
        y = int(window.GetSize()[1] - qt_pos.y() - 1)
        self.picker.Pick(x, y, 0, self.renderer)
        actor = self.picker.GetActor()
        if actor is None:
            return None
        return self.actor_lookup.get(actor.GetAddressAsString(""))

    def _update_hover_face(self, qt_pos):
        self._set_hover_face(self._pick_face_id(qt_pos))

    def _set_hover_face(self, face_id):
        if self.hover_face_id == face_id:
            return
        old_face = self.hover_face_id
        self.hover_face_id = face_id
        if old_face is not None:
            self._apply_actor_style(old_face)
        if self.hover_face_id is not None:
            self._apply_actor_style(self.hover_face_id)
        self._update_current_face_label()
        self.vtk_widget.GetRenderWindow().Render()

    def _assign_face_from_click(self, qt_pos):
        face_id = self._pick_face_id(qt_pos)
        if face_id is None:
            return
        self.selected_face_id = face_id
        instance_id = self._ensure_current_instance(self.current_category_id)
        self.set_face_instance(face_id, instance_id, record_history=True)
        self._select_face_in_list(face_id)
        self._update_current_face_label()

    def _select_face_in_list(self, face_id):
        for row in range(self.face_list.count()):
            item = self.face_list.item(row)
            if item.data(Qt.UserRole) == face_id:
                self.face_list.setCurrentRow(row)
                return

    def on_face_item_changed(self, current, previous):
        if current is None:
            self.selected_face_id = None
        else:
            self.selected_face_id = current.data(Qt.UserRole)
        if previous is not None:
            self._apply_actor_style(previous.data(Qt.UserRole))
        if self.selected_face_id is not None:
            self._apply_actor_style(self.selected_face_id)
        self._update_current_face_label()
        self.vtk_widget.GetRenderWindow().Render()

    def _remove_empty_instance(self, instance_id):
        if instance_id is None:
            return
        entry = self._get_current_entry()
        if entry is None:
            return
        if instance_id in entry.get("face_to_instance", {}).values():
            return
        removed_instance = self._get_instance_by_id(instance_id)
        entry["instances"] = [item for item in entry.get("instances", []) if item["instance_id"] != instance_id]
        if self.current_instance_id == instance_id:
            self.current_instance_id = None
        if removed_instance is not None:
            self._renumber_instances(removed_instance["category_id"])

    def set_face_instance(self, face_id, instance_id, record_history=True):
        record = self.face_lookup.get(face_id)
        if record is None:
            return
        old_instance_id = record.get("instance_id")
        if old_instance_id == instance_id:
            return
        entry = self._get_current_entry()
        if entry is None:
            return
        if record_history:
            action = {
                "face_id": face_id,
                "old_instance_id": old_instance_id,
                "new_instance_id": instance_id,
            }
            self.undo_stack.append(action)
            entry["undo_stack"] = list(self.undo_stack)

        record["instance_id"] = instance_id
        if instance_id is None:
            entry["face_to_instance"].pop(face_id, None)
            record["label_id"] = None
        else:
            entry["face_to_instance"][face_id] = instance_id
            instance = self._get_instance_by_id(instance_id)
            record["label_id"] = instance["category_id"] if instance is not None else None

        self._remove_empty_instance(old_instance_id)
        self._apply_actor_style(face_id)
        self._refresh_instance_list()
        self._refresh_face_list()
        self._update_summary()
        self._update_current_instance_label()
        self._update_current_face_label()
        self.vtk_widget.GetRenderWindow().Render()

    def set_face_label(self, face_id, label_id, record_history=True):
        if label_id is None:
            self.set_face_instance(face_id, None, record_history=record_history)
            return
        instance_id = self._ensure_current_instance(label_id)
        self.set_face_instance(face_id, instance_id, record_history=record_history)

    def clear_selected_face_label(self):
        if self.selected_face_id is None:
            QMessageBox.information(self, "提示", "请先在模型或右侧列表中选中一个面。")
            return
        self.set_face_instance(self.selected_face_id, None, record_history=True)

    def undo_last_action(self):
        if not self.undo_stack:
            QMessageBox.information(self, "提示", "当前没有可撤回的标注操作。")
            return
        action = self.undo_stack.pop()
        self.model_dataset[self.step_path]["undo_stack"] = list(self.undo_stack)
        self.set_face_instance(action["face_id"], action.get("old_instance_id"), record_history=False)
        self.selected_face_id = action["face_id"]
        self._select_face_in_list(action["face_id"])

    def clear_all_labels(self):
        if not self.face_records:
            return
        reply = QMessageBox.question(self, "确认", "确定要清空当前模型的全部标注吗？")
        if reply != QMessageBox.Yes:
            return
        self.undo_stack = []
        entry = self.model_dataset.get(self.step_path)
        if entry is not None:
            entry["undo_stack"] = []
            entry["labels"] = {}
            entry["face_to_instance"] = {}
            entry["instances"] = []
            entry["instance_counters"] = {}
        self.current_instance_id = None
        for record in self.face_records:
            record["label_id"] = None
            record["instance_id"] = None
            self._apply_actor_style(record["face_id"])
        self._refresh_instance_list()
        self._refresh_face_list()
        self._update_summary()
        self._update_current_instance_label()
        self._update_current_face_label()
        self.vtk_widget.GetRenderWindow().Render()

    def _on_similarity_changed(self, value):
        self.similarity_threshold = value / 100.0
        self.label_similarity.setText(f"当前阈值: {self.similarity_threshold:.2f}")

    def add_to_template(self):
        if self.selected_face_id is None:
            QMessageBox.warning(self, "提示", "请先在模型或右侧列表中选中一个已标注的面。")
            return
        record = self.face_lookup.get(self.selected_face_id)
        if record is None or record["label_id"] is None:
            QMessageBox.warning(self, "提示", "选中的面尚未标注类别，请先用左键单击给它标一个类型。")
            return
        instance_id = record.get("instance_id")
        if instance_id is None:
            QMessageBox.warning(self, "提示", "当前面还没有归属于实例，请先创建实例并分配面。")
            return
        instance_face_ids = [
            r["face_id"] for r in self.face_records if r.get("instance_id") == instance_id
        ]
        for face_id in instance_face_ids:
            self.template_face_ids.add(face_id)
            self._apply_actor_style(face_id)
        self._refresh_template_list()
        self.vtk_widget.GetRenderWindow().Render()

    def remove_from_template(self):
        if self.selected_face_id is None or self.selected_face_id not in self.template_face_ids:
            QMessageBox.warning(self, "提示", "请先选中一个已在模板列表中的面。")
            return
        record = self.face_lookup.get(self.selected_face_id)
        instance_id = record.get("instance_id") if record is not None else None
        if instance_id is None:
            self.template_face_ids.discard(self.selected_face_id)
            self._apply_actor_style(self.selected_face_id)
        else:
            for face_id in list(self.template_face_ids):
                target_record = self.face_lookup.get(face_id)
                if target_record is not None and target_record.get("instance_id") == instance_id:
                    self.template_face_ids.discard(face_id)
                    self._apply_actor_style(face_id)
        self._refresh_template_list()
        self.vtk_widget.GetRenderWindow().Render()

    def _on_template_item_clicked(self, item):
        face_id = item.data(Qt.UserRole)
        if face_id is not None:
            self.selected_face_id = face_id
            self._apply_actor_style(face_id)
            self._select_face_in_list(face_id)
            self._update_current_face_label()
            self.vtk_widget.GetRenderWindow().Render()

    def _refresh_template_list(self):
        self.template_list.clear()
        for fid in sorted(self.template_face_ids):
            record = self.face_lookup.get(fid)
            if record is None:
                continue
            label_id = record["label_id"]
            category = self.category_by_id.get(label_id, {})
            instance = self._get_instance_by_id(record.get("instance_id"))
            text = f"F{fid:04d} | {instance['instance_name'] if instance else '未分组'} | {category.get('display_name', '?')}"
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, fid)
            if label_id:
                item.setBackground(QBrush(QColor(*category.get("color", (200, 200, 200)))))
            self.template_list.addItem(item)

    def learn_and_label_similar(self):
        if not self.face_records:
            QMessageBox.warning(self, "提示", "请先加载 STEP 模型。")
            return
        if not self.template_face_ids:
            QMessageBox.warning(self, "提示", "模板实例为空。\n请先选中一个实例中的面，再点“加入模板面列表”。")
            return

        template_records = [self.face_lookup[fid] for fid in self.template_face_ids if fid in self.face_lookup]
        template_instance_ids = {record.get("instance_id") for record in template_records if record.get("instance_id") is not None}
        if len(template_instance_ids) != 1:
            QMessageBox.warning(self, "提示", "实例学习要求模板面全部来自同一个实例。")
            return

        template_instance_id = next(iter(template_instance_ids))
        template_instance = self._get_instance_by_id(template_instance_id)
        if template_instance is None:
            QMessageBox.warning(self, "提示", "模板实例无效。")
            return

        template_face_ids = [
            record["face_id"] for record in self.face_records if record.get("instance_id") == template_instance_id
        ]
        if not template_face_ids:
            QMessageBox.warning(self, "提示", "模板实例没有有效面。")
            return

        threshold = self.similarity_threshold
        reply = QMessageBox.question(
            self,
            "确认实例学习",
            f"将以实例 {template_instance['instance_name']} 为模板，\n"
            f"使用 {len(template_face_ids)} 个面搜索相似实例。\n"
            f"相似度阈值 = {threshold:.2f}\n\n是否继续？",
        )
        if reply != QMessageBox.Yes:
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            all_matches = find_similar_feature_groups(
                template_face_ids,
                self.face_records,
                self.face_lookup,
                similarity_threshold=threshold,
            )
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "实例学习失败", str(exc))
            return
        QApplication.restoreOverrideCursor()

        created_instances = 0
        applied_faces = 0
        skipped_groups = 0
        used_face_ids = set()
        target_category_id = template_instance["category_id"]

        for mapping in all_matches:
            matched_face_ids = sorted(set(mapping.values()))
            if not matched_face_ids:
                continue
            if any(face_id in used_face_ids for face_id in matched_face_ids):
                skipped_groups += 1
                continue
            if any(self.face_lookup[face_id].get("instance_id") is not None for face_id in matched_face_ids):
                skipped_groups += 1
                continue

            new_instance_id = self._create_instance(target_category_id, auto_select=False)
            if new_instance_id is None:
                skipped_groups += 1
                continue

            for face_id in matched_face_ids:
                self.set_face_instance(face_id, new_instance_id, record_history=True)
                used_face_ids.add(face_id)
                applied_faces += 1
            created_instances += 1

        self._refresh_instance_list()
        self._refresh_face_list()
        self._update_summary()
        self._update_current_instance_label()
        self.vtk_widget.GetRenderWindow().Render()

        if created_instances:
            QMessageBox.information(
                self,
                "实例学习完成",
                f"模板实例: {template_instance['instance_name']}\n"
                f"新建实例数: {created_instances}\n"
                f"标注面数: {applied_faces}\n"
                f"跳过分组数: {skipped_groups}",
            )
        else:
            QMessageBox.information(
                self,
                "未找到",
                "没有找到可新建为独立实例的相似结构。\n可能原因：阈值过高，或候选面已属于其他实例。",
            )

    def _apply_actor_style(self, face_id):
        record = self.face_lookup.get(face_id)
        if record is None:
            return
        actor = record["actor"]
        prop = actor.GetProperty()
        prop.LightingOn()

        label_id = record["label_id"]
        if label_id is None:
            base_color = rgb_to_float(DEFAULT_FACE_COLOR)
        else:
            base_color = rgb_to_float(self.category_by_id[label_id]["color"])

        prop.SetColor(*base_color)
        prop.SetOpacity(1.0)
        prop.SetAmbient(0.25)
        prop.SetDiffuse(0.75)
        prop.SetSpecular(0.05)
        prop.SetEdgeVisibility(False)
        prop.SetLineWidth(1.0)

        if face_id == self.selected_face_id:
            prop.SetEdgeVisibility(True)
            prop.SetEdgeColor(1.0, 0.6, 0.0)
            prop.SetLineWidth(3.5)
        elif face_id in self.template_face_ids:
            prop.SetEdgeVisibility(True)
            prop.SetEdgeColor(0.0, 1.0, 0.5)
            prop.SetLineWidth(2.8)
        elif face_id == self.hover_face_id:
            prop.SetEdgeVisibility(True)
            prop.SetEdgeColor(1.0, 0.85, 0.0)
            prop.SetLineWidth(2.5)

    def _refresh_face_list(self):
        previous_face_id = self.selected_face_id
        self.face_list.clear()
        labeled = [record for record in self.face_records if record["label_id"] is not None]
        labeled.sort(key=lambda item: item["face_id"])
        for record in labeled:
            category = self.category_by_id[record["label_id"]]
            instance = self._get_instance_by_id(record.get("instance_id"))
            center = record["center"]
            text = (
                f"F{record['face_id']:04d} | {instance['instance_name'] if instance else '未分组'} | "
                f"{category['display_name']} / {category['name']} | "
                f"C=({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})"
            )
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, record["face_id"])
            item.setBackground(QBrush(QColor(*category["color"])))
            self.face_list.addItem(item)
        if previous_face_id is not None:
            self._select_face_in_list(previous_face_id)

    def _update_summary(self):
        total = len(self.face_records)
        labeled = sum(1 for record in self.face_records if record["label_id"] is not None)
        instance_count = len(self._get_current_entry().get("instances", [])) if self._get_current_entry() else 0
        self.summary_label.setText(
            f"面统计:\n总面数: {total}\n已标注: {labeled}\n未标注: {total - labeled}\n实例数: {instance_count}"
        )

    def _update_current_face_label(self):
        face_id = self.selected_face_id if self.selected_face_id is not None else self.hover_face_id
        if face_id is None:
            self.current_face_label.setText("当前面: 无")
            return
        record = self.face_lookup.get(face_id)
        if record is None:
            self.current_face_label.setText("当前面: 无")
            return
        label_text = "未标注"
        instance_text = "无"
        if record["label_id"] is not None:
            category = self.category_by_id[record["label_id"]]
            label_text = f"{category['display_name']} / {category['name']}"
            instance = self._get_instance_by_id(record.get("instance_id"))
            if instance is not None:
                instance_text = instance["instance_name"]
        prefix = "选中面" if face_id == self.selected_face_id else "预选面"
        center = record["center"]
        self.current_face_label.setText(
            f"{prefix}: F{face_id:04d}\n"
            f"实例: {instance_text}\n"
            f"标签: {label_text}\n"
            f"中心: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})"
        )

    def _get_export_instances(self):
        entry = self._get_current_entry()
        if entry is None:
            return []
        result = []
        face_to_instance = entry.get("face_to_instance", {})
        valid_face_ids = set(self.face_lookup.keys())
        for instance in entry.get("instances", []):
            face_ids = sorted(
                face_id
                for face_id, instance_id in face_to_instance.items()
                if instance_id == instance["instance_id"] and face_id in valid_face_ids
            )
            if not face_ids:
                continue
            result.append(
                {
                    "instance_id": instance["instance_id"],
                    "instance_name": instance["instance_name"],
                    "category_id": instance["category_id"],
                    "face_ids": face_ids,
                }
            )
        return result

    def _on_num_views_changed(self, text):
        """视角数量combo变化时更新按钮文字"""
        n = int(text)
        self.btn_export_gray.setText(f"4. 导出{n}视角灰度图")

    def _get_current_view_directions(self):
        """根据combo选择返回对应的视角方向列表"""
        if self.combo_num_views.currentText() == "24":
            return get_24_face_normal_directions()
        return get_dodecahedron_view_directions()

    def _get_manifest_path(self, base_dir):
        """获取共享导出清单路径，放在输出目录上一级"""
        parent = os.path.dirname(base_dir)
        return os.path.join(parent, "export_manifest.json")

    def _load_manifest(self, manifest_path):
        """加载共享清单，不存在则返回None。自动将旧版 type 字段迁移为 types 列表。"""
        if not os.path.exists(manifest_path):
            return None
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            # 向后兼容：旧版 entries 用 "type": "xxx"，新版用 "types": ["xxx"]
            for img_name, entry in manifest.get("entries", {}).items():
                if "type" in entry and "types" not in entry:
                    entry["types"] = [entry.pop("type")]
                elif "type" in entry and "types" in entry:
                    entry.pop("type")
            return manifest
        except Exception:
            return None

    def _save_manifest(self, manifest_path, manifest):
        """保存共享清单"""
        with open(manifest_path, "w", encoding="utf-8") as f:
            std_json.dump(manifest, f, ensure_ascii=False, indent=2)

    def _validate_manifest(self, manifest, step_paths, start_id, num_views):
        """验证当前模型列表和编号参数是否与已有清单一致。
        返回 (ok, message)，ok=True 表示一致可以继续。
        """
        if manifest is None:
            return True, "无已有清单，将创建新清单"

        if manifest.get("start_id") != start_id:
            return False, (
                f"起始编号不一致：\n"
                f"已有清单起始编号 = {manifest.get('start_id')}\n"
                f"当前设置起始编号 = {start_id}\n"
                f"请改为一致，或删除 export_manifest.json 后重新全部导出。"
            )

        if manifest.get("num_views") != num_views:
            return False, (
                f"视角数量不一致：\n"
                f"已有清单视角数 = {manifest.get('num_views')}\n"
                f"当前设置视角数 = {num_views}\n"
                f"请改为一致，或删除 export_manifest.json 后重新全部导出。"
            )

        manifest_paths = [m["step_file"] for m in manifest.get("models", [])]
        current_paths = [normalize_json_path(p) for p in step_paths]

        if manifest_paths != current_paths:
            return False, (
                "模型列表与已有清单不一致：\n"
                "请保证模型导入顺序和数量不变，\n"
                "或删除 export_manifest.json 后重新全部导出。"
            )

        return True, "与已有清单一致"

    def _build_manifest(self, step_paths, start_id, num_views):
        """根据当前模型列表创建新的共享清单框架"""
        models = []
        for idx, path in enumerate(step_paths, start=1):
            models.append({
                "index": idx,
                "step_file": normalize_json_path(path),
                "model_name": os.path.splitext(os.path.basename(path))[0],
            })
        return {
            "start_id": start_id,
            "num_views": num_views,
            "models": models,
            "entries": {},
            "_version": 2,
        }

    def reset_camera(self):
        if not self.face_records:
            return
        self.renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()

    def _set_camera_for_view(self, camera, center, dist, direction, aspect_ratio):
        viewup = get_viewup(direction)
        camera.SetFocalPoint(*center)
        camera.SetPosition(
            center[0] + direction[0] * dist,
            center[1] + direction[1] * dist,
            center[2] + direction[2] * dist,
        )
        camera.SetViewUp(*viewup)
        camera.SetParallelScale(
            max(
                get_parallel_scale(
                    self.model_bounds,
                    center,
                    direction,
                    viewup,
                    aspect_ratio=aspect_ratio,
                    margin=1.10,
                ),
                0.01,
            )
        )

    def build_coco_dataset(self, image_output_dir):
        if not self.step_path or not self.face_records:
            return None

        render_window = self.vtk_widget.GetRenderWindow()
        old_size = render_window.GetSize()
        old_camera = vtk.vtkCamera()
        old_camera.DeepCopy(self.renderer.GetActiveCamera())
        old_background = self.renderer.GetBackground()
        old_selected = self.selected_face_id
        old_hover = self.hover_face_id

        export_instances = self._get_export_instances()
        output_data = {
            "images": [],
            "annotations": [],
            "categories": [{"id": c["id"], "name": c["name"]} for c in self.categories],
        }

        self.selected_face_id = None
        self.hover_face_id = None
        for record in self.face_records:
            self._apply_actor_style(record["face_id"])

        render_window.SetSize(OUTPUT_WIDTH, OUTPUT_HEIGHT)
        camera = self.renderer.GetActiveCamera()
        center = (
            (self.model_bounds[0] + self.model_bounds[1]) / 2,
            (self.model_bounds[2] + self.model_bounds[3]) / 2,
            (self.model_bounds[4] + self.model_bounds[5]) / 2,
        )
        max_dim = max(
            self.model_bounds[1] - self.model_bounds[0],
            self.model_bounds[3] - self.model_bounds[2],
            self.model_bounds[5] - self.model_bounds[4],
        )
        if max_dim < 0.001:
            max_dim = 1.0

        camera.ParallelProjectionOn()
        dist = max_dim * 3
        aspect_ratio = OUTPUT_WIDTH / OUTPUT_HEIGHT
        annotation_id = 1

        try:
            render_window.Render()
            render_window.Render()
            for image_index, direction in enumerate(self._get_current_view_directions(), start=1):
                self._set_camera_for_view(camera, center, dist, direction, aspect_ratio)
                self.renderer.ResetCameraClippingRange()
                render_window.Render()

                image_name = f"dodecahedron_view_{image_index:02d}.png"
                image_path = os.path.join(image_output_dir, image_name)
                color_rgb = capture_render_window_rgb(render_window)
                write_png_rgb(image_path, color_rgb)

                output_data["images"].append(
                    {
                        "id": int(image_index),
                        "file_name": image_name,
                        "width": int(OUTPUT_WIDTH),
                        "height": int(OUTPUT_HEIGHT),
                    }
                )

                if not export_instances:
                    continue

                self.renderer.SetBackground(0.0, 0.0, 0.0)
                for instance_item in export_instances:
                    for record in self.face_records:
                        actor = record["actor"]
                        prop = actor.GetProperty()
                        actor.SetVisibility(True)
                        prop.SetOpacity(1.0)
                        prop.SetEdgeVisibility(False)
                        prop.SetLineWidth(1.0)
                        prop.LightingOff()
                        prop.SetAmbient(1.0)
                        prop.SetDiffuse(0.0)
                        prop.SetSpecular(0.0)
                        if record["face_id"] in instance_item["face_ids"]:
                            prop.SetColor(1.0, 1.0, 1.0)
                        else:
                            prop.SetColor(0.0, 0.0, 0.0)

                    self.renderer.ResetCameraClippingRange()
                    render_window.Render()
                    mask_rgb = capture_render_window_rgb(render_window)
                    binary_mask = np.where(np.any(mask_rgb > 0, axis=2), 255, 0).astype(np.uint8)
                    annotation = build_coco_annotation(
                        binary_mask,
                        image_id=image_index,
                        category_id=instance_item["category_id"],
                        annotation_id=annotation_id,
                    )
                    if annotation is not None:
                        annotation["instance_name"] = instance_item["instance_name"]
                        annotation["instance_id_3d"] = instance_item["instance_id"]
                        output_data["annotations"].append(annotation)
                        annotation_id += 1

                self.renderer.SetBackground(*old_background)
                for record in self.face_records:
                    self._apply_actor_style(record["face_id"])
                self.renderer.ResetCameraClippingRange()
                render_window.Render()
        finally:
            render_window.SetSize(old_size[0], old_size[1])
            camera.DeepCopy(old_camera)
            self.renderer.SetBackground(*old_background)
            self.selected_face_id = old_selected
            self.hover_face_id = old_hover
            for record in self.face_records:
                self._apply_actor_style(record["face_id"])
            self.renderer.ResetCameraClippingRange()
            render_window.Render()

        return output_data

    def build_gray_view_images(self, image_output_dir):
        """Export 12 grayscale views using the same processing flow as render_dodecahedron_views.py."""
        if not self.step_path or not self.face_records:
            return 0

        render_window = self.vtk_widget.GetRenderWindow()
        old_size = render_window.GetSize()
        old_camera = vtk.vtkCamera()
        old_camera.DeepCopy(self.renderer.GetActiveCamera())
        old_background = self.renderer.GetBackground()
        old_selected = self.selected_face_id
        old_hover = self.hover_face_id

        feature_edge_actor = build_feature_edge_actor(self.face_records)
        self.selected_face_id = None
        self.hover_face_id = None

        render_window.SetSize(OUTPUT_WIDTH, OUTPUT_HEIGHT)
        camera = self.renderer.GetActiveCamera()
        center = (
            (self.model_bounds[0] + self.model_bounds[1]) / 2,
            (self.model_bounds[2] + self.model_bounds[3]) / 2,
            (self.model_bounds[4] + self.model_bounds[5]) / 2,
        )
        max_dim = max(
            self.model_bounds[1] - self.model_bounds[0],
            self.model_bounds[3] - self.model_bounds[2],
            self.model_bounds[5] - self.model_bounds[4],
        )
        if max_dim < 0.001:
            max_dim = 1.0

        camera.ParallelProjectionOn()
        dist = max_dim * 3
        aspect_ratio = OUTPUT_WIDTH / OUTPUT_HEIGHT

        try:
            self.renderer.SetBackground(1.0, 1.0, 1.0)
            self.renderer.AddActor(feature_edge_actor)
            for record in self.face_records:
                actor = record["actor"]
                prop = actor.GetProperty()
                prop.LightingOn()
                prop.SetColor(*rgb_to_float(DEFAULT_FACE_COLOR))
                prop.SetOpacity(1.0)
                prop.SetAmbient(0.25)
                prop.SetDiffuse(0.75)
                prop.SetSpecular(0.05)
                prop.SetEdgeVisibility(False)
                prop.SetLineWidth(1.0)
            render_window.Render()
            render_window.Render()
            for image_index, direction in enumerate(self._get_current_view_directions(), start=1):
                self._set_camera_for_view(camera, center, dist, direction, aspect_ratio)
                self.renderer.ResetCameraClippingRange()
                render_window.Render()

                gray_name = f"dodecahedron_view_{image_index:02d}.png"
                gray_path = os.path.join(image_output_dir, gray_name)
                rgb_image = capture_render_window_rgb(render_window)
                gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
                gray_image = enhance_gray_image(gray_image)
                write_png_gray(gray_path, gray_image)
        finally:
            self.renderer.RemoveActor(feature_edge_actor)
            render_window.SetSize(old_size[0], old_size[1])
            camera.DeepCopy(old_camera)
            self.renderer.SetBackground(*old_background)
            self.selected_face_id = old_selected
            self.hover_face_id = old_hover
            for record in self.face_records:
                self._apply_actor_style(record["face_id"])
            self.renderer.ResetCameraClippingRange()
            render_window.Render()

        return len(self._get_current_view_directions())

    def load_annotation_json(self):
        json_path, _ = QFileDialog.getOpenFileName(
            self,
            "选择标注 JSON",
            "",
            "JSON Files (*.json)",
        )
        if not json_path:
            return

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                payload = std_json.load(f)
        except Exception as exc:
            QMessageBox.critical(self, "加载失败", f"无法读取 JSON:\n{exc}")
            return

        if "faces" not in payload and "annotations" in payload and "images" in payload:
            QMessageBox.warning(
                self,
                "提示",
                "当前导出的 COCO JSON 主要用于数据集训练，不能直接回载成三维面的逐面标签。",
            )
            return

        step_path = payload.get("step_file", "")
        if not self.step_path:
            if step_path and os.path.exists(step_path):
                self.add_step_files([step_path])
                self.model_list.setCurrentRow(self.step_paths.index(step_path))
            else:
                QMessageBox.warning(self, "提示", "请先加载与该 JSON 对应的 STEP 模型。")
                return
        elif step_path and os.path.normcase(step_path) != os.path.normcase(self.step_path):
            reply = QMessageBox.question(
                self,
                "模型不一致",
                "JSON 对应的 STEP 文件与当前模型不同，是否切换到 JSON 里的 STEP 文件？",
            )
            if reply == QMessageBox.Yes and os.path.exists(step_path):
                self.add_step_files([step_path])
                self.model_list.setCurrentRow(self.step_paths.index(step_path))
            else:
                return

        entry = self.model_dataset.get(self.step_path)
        if entry is None:
            return
        entry["labels"] = {}
        entry["face_to_instance"] = {}
        entry["instances"] = []
        entry["instance_counters"] = {}
        entry["undo_stack"] = []
        self.undo_stack = []

        if "instances" in payload and "face_to_instance" in payload:
            entry["instances"] = payload.get("instances", [])
            entry["face_to_instance"] = {
                int(face_id): instance_id for face_id, instance_id in payload.get("face_to_instance", {}).items()
            }
        else:
            faces = payload.get("faces", [])
            for item in faces:
                face_id = item.get("face_id")
                label_id = item.get("type_id")
                if face_id in self.face_lookup and label_id is not None:
                    instance_id = self._create_instance(label_id, auto_select=False)
                    entry["face_to_instance"][face_id] = instance_id

        for record in self.face_records:
            record["instance_id"] = entry["face_to_instance"].get(record["face_id"])
            instance = self._get_instance_by_id(record["instance_id"])
            record["label_id"] = instance["category_id"] if instance is not None else None
            self._apply_actor_style(record["face_id"])

        self._renumber_instances()
        self._refresh_instance_list()
        self._refresh_face_list()
        self._update_summary()
        self._update_current_instance_label()
        self._update_current_face_label()
        self.vtk_widget.GetRenderWindow().Render()
        QMessageBox.information(self, "成功", "标注 JSON 已加载。")

    def export_inference_views(self):
        """生成语义染色图 + 独立着色图（调用 generate_inference_views 模块）"""
        if not self.step_paths:
            QMessageBox.warning(self, "提示", "请先导入至少一个 STEP 模型。")
            return

        output_dir = QFileDialog.getExistingDirectory(self, "选择输出目录（将在此目录下创建 semantic_views 和 unique_views 文件夹）")
        if not output_dir:
            return

        semantic_dir = os.path.join(output_dir, "semantic_views")
        unique_dir = os.path.join(output_dir, "unique_views")
        os.makedirs(semantic_dir, exist_ok=True)
        os.makedirs(unique_dir, exist_ok=True)

        # 导入 generate_inference_views 模块的关键函数
        tool_dir = os.path.dirname(os.path.abspath(__file__))
        if tool_dir not in sys.path:
            sys.path.insert(0, tool_dir)
        import colorsys
        import generate_inference_views as giv

        num_views = int(self.combo_num_views.currentText())
        if num_views not in (12, 24):
            QMessageBox.warning(self, "提示",
                f"当前仅支持 12 或 24 视角。\n"
                f"视角数量已自动切换为 12。")
            self.combo_num_views.setCurrentText("12")
            num_views = 12

        # 获取视角方向列表
        if num_views == 24:
            import color_step_24face_edge as c24
            view_directions = c24.get_icositetrahedron_face_normal_directions()
        else:
            view_directions = giv.get_dodecahedron_view_directions()

        global_img_id = self.spin_start_id.value()
        start_id = self.spin_start_id.value()
        export_count = 0
        image_index_map = {}

        # 共享清单：验证一致性
        manifest_path = self._get_manifest_path(output_dir)
        manifest = self._load_manifest(manifest_path)
        ok, msg = self._validate_manifest(manifest, self.step_paths, start_id, num_views)
        if not ok:
            QMessageBox.warning(self, "导出中断", msg)
            return
        if manifest is None:
            manifest = self._build_manifest(self.step_paths, start_id, num_views)

        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.status_label.setText("正在生成语义染色图 + 独立着色图...")
        QApplication.processEvents()

        try:
            for step_idx, step_path in enumerate(self.step_paths, start=1):
                basename = os.path.splitext(os.path.basename(step_path))[0]
                model_name = f"model_{step_idx:04d}_{basename}"
                self.status_label.setText(f"[{step_idx}/{len(self.step_paths)}] 处理: {basename}")
                QApplication.processEvents()

                # 1. 读取STEP，提取面/边几何信息
                data = giv.load_step_and_compute_data(step_path)
                if data is None:
                    continue

                # 2. 计算独立着色（黄金角HSV）
                unique_colors = giv.compute_unique_colors(data["face_count"], view_index=1)

                # 3. 计算12个视角的着色方案（基于独立颜色做色相偏移）
                views_colors = []
                for v in range(1, num_views + 1):
                    hue_offset = (v - 1) * giv.HUE_OFFSET_STEP
                    view_palette = []
                    for r, g, b in unique_colors:
                        h, s, val = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
                        h = (h + hue_offset / 360.0) % 1.0
                        nr, ng, nb = colorsys.hsv_to_rgb(h, s, val)
                        view_palette.append((int(nr * 255), int(ng * 255), int(nb * 255)))
                    views_colors.append(view_palette)

                # 4. 生成文件路径（全局编号）
                semantic_paths = []
                unique_paths = []
                for view_idx in range(1, num_views + 1):
                    image_name = f"{global_img_id:06d}.png"
                    semantic_paths.append(os.path.join(semantic_dir, image_name))
                    unique_paths.append(os.path.join(unique_dir, image_name))
                    image_index_map[image_name] = {
                        "step_file": normalize_json_path(step_path),
                        "model_name": basename,
                        "view_index": view_idx,
                        "num_views": num_views,
                    }
                    global_img_id += 1

                # 5. 渲染
                giv.render_12_views(
                    data["faces"], data["edges"], data["bounds"],
                    semantic_paths, unique_paths, views_colors,
                    view_directions=view_directions,
                )
                # 写入共享清单条目（追加类型，不覆盖）
                for view_idx in range(1, num_views + 1):
                    img_id = global_img_id - num_views + view_idx - 1
                    img_name = f"{img_id:06d}.png"
                    entry = manifest["entries"].get(img_name, {})
                    entry["model_index"] = step_idx
                    entry["view_index"] = view_idx
                    entry["num_views"] = num_views
                    types = entry.get("types", [])
                    if "semantic+unique" not in types:
                        types.append("semantic+unique")
                    entry["types"] = types
                    manifest["entries"][img_name] = entry
                export_count += 1

            index_map_path = os.path.join(output_dir, "image_index_map.json")
            with open(index_map_path, "w", encoding="utf-8") as f:
                std_json.dump(image_index_map, f, ensure_ascii=False, indent=2)

            self._save_manifest(manifest_path, manifest)
        except Exception as exc:
            QMessageBox.critical(self, "导出失败", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()
            self.status_label.setText("就绪")

        total_images = export_count * num_views * 2
        QMessageBox.information(
            self,
            "导出成功",
            f"已完成 {export_count} 个模型的导出。\n"
            f"语义染色图: {export_count * num_views} 张（{semantic_dir}）\n"
            f"独立着色图: {export_count * num_views} 张（{unique_dir}）\n"
            f"合计: {total_images} 张图片",
        )

    def export_gb_encoded_views(self):
        """导出GB编码染色图（R=面类型，G=face_id低8位，B=face_id高8位），支持12/24视角"""
        if not self.step_paths:
            QMessageBox.warning(self, "提示", "请先导入至少一个 STEP 模型。")
            return

        output_dir = QFileDialog.getExistingDirectory(self, "选择输出目录（将在此目录下创建 encoded_views 文件夹）")
        if not output_dir:
            return

        encoded_dir = os.path.join(output_dir, "encoded_views")
        os.makedirs(encoded_dir, exist_ok=True)

        # 导入编码模块
        tool_dir = os.path.dirname(os.path.abspath(__file__))
        if tool_dir not in sys.path:
            sys.path.insert(0, tool_dir)
        import generate_encoded_face_views as gefv

        num_views = int(self.combo_num_views.currentText())
        if num_views not in (12, 24):
            QMessageBox.warning(self, "提示", f"当前仅支持 12 或 24 视角。\n视角数量已自动切换为 12。")
            self.combo_num_views.setCurrentText("12")
            num_views = 12

        # 获取视角方向
        if num_views == 24:
            import color_step_24face_edge as c24
            view_directions = c24.get_icositetrahedron_face_normal_directions()
        else:
            view_directions = gefv.get_dodecahedron_view_directions()

        global_img_id = self.spin_start_id.value()
        start_id = self.spin_start_id.value()
        export_count = 0

        # 共享清单：验证一致性
        manifest_path = self._get_manifest_path(output_dir)
        manifest = self._load_manifest(manifest_path)
        ok, msg = self._validate_manifest(manifest, self.step_paths, start_id, num_views)
        if not ok:
            QMessageBox.warning(self, "导出中断", msg)
            return
        if manifest is None:
            manifest = self._build_manifest(self.step_paths, start_id, num_views)

        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.status_label.setText("正在生成GB编码染色图...")
        QApplication.processEvents()

        face_encoding_map = {
            "encoding_rule": {
                "background_rgb": [255, 255, 255],
                "R": "type base + area offset: TYPE_GAP=51, area has 51 levels",
                "G_B": "face_id adaptive KxK grid mapping, K=ceil(sqrt(num_faces))",
                "decode": "face_id is recovered from model.gb_mapping by (G,B); type_id and area_ratio are recovered from R",
                "type_id": "0=Plane, 1=Cylinder, 2=Cone, 3=Sphere, 4=Other",
            },
            "models": {},
        }
        camera_views = {}

        try:
            for step_idx, step_path in enumerate(self.step_paths, start=1):
                basename = os.path.splitext(os.path.basename(step_path))[0]
                model_name = f"model_{step_idx:04d}_{basename}"
                self.status_label.setText(f"[{step_idx}/{len(self.step_paths)}] 处理: {basename}")
                QApplication.processEvents()

                # 1. 读取STEP，提取面信息并编码RGB
                data = gefv.load_step_and_encode_faces(step_path)
                if data is None:
                    continue

                # 2. 生成文件路径（全局编号）
                image_paths = []
                image_names = []
                for _ in range(num_views):
                    image_name = f"{global_img_id:06d}.png"
                    image_names.append(image_name)
                    image_paths.append(os.path.join(encoded_dir, image_name))
                    global_img_id += 1

                # 3. 渲染GB编码视图
                camera_records = gefv.render_encoded_views(
                    data["faces"], data["bounds"], image_paths,
                    view_directions=view_directions,
                )

                # 4. 记录编码映射
                faces_json = {}
                gb_reverse = {
                    (int(v["G"]), int(v["B"])): int(fid)
                    for fid, v in data.get("gb_mapping", {}).items()
                }
                for item in data["faces"]:
                    r, g, b = item["encoded_rgb"]
                    faces_json[str(item["face_id"])] = {
                        "face_type": item["face_type"],
                        "type_id": item.get("type_id"),
                        "area": item["area"],
                        "area_ratio": item.get("area_ratio"),
                        "encoded_rgb": [int(r), int(g), int(b)],
                        "r_type_area": int(r),
                        "g_grid": int(g),
                        "b_grid": int(b),
                        "decoded_face_id": gb_reverse.get((int(g), int(b))),
                    }

                face_encoding_map["models"][model_name] = {
                    "model_name": basename,
                    "step_file": normalize_json_path(step_path),
                    "max_area": data["max_area"],
                    "max_face_id": data["max_face_id"],
                    "face_count": data["face_count"],
                    "encoder_config": data.get("encoder_config", {}),
                    "gb_mapping": data.get("gb_mapping", {}),
                    "images": image_names,
                    "faces": faces_json,
                }

                for image_name, camera_record in zip(image_names, camera_records):
                    camera_views[image_name] = {
                        "model_name": basename,
                        "step_file": normalize_json_path(step_path),
                        **camera_record,
                    }

                # 写入共享清单条目（追加类型，不覆盖）
                for view_idx in range(1, num_views + 1):
                    img_id = global_img_id - num_views + view_idx - 1
                    img_name = f"{img_id:06d}.png"
                    entry = manifest["entries"].get(img_name, {})
                    entry["model_index"] = step_idx
                    entry["view_index"] = view_idx
                    entry["num_views"] = num_views
                    types = entry.get("types", [])
                    if "gb_encoded" not in types:
                        types.append("gb_encoded")
                    entry["types"] = types
                    manifest["entries"][img_name] = entry
                export_count += 1

            # 保存映射文件
            face_map_path = os.path.join(output_dir, "face_encoding_map.json")
            camera_path = os.path.join(output_dir, "camera_views.json")
            with open(face_map_path, "w", encoding="utf-8") as f:
                std_json.dump(face_encoding_map, f, ensure_ascii=False, indent=2)
            with open(camera_path, "w", encoding="utf-8") as f:
                std_json.dump(camera_views, f, ensure_ascii=False, indent=2)

            self._save_manifest(manifest_path, manifest)
        except Exception as exc:
            QMessageBox.critical(self, "导出失败", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()
            self.status_label.setText("就绪")

        total_images = export_count * num_views
        QMessageBox.information(
            self,
            "导出成功",
            f"已完成 {export_count} 个模型的导出。\n"
            f"GB编码染色图: {total_images} 张（{encoded_dir}）\n"
            f"face_encoding_map.json: {face_map_path}\n"
            f"camera_views.json: {camera_path}",
        )

    def export_gray_package(self):
        if not self.step_paths:
            QMessageBox.warning(self, "提示", "请先导入至少一个 STEP 模型。")
            return

        output_dir = QFileDialog.getExistingDirectory(self, "选择保存灰度图的文件夹")
        if not output_dir:
            return

        original_path = self.step_path
        export_count = 0
        image_count = 0
        try:
            for step_path in self.step_paths:
                if os.path.normcase(step_path) != os.path.normcase(self.step_path):
                    self.load_step_file(step_path)

                base_name = os.path.splitext(os.path.basename(self.step_path))[0]
                gray_dir = os.path.join(output_dir, f"{base_name}_gray_views")
                os.makedirs(gray_dir, exist_ok=True)
                image_count += self.build_gray_view_images(gray_dir)
                export_count += 1
        except Exception as exc:
            QMessageBox.critical(self, "灰度图导出失败", str(exc))
            return
        finally:
            if original_path and os.path.normcase(original_path) != os.path.normcase(self.step_path):
                self.load_step_file(original_path)

        QMessageBox.information(
            self,
            "灰度图导出成功",
            f"已完成 {export_count} 个零件的灰度图导出，共生成 {image_count} 张 8位灰度图。",
        )

    def export_global_gray_coco(self):
        if not self.model_dataset:
            QMessageBox.warning(self, "提示", "请先加载至少一个 STEP 模型。")
            return

        out_dir = QFileDialog.getExistingDirectory(self, "选择数据集保存目录 (将在此目录下创建 images 文件夹和 annotations.json)")
        if not out_dir:
            return

        images_dir = os.path.join(out_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

        coco_data = {
            "images": [],
            "annotations": [],
            "categories": self.categories
        }

        global_img_id = self.spin_start_id.value()
        global_ann_id = 1
        start_id = self.spin_start_id.value()

        # 共享清单：验证一致性
        manifest_path = self._get_manifest_path(out_dir)
        manifest = self._load_manifest(manifest_path)
        num_views_for_manifest = len(self._get_current_view_directions())
        ok, msg = self._validate_manifest(manifest, self.step_paths, start_id, num_views_for_manifest)
        if not ok:
            QMessageBox.warning(self, "导出中断", msg)
            return
        if manifest is None:
            manifest = self._build_manifest(self.step_paths, start_id, num_views_for_manifest)

        # Remember current state
        original_path = self.step_path

        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.status_label.setText("正在导出全局 COCO 灰度数据集...")
        QApplication.processEvents()

        try:
            dirs = self._get_current_view_directions()

            for step_path in self.step_paths:
                # 按左侧模型列表顺序导出，保证和其他导出功能编号一致
                self.load_step_file(step_path)
                QApplication.processEvents()

                # 1. Convert shape to STL
                BRepMesh_IncrementalMesh(self.shape, 0.1)
                fd, temp_stl = tempfile.mkstemp(suffix=".stl")
                os.close(fd)
                writer = StlAPI_Writer()
                writer.Write(self.shape, temp_stl)
                mesh = pv.read(temp_stl)

                # Feature edges for gray render
                feature_edges = mesh.extract_feature_edges(
                    boundary_edges=True,
                    feature_edges=True,
                    manifold_edges=False,
                    non_manifold_edges=True,
                    feature_angle=25,
                )

                bb = mesh.bounds
                cx = (bb[0] + bb[1]) / 2
                cy = (bb[2] + bb[3]) / 2
                cz = (bb[4] + bb[5]) / 2
                focal = (cx, cy, cz)
                max_dim = max(bb[1] - bb[0], bb[3] - bb[2], bb[5] - bb[4])
                if max_dim < 0.001:
                    max_dim = 1.0
                dist = max_dim * 3

                export_instances = self._get_export_instances()
                face_meshes = {}
                for r in self.face_records:
                    if r["label_id"] is None:
                        continue
                    fd_f, temp_stl_f = tempfile.mkstemp(suffix=".stl")
                    os.close(fd_f)
                    writer.Write(r["face"], temp_stl_f)
                    face_meshes[r["face_id"]] = (pv.read(temp_stl_f), temp_stl_f)

                plotter = pv.Plotter(off_screen=True, window_size=[OUTPUT_WIDTH, OUTPUT_HEIGHT])
                plotter.disable_anti_aliasing()
                plotter.camera.SetParallelProjection(True)

                # Fix for the first frame being blank in off-screen rendering
                # Create a temporary file and save a dummy screenshot to force the complete rendering pipeline
                plotter.add_mesh(mesh, color='white')
                dummy_png = os.path.join(images_dir, "dummy_init.png")
                plotter.screenshot(dummy_png)
                if os.path.exists(dummy_png):
                    os.remove(dummy_png)
                plotter.clear()

                for i, d in enumerate(dirs):
                    img_name = f"{global_img_id:06d}.png"
                    img_path = os.path.join(images_dir, img_name)

                    cam_pos = (cx + d[0] * dist, cy + d[1] * dist, cz + d[2] * dist)
                    viewup = get_viewup(d)
                    parallel_scale = get_parallel_scale(
                        bb, focal, d, viewup, aspect_ratio=OUTPUT_WIDTH / OUTPUT_HEIGHT, margin=1.10
                    )

                    # --- Render Gray Image ---
                    plotter.clear()
                    plotter.set_background('white')
                    plotter.add_mesh(mesh, color='lightgray', smooth_shading=False, lighting=True)
                    if feature_edges.n_points > 0:
                        plotter.add_mesh(feature_edges, color='black', line_width=3.0, lighting=False)
                    plotter.enable_lightkit()
                    plotter.camera_position = [cam_pos, focal, viewup]
                    plotter.camera.SetParallelScale(max(parallel_scale, 0.01))
                    plotter.render()

                    temp_png = os.path.join(images_dir, "temp_gray.png")
                    plotter.screenshot(temp_png)

                    # Post-process gray image
                    img_bgr = cv2.imdecode(np.fromfile(temp_png, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if img_bgr is not None:
                        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                        gray = enhance_gray_image(gray)
                        cv2.imencode('.png', gray)[1].tofile(img_path)
                    if os.path.exists(temp_png):
                        os.remove(temp_png)

                    # Add image info to COCO
                    coco_data["images"].append({
                        "id": global_img_id,
                        "file_name": img_name,
                        "width": OUTPUT_WIDTH,
                        "height": OUTPUT_HEIGHT
                    })

                    # --- Render Masks ---
                    plotter.clear()
                    plotter.set_background('black')
                    plotter.add_mesh(mesh, color='black', smooth_shading=False, lighting=False)

                    for instance_item in export_instances:
                        actors = []
                        for face_id in instance_item["face_ids"]:
                            mesh_info = face_meshes.get(face_id)
                            if mesh_info is None:
                                continue
                            f_mesh, _ = mesh_info
                            actors.append(plotter.add_mesh(f_mesh, color='white', smooth_shading=False, lighting=False))
                        plotter.render()

                        temp_mask = os.path.join(images_dir, "temp_mask.png")
                        plotter.screenshot(temp_mask)
                        for actor in actors:
                            plotter.remove_actor(actor)

                        mask_img = cv2.imdecode(np.fromfile(temp_mask, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                        if mask_img is not None:
                            _, bin_mask = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)
                            contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            
                            segmentation = []
                            for contour in contours:
                                if contour.shape[0] >= 3:
                                    segmentation.append(contour.flatten().tolist())
                            
                            if segmentation:
                                x, y, w, h = cv2.boundingRect(bin_mask)
                                area = float(cv2.contourArea(contours[0])) if contours else 0.0
                                for cnt in contours[1:]:
                                    area += float(cv2.contourArea(cnt))

                                coco_data["annotations"].append({
                                    "id": global_ann_id,
                                    "image_id": global_img_id,
                                    "category_id": instance_item["category_id"],
                                    "instance_name": instance_item["instance_name"],
                                    "instance_id_3d": instance_item["instance_id"],
                                    "segmentation": segmentation,
                                    "area": area,
                                    "bbox": [int(x), int(y), int(w), int(h)],
                                    "iscrowd": 0
                                })
                                global_ann_id += 1
                        if os.path.exists(temp_mask):
                            os.remove(temp_mask)

                    global_img_id += 1

                plotter.close()
                os.remove(temp_stl)
                for _, temp_stl_f in face_meshes.values():
                    os.remove(temp_stl_f)

            # Write JSON
            json_path = os.path.join(out_dir, "annotations.json")
            with open(json_path, "w", encoding="utf-8") as f:
                std_json.dump(coco_data, f, ensure_ascii=False, indent=2)

            self._save_manifest(manifest_path, manifest)

        except Exception as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "导出失败", str(exc))
            self.status_label.setText("导出异常")
            return

        # Restore context
        if original_path and original_path in self.model_dataset:
            self.load_step_file(original_path)
            self.template_face_ids.clear()
            self._refresh_template_list()

        QApplication.restoreOverrideCursor()
        self.status_label.setText("全局 COCO 灰度数据集导出完成")
        QMessageBox.information(
            self,
            "导出成功",
            f"成功导出到 {out_dir}\n"
            f"共生成 {len(coco_data['images'])} 张灰度图片\n"
            f"共生成 {len(coco_data['annotations'])} 个标注记录"
        )

    def export_coco_mask_labels(self):
        """导出多灰度值掩码图 + class_map.json。

        功能说明：
        - 对数据集中的每个 STEP 模型，从12个视角生成一张多灰度值掩码图
        - 每个实例分配一个唯一的灰度值 (0, 1, 2, ...)
        - 未标注的面为背景（灰度值 255，白色）
        - 同时输出 class_map.json，记录每个掩码图中实例灰度值对应的类别

        输出目录结构：
            output_dir/
                masks/          ← 每视角一张多灰度值掩码图 PNG
                class_map.json  ← 掩码图 → {灰度值: 类别ID} 的映射
        """
        # ============================================================
        # 步骤1: 前置检查 — 确认已加载模型数据
        # ============================================================
        if not self.model_dataset:
            QMessageBox.warning(self, "提示", "请先加载至少一个 STEP 模型。")
            return

        # ============================================================
        # 步骤2: 选择输出目录，创建 masks 子文件夹
        # ============================================================
        out_dir = QFileDialog.getExistingDirectory(
            self, "选择掩码标签保存目录"
        )
        if not out_dir:
            return

        masks_dir = os.path.join(out_dir, "masks")
        os.makedirs(masks_dir, exist_ok=True)

        # ============================================================
        # 步骤3: 初始化 class_map 数据结构
        # class_map: { "000001.png": { "0": 0, "1": 1, ... }, ... }
        # 键 = 掩码图文件名, 值 = {灰度值(字符串): category_id}
        # ============================================================
        class_map = {}

        # 图片 ID 起始值
        global_img_id = self.spin_start_id.value()
        start_id = self.spin_start_id.value()

        # 共享清单：验证一致性
        manifest_path = self._get_manifest_path(out_dir)
        manifest = self._load_manifest(manifest_path)
        num_views_for_manifest = len(self._get_current_view_directions())
        ok, msg = self._validate_manifest(manifest, self.step_paths, start_id, num_views_for_manifest)
        if not ok:
            QMessageBox.warning(self, "导出中断", msg)
            return
        if manifest is None:
            manifest = self._build_manifest(self.step_paths, start_id, num_views_for_manifest)

        # 保存当前加载的模型路径，导出结束后恢复
        original_path = self.step_path

        # 显示等待光标，更新状态栏
        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.status_label.setText("正在导出掩码图标签...")
        QApplication.processEvents()

        try:
            # ============================================================
            # 步骤4: 获取视角方向
            # ============================================================
            dirs = self._get_current_view_directions()

            # ============================================================
            # 步骤5: 遍历数据集中的每个 STEP 模型
            # ============================================================
            image_index_map = {}

            for step_path in self.step_paths:
                # 5.1 按左侧模型列表顺序切换模型，保证和多视角图导出编号一致
                self.load_step_file(step_path)
                QApplication.processEvents()

                # 5.2 将 STEP 形状转换为 STL 格式（PyVista 可读取）
                BRepMesh_IncrementalMesh(self.shape, 0.1)
                fd, temp_stl = tempfile.mkstemp(suffix=".stl")
                os.close(fd)
                writer = StlAPI_Writer()
                writer.Write(self.shape, temp_stl)
                mesh = pv.read(temp_stl)

                # 5.3 计算模型包围盒，用于设置相机参数
                bb = mesh.bounds
                cx = (bb[0] + bb[1]) / 2
                cy = (bb[2] + bb[3]) / 2
                cz = (bb[4] + bb[5]) / 2
                focal = (cx, cy, cz)
                max_dim = max(bb[1] - bb[0], bb[3] - bb[2], bb[5] - bb[4])
                if max_dim < 0.001:
                    max_dim = 1.0
                dist = max_dim * 3

                # 5.4 获取当前模型的标注实例信息
                export_instances = self._get_export_instances()
                if len(export_instances) > 255:
                    raise ValueError("单个模型的实例数不能超过 255，因为 255 保留给背景。")

                # 为当前模型中的每个实例分配唯一灰度值：0, 1, 2, ...
                instance_gray_map = {}
                for idx, instance_item in enumerate(export_instances):
                    instance_gray_map[instance_item["instance_id"]] = idx

                # 建立 face_id -> gray_value 的映射，未标注面默认为背景 255。
                face_gray_map = {}
                for instance_item in export_instances:
                    gray_value = instance_gray_map[instance_item["instance_id"]]
                    for face_id in instance_item["face_ids"]:
                        face_gray_map[face_id] = gray_value

                # 5.5 为每个面生成单独的 PyVista 网格，并准备对应的灰度颜色。
                # 这里按“整模一次性渲染”的思路处理：
                # - 标注面使用所属实例的灰度值
                # - 未标注面使用背景值 255（白色）
                # 这样每个面只会被画一次，遮挡关系由深度缓冲自然决定，
                # 不会再出现“逐实例单独渲染导致隐藏部分漏出来”的问题。
                face_meshes = {}
                face_render_items = []
                for r in self.face_records:
                    fd_f, temp_stl_f = tempfile.mkstemp(suffix=".stl")
                    os.close(fd_f)
                    writer.Write(r["face"], temp_stl_f)
                    face_mesh = pv.read(temp_stl_f)
                    face_meshes[r["face_id"]] = (face_mesh, temp_stl_f)

                    gray_value = face_gray_map.get(r["face_id"], 255)
                    gray_float = gray_value / 255.0
                    face_render_items.append((face_mesh, (gray_float, gray_float, gray_float)))

                # ============================================================
                # 步骤6: 创建 PyVista 离屏渲染器
                # ============================================================
                plotter = pv.Plotter(
                    off_screen=True,
                    window_size=[OUTPUT_WIDTH, OUTPUT_HEIGHT]
                )
                plotter.disable_anti_aliasing()
                plotter.camera.SetParallelProjection(True)

                # 6.1 预渲染一帧到内存，修复首帧空白的 bug
                plotter.add_mesh(mesh, color='white')
                plotter.render()
                _ = plotter.screenshot(None)
                plotter.clear()

                # ============================================================
                # 步骤7: 遍历12个视角，为每个视角生成一张多灰度值掩码图
                # ============================================================
                for i, d in enumerate(dirs, start=1):
                    img_name = f"{global_img_id:06d}.png"
                    img_path = os.path.join(masks_dir, img_name)
                    image_index_map[img_name] = {
                        "step_file": normalize_json_path(step_path),
                        "model_name": os.path.splitext(os.path.basename(step_path))[0],
                        "view_index": i,
                        "num_views": len(dirs),
                    }

                    # 7.1 先计算当前视角的相机参数
                    cam_pos = (cx + d[0] * dist, cy + d[1] * dist, cz + d[2] * dist)
                    viewup = get_viewup(d)
                    parallel_scale = get_parallel_scale(
                        bb, focal, d, viewup,
                        aspect_ratio=OUTPUT_WIDTH / OUTPUT_HEIGHT,
                        margin=1.10
                    )

                    # ============================================================
                    # 步骤8: 按方案 A 整模一次性渲染当前视角的掩码图
                    # - 每个实例一个灰度值
                    # - 未标注面为白色背景(255)
                    # - 所有面只画一次，遮挡关系自然正确
                    # ============================================================
                    plotter.clear()
                    plotter.set_background('white')
                    for face_mesh, face_color in face_render_items:
                        plotter.add_mesh(
                            face_mesh,
                            color=face_color,
                            smooth_shading=False,
                            lighting=False,
                        )

                    plotter.camera_position = [cam_pos, focal, viewup]
                    plotter.camera.SetParallelScale(max(parallel_scale, 0.01))
                    plotter.render()

                    screenshot_arr = plotter.screenshot(None)
                    if screenshot_arr is None:
                        raise RuntimeError(f"截图失败: {img_name}")

                    mask_gray = cv2.cvtColor(screenshot_arr, cv2.COLOR_BGR2GRAY)

                    # ============================================================
                    # 步骤9: 保存当前视角的掩码图 PNG
                    # ============================================================
                    cv2.imencode('.png', mask_gray)[1].tofile(img_path)

                    # ============================================================
                    # 步骤10: 记录 class_map（灰度值 → 类别ID）
                    # ============================================================
                    instance_class_map = {}
                    for instance_item in export_instances:
                        gray_value = instance_gray_map[instance_item["instance_id"]]
                        instance_class_map[str(gray_value)] = instance_item["category_id"]
                    class_map[img_name] = instance_class_map

                    # 写入共享清单条目（追加类型，不覆盖）
                    try:
                        model_index = self.step_paths.index(step_path) + 1
                    except ValueError:
                        model_index = 0
                    entry = manifest["entries"].get(img_name, {})
                    entry["model_index"] = model_index
                    entry["view_index"] = i
                    entry["num_views"] = len(dirs)
                    types = entry.get("types", [])
                    if "mask" not in types:
                        types.append("mask")
                    entry["types"] = types
                    manifest["entries"][img_name] = entry

                    # 图片 ID 自增
                    global_img_id += 1

                # ============================================================
                # 步骤11: 清理当前模型的临时资源
                # ============================================================
                plotter.close()
                os.remove(temp_stl)
                for _, temp_stl_f in face_meshes.values():
                    os.remove(temp_stl_f)

            # ============================================================
            # 步骤12: 写入 class_map.json
            # ============================================================
            json_path = os.path.join(out_dir, "class_map.json")
            with open(json_path, "w", encoding="utf-8") as f:
                std_json.dump(class_map, f, ensure_ascii=False, indent=2)

            index_map_path = os.path.join(out_dir, "image_index_map.json")
            with open(index_map_path, "w", encoding="utf-8") as f:
                std_json.dump(image_index_map, f, ensure_ascii=False, indent=2)

            self._save_manifest(manifest_path, manifest)

        except Exception as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "导出失败", str(exc))
            self.status_label.setText("导出异常")
            return

        # ============================================================
        # 步骤13: 恢复原始模型状态
        # ============================================================
        if original_path and original_path in self.model_dataset:
            self.load_step_file(original_path)
            self.template_face_ids.clear()
            self._refresh_template_list()

        # ============================================================
        # 步骤14: 完成提示
        # ============================================================
        QApplication.restoreOverrideCursor()
        self.status_label.setText("掩码图标签导出完成")
        QMessageBox.information(
            self,
            "导出成功",
            f"成功导出到 {out_dir}\n"
            f"共生成 {len(class_map)} 张多灰度值掩码图\n"
            f"标注文件: class_map.json"
        )

    # ================================================================
    # 特征预识别模块
    # ================================================================

    def _run_feature_pre_recognition(self):
        """调用Mask2Former推理流水线，预识别当前模型的特征标签"""
        if not self.step_path:
            QMessageBox.warning(self, "提示", "请先加载 STEP 模型。")
            return
        if not self.face_records:
            QMessageBox.warning(self, "提示", "当前模型无面数据。")
            return

        # 确认模型权重路径
        default_model = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "..", "mask2former_syb",
            "models", "finetuned_instance_model_v610"))
        
        model_dir = QFileDialog.getExistingDirectory(
            self, "选择模型权重目录", default_model)
        if not model_dir:
            model_dir = default_model
        if not os.path.exists(model_dir):
            QMessageBox.warning(self, "提示", f"模型权重目录不存在: {model_dir}")
            return

        # 输出目录
        output_dir = os.path.join(os.path.dirname(self.step_path), "pre_recognition_output")
        basename = os.path.splitext(os.path.basename(self.step_path))[0]
        step_output = os.path.join(output_dir, basename)

        reply = QMessageBox.question(
            self, "确认预识别",
            f"模型: {os.path.basename(self.step_path)}\n"
            f"权重: {os.path.basename(model_dir)}\n"
            f"输出: {step_output}\n\n"
            f"将生成12视角图并运行Mask2Former推理。\n是否继续？")
        if reply != QMessageBox.Yes:
            return

        self.btn_pre_recognize.setEnabled(False)
        self.pre_recog_progress.setValue(0)
        self.pre_recog_progress.setFormat("准备中...")
        self.pre_recog_status.setText("正在执行预识别...")
        QApplication.processEvents()

        import generate_inference_views as giv
        from back_project_to_step import (
            generate_unique_colors, back_project_single_view,
            aggregate_votes, CLASS_NAMES
        )

        t_start = time.time()
        self._pre_recog_t_start = t_start
        timing_info = []

        def _update_progress(pct, step_text, eta_override=None):
            elapsed = time.time() - t_start
            if eta_override is not None:
                eta = eta_override
            elif pct > 0:
                eta = elapsed / pct * (100 - pct)
            else:
                eta = None
            if eta is not None:
                self.pre_recog_progress.setFormat(f"{step_text} | 已用 {elapsed:.1f}s | 预计剩余 {eta:.1f}s")
            else:
                self.pre_recog_progress.setFormat(f"{step_text} | 已用 {elapsed:.1f}s")
            self.pre_recog_progress.setValue(int(pct))
            QApplication.processEvents()

        try:
            # Step 1: 生成多视角图 (0% ~ 30%)
            self.pre_recog_status.setText("Step 1/3: 生成多视角图...")
            _update_progress(2, "加载STEP...")

            t_step = time.time()

            data = giv.load_step_and_compute_data(self.step_path)
            if data is None:
                raise RuntimeError("无法读取STEP文件")

            num_faces = data["face_count"]
            directions = giv.get_dodecahedron_view_directions()
            num_views = len(directions)

            fixed_colors, color_to_face_id = generate_unique_colors(num_faces)
            views_colors = [fixed_colors] * num_views

            semantic_dir = os.path.join(step_output, "semantic_views")
            unique_dir = os.path.join(step_output, "unique_views")
            os.makedirs(semantic_dir, exist_ok=True)
            os.makedirs(unique_dir, exist_ok=True)
            semantic_paths = [os.path.join(semantic_dir, f"{v+1:06d}.png") for v in range(num_views)]
            unique_paths = [os.path.join(unique_dir, f"{v+1:06d}.png") for v in range(num_views)]

            _update_progress(5, f"渲染 {num_views} 视角...")
            giv.render_12_views(
                data["faces"], data["edges"], data["bounds"],
                semantic_paths, unique_paths, views_colors,
                view_directions=directions,
            )

            mapping_path = os.path.join(step_output, "color_face_id_map.json")
            serializable_map = {f"{r},{g},{b}": fid for (r, g, b), fid in color_to_face_id.items()}
            with open(mapping_path, "w", encoding="utf-8") as f:
                std_json.dump(serializable_map, f)

            t_step2 = time.time()
            timing_info.append(f"Step 1 多视角渲染: {t_step2 - t_step:.2f}s")
            _update_progress(30, "渲染完成")

            # Step 2: Mask2Former推理 (30% ~ 80%)
            self.pre_recog_status.setText("Step 2/3: Mask2Former推理...")
            _update_progress(32, "加载模型并推理...")

            pred_dir = os.path.join(step_output, "pred_masks")
            self._run_inference_subprocess(semantic_dir, model_dir, pred_dir, "cuda")

            t_step3 = time.time()
            timing_info.append(f"Step 2 Mask2Former推理: {t_step3 - t_step2:.2f}s")
            _update_progress(80, "推理完成")

            # Step 3: 回传3D (80% ~ 95%)
            self.pre_recog_status.setText("Step 3/3: 回传3D面标签...")
            _update_progress(82, "回传面标签...")

            with open(mapping_path, "r", encoding="utf-8") as f:
                raw_map = std_json.load(f)
            c2f = {}
            for key, fid in raw_map.items():
                r, g, b = [int(x) for x in key.split(",")]
                c2f[(r, g, b)] = fid

            from back_project_to_step import back_project_single_view as bpv
            from back_project_to_step import aggregate_votes as av

            face_votes = {}
            for vi in range(1, num_views + 1):
                iname = f"{vi:06d}.png"
                sp = os.path.join(pred_dir, f"{iname}_seg.npy")
                ip = os.path.join(pred_dir, f"{iname}_info.json")
                up = os.path.join(unique_dir, iname)
                if all(os.path.exists(p) for p in [sp, ip, up]):
                    bpv(sp, ip, up, face_votes, c2f)

            _update_progress(92, "投票聚合...")
            face_labels = av(face_votes, num_faces)

            t_step4 = time.time()
            timing_info.append(f"Step 3 回传3D: {t_step4 - t_step3:.2f}s")

            t_total = t_step4 - t_start
            timing_info.append(f"总耗时: {t_total:.2f}s")
            _update_progress(100, "完成")

            # 保存结果
            label_path = os.path.join(step_output, "face_label.json")
            with open(label_path, "w", encoding="utf-8") as f:
                std_json.dump(face_labels, f, ensure_ascii=False, indent=2)

            self._pre_recognition_results = face_labels
            self._pre_recognition_step_path = self.step_path

            from collections import Counter
            cc = Counter(info["class_id"] for info in face_labels.values())
            stats = []
            for cls_id in sorted(cc.keys()):
                cls_name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")
                stats.append(f"  {cls_name}: {cc[cls_id]}")

            detected = sum(1 for v in face_labels.values() if v["class_id"] > 0)
            timing_str = "\n".join(timing_info)
            self.pre_recog_status.setText(
                f"预识别完成! 检测到 {detected}/{num_faces} 个面\n"
                + "\n".join(stats)
                + f"\n\n{timing_str}")
            self.pre_recog_progress.setFormat(f"完成 | 总耗时 {t_total:.1f}s")
            self.btn_apply_predictions.setEnabled(True)

            QMessageBox.information(
                self, "预识别完成",
                f"检测到 {detected}/{num_faces} 个面\n\n"
                + "\n".join(stats)
                + f"\n\n{timing_str}\n\n"
                f"点击「✅ 应用预识别结果」将预测标签应用到模型。")

        except Exception as e:
            self.pre_recog_status.setText(f"预识别失败: {str(e)[:100]}")
            self.pre_recog_progress.setFormat("失败")
            self.pre_recog_progress.setValue(0)
            QMessageBox.critical(self, "预识别失败", str(e))
        finally:
            self.btn_pre_recognize.setEnabled(True)

    def _run_inference_subprocess(self, semantic_dir, model_dir, pred_dir, device="cuda"):
        """在子进程中运行Mask2Former推理，逐图输出进度"""
        import subprocess as sp
        os.makedirs(pred_dir, exist_ok=True)

        # 子进程脚本：每推理完一张图输出 PROGRESS:N/TOTAL
        script = f'''
import os, json, torch, numpy as np
from PIL import Image
from transformers import Mask2FormerForUniversalSegmentation, Mask2FormerImageProcessor

def main():
    processor = Mask2FormerImageProcessor.from_pretrained(r"{model_dir}")
    model = Mask2FormerForUniversalSegmentation.from_pretrained(r"{model_dir}")
    model.eval()
    dev = torch.device("{device}" if torch.cuda.is_available() else "cpu")
    model = model.to(dev)
    image_files = sorted([f for f in os.listdir(r"{semantic_dir}") if f.endswith(".png")])
    total = len(image_files)
    for idx, img_name in enumerate(image_files, 1):
        image = Image.open(os.path.join(r"{semantic_dir}", img_name)).convert("RGB")
        inputs = processor(images=image, return_tensors="pt", padding=True)
        inputs = {{k: v.to(dev) for k, v in inputs.items()}}
        with torch.no_grad():
            outputs = model(**inputs)
        result = processor.post_process_instance_segmentation(
            outputs, target_sizes=[image.size[::-1]], threshold=0.5, mask_threshold=0.5
        )[0]
        segmentation = result["segmentation"]
        if isinstance(segmentation, torch.Tensor):
            segmentation = segmentation.cpu().numpy()
        segments_info = result["segments_info"]
        np.save(os.path.join(r"{pred_dir}", f"{{img_name}}_seg.npy"),
                np.array(segmentation, dtype=np.uint16))
        info_data = [
            {{"id": int(s["id"]), "label_id": int(s["label_id"]),
              "score": float(s.get("score", 0.0))}} for s in segments_info
        ]
        with open(os.path.join(r"{pred_dir}", f"{{img_name}}_info.json"), "w") as f:
            json.dump(info_data, f)
        print(f"PROGRESS:{{idx}}/{{total}}", flush=True)

if __name__ == "__main__":
    main()
'''
        temp_script = os.path.join(pred_dir, "_temp_inference.py")
        with open(temp_script, "w", encoding="utf-8") as f:
            f.write(script)

        python_exe = r"D:\ProgramData\Miniconda3\envs\vlm_afr\python.exe"
        if not os.path.exists(python_exe):
            python_exe = sys.executable

        try:
            proc = sp.Popen(
                [python_exe, "-u", temp_script],
                stdout=sp.PIPE, stderr=sp.PIPE,
                text=True, bufsize=1
            )
            t_infer_start = time.time()
            for line in proc.stdout:
                line = line.strip()
                if line.startswith("PROGRESS:"):
                    parts = line.split(":")[1].split("/")
                    cur, total = int(parts[0]), int(parts[1])
                    pct = 32 + (cur / total) * 46
                    # 按每图速度估算剩余时间
                    elapsed_infer = time.time() - t_infer_start
                    if cur > 0:
                        time_per_img = elapsed_infer / cur
                        remaining_imgs = total - cur
                        eta_infer = time_per_img * remaining_imgs
                    else:
                        eta_infer = None
                    self.pre_recog_progress.setValue(int(pct))
                    elapsed_total = time.time() - self._pre_recog_t_start
                    if eta_infer is not None:
                        self.pre_recog_progress.setFormat(
                            f"推理 {cur}/{total} | 已用 {elapsed_total:.1f}s | 预计剩余 {eta_infer:.1f}s")
                    else:
                        self.pre_recog_progress.setFormat(
                            f"推理 {cur}/{total} | 已用 {elapsed_total:.1f}s")
                    QApplication.processEvents()

            proc.wait()
            if proc.returncode != 0:
                stderr = proc.stderr.read() if proc.stderr else ""
                raise RuntimeError(f"推理失败: {stderr[:500]}")
        finally:
            proc.kill() if proc.poll() is None else None
            if os.path.exists(temp_script):
                os.remove(temp_script)

    def _apply_pre_recognition_results(self):
        """将预识别结果应用到当前模型的面标注"""
        if not hasattr(self, '_pre_recognition_results'):
            QMessageBox.warning(self, "提示", "请先执行特征预识别。")
            return
        if not self.step_path:
            QMessageBox.warning(self, "提示", "请先加载 STEP 模型。")
            return
        if self.step_path != self._pre_recognition_step_path:
            QMessageBox.warning(self, "提示",
                "预识别结果与当前加载的模型不匹配。\n请重新执行预识别。")
            return

        face_labels = self._pre_recognition_results
        from back_project_to_step import CLASS_NAMES

        detected = sum(1 for v in face_labels.values() if v["class_id"] > 0)
        reply = QMessageBox.question(
            self, "确认应用",
            f"将把 {detected} 个面的预测标签应用到当前模型。\n"
            f"已有标注的面不会被覆盖。\n\n是否继续？")
        if reply != QMessageBox.Yes:
            return

        applied = 0
        skipped = 0
        for face_id_str, info in face_labels.items():
            face_id = int(face_id_str)
            class_id = info.get("class_id", 0)
            if class_id <= 0:
                continue
            if face_id not in self.face_lookup:
                skipped += 1
                continue
            record = self.face_lookup[face_id]
            # 跳过已有标注的面
            if record.get("instance_id") is not None:
                skipped += 1
                continue

            self.set_face_label(face_id, class_id, record_history=True)
            applied += 1

        self.pre_recog_status.setText(
            f"已应用: {applied} 个面\n跳过: {skipped} 个面（已有标注或未找到）")

        QMessageBox.information(
            self, "应用完成",
            f"已应用 {applied} 个面的预测标签\n"
            f"跳过 {skipped} 个面\n\n"
            f"请检查标注结果并手动修正不准确的预测。")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LabelTool()
    window.show()
    sys.exit(app.exec_())
