import sys
import os
import math
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
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
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
    image_filter.Update()

    vtk_image = image_filter.GetOutput()
    width, height, _ = vtk_image.GetDimensions()
    scalars = vtk_image.GetPointData().GetScalars()
    array = vtk_to_numpy(scalars).reshape(height, width, -1)
    array = np.flipud(array)
    if array.shape[2] > 3:
        array = array[:, :, :3]
    return array.copy()


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

        self._build_ui()
        self._init_vtk()

    def _build_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        left_panel = QVBoxLayout()
        self.btn_load_step = QPushButton("1. 导入多个 STEP 模型")
        self.btn_load_step.clicked.connect(self.open_step_dialog)
        left_panel.addWidget(self.btn_load_step)

        self.btn_load_json = QPushButton("2. 加载标注 JSON")
        self.btn_load_json.clicked.connect(self.load_annotation_json)
        left_panel.addWidget(self.btn_load_json)

        self.btn_export = QPushButton("3. 导出 COCO JSON + 12视角图片")
        self.btn_export.clicked.connect(self.export_package)
        left_panel.addWidget(self.btn_export)

        self.btn_export_gray = QPushButton("4. 导出12视角灰度图")
        self.btn_export_gray.clicked.connect(self.export_gray_package)
        left_panel.addWidget(self.btn_export_gray)

        # 新增 COCO 灰度数据集导出
        hbox_export_gray_coco = QHBoxLayout()
        hbox_export_gray_coco.addWidget(QLabel("起始图片编号:"))
        self.spin_start_id = QSpinBox()
        self.spin_start_id.setRange(1, 999999)
        self.spin_start_id.setValue(1)
        hbox_export_gray_coco.addWidget(self.spin_start_id)
        left_panel.addLayout(hbox_export_gray_coco)

        self.btn_export_gray_coco = QPushButton("5. 导出全局 COCO 灰度数据集")
        self.btn_export_gray_coco.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        self.btn_export_gray_coco.clicked.connect(self.export_global_gray_coco)
        left_panel.addWidget(self.btn_export_gray_coco)

        self.btn_reset_camera = QPushButton("重置三维视角")
        self.btn_reset_camera.clicked.connect(self.reset_camera)
        left_panel.addWidget(self.btn_reset_camera)

        left_panel.addWidget(QLabel("已导入模型列表:"))
        self.model_list = QListWidget()
        self.model_list.currentRowChanged.connect(self.on_model_switched)
        left_panel.addWidget(self.model_list)

        left_panel.addWidget(QLabel("当前模型:"))
        self.model_label = QLabel("未加载")
        self.model_label.setWordWrap(True)
        left_panel.addWidget(self.model_label)

        left_panel.addWidget(QLabel("预设类型 (左键单击面即可标注当前类型):"))
        self.category_list = QListWidget()
        self.category_list.currentRowChanged.connect(self.on_category_changed)
        left_panel.addWidget(self.category_list)
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
            "- 鼠标移动: 实时高亮预选面\n"
            "- 按钮3: 批量导出彩色12视角和COCO JSON\n"
            "- 按钮4: 批量导出灰度12视角"
        )
        self.status_label.setWordWrap(True)
        left_panel.addWidget(self.status_label)
        left_panel.addStretch(1)

        center_panel = QVBoxLayout()
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        self.vtk_widget.setMouseTracking(True)
        self.vtk_widget.installEventFilter(self)
        center_panel.addWidget(self.vtk_widget)

        right_panel = QVBoxLayout()
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
        self._refresh_template_list()

        for record in self.face_records:
            actor_key = record["actor"].GetAddressAsString("")
            self.actor_lookup[actor_key] = record["face_id"]
            record["actor"].PickableOn()
            record["label_id"] = entry["labels"].get(record["face_id"])
            self.renderer.AddActor(record["actor"])
            self._apply_actor_style(record["face_id"])

        self.renderer.ResetCamera()
        self.model_bounds = self.renderer.ComputeVisiblePropBounds()
        self.vtk_widget.GetRenderWindow().Render()

        self.model_label.setText(step_path)
        self._refresh_face_list()
        self._update_summary()
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
        self.set_face_label(face_id, self.current_category_id, record_history=True)
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

    def set_face_label(self, face_id, label_id, record_history=True):
        record = self.face_lookup.get(face_id)
        if record is None:
            return
        old_label = record["label_id"]
        if old_label == label_id:
            return
        entry = self.model_dataset.get(self.step_path)
        if entry is None:
            return
        if record_history:
            action = {"face_id": face_id, "old_label": old_label, "new_label": label_id}
            self.undo_stack.append(action)
            entry["undo_stack"] = list(self.undo_stack)
        record["label_id"] = label_id
        if label_id is None:
            entry["labels"].pop(face_id, None)
        else:
            entry["labels"][face_id] = label_id
        self._apply_actor_style(face_id)
        self._refresh_face_list()
        self._update_summary()
        self._update_current_face_label()
        self.vtk_widget.GetRenderWindow().Render()

    def clear_selected_face_label(self):
        if self.selected_face_id is None:
            QMessageBox.information(self, "提示", "请先在模型或右侧列表中选中一个面。")
            return
        self.set_face_label(self.selected_face_id, None, record_history=True)

    def undo_last_action(self):
        if not self.undo_stack:
            QMessageBox.information(self, "提示", "当前没有可撤回的标注操作。")
            return
        action = self.undo_stack.pop()
        self.model_dataset[self.step_path]["undo_stack"] = list(self.undo_stack)
        self.set_face_label(action["face_id"], action["old_label"], record_history=False)
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
        for record in self.face_records:
            record["label_id"] = None
            self._apply_actor_style(record["face_id"])
        self._refresh_face_list()
        self._update_summary()
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
        self.template_face_ids.add(self.selected_face_id)
        self._apply_actor_style(self.selected_face_id)
        self._refresh_template_list()
        self.vtk_widget.GetRenderWindow().Render()

    def remove_from_template(self):
        if self.selected_face_id is None or self.selected_face_id not in self.template_face_ids:
            QMessageBox.warning(self, "提示", "请先选中一个已在模板列表中的面。")
            return
        self.template_face_ids.discard(self.selected_face_id)
        self._apply_actor_style(self.selected_face_id)
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
            text = f"F{fid:04d} | {category.get('display_name', '?')}"
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
            QMessageBox.warning(self, "提示", "模板面列表为空。\n请先用左键单击选中已标注面，再点「加入模板面列表」。")
            return

        template_records = [self.face_lookup[fid] for fid in self.template_face_ids if fid in self.face_lookup]
        if not template_records:
            QMessageBox.warning(self, "提示", "模板面无效。")
            return

        label_ids = set(r["label_id"] for r in template_records if r["label_id"] is not None)
        if not label_ids:
            QMessageBox.warning(self, "提示", "模板面中至少需要一个已标注类别的面。")
            return

        threshold = self.similarity_threshold
        summary_parts = []
        for lid in sorted(label_ids):
            cat = self.category_by_id.get(lid, {})
            summary_parts.append(cat.get("display_name", f"ID={lid}"))
        cat_summary = " / ".join(summary_parts)

        reply = QMessageBox.question(
            self,
            "确认特征学习",
            f"将以 {len(template_records)} 个模板面为参考（{cat_summary}），\n"
            f"在 {len(self.face_records)} 个面中搜索结构相似的区域。\n"
            f"相似度阈值 = {threshold:.2f}\n\n是否继续？",
        )
        if reply != QMessageBox.Yes:
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            all_matches = find_similar_feature_groups(
                list(self.template_face_ids),
                self.face_records,
                self.face_lookup,
                similarity_threshold=threshold,
            )
        except Exception as exc:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "特征学习失败", str(exc))
            return
        QApplication.restoreOverrideCursor()

        applied = 0
        for mapping in all_matches:
            for t_fid, c_fid in mapping.items():
                t_record = self.face_lookup.get(t_fid)
                if not t_record or t_record["label_id"] is None:
                    continue
                label_id = t_record["label_id"]
                
                # Check if target already has a label, skip if we don't want to overwrite or just overwrite
                c_record = self.face_lookup.get(c_fid)
                if c_record and c_record["label_id"] != label_id:
                    self.set_face_label(c_fid, label_id, record_history=True)
                    applied += 1

        self._refresh_face_list()
        self._update_summary()
        self.vtk_widget.GetRenderWindow().Render()
        if applied:
            QMessageBox.information(
                self,
                "特征学习完成",
                f"模板: {len(template_records)} 个面（{cat_summary}）\n"
                f"找到 {len(all_matches)} 个结构相似的特征组。\n"
                f"共成功标注 {applied} 个面。",
            )
        else:
            QMessageBox.information(self, "未找到", "在当前阈值下没有找到结构一致的特征组。\n请尝试降低相似度阈值或检查模板选择。")

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
            center = record["center"]
            text = (
                f"F{record['face_id']:04d} | {category['display_name']} / {category['name']} | "
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
        self.summary_label.setText(
            f"面统计:\n总面数: {total}\n已标注: {labeled}\n未标注: {total - labeled}"
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
        if record["label_id"] is not None:
            category = self.category_by_id[record["label_id"]]
            label_text = f"{category['display_name']} / {category['name']}"
        prefix = "选中面" if face_id == self.selected_face_id else "预选面"
        center = record["center"]
        self.current_face_label.setText(
            f"{prefix}: F{face_id:04d}\n"
            f"标签: {label_text}\n"
            f"中心: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})"
        )

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

        labeled_records = [record for record in self.face_records if record["label_id"] is not None]
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
            for image_index, direction in enumerate(get_dodecahedron_view_directions(), start=1):
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

                if not labeled_records:
                    continue

                self.renderer.SetBackground(0.0, 0.0, 0.0)
                for target_record in labeled_records:
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
                        if record["face_id"] == target_record["face_id"]:
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
                        category_id=target_record["label_id"],
                        annotation_id=annotation_id,
                    )
                    if annotation is not None:
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

            for image_index, direction in enumerate(get_dodecahedron_view_directions(), start=1):
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

        return len(get_dodecahedron_view_directions())

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

        faces = payload.get("faces", [])
        entry = self.model_dataset.get(self.step_path)
        if entry is None:
            return
        entry["labels"] = {}
        entry["undo_stack"] = []
        self.undo_stack = []
        for item in faces:
            face_id = item.get("face_id")
            label_id = item.get("type_id")
            if face_id in self.face_lookup:
                self.face_lookup[face_id]["label_id"] = label_id
                if label_id is not None:
                    entry["labels"][face_id] = label_id
                self._apply_actor_style(face_id)

        self._refresh_face_list()
        self._update_summary()
        self._update_current_face_label()
        self.vtk_widget.GetRenderWindow().Render()
        QMessageBox.information(self, "成功", "标注 JSON 已加载。")

    def export_package(self):
        if not self.step_paths:
            QMessageBox.warning(self, "提示", "请先导入至少一个 STEP 模型。")
            return

        output_dir = QFileDialog.getExistingDirectory(self, "选择导出文件夹")
        if not output_dir:
            return

        original_path = self.step_path
        export_count = 0
        try:
            for step_path in self.step_paths:
                if os.path.normcase(step_path) != os.path.normcase(self.step_path):
                    self.load_step_file(step_path)

                unlabeled = [record["face_id"] for record in self.face_records if record["label_id"] is None]
                if unlabeled:
                    reply = QMessageBox.question(
                        self,
                        "存在未标注面",
                        f"{os.path.basename(step_path)} 仍有 {len(unlabeled)} 个面未标注，是否继续导出该零件？",
                    )
                    if reply != QMessageBox.Yes:
                        continue

                base_name = os.path.splitext(os.path.basename(self.step_path))[0]
                json_path = os.path.join(output_dir, f"{base_name}_labels.json")
                image_dir = os.path.join(output_dir, f"{base_name}_views")
                os.makedirs(image_dir, exist_ok=True)

                payload = self.build_coco_dataset(image_dir)
                with open(json_path, "w", encoding="utf-8") as f:
                    std_json.dump(payload, f, ensure_ascii=False, indent=2)
                export_count += 1
        except Exception as exc:
            QMessageBox.critical(self, "导出失败", str(exc))
            return
        finally:
            if original_path and os.path.normcase(original_path) != os.path.normcase(self.step_path):
                self.load_step_file(original_path)

        QMessageBox.information(
            self,
            "导出成功",
            f"已完成 {export_count} 个零件的 JSON 与 12视角图片导出。",
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

        # Remember current state
        original_path = self.step_path

        QApplication.setOverrideCursor(Qt.WaitCursor)
        self.status_label.setText("正在导出全局 COCO 灰度数据集...")
        QApplication.processEvents()

        try:
            dirs = get_dodecahedron_view_directions()

            for step_path in self.model_dataset.keys():
                # 切换当前加载的模型，以便利用现成的 self.shape 和 self.face_records
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

                # Convert labeled faces to STL for mask rendering
                labeled_records = [r for r in self.face_records if r["label_id"] is not None]
                face_meshes = []
                for r in labeled_records:
                    fd_f, temp_stl_f = tempfile.mkstemp(suffix=".stl")
                    os.close(fd_f)
                    writer.Write(r["face"], temp_stl_f)
                    face_meshes.append((r, pv.read(temp_stl_f), temp_stl_f))

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
                    plotter.camera_position = [cam_pos, focal, viewup]
                    parallel_scale = get_parallel_scale(
                        bb, focal, d, viewup, aspect_ratio=OUTPUT_WIDTH / OUTPUT_HEIGHT, margin=1.10
                    )
                    plotter.camera.SetParallelScale(max(parallel_scale, 0.01))

                    # --- Render Gray Image ---
                    plotter.clear()
                    plotter.set_background('white')
                    plotter.add_mesh(mesh, color='lightgray', smooth_shading=False, lighting=True)
                    if feature_edges.n_points > 0:
                        plotter.add_mesh(feature_edges, color='black', line_width=3.0, lighting=False)
                    plotter.enable_lightkit()
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

                    for r, f_mesh, _ in face_meshes:
                        actor = plotter.add_mesh(f_mesh, color='white', smooth_shading=False, lighting=False)
                        plotter.render()

                        temp_mask = os.path.join(images_dir, "temp_mask.png")
                        plotter.screenshot(temp_mask)
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
                                    "category_id": r["label_id"],
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
                for _, _, temp_stl_f in face_meshes:
                    os.remove(temp_stl_f)

            # Write JSON
            json_path = os.path.join(out_dir, "annotations.json")
            with open(json_path, "w", encoding="utf-8") as f:
                std_json.dump(coco_data, f, ensure_ascii=False, indent=2)

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

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LabelTool()
    window.show()
    sys.exit(app.exec_())
