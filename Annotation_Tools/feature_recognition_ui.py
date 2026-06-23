# -*- coding: utf-8 -*-
"""
3D特征识别UI界面
功能：选择STEP文件 → 选择模型权重 → 特征识别 → 3D模型预览 + 特征高亮演示
"""
import json
import os
import sys
import math
import threading
import subprocess
import traceback

import numpy as np
import pyvista as pv
from PIL import Image

try:
    import vtk
    from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
    VTK_AVAILABLE = True
except ImportError:
    VTK_AVAILABLE = False

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QComboBox, QTextEdit,
    QGroupBox, QProgressBar, QSplitter, QFrame, QMessageBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QTabWidget,
    QCheckBox
)
from PyQt5.QtCore import Qt, pyqtSignal, QObject, QSize
from PyQt5.QtGui import QPixmap, QFont, QColor, QIcon, QImage


# ==================== 常量 ====================

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

CLASS_COLORS_RGB = {
    1: (255, 60, 60),
    2: (60, 200, 60),
    3: (60, 60, 255),
    4: (220, 200, 40),
    5: (255, 128, 0),
    6: (128, 0, 255),
    7: (0, 200, 200),
}

DEFAULT_FACE_COLOR = (180, 180, 200)
UNKNOWN_FACE_COLOR = (200, 200, 200)
OUTPUT_WIDTH = 1024
OUTPUT_HEIGHT = 1024


# ==================== 工具函数 ====================

def rgb_to_float(rgb):
    return tuple(c / 255.0 for c in rgb)


# ==================== 信号桥接 ====================

class WorkerSignals(QObject):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)
    image_signal = pyqtSignal(str, str)


# ==================== 主界面 ====================

class FeatureRecognitionUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D特征识别系统")
        self.setMinimumSize(1200, 800)
        self.worker_signals = WorkerSignals()
        self.worker_signals.log_signal.connect(self.append_log)
        self.worker_signals.progress_signal.connect(self.update_progress)
        self.worker_signals.finished_signal.connect(self.on_finished)
        self.worker_signals.error_signal.connect(self.on_error)
        self.worker_signals.image_signal.connect(self.show_image)

        self.current_output_dir = None
        self._face_records = []      # 当前加载的STEP面信息
        self._face_labels = {}       # 识别结果 {face_id: info}
        self._step_path = ""
        self._init_ui()

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # ---- 顶部：文件和配置区 ----
        config_group = QGroupBox("配置")
        config_layout = QVBoxLayout()

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("STEP文件:"))
        self.step_path_edit = QTextEdit()
        self.step_path_edit.setMaximumHeight(30)
        self.step_path_edit.setPlaceholderText("选择STEP文件或文件夹...")
        row1.addWidget(self.step_path_edit)
        self.btn_select_file = QPushButton("选择文件")
        self.btn_select_file.setFixedWidth(100)
        self.btn_select_file.clicked.connect(self.select_step_file)
        row1.addWidget(self.btn_select_file)
        self.btn_select_dir = QPushButton("选择文件夹")
        self.btn_select_dir.setFixedWidth(100)
        self.btn_select_dir.clicked.connect(self.select_step_dir)
        row1.addWidget(self.btn_select_dir)
        config_layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("模型权重:"))
        self.model_path_edit = QTextEdit()
        self.model_path_edit.setMaximumHeight(30)
        default_model = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "..", "mask2former_syb",
            "models", "finetuned_instance_model_v610"))
        self.model_path_edit.setText(default_model)
        row2.addWidget(self.model_path_edit)
        self.btn_select_model = QPushButton("选择权重")
        self.btn_select_model.setFixedWidth(100)
        self.btn_select_model.clicked.connect(self.select_model)
        row2.addWidget(self.btn_select_model)
        config_layout.addLayout(row2)

        row3 = QHBoxLayout()
        row3.addWidget(QLabel("输出目录:"))
        self.output_path_edit = QTextEdit()
        self.output_path_edit.setMaximumHeight(30)
        self.output_path_edit.setText(os.path.join(
            os.path.dirname(__file__), "..", "pipeline_output"))
        row3.addWidget(self.output_path_edit)
        self.btn_select_output = QPushButton("选择目录")
        self.btn_select_output.setFixedWidth(100)
        self.btn_select_output.clicked.connect(self.select_output)
        row3.addWidget(self.btn_select_output)
        row3.addWidget(QLabel("设备:"))
        self.combo_device = QComboBox()
        self.combo_device.addItems(["cuda", "cpu"])
        self.combo_device.setFixedWidth(80)
        row3.addWidget(self.combo_device)
        config_layout.addLayout(row3)

        config_group.setLayout(config_layout)
        main_layout.addWidget(config_group)

        # ---- 中部：按钮和进度 ----
        btn_layout = QHBoxLayout()
        self.btn_run = QPushButton("  开始识别  ")
        self.btn_run.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-size: 14px; "
            "font-weight: bold; padding: 8px 24px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #45a049; }"
            "QPushButton:disabled { background-color: #cccccc; }"
        )
        self.btn_run.clicked.connect(self.run_recognition)
        btn_layout.addWidget(self.btn_run)

        self.btn_stop = QPushButton("  停止  ")
        self.btn_stop.setStyleSheet(
            "QPushButton { background-color: #f44336; color: white; font-size: 14px; "
            "padding: 8px 24px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #da190b; }"
            "QPushButton:disabled { background-color: #cccccc; }"
        )
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_recognition)
        btn_layout.addWidget(self.btn_stop)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        btn_layout.addWidget(self.progress_bar)

        main_layout.addLayout(btn_layout)

        # ---- 下部：结果展示区 ----
        splitter = QSplitter(Qt.Vertical)

        # 日志区
        log_group = QGroupBox("运行日志")
        log_layout = QVBoxLayout()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
        log_layout.addWidget(self.log_text)
        log_group.setLayout(log_layout)
        splitter.addWidget(log_group)

        # 标签页
        self.tabs = QTabWidget()

        # Tab 0: 3D模型预览
        self._create_3d_viewer_tab()
        # Tab 1: 识别特征演示
        self._create_feature_demo_tab()
        # Tab 2: 识别结果表格
        self._create_result_table_tab()
        # Tab 3: 多视角图预览
        self._create_image_tab("多视角预览", "识别完成后显示多视角渲染图")
        # Tab 4: 语义染色图
        self._create_image_tab("语义染色图", "识别完成后显示语义染色图")
        # Tab 5: Unique面ID图
        self._create_image_tab("Unique面ID图", "识别完成后显示unique面ID图")
        # Tab 6: 推理掩码
        self._create_image_tab("推理掩码", "识别完成后显示推理掩码")

        splitter.addWidget(self.tabs)
        splitter.setSizes([200, 500])

        main_layout.addWidget(splitter)

    # ======== 3D模型预览 Tab ========

    def _create_3d_viewer_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        if VTK_AVAILABLE:
            self.vtk_widget_preview = QVTKRenderWindowInteractor(widget)
            layout.addWidget(self.vtk_widget_preview)

            info_layout = QHBoxLayout()
            self.preview_info = QLabel("未加载模型 | 鼠标拖动旋转 | 滚轮缩放")
            self.preview_info.setStyleSheet("QLabel { color: #666; padding: 4px; }")
            info_layout.addWidget(self.preview_info)
            info_layout.addStretch()
            layout.addLayout(info_layout)

            self.renderer_preview = vtk.vtkRenderer()
            self.renderer_preview.SetBackground(0.9, 0.9, 0.92)
            self.vtk_widget_preview.GetRenderWindow().AddRenderer(self.renderer_preview)
            style = vtk.vtkInteractorStyleTrackballCamera()
            self.vtk_widget_preview.GetRenderWindow().GetInteractor().SetInteractorStyle(style)
            self.vtk_widget_preview.Initialize()
        else:
            self.preview_label_fb = QLabel("(需要安装VTK模块)")
            self.preview_label_fb.setAlignment(Qt.AlignCenter)
            layout.addWidget(self.preview_label_fb)

        self.tabs.addTab(widget, "3D模型预览")

    # ======== 识别特征演示 Tab ========

    def _create_feature_demo_tab(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)

        if VTK_AVAILABLE:
            self.vtk_widget_demo = QVTKRenderWindowInteractor(widget)
            layout.addWidget(self.vtk_widget_demo)

            ctrl_line = QHBoxLayout()
            self.chk_show_unrecognized = QCheckBox("显示未识别面")
            self.chk_show_unrecognized.setChecked(True)
            self.chk_show_unrecognized.stateChanged.connect(self._refresh_feature_demo)
            ctrl_line.addWidget(self.chk_show_unrecognized)

            self.chk_highlight_selected = QCheckBox("高亮选中的面")
            self.chk_highlight_selected.setChecked(True)
            self.chk_highlight_selected.stateChanged.connect(self._refresh_feature_demo)
            ctrl_line.addWidget(self.chk_highlight_selected)

            self.demo_info = QLabel("执行识别后在此显示特征分类 | 鼠标拖动旋转 | 滚轮缩放")
            self.demo_info.setStyleSheet("QLabel { color: #666; padding: 4px; }")
            ctrl_line.addWidget(self.demo_info)
            ctrl_line.addStretch()
            layout.addLayout(ctrl_line)

            self.renderer_demo = vtk.vtkRenderer()
            self.renderer_demo.SetBackground(0.15, 0.15, 0.18)
            self.vtk_widget_demo.GetRenderWindow().AddRenderer(self.renderer_demo)
            style = vtk.vtkInteractorStyleTrackballCamera()
            self.vtk_widget_demo.GetRenderWindow().GetInteractor().SetInteractorStyle(style)
            self.vtk_widget_demo.Initialize()
        else:
            self.demo_label_fb = QLabel("(需要安装VTK模块)")
            self.demo_label_fb.setAlignment(Qt.AlignCenter)
            layout.addWidget(self.demo_label_fb)

        self.tabs.addTab(widget, "识别特征演示")
        self._demo_highlight_face_id = None

    # ======== 结果表格 Tab ========

    def _create_result_table_tab(self):
        self.result_table = QTableWidget()
        self.result_table.setColumnCount(7)
        self.result_table.setHorizontalHeaderLabels(
            ["面ID", "类别ID", "类别名称", "置信度", "投票数", "主要投票", "平均分数"])
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.result_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.result_table.setSelectionMode(QTableWidget.SingleSelection)
        self.result_table.currentCellChanged.connect(self._on_table_row_changed)
        self.tabs.addTab(self.result_table, "识别结果")

    # ======== 图片预览 Tab ========

    def _create_image_tab(self, title, placeholder):
        label = QLabel(placeholder)
        label.setAlignment(Qt.AlignCenter)
        label.setMinimumHeight(400)
        label.setStyleSheet("QLabel { background-color: #f0f0f0; border: 1px solid #ccc; }")
        setattr(self, f"_imgtab_{title}", label)
        self.tabs.addTab(label, title)

    # ---- 文件选择 ----

    def select_step_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择STEP文件", "",
            "STEP文件 (*.step *.stp *.STEP *.STP);;所有文件 (*)")
        if path:
            self.step_path_edit.setText(path)
            self._try_load_step_preview(path)

    def select_step_dir(self):
        path = QFileDialog.getExistingDirectory(self, "选择STEP文件夹")
        if path:
            self.step_path_edit.setText(path)
            import glob as glob_mod
            for ext in ("*.step", "*.stp", "*.STEP", "*.STP"):
                files = glob_mod.glob(os.path.join(path, ext))
                if files:
                    self._try_load_step_preview(sorted(files)[0])
                    break

    def select_model(self):
        path = QFileDialog.getExistingDirectory(self, "选择模型权重目录")
        if path:
            self.model_path_edit.setText(path)

    def select_output(self):
        path = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if path:
            self.output_path_edit.setText(path)

    # ======== 3D模型加载（VTK） ========

    def _load_step_faces_vtk(self, step_path):
        """加载STEP文件，为每个面创建VTK Actor"""
        from OCP.STEPControl import STEPControl_Reader
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
        from OCP.TopoDS import TopoDS
        from OCP.TopLoc import TopLoc_Location
        from OCP.BRep import BRep_Tool
        from OCP.BRepMesh import BRepMesh_IncrementalMesh

        reader = STEPControl_Reader()
        if reader.ReadFile(step_path) != 1:
            return None

        reader.TransferRoots()
        shape = reader.OneShape()
        BRepMesh_IncrementalMesh(shape, 0.1)

        records = []
        exp = TopExp_Explorer(shape, TopAbs_FACE)
        face_id = 1
        while exp.More():
            face = TopoDS.Face_s(exp.Current())
            location = TopLoc_Location()
            tri = BRep_Tool.Triangulation_s(face, location)
            if tri is not None and tri.NbTriangles() > 0:
                transform = location.Transformation()
                points = vtk.vtkPoints()
                cells = vtk.vtkCellArray()
                for i in range(1, tri.NbNodes() + 1):
                    pt = tri.Node(i)
                    try:
                        pt = pt.Transformed(transform)
                    except Exception:
                        pass
                    points.InsertNextPoint(pt.X(), pt.Y(), pt.Z())
                is_rev = face.Orientation() == TopAbs_REVERSED
                for i in range(1, tri.NbTriangles() + 1):
                    t = tri.Triangle(i)
                    tri_cell = vtk.vtkTriangle()
                    n1, n2, n3 = t.Value(1) - 1, t.Value(2) - 1, t.Value(3) - 1
                    if is_rev:
                        n2, n3 = n3, n2
                    tri_cell.GetPointIds().SetId(0, n1)
                    tri_cell.GetPointIds().SetId(1, n2)
                    tri_cell.GetPointIds().SetId(2, n3)
                    cells.InsertNextCell(tri_cell)

                polydata = vtk.vtkPolyData()
                polydata.SetPoints(points)
                polydata.SetPolys(cells)

                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputData(polydata)
                actor = vtk.vtkActor()
                actor.SetMapper(mapper)
                actor.GetProperty().SetInterpolationToPhong()
                actor.GetProperty().LightingOn()
                actor.GetProperty().SetAmbient(0.25)
                actor.GetProperty().SetDiffuse(0.75)
                actor.GetProperty().SetSpecular(0.1)
                actor.GetProperty().SetEdgeVisibility(True)
                actor.GetProperty().SetEdgeColor(0.3, 0.3, 0.3)
                actor.GetProperty().SetLineWidth(0.5)

                records.append({
                    "face_id": face_id,
                    "actor": actor,
                    "actor_address": actor.GetAddressAsString(""),
                })
                face_id += 1
            exp.Next()

        return records

    def _try_load_step_preview(self, step_path):
        """加载STEP文件并显示3D预览"""
        if not os.path.exists(step_path) or not VTK_AVAILABLE:
            return
        self.preview_info.setText("正在加载模型...")
        QApplication.processEvents()

        try:
            records = self._load_step_faces_vtk(step_path)
            if records:
                self._face_records = records
                self._face_labels = {}
                self._step_path = step_path

                self.renderer_preview.RemoveAllViewProps()
                for rec in records:
                    prop = rec["actor"].GetProperty()
                    prop.SetColor(*rgb_to_float(DEFAULT_FACE_COLOR))
                    self.renderer_preview.AddActor(rec["actor"])

                self.renderer_preview.ResetCamera()
                self.vtk_widget_preview.GetRenderWindow().Render()
                self.preview_info.setText(
                    f"已加载: {os.path.basename(step_path)} | {len(records)} 个面 | 鼠标拖动旋转 | 滚轮缩放")
            else:
                self.preview_info.setText("加载失败")
        except Exception as e:
            self.preview_info.setText(f"加载失败: {e}")

    # ======== 3D特征演示渲染 ========

    def _refresh_feature_demo(self):
        """刷新特征演示VTK视图"""
        if not VTK_AVAILABLE or not self._face_records:
            return

        show_unrecog = self.chk_show_unrecognized.isChecked()
        highlight = self.chk_highlight_selected.isChecked()
        highlight_id = self._demo_highlight_face_id if highlight else None

        self.renderer_demo.RemoveAllViewProps()
        for rec in self._face_records:
            fid = rec["face_id"]
            info = self._face_labels.get(fid, {})
            class_id = info.get("class_id", 0)

            if class_id == 0 and not show_unrecog:
                continue

            actor = rec["actor"]
            prop = actor.GetProperty()

            if highlight_id is not None and fid == highlight_id:
                prop.SetColor(1.0, 1.0, 0.2)
                prop.SetAmbient(0.5)
                prop.SetDiffuse(1.0)
                prop.SetSpecular(0.4)
                prop.SetEdgeColor(1.0, 0.2, 0.0)
                prop.SetLineWidth(3.0)
                prop.SetEdgeVisibility(True)
            elif class_id > 0 and class_id in CLASS_COLORS_RGB:
                color = CLASS_COLORS_RGB[class_id]
                prop.SetColor(*rgb_to_float(color))
                prop.SetAmbient(0.3)
                prop.SetDiffuse(0.7)
                prop.SetSpecular(0.1)
                prop.SetEdgeColor(0.4, 0.4, 0.4)
                prop.SetLineWidth(0.5)
                prop.SetEdgeVisibility(False)
            else:
                prop.SetColor(*rgb_to_float(UNKNOWN_FACE_COLOR))
                prop.SetAmbient(0.2)
                prop.SetDiffuse(0.6)
                prop.SetSpecular(0.05)
                prop.SetEdgeColor(0.3, 0.3, 0.3)
                prop.SetLineWidth(0.5)
                prop.SetEdgeVisibility(False)

            self.renderer_demo.AddActor(actor)

        self.renderer_demo.ResetCamera()
        self.vtk_widget_demo.GetRenderWindow().Render()

    def _on_table_row_changed(self, current_row, _prev_col, _prev_row, _new_col):
        """点击表格行 → 高亮对应面"""
        if current_row < 0 or not VTK_AVAILABLE or not self._face_records:
            return
        item = self.result_table.item(current_row, 0)
        if item is None:
            return
        try:
            face_id = int(item.text())
        except ValueError:
            return

        self._demo_highlight_face_id = face_id
        self._refresh_feature_demo()

    # ======== 日志和进度 ========

    def append_log(self, msg):
        self.log_text.append(msg)
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum())

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def show_image(self, path, title):
        if not os.path.exists(path):
            return
        pixmap = QPixmap(path)
        if pixmap.isNull():
            return
        scaled = pixmap.scaled(800, 800, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        for tab_name in ["多视角预览", "语义染色图", "Unique面ID图", "推理掩码"]:
            lbl = getattr(self, f"_imgtab_{tab_name}", None)
            if lbl and title == tab_name:
                lbl.setPixmap(scaled)

    # ======== 运行识别 ========

    def run_recognition(self):
        step_path = self.step_path_edit.toPlainText().strip()
        model_dir = self.model_path_edit.toPlainText().strip()
        output_dir = self.output_path_edit.toPlainText().strip()
        device = self.combo_device.currentText()

        if not step_path:
            QMessageBox.warning(self, "提示", "请选择STEP文件或文件夹")
            return
        if not os.path.exists(step_path):
            QMessageBox.warning(self, "提示", f"路径不存在: {step_path}")
            return
        if not model_dir or not os.path.exists(model_dir):
            QMessageBox.warning(self, "提示", f"模型权重目录不存在: {model_dir}")
            return

        self.current_output_dir = output_dir
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress_bar.setValue(0)
        self.log_text.clear()
        self.result_table.setRowCount(0)

        self._worker_thread = threading.Thread(
            target=self._run_pipeline,
            args=(step_path, model_dir, output_dir, device),
            daemon=True)
        self._worker_thread.start()

    def _run_pipeline(self, step_path, model_dir, output_dir, device):
        try:
            sig = self.worker_signals
            import glob as glob_mod

            step_files = []
            if os.path.isfile(step_path):
                step_files.append(step_path)
            else:
                for ext in ("*.step", "*.stp", "*.STEP", "*.STP"):
                    step_files.extend(glob_mod.glob(os.path.join(step_path, ext)))
                step_files.sort()

            if not step_files:
                sig.error_signal.emit("未找到STEP文件")
                return

            sig.log_signal.emit(f"共 {len(step_files)} 个STEP文件")
            sig.log_signal.emit(f"模型权重: {model_dir}")
            sig.log_signal.emit(f"设备: {device}\n")

            total = len(step_files)
            all_results = {}

            for idx, step_file in enumerate(step_files):
                basename = os.path.splitext(os.path.basename(step_file))[0]
                sig.log_signal.emit(f"[{idx+1}/{total}] 处理: {basename}")
                step_output = os.path.join(output_dir, basename)
                os.makedirs(step_output, exist_ok=True)

                import generate_inference_views as giv
                from back_project_to_step import (
                    back_project_single_view, aggregate_votes, colorize_step,
                    generate_unique_colors, 
                )

                # Step 1 - 严格参考 label_tool_instance.py 的渲染方式
                sig.log_signal.emit("  [Step 1] 生成多视角图...")
                sig.progress_signal.emit(int((idx / total) * 80))
                data = giv.load_step_and_compute_data(step_file)
                if data is None:
                    sig.log_signal.emit("  跳过: 无法读取STEP")
                    continue
                num_faces = data["face_count"]
                directions = giv.get_dodecahedron_view_directions()
                num_views = len(directions)

                # 固定颜色（无 hue offset），用于回传face_id
                fixed_colors, color_to_face_id = generate_unique_colors(num_faces)
                # views_colors: 12个视角都使用相同固定颜色
                views_colors = [fixed_colors] * num_views

                # 构建输出路径
                semantic_dir = os.path.join(step_output, "semantic_views")
                unique_dir = os.path.join(step_output, "unique_views")
                os.makedirs(semantic_dir, exist_ok=True)
                os.makedirs(unique_dir, exist_ok=True)
                semantic_paths = [os.path.join(semantic_dir, f"{v+1:06d}.png") for v in range(num_views)]
                unique_paths = [os.path.join(unique_dir, f"{v+1:06d}.png") for v in range(num_views)]

                # 调用 render_12_views（含边渲染，与label_tool_instance一致）
                giv.render_12_views(
                    data["faces"], data["edges"], data["bounds"],
                    semantic_paths, unique_paths, views_colors,
                    view_directions=directions,
                )

                # 保存颜色→face_id映射（用于Step 3回传）
                mapping_path = os.path.join(step_output, "color_face_id_map.json")
                serializable_map = {f"{r},{g},{b}": fid for (r, g, b), fid in color_to_face_id.items()}
                with open(mapping_path, "w", encoding="utf-8") as f:
                    json.dump(serializable_map, f)

                sig.log_signal.emit(f"  面数: {num_faces}, 视角数: {num_views}")
                sig.log_signal.emit(f"  语义图: {semantic_dir}")
                sig.log_signal.emit(f"  Unique图: {unique_dir}")
                sig.log_signal.emit(f"  颜色映射: {mapping_path}")

                # Step 2
                sig.log_signal.emit("  [Step 2] Mask2Former推理...")
                sig.progress_signal.emit(int(((idx + 0.4) / total) * 80))
                pred_dir = os.path.join(step_output, "pred_masks")
                self._run_inference_subprocess(semantic_dir, model_dir, pred_dir, device, sig)

                # Step 3
                sig.log_signal.emit("  [Step 3] 回传3D面标签...")
                sig.progress_signal.emit(int(((idx + 0.7) / total) * 80))
                from collections import defaultdict
                mapping_path = os.path.join(step_output, "color_face_id_map.json")
                with open(mapping_path, "r", encoding="utf-8") as f:
                    raw_map = json.load(f)
                color_to_face_id = {}
                for key, fid in raw_map.items():
                    r, g, b = [int(x) for x in key.split(",")]
                    color_to_face_id[(r, g, b)] = fid

                face_votes = {}
                for vi in range(1, len(directions) + 1):
                    iname = f"{vi:06d}.png"
                    sp = os.path.join(pred_dir, f"{iname}_seg.npy")
                    ip = os.path.join(pred_dir, f"{iname}_info.json")
                    up = os.path.join(unique_dir, iname)
                    if all(os.path.exists(p) for p in [sp, ip, up]):
                        back_project_single_view(sp, ip, up, face_votes, color_to_face_id)

                face_labels = aggregate_votes(face_votes, num_faces)
                with open(os.path.join(step_output, "face_label.json"), "w", encoding="utf-8") as f:
                    json.dump(face_labels, f, ensure_ascii=False, indent=2)

                from collections import Counter
                cc = Counter(info["class_id"] for info in face_labels.values())
                for cls_id in sorted(cc.keys()):
                    sig.log_signal.emit(f"    {CLASS_NAMES.get(cls_id, f'class_{cls_id}')}: {cc[cls_id]} 个面")

                # Step 4
                sig.log_signal.emit("  [Step 4] 生成彩色STEP...")
                sig.progress_signal.emit(int(((idx + 0.9) / total) * 80))
                colored_path = os.path.join(step_output, f"{basename}_colored.step")
                colorize_step(step_file, face_labels, colored_path)
                sig.log_signal.emit(f"  完成: {basename}\n")

                all_results[basename] = {
                    "face_labels": face_labels,
                    "step_output": step_output,
                    "step_file": step_file,
                }

            sig.progress_signal.emit(100)
            sig.finished_signal.emit(all_results)

        except Exception as e:
            sig.error_signal.emit(traceback.format_exc())

    def _run_inference_subprocess(self, semantic_dir, model_dir, pred_dir, device, sig):
        import subprocess as sp
        os.makedirs(pred_dir, exist_ok=True)

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
    for img_name in image_files:
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
            result = sp.run([python_exe, temp_script],
                            capture_output=True, text=True, check=True)
            if result.stdout.strip():
                for line in result.stdout.strip().split("\n"):
                    sig.log_signal.emit(f"    {line}")
        except sp.CalledProcessError as e:
            sig.log_signal.emit(f"  推理错误: {e.stderr[:500] if e.stderr else str(e)}")
        finally:
            if os.path.exists(temp_script):
                os.remove(temp_script)

    # ======== 完成回调 ========

    def on_error(self, msg):
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.append_log(f"[ERROR] {msg}")

    def on_finished(self, all_results):
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress_bar.setValue(100)
        self.append_log("=" * 50)
        self.append_log("全部完成!")
        self.append_log("")

        # 填充结果表格（用第一个模型的数据）
        if all_results:
            first_model = list(all_results.values())[0]
            self._face_labels = first_model["face_labels"]
            self._fill_result_table(self._face_labels)
            self._refresh_feature_demo()

            # 重新加载第一个模型的面用于VTK预览（如果还没加载）
            step_file = first_model.get("step_file")
            if step_file and os.path.exists(step_file):
                if not self._face_records or self._step_path != step_file:
                    self._try_load_step_preview(step_file)

            # 加载预览图
            step_output = first_model.get("step_output", "")
            if step_output:
                self._load_preview_images(step_output)

    def _fill_result_table(self, face_labels):
        self.result_table.setRowCount(0)
        for face_id_str, info in sorted(face_labels.items(),
                                          key=lambda x: int(x[0])):
            face_id = int(face_id_str)
            class_id = info.get("class_id", 0)
            class_name = info.get("class_name", "未检测到")
            confidence = info.get("confidence", 0.0)
            votes = info.get("votes", {})

            main_vote = "-"
            avg_score = 0.0
            vote_count = 0
            if votes:
                best_cls = max(votes.items(), key=lambda x: x[1]["view_count"])
                main_vote = f"{best_cls[1]['class_name']}({best_cls[1]['view_count']}票)"
                avg_score = best_cls[1].get("avg_score", 0)
                vote_count = sum(v["view_count"] for v in votes.values())

            row = self.result_table.rowCount()
            self.result_table.insertRow(row)
            self.result_table.setItem(row, 0, QTableWidgetItem(str(face_id)))
            self.result_table.setItem(row, 1, QTableWidgetItem(str(class_id)))

            name_item = QTableWidgetItem(class_name)
            if class_id > 0:
                color_map = {1: "#ff3c3c", 2: "#3cc83c",
                             3: "#3c3cff", 4: "#dcc828"}
                name_item.setForeground(QColor(color_map.get(class_id, "#000000")))
            self.result_table.setItem(row, 2, name_item)
            self.result_table.setItem(row, 3, QTableWidgetItem(f"{confidence:.2f}"))
            self.result_table.setItem(row, 4, QTableWidgetItem(str(vote_count)))
            self.result_table.setItem(row, 5, QTableWidgetItem(main_vote))
            self.result_table.setItem(row, 6, QTableWidgetItem(f"{avg_score:.4f}"))

    def _load_preview_images(self, step_output):
        semantic_dir = os.path.join(step_output, "semantic_views")
        if os.path.isdir(semantic_dir):
            imgs = sorted([f for f in os.listdir(semantic_dir) if f.endswith(".png")])
            if imgs:
                p = QPixmap(os.path.join(semantic_dir, imgs[0])).scaled(
                    800, 800, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                lbl = getattr(self, "_imgtab_语义染色图", None)
                if lbl:
                    lbl.setPixmap(p)

        unique_dir = os.path.join(step_output, "unique_views")
        if os.path.isdir(unique_dir):
            imgs = sorted([f for f in os.listdir(unique_dir) if f.endswith(".png")])
            if imgs:
                p = QPixmap(os.path.join(unique_dir, imgs[0])).scaled(
                    800, 800, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                lbl = getattr(self, "_imgtab_Unique面ID图", None)
                if lbl:
                    lbl.setPixmap(p)

    def stop_recognition(self):
        if hasattr(self, '_worker_thread') and self._worker_thread.is_alive():
            self.append_log("正在停止...")
            self.btn_stop.setEnabled(False)
            self.btn_run.setEnabled(True)


# ==================== 启动 ====================

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = FeatureRecognitionUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
