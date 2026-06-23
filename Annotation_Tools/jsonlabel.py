# -*- coding: utf-8 -*-
# 用于标注json文件
# 该工具用于标注json文件，每个json文件对应一个图片，图片中包含多个对象，每个对象都有一个矩形框和一个标签。
# 标签可以是任意字符串，例如"car"、"person"等。
# 标注完成后，每个json文件会包含一个矩形框列表，每个矩形框都有一个标签和一个坐标列表。
# 输入的格式是图片
import sys
import os
import json
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QListWidget, QInputDialog, QMessageBox, 
                             QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, 
                             QGraphicsPolygonItem, QGraphicsTextItem)
from PyQt5.QtGui import QPixmap, QColor, QPolygonF, QPen, QBrush, QPainter, QFont
from PyQt5.QtCore import Qt, QPointF

class ImageViewer(QGraphicsView):
    def __init__(self, parent=None):
        super(ImageViewer, self).__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        self.setRenderHint(QPainter.Antialiasing if hasattr(Qt, 'Antialiasing') else 0)
        
        # 启用以鼠标为中心的缩放锚点
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        
        self.main_window = parent
        self._is_panning = False
        self._pan_start_x = 0
        self._pan_start_y = 0

    def wheelEvent(self, event):
        """鼠标滚轮实现无极缩放"""
        zoom_in_factor = 1.15
        zoom_out_factor = 1 / zoom_in_factor
        if event.angleDelta().y() > 0:
            self.scale(zoom_in_factor, zoom_in_factor)
        else:
            self.scale(zoom_out_factor, zoom_out_factor)

    def mousePressEvent(self, event):
        """左键点击获取坐标进行分割，右键按住拖拽画布"""
        if event.button() == Qt.LeftButton and self.pixmap_item.pixmap().width() > 0:
            scene_pos = self.mapToScene(event.pos())
            x, y = int(scene_pos.x()), int(scene_pos.y())
            if self.main_window:
                self.main_window.auto_segment(x, y)
        elif event.button() == Qt.RightButton:
            self._is_panning = True
            self._pan_start_x = event.x()
            self._pan_start_y = event.y()
            self.setCursor(Qt.ClosedHandCursor)
        super(ImageViewer, self).mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """处理右键平移逻辑"""
        if self._is_panning:
            horizontal_bar = self.horizontalScrollBar()
            vertical_bar = self.verticalScrollBar()
            horizontal_bar.setValue(horizontal_bar.value() - (event.x() - self._pan_start_x))
            vertical_bar.setValue(vertical_bar.value() - (event.y() - self._pan_start_y))
            self._pan_start_x = event.x()
            self._pan_start_y = event.y()
        super(ImageViewer, self).mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """释放右键恢复鼠标样式"""
        if event.button() == Qt.RightButton:
            self._is_panning = False
            self.setCursor(Qt.ArrowCursor)
        super(ImageViewer, self).mouseReleaseEvent(event)

    def set_image(self, image_path):
        """加载图片并重置视图状态"""
        self.scene.clear()
        self.pixmap_item = QGraphicsPixmapItem(QPixmap(image_path))
        self.scene.addItem(self.pixmap_item)
        self.resetTransform()
        self.setSceneRect(self.pixmap_item.boundingRect())
        self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)


class LabelTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CAD特征实例分割标注工具 (多图批处理 + 纯正COCO格式)")
        self.setGeometry(50, 50, 1400, 850)

        # 核心数据管理（支持多张图片切换及记忆）
        self.dataset = {}   
        self.image_paths = [] 
        
        self.current_image_path = ""
        self.cv_image = None
        self.instances = [] 
        self.instance_id_counter = 1

        # ==========================================
        # 标签库：更新为指定的 4 个标准英文特征
        # ==========================================
        self.categories = [
            {"id": 1, "name": "Wide Slot", "color": (255, 0, 0)},       # 宽体槽 (红)
            {"id": 2, "name": "Closed Slot", "color": (0, 255, 0)},     # 封闭槽 (绿)
            {"id": 3, "name": "Open Slot", "color": (0, 0, 255)},       # 开放槽 (蓝)
            {"id": 4, "name": "Hole", "color": (255, 165, 0)}           # 孔 (橙)
        ]

        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # --- 左侧面板 (文件与批处理) ---
        left_panel = QVBoxLayout()
        self.btn_load = QPushButton("📂 1. 导入多张图片")
        self.btn_load.setStyleSheet("font-weight: bold; padding: 10px; background-color: #e0f7fa;")
        self.btn_load.clicked.connect(self.load_images)
        
        left_panel.addWidget(self.btn_load)
        left_panel.addWidget(QLabel("🖼️ 图片列表:"))
        
        self.image_list_widget = QListWidget()
        self.image_list_widget.currentRowChanged.connect(self.on_image_switched)
        left_panel.addWidget(self.image_list_widget)
        
        self.btn_export_json = QPushButton("📄 批量导出 COCO JSON")
        self.btn_export_json.clicked.connect(self.export_json)
        
        self.btn_export_png = QPushButton("🖼️ 批量导出 PNG Mask (白底黑特征)")
        self.btn_export_png.clicked.connect(self.export_png)

        left_panel.addWidget(self.btn_export_json)
        left_panel.addWidget(self.btn_export_png)

        # --- 中间面板 (视图) ---
        center_panel = QVBoxLayout()
        self.viewer = ImageViewer(self)
        center_panel.addWidget(QLabel("💡 操作提示：滚轮缩放 | 右键拖拽 | 左键点击面（同一实例可包含多个面）"))
        center_panel.addWidget(self.viewer)

        # --- 右侧面板 (标签与纠错工具) ---
        right_panel = QVBoxLayout()
        right_panel.addWidget(QLabel("🏷️ 1. 标签库 (Categories):"))
        self.category_list = QListWidget()
        self.update_category_list()
        right_panel.addWidget(self.category_list)
        
        self.btn_add_category = QPushButton("➕ 增加新标签")
        self.btn_add_category.clicked.connect(self.add_category)
        right_panel.addWidget(self.btn_add_category)

        right_panel.addWidget(QLabel("-" * 40))

        right_panel.addWidget(QLabel("📦 2. 图像上的实际物体 (Instances):"))
        self.btn_create_instance = QPushButton("👇 用上方选中的标签【新建一个实例】")
        self.btn_create_instance.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; padding: 5px;")
        self.btn_create_instance.clicked.connect(self.create_instance)
        right_panel.addWidget(self.btn_create_instance)

        self.instance_list_widget = QListWidget()
        right_panel.addWidget(self.instance_list_widget)
        
        self.btn_undo_face = QPushButton("↩️ 撤销该实例上一次点错的【面】")
        self.btn_undo_face.clicked.connect(self.undo_last_face)
        right_panel.addWidget(self.btn_undo_face)

        self.btn_change_category = QPushButton("🔄 将选中实例的【类别】改为上方标签")
        self.btn_change_category.clicked.connect(self.change_instance_category)
        right_panel.addWidget(self.btn_change_category)
        
        self.btn_delete_instance = QPushButton("❌ 删除选中的【整个实例】")
        self.btn_delete_instance.clicked.connect(self.delete_instance)
        right_panel.addWidget(self.btn_delete_instance)

        self.btn_clear_instances = QPushButton("🗑️ 清空当前图片的所有实例")
        self.btn_clear_instances.setStyleSheet("color: red;")
        self.btn_clear_instances.clicked.connect(self.clear_instances)
        right_panel.addWidget(self.btn_clear_instances)

        # 比例布局
        main_layout.addLayout(left_panel, 2)
        main_layout.addLayout(center_panel, 6)
        main_layout.addLayout(right_panel, 3)

    def update_category_list(self):
        self.category_list.clear()
        for cat in self.categories:
            self.category_list.addItem(f"[{cat['id']}] {cat['name']}")
        if self.categories:
            self.category_list.setCurrentRow(0)

    def add_category(self):
        text, ok = QInputDialog.getText(self, '添加类别', '请输入标签名称:')
        if ok and text:
            new_id = max([c['id'] for c in self.categories] + [0]) + 1
            color = tuple(np.random.randint(50, 255, 3).tolist())
            self.categories.append({"id": new_id, "name": text, "color": color})
            self.update_category_list()

    def load_images(self):
        options = QFileDialog.Options()
        files, _ = QFileDialog.getOpenFileNames(self, "选择多张图片", "", "Images (*.png *.jpg *.jpeg *.bmp)", options=options)
        if files:
            for f in files:
                if f not in self.dataset:
                    img_id = len(self.dataset) + 1
                    self.dataset[f] = {'id': img_id, 'instances': [], 'counter': 1}
                    self.image_paths.append(f)
                    self.image_list_widget.addItem(os.path.basename(f))
            if self.image_list_widget.count() > 0 and self.image_list_widget.currentRow() < 0:
                self.image_list_widget.setCurrentRow(0)

    def on_image_switched(self, row):
        """列表切换时加载对应的图片及其标注记忆"""
        if row < 0 or row >= len(self.image_paths): return
        path = self.image_paths[row]
        self.current_image_path = path
        self.cv_image = cv2.imread(path)
        
        # 激活当前图像的存储指针
        self.instances = self.dataset[path]['instances']
        self.instance_id_counter = self.dataset[path]['counter']
        
        self.viewer.set_image(path)
        self.update_instance_list_ui()
        self.redraw_all()

    def create_instance(self):
        if not self.current_image_path: return
        current_cat_idx = self.category_list.currentRow()
        if current_cat_idx < 0: return
        category = self.categories[current_cat_idx]
        
        new_instance = {
            "id": self.instance_id_counter,
            "category_id": category['id'],
            "category_name": category['name'],
            "color": category['color'],
            "contours": [] 
        }
        self.instances.append(new_instance)
        self.instance_id_counter += 1
        self.dataset[self.current_image_path]['counter'] = self.instance_id_counter 
        self.update_instance_list_ui()
        self.instance_list_widget.setCurrentRow(len(self.instances) - 1)

    def update_instance_list_ui(self):
        self.instance_list_widget.clear()
        for inst in self.instances:
            face_count = len(inst['contours'])
            self.instance_list_widget.addItem(f"ID: {inst['id']} | {inst['category_name']} (Faces: {face_count})")

    def auto_segment(self, x, y):
        """核心：漫水填充提取同色面连通域"""
        if self.cv_image is None: return
        h, w = self.cv_image.shape[:2]
        if x < 0 or x >= w or y < 0 or y >= h: return

        current_inst_idx = self.instance_list_widget.currentRow()
        if current_inst_idx < 0:
            QMessageBox.warning(self, "提示", "请先在右侧点击【新建一个实例】！")
            return
            
        target_instance = self.instances[current_inst_idx]
        mask = np.zeros((h + 2, w + 2), np.uint8)
        tolerance = (15, 15, 15) 
        cv2.floodFill(self.cv_image, mask, (x, y), (255, 255, 255), tolerance, tolerance, cv2.FLOODFILL_FIXED_RANGE)
        
        mask_roi = mask[1:-1, 1:-1]
        contours, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: return
        
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) < 20: return 

        target_instance['contours'].append(largest_contour)
        self.update_instance_list_ui()
        self.instance_list_widget.setCurrentRow(current_inst_idx) 
        self.redraw_all()

    def undo_last_face(self):
        current_inst_idx = self.instance_list_widget.currentRow()
        if current_inst_idx < 0: return
        target_instance = self.instances[current_inst_idx]
        if len(target_instance['contours']) > 0:
            target_instance['contours'].pop()
            self.update_instance_list_ui()
            self.instance_list_widget.setCurrentRow(current_inst_idx)
            self.redraw_all()

    def delete_instance(self):
        current_inst_idx = self.instance_list_widget.currentRow()
        if current_inst_idx < 0: return
        del self.instances[current_inst_idx]
        self.update_instance_list_ui()
        if self.instances:
            self.instance_list_widget.setCurrentRow(min(current_inst_idx, len(self.instances) - 1))
        self.redraw_all()

    def change_instance_category(self):
        inst_idx = self.instance_list_widget.currentRow()
        cat_idx = self.category_list.currentRow()
        if inst_idx < 0 or cat_idx < 0: return
        
        target_cat = self.categories[cat_idx]
        target_inst = self.instances[inst_idx]
        target_inst['category_id'] = target_cat['id']
        target_inst['category_name'] = target_cat['name']
        target_inst['color'] = target_cat['color']
        
        self.update_instance_list_ui()
        self.instance_list_widget.setCurrentRow(inst_idx)
        self.redraw_all()

    def redraw_all(self):
        if not self.current_image_path: return
        # 清除之前的标注图形，保留底层图片对象
        for item in self.viewer.scene.items():
            if item != self.viewer.pixmap_item:
                self.viewer.scene.removeItem(item)

        for inst in self.instances:
            if not inst['contours']: continue
            r, g, b = inst['color']
            
            for contour in inst['contours']:
                q_polygon = QPolygonF()
                for point in contour:
                    q_polygon.append(QPointF(float(point[0][0]), float(point[0][1])))
                
                poly_item = QGraphicsPolygonItem(q_polygon)
                poly_item.setPen(QPen(QColor(r, g, b, 255), 2))
                poly_item.setBrush(QBrush(QColor(r, g, b, 120))) 
                self.viewer.scene.addItem(poly_item)

            # 在合并外框上显示 ID
            all_points = np.vstack(inst['contours'])
            bx, by, bw, bh = cv2.boundingRect(all_points)
            
            text_item = QGraphicsTextItem(f"ID:{inst['id']} {inst['category_name']}")
            text_item.setDefaultTextColor(QColor(255, 255, 255))
            text_item.setFont(QFont("Arial", 10, QFont.Bold))
            text_item.setPos(bx, by)
            self.viewer.scene.addRect(text_item.boundingRect().translated(bx, by), 
                                      QPen(Qt.NoPen), QBrush(QColor(0, 0, 0, 150)))
            self.viewer.scene.addItem(text_item)

    def clear_instances(self):
        self.instances.clear() 
        self.instance_id_counter = 1
        self.dataset[self.current_image_path]['counter'] = 1
        self.update_instance_list_ui()
        self.redraw_all()

    # ==========================================
    # 严格的 COCO 顶点格式解析与纯色掩码导出
    # ==========================================
    def export_json(self):
        if not self.dataset: return
        
        output_data = {
            "images": [],
            "annotations": [],
            "categories": [{"id": c["id"], "name": c["name"]} for c in self.categories]
        }

        anno_id_counter = 1

        for img_path, data in self.dataset.items():
            if not data['instances']: continue 
            
            cv_img = cv2.imread(img_path)
            if cv_img is None: continue
            h, w = cv_img.shape[:2]
            
            img_id = data['id']
            output_data["images"].append({
                "id": img_id, 
                "file_name": os.path.basename(img_path), 
                "width": int(w), 
                "height": int(h)
            })

            for inst in data['instances']:
                if not inst['contours']: continue 
                
                # 合并计算边界框
                all_points = np.vstack(inst['contours'])
                bx, by, bw, bh = cv2.boundingRect(all_points)
                total_area = sum([cv2.contourArea(c) for c in inst['contours']])
                
                # 严密打平嵌套，输出纯坐标数组避免触发 RLE
                segmentation_polygons = []
                for contour in inst['contours']:
                    flattened_pts = contour.reshape(-1, 2)
                    poly_coord_list = []
                    for pt in flattened_pts:
                        poly_coord_list.append(float(pt[0])) 
                        poly_coord_list.append(float(pt[1])) 
                    
                    # 只有构成封闭几何的多边形(至少3个点=6个数值)才录入
                    if len(poly_coord_list) >= 6:
                        segmentation_polygons.append(poly_coord_list)

                if not segmentation_polygons: continue

                anno = {
                    "id": int(anno_id_counter),
                    "image_id": int(img_id),
                    "category_id": int(inst['category_id']),
                    "bbox": [int(bx), int(by), int(bw), int(bh)],
                    "segmentation": segmentation_polygons,       
                    "area": float(total_area),
                    "iscrowd": 0
                }
                output_data["annotations"].append(anno)
                anno_id_counter += 1

        save_path, _ = QFileDialog.getSaveFileName(self, "保存标准COCO数据集 JSON", "coco_dataset.json", "JSON Files (*.json)")
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            QMessageBox.information(self, "成功", f"标准格式 JSON 导出成功！\n已被清洗为纯粹的 [x, y] 多边形轮廓。")

    def export_png(self):
        if not self.dataset: return
        
        dir_path = QFileDialog.getExistingDirectory(self, "选择保存 Mask 的文件夹")
        if not dir_path: return

        count = 0
        for img_path, data in self.dataset.items():
            if not data['instances']: continue
            
            cv_img = cv2.imread(img_path)
            if cv_img is None: continue
            h, w = cv_img.shape[:2]
            
            # 【白底黑特征】设计
            mask_image = np.ones((h, w), dtype=np.uint8) * 255
            
            for inst in data['instances']:
                if inst['contours']:
                    cv2.drawContours(mask_image, inst['contours'], -1, 0, thickness=cv2.FILLED)

            base_name = os.path.splitext(os.path.basename(img_path))[0]
            save_path = os.path.join(dir_path, f"{base_name}_mask.png")
            cv2.imwrite(save_path, mask_image)
            count += 1

        QMessageBox.information(self, "成功", f"批量导出成功！\n共生成了 {count} 张白底黑特征的规范 Mask。")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = LabelTool()
    window.show()
    sys.exit(app.exec_())