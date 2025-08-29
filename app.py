# -*- coding: utf-8 -*-
import os
import sys
import time
import traceback
import tempfile
import shutil
from typing import List, Tuple
import subprocess
import json
import numpy as np
import cv2
import math
import onnxruntime as ort
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QColor, QFont, QPainter
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QListWidget,
    QFileDialog, QHBoxLayout, QVBoxLayout, QListWidgetItem, QProgressBar,
    QTextEdit, QDialog, QMessageBox, QRadioButton, QDoubleSpinBox,
    QColorDialog, QFormLayout, QGroupBox, QStatusBar, QGridLayout, QSplashScreen
)

# ----------------------------- 基本路径与配置文件 -----------------------------
def get_base_dir():
    """
    返回适合读写的基础目录：
    - PyInstaller onefile：程序所在目录（便于写 processed_images、config.json）
    - 其他：源文件同级目录
    """
    if getattr(sys, 'frozen', False):
        return os.path.dirname(os.path.abspath(sys.executable))
    else:
        return os.path.dirname(os.path.abspath(__file__))

def get_resource_path(*parts):
    """
    返回打包后可读资源路径：
    - onefile：sys._MEIPASS 解包目录
    - 其他：源码目录
    """
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        base = sys._MEIPASS
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, *parts)

BASE_DIR = get_base_dir()
CONFIG_PATH = os.path.join(BASE_DIR, "config.json")

# ----------------------------- 全局默认参数（会被 config 覆盖） -----------------------------
# 推理阈值
CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
MASK_THRESHOLD = 0.5

# 可视化颜色 BGR (注意内部为 BGR)
MASK_COLOR = (0, 255, 0)
BBOX_COLOR = (255, 0, 0)
DET_COLOR = (0, 255, 0)          # 计数：绿色
EXCLUDED_COLOR = (0, 0, 255)     # 排除：红色

# 掩膜透明度（overlay）
ALPHA = 0.6

# 边界/带宽比例
BAND_WIDTH_RATIO = 0.05

# 计数带配色
BAND_COLOR_ALLOW = (0, 255, 0)
BAND_COLOR_BLOCK = (0, 0, 255)
BAND_ALPHA = 0.4

# ---------- 启动画面（Splash） ----------
SHOW_SPLASH = True          # 是否显示启动页
SPLASH_MIN_MS = 1000        # 启动画面最短显示毫秒数
SPLASH_IMAGE = "splash.png" # 你的图片文件名（默认放在程序同目录；也可写绝对路径）

# ----------------------------- 模型路径 & 结果路径 -----------------------------
MODEL_DIR = os.path.join(BASE_DIR, "model")
DET_MODEL_PATH = os.path.join(MODEL_DIR, "yeast.onnx")
SEG_MODEL_PATH = os.path.join(MODEL_DIR, "mid.onnx")

# 处理结果保存目录（程序启动时会清空）
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_images")

# ----------------------------- 配置读写 -----------------------------
DEFAULT_CONFIG = {
    "CONF_THRESHOLD": CONF_THRESHOLD,
    "IOU_THRESHOLD": IOU_THRESHOLD,
    "MASK_THRESHOLD": MASK_THRESHOLD,
    "MASK_COLOR": "#00ff00",
    "BBOX_COLOR": "#ff0000",
    "DET_COLOR": "#00ff00",
    "EXCLUDED_COLOR": "#0000ff",
    "ALPHA": ALPHA,
    "BAND_WIDTH_RATIO": BAND_WIDTH_RATIO,
    "BAND_COLOR_ALLOW": "#00ff00",
    "BAND_COLOR_BLOCK": "#0000ff",
    "BAND_ALPHA": BAND_ALPHA,
    # 新增 Splash 配置
    "SHOW_SPLASH": SHOW_SPLASH,
    "SPLASH_MIN_MS": SPLASH_MIN_MS,
    "SPLASH_IMAGE": SPLASH_IMAGE
}

def _hex_to_bgr(hexstr: str):
    c = QColor(hexstr)
    r, g, b, _ = c.getRgb()
    return (b, g, r)

def _bgr_to_hex(bgr: Tuple[int, int, int]):
    b, g, r = bgr
    return '#{0:02x}{1:02x}{2:02x}'.format(r, g, b)

def load_config():
    global CONF_THRESHOLD, IOU_THRESHOLD, MASK_THRESHOLD
    global MASK_COLOR, BBOX_COLOR, DET_COLOR, EXCLUDED_COLOR
    global ALPHA, BAND_WIDTH_RATIO, BAND_COLOR_ALLOW, BAND_COLOR_BLOCK, BAND_ALPHA
    global SHOW_SPLASH, SPLASH_MIN_MS, SPLASH_IMAGE

    cfg = DEFAULT_CONFIG.copy()
    try:
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
                file_cfg = json.load(f)
            cfg.update(file_cfg)
        else:
            with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
                json.dump(cfg, f, indent=4, ensure_ascii=False)
    except Exception:
        try:
            with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
                json.dump(cfg, f, indent=4, ensure_ascii=False)
        except Exception:
            pass

    try:
        CONF_THRESHOLD = float(cfg.get("CONF_THRESHOLD", CONF_THRESHOLD))
        IOU_THRESHOLD = float(cfg.get("IOU_THRESHOLD", IOU_THRESHOLD))
        MASK_THRESHOLD = float(cfg.get("MASK_THRESHOLD", MASK_THRESHOLD))
        ALPHA = float(cfg.get("ALPHA", ALPHA))
        BAND_WIDTH_RATIO = float(cfg.get("BAND_WIDTH_RATIO", BAND_WIDTH_RATIO))
        BAND_ALPHA = float(cfg.get("BAND_ALPHA", BAND_ALPHA))

        MASK_COLOR = _hex_to_bgr(cfg.get("MASK_COLOR", DEFAULT_CONFIG["MASK_COLOR"]))
        BBOX_COLOR = _hex_to_bgr(cfg.get("BBOX_COLOR", DEFAULT_CONFIG["BBOX_COLOR"]))
        DET_COLOR = _hex_to_bgr(cfg.get("DET_COLOR", DEFAULT_CONFIG["DET_COLOR"]))
        EXCLUDED_COLOR = _hex_to_bgr(cfg.get("EXCLUDED_COLOR", DEFAULT_CONFIG["EXCLUDED_COLOR"]))
        BAND_COLOR_ALLOW = _hex_to_bgr(cfg.get("BAND_COLOR_ALLOW", DEFAULT_CONFIG["BAND_COLOR_ALLOW"]))
        BAND_COLOR_BLOCK = _hex_to_bgr(cfg.get("BAND_COLOR_BLOCK", DEFAULT_CONFIG["BAND_COLOR_BLOCK"]))

        SHOW_SPLASH = bool(cfg.get("SHOW_SPLASH", SHOW_SPLASH))
        SPLASH_MIN_MS = int(cfg.get("SPLASH_MIN_MS", SPLASH_MIN_MS))
        SPLASH_IMAGE = cfg.get("SPLASH_IMAGE", SPLASH_IMAGE)
    except Exception:
        pass

def save_config():
    cfg = {
        "CONF_THRESHOLD": CONF_THRESHOLD,
        "IOU_THRESHOLD": IOU_THRESHOLD,
        "MASK_THRESHOLD": MASK_THRESHOLD,
        "MASK_COLOR": _bgr_to_hex(MASK_COLOR),
        "BBOX_COLOR": _bgr_to_hex(BBOX_COLOR),
        "DET_COLOR": _bgr_to_hex(DET_COLOR),
        "EXCLUDED_COLOR": _bgr_to_hex(EXCLUDED_COLOR),
        "ALPHA": ALPHA,
        "BAND_WIDTH_RATIO": BAND_WIDTH_RATIO,
        "BAND_COLOR_ALLOW": _bgr_to_hex(BAND_COLOR_ALLOW),
        "BAND_COLOR_BLOCK": _bgr_to_hex(BAND_COLOR_BLOCK),
        "BAND_ALPHA": BAND_ALPHA,
        "SHOW_SPLASH": SHOW_SPLASH,
        "SPLASH_MIN_MS": SPLASH_MIN_MS,
        "SPLASH_IMAGE": SPLASH_IMAGE
    }
    try:
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, indent=4, ensure_ascii=False)
    except Exception:
        pass

load_config()

def resolve_splash_path() -> str:
    # 绝对路径：直接使用
    if os.path.isabs(SPLASH_IMAGE) and os.path.exists(SPLASH_IMAGE):
        return SPLASH_IMAGE
    # 先在可读资源目录找（适配 PyInstaller onefile）
    cand1 = get_resource_path(SPLASH_IMAGE)
    if os.path.exists(cand1):
        return cand1
    # 再在可写的 BASE_DIR 找
    cand2 = os.path.join(BASE_DIR, SPLASH_IMAGE)
    if os.path.exists(cand2):
        return cand2
    # 兜底：返回资源目录下的同名文件（即便不存在，后面会生成占位图）
    return cand1

# ----------------------------- 工具函数 -----------------------------
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def letterbox_preprocess(img, target_size=(640, 640)):
    orig_h, orig_w = img.shape[:2]
    target_w, target_h = target_size
    scale = min(target_w / orig_w, target_h / orig_h)
    nw, nh = int(round(orig_w * scale)), int(round(orig_h * scale))
    dw = (target_w - nw) // 2
    dh = (target_h - nh) // 2

    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target_h, target_w, 3), 128, dtype=np.uint8)
    canvas[dh:dh + nh, dw:dw + nw] = resized

    blob = canvas.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[np.newaxis, ...]
    return blob, (scale, dw, dh), (orig_h, orig_w), (nh, nw)

def nms(boxes, scores, iou_threshold=0.5):
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
    return keep

def generate_masks(masks_coeff, protos, boxes, padding_info, original_shape):
    scale, dw, dh, scaled_size = padding_info
    nh, nw = scaled_size
    orig_h, orig_w = original_shape

    protos_flat = protos.reshape(protos.shape[0], -1).astype(np.float32)

    final_masks = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2, conf = box
        x1 = max(0, min(x1, 640))
        y1 = max(0, min(y1, 640))
        x2 = max(0, min(x2, 640))
        y2 = max(0, min(y2, 640))

        coeff = masks_coeff[i:i + 1].astype(np.float32)
        mask160 = coeff @ protos_flat
        mask160 = 1 / (1 + np.exp(-mask160))
        mask160 = mask160.reshape(160, 160)

        mask640 = cv2.resize(mask160, (640, 640), interpolation=cv2.INTER_LINEAR)
        cropped_mask = mask640[dh:dh + nh, dw:dw + nw]

        if scale != 1:
            cropped_mask = cv2.resize(cropped_mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        final_mask = np.zeros((orig_h, orig_w), dtype=np.float32)

        orig_x1 = int(np.floor((x1 - dw) / scale))
        orig_y1 = int(np.floor((y1 - dh) / scale))
        orig_x2 = int(np.ceil((x2 - dw) / scale))
        orig_y2 = int(np.ceil((y2 - dh) / scale))

        orig_x1 = max(0, min(orig_x1, orig_w - 1))
        orig_y1 = max(0, min(orig_y1, orig_h - 1))
        orig_x2 = max(0, min(orig_x2, orig_w))
        orig_y2 = max(0, min(orig_y2, orig_h))
        orig_box_w = orig_x2 - orig_x1
        orig_box_h = orig_y2 - orig_y1

        if orig_box_w > 0 and orig_box_h > 0:
            mask_roi = cropped_mask[orig_y1:orig_y2, orig_x1:orig_x2]
            if mask_roi.shape[0] == orig_box_h and mask_roi.shape[1] == orig_box_w:
                final_mask[orig_y1:orig_y2, orig_x1:orig_x2] = mask_roi

        _, mask_bin = cv2.threshold(final_mask, MASK_THRESHOLD, 255, cv2.THRESH_BINARY)
        mask_bin = mask_bin.astype(np.uint8)
        final_masks.append(mask_bin)
    return final_masks

def process_seg_outputs(outputs, original_shape, padding_info):
    output0, output1 = outputs
    outputs_transposed = output0[0].transpose()

    boxes = []
    masks_coeff = []
    mask_protos = output1[0]

    for i in range(outputs_transposed.shape[0]):
        row = outputs_transposed[i]
        cx, cy, w, h = row[:4]
        confidence = row[4]
        if confidence < CONF_THRESHOLD:
            continue
        boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, confidence])
        masks_coeff.append(row[5:37])

    if not boxes:
        return np.array([]), []

    boxes = np.array(boxes)
    masks_coeff = np.array(masks_coeff)

    keep = nms(boxes[:, :4], boxes[:, 4], IOU_THRESHOLD)
    boxes = boxes[keep]
    masks_coeff = masks_coeff[keep]

    full_padding_info = (padding_info[0], padding_info[1], padding_info[2], padding_info[3])
    final_masks = generate_masks(masks_coeff, mask_protos, boxes, full_padding_info, original_shape)
    return boxes, final_masks

# ----------------------------- 旋转矩形“均值边界”细化 -----------------------------
def _normalize_angle_to_global_right(ang_deg: float) -> float:
    if math.cos(math.radians(ang_deg)) < 0:
        ang_deg += 180.0
    if ang_deg >= 180.0:
        ang_deg -= 360.0
    if ang_deg < -180.0:
        ang_deg += 360.0
    return ang_deg

def _refine_rect_with_avg_edges(union_mask: np.ndarray,
                                rect: Tuple[Tuple[float, float], Tuple[float, float], float]
                                ) -> Tuple[float, float, float, float, float, np.ndarray]:
    (cx, cy), (rw, rh), ang = rect
    ang_norm = _normalize_angle_to_global_right(ang)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edge = cv2.morphologyEx(union_mask, cv2.MORPH_GRADIENT, kernel)
    pts = cv2.findNonZero(edge)
    if pts is None or len(pts) < 100:
        box_points = cv2.boxPoints(((cx, cy), (rw, rh), ang_norm)).astype(np.int32)
        return float(cx), float(cy), float(rw), float(rh), float(ang_norm), box_points

    theta = math.radians(-ang_norm)
    cos_t, sin_t = math.cos(theta), math.sin(theta)

    pts = pts.reshape(-1, 2).astype(np.float32)
    dx = pts[:, 0] - cx
    dy = pts[:, 1] - cy
    u = dx * cos_t - dy * sin_t
    v = dx * sin_t + dy * cos_t

    def mean_of_tail(arr: np.ndarray, low_q=0.10, high_q=0.90):
        if arr.size == 0:
            return 0.0, 0.0
        ql = np.quantile(arr, low_q)
        qh = np.quantile(arr, high_q)
        left_mean = np.mean(arr[arr <= ql]) if np.any(arr <= ql) else np.min(arr)
        right_mean = np.mean(arr[arr >= qh]) if np.any(arr >= qh) else np.max(arr)
        return float(left_mean), float(right_mean)

    u_left, u_right = mean_of_tail(u, 0.10, 0.90)
    v_top, v_bottom = mean_of_tail(v, 0.10, 0.90)

    u_c = 0.5 * (u_left + u_right)
    v_c = 0.5 * (v_top + v_bottom)
    w_ref = max(1.0, (u_right - u_left))
    h_ref = max(1.0, (v_bottom - v_top))

    theta_inv = math.radians(ang_norm)
    cos_i, sin_i = math.cos(theta_inv), math.sin(theta_inv)
    gx = u_c * cos_i - v_c * sin_i + cx
    gy = u_c * sin_i + v_c * cos_i + cy

    refined_rect = ((gx, gy), (w_ref, h_ref), ang_norm)
    box_points = cv2.boxPoints(refined_rect).astype(np.int32)
    return float(gx), float(gy), float(w_ref), float(h_ref), float(ang_norm), box_points

# ----------------------------- ONNX Wrapper -----------------------------
class ONNXModelWrapper:
    def __init__(self, model_path: str, providers=None):
        self.model_path = model_path
        self.providers = providers
        self.session = None
        self.input_names = []
        self.output_names = []
        self._load()

    def _load(self):
        if self.providers is None:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = self.providers
        try:
            self.session = ort.InferenceSession(self.model_path, providers=providers)
        except Exception:
            self.session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]

    def run(self, feed_dict: dict, output_names: List[str] = None):
        if output_names is None:
            output_names = self.output_names
        return self.session.run(output_names, feed_dict)

# ----------------------------- 多边形裁剪窗口（人工裁剪） -----------------------------
class PolygonCropper(QDialog):
    def __init__(self, bgr_image: np.ndarray, parent=None, max_display=(1200, 900)):
        super().__init__(parent)
        self.setWindowTitle("多边形裁剪 (左键点选, 右键或完成结束)")
        self.orig_bgr = bgr_image.copy()
        self.points = []
        self.max_display = max_display

        self.display_img_rgb, self.display_scale = self._get_display_image(self.orig_bgr, max_display)
        h_disp, w_disp = self.display_img_rgb.shape[:2]

        self.label = QLabel()
        self.label.setFixedSize(w_disp, h_disp)
        self.pix = self._numpy_to_qpixmap(self.display_img_rgb)
        self.label.setPixmap(self.pix)
        self.label.setCursor(Qt.CrossCursor)

        btn_finish = QPushButton("完成")
        btn_clear = QPushButton("清除")
        btn_undo = QPushButton("撤销")
        btn_cancel = QPushButton("取消")

        btn_finish.clicked.connect(self.on_finish)
        btn_clear.clicked.connect(self.on_clear)
        btn_undo.clicked.connect(self.on_undo)
        btn_cancel.clicked.connect(self.reject)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(btn_clear)
        btn_layout.addWidget(btn_undo)
        btn_layout.addStretch(1)
        btn_layout.addWidget(btn_cancel)
        btn_layout.addWidget(btn_finish)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addLayout(btn_layout)
        self.setLayout(layout)

        self.label.mousePressEvent = self._mouse_press_event
        self.label.mouseMoveEvent = self._mouse_move_event

        self._redraw_display()
        self.last_mouse_pos = None

    def _get_display_image(self, bgr, max_display):
        h, w = bgr.shape[:2]
        mw, mh = max_display
        scale = min(1.0, min(mw / w, mh / h))
        if scale < 1.0:
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            disp = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            disp = bgr.copy()
        disp_rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        return disp_rgb, scale

    def _numpy_to_qpixmap(self, rgb):
        h, w = rgb.shape[:2]
        bytes_per_line = 3 * w
        qimg = QImage(rgb.data.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg)

    def _mouse_press_event(self, event):
        x = event.pos().x()
        y = event.pos().y()
        if event.button() == Qt.LeftButton:
            orig_x = int(round(x / self.display_scale))
            orig_y = int(round(y / self.display_scale))
            orig_x = max(0, min(orig_x, self.orig_bgr.shape[1] - 1))
            orig_y = max(0, min(orig_y, self.orig_bgr.shape[0] - 1))
            self.points.append((orig_x, orig_y))
            self._redraw_display()
        elif event.button() == Qt.RightButton:
            if len(self.points) >= 3:
                self.accept()
            else:
                self.accept()

    def _mouse_move_event(self, event):
        self.last_mouse_pos = (event.pos().x(), event.pos().y())
        self._redraw_display()

    def on_clear(self):
        self.points = []
        self._redraw_display()

    def on_undo(self):
        if self.points:
            self.points.pop()
            self._redraw_display()

    def on_finish(self):
        self.accept()

    def _redraw_display(self):
        disp = self.display_img_rgb.copy()
        if len(self.points) > 0:
            disp_pts = [(int(round(x * self.display_scale)), int(round(y * self.display_scale))) for (x, y) in self.points]
            for i, (px, py) in enumerate(disp_pts):
                cv2.circle(disp, (px, py), 4, (255, 0, 0), -1)
                if i > 0:
                    cv2.line(disp, disp_pts[i - 1], (px, py), (0, 255, 0), 2)
            if len(disp_pts) >= 3:
                cv2.line(disp, disp_pts[-1], disp_pts[0], (0, 255, 0), 2)
            if self.last_mouse_pos is not None and len(disp_pts) > 0:
                lx, ly = self.last_mouse_pos
                cv2.line(disp, disp_pts[-1], (int(lx), int(ly)), (0, 255, 255), 1)

        pix = self._numpy_to_qpixmap(disp)
        self.label.setPixmap(pix)

    def get_cropped_image(self) -> np.ndarray:
        if len(self.points) < 3:
            return self.orig_bgr.copy()
        mask = np.zeros(self.orig_bgr.shape[:2], dtype=np.uint8)
        pts = np.array(self.points, np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 255)
        cropped = cv2.bitwise_and(self.orig_bgr, self.orig_bgr, mask=mask)
        return cropped

# ----------------------------- 启动画面管理 -----------------------------
class SplashManager:
    """
    轻量封装 QSplashScreen：
    - 全程使用你提供的图片作为背景
    - 在模型加载等阶段显示进度文字
    - 最少显示 SPLASH_MIN_MS 毫秒
    """
    def __init__(self, app: QApplication):
        self.app = app
        self.splash = None
        self.start_ts = None

    def start(self):
        img_path = resolve_splash_path()
        pix = QPixmap(img_path)
        if pix.isNull():
            # 兜底生成一张占位图
            pix = QPixmap(900, 540)
            pix.fill(QColor("#F5F7FA"))
            painter = QPainter(pix)
            painter.setPen(QColor("#2d3b36"))
            font = QFont()
            font.setPointSize(20)
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(pix.rect(), Qt.AlignCenter, "CellCounter Pro")
            painter.end()
        # 尽可能铺满但保持比例
        try:
            screen = self.app.primaryScreen()
            if screen is not None:
                geom = screen.geometry()
                w = int(geom.width() * 0.5)
                h = int(geom.height() * 0.5)
                w = max(720, w)
                h = max(420, h)
                pix = pix.scaled(w, h, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
        except Exception:
            pass

        self.splash = QSplashScreen(pix)
        self.splash.setWindowFlag(Qt.FramelessWindowHint, True)
        self.splash.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        self.splash.show()
        self.app.processEvents()
        self.start_ts = time.time()

    def message(self, text: str):
        if not self.splash:
            return
        # 底部居中显示加载文本
        self.splash.showMessage(
            text,
            alignment=Qt.AlignHCenter | Qt.AlignBottom,
            color=Qt.white
        )
        self.app.processEvents()

    def finish(self, main_window: QMainWindow):
        if not self.splash:
            return
        # 确保最短显示时长
        elapsed = int((time.time() - self.start_ts) * 1000) if self.start_ts else 0
        wait_ms = max(0, SPLASH_MIN_MS - elapsed)
        if wait_ms > 0:
            QtCore.QThread.msleep(wait_ms)
        self.splash.finish(main_window)
        self.splash = None

# ----------------------------- 处理线程 -----------------------------
class ProcessWorker(QtCore.QThread):
    progress = pyqtSignal(int)
    single_result = pyqtSignal(str, object, int)
    finished_all = pyqtSignal()
    log = pyqtSignal(str)
    status = pyqtSignal(str)

    def __init__(self, files: List[str], seg_wrapper: ONNXModelWrapper, det_wrapper: ONNXModelWrapper,
                 parent=None, det_conf_thresh=0.25, seg_thresh=0.5, iou_thresh=0.45, mode='auto',
                 processed_dir: str = PROCESSED_DIR):
        super().__init__(parent)
        self.files = files
        self.seg = seg_wrapper
        self.det = det_wrapper
        self._is_running = True
        self.det_conf_thresh = det_conf_thresh
        self.seg_thresh = seg_thresh
        self.iou_thresh = iou_thresh
        self.mode = mode
        self.processed_dir = processed_dir

    def stop(self):
        self._is_running = False

    def run(self):
        total = len(self.files)

        def draw_rotated_band(img, cx, cy, w, h, angle_deg, band_width, side, allow=True):
            color = BAND_COLOR_ALLOW if allow else BAND_COLOR_BLOCK
            bw = band_width
            if side == 'left':
                rect = ((cx - (w / 2 - bw / 2) * math.cos(math.radians(angle_deg)),
                         cy - (w / 2 - bw / 2) * math.sin(math.radians(angle_deg))),
                        (bw, h), angle_deg)
            elif side == 'right':
                rect = ((cx + (w / 2 - bw / 2) * math.cos(math.radians(angle_deg)),
                         cy + (w / 2 - bw / 2) * math.sin(math.radians(angle_deg))),
                        (bw, h), angle_deg)
            elif side == 'top':
                rect = ((cx - (h / 2 - bw / 2) * math.sin(math.radians(angle_deg)),
                         cy + (h / 2 - bw / 2) * math.cos(math.radians(angle_deg))),
                        (w, bw), angle_deg)
            elif side == 'bottom':
                rect = ((cx + (h / 2 - bw / 2) * math.sin(math.radians(angle_deg)),
                         cy - (h / 2 - bw / 2) * math.cos(math.radians(angle_deg))),
                        (w, bw), angle_deg)
            else:
                return

            box = cv2.boxPoints(rect).astype(np.int32)
            overlay = img.copy()
            cv2.fillPoly(overlay, [box], color)
            cv2.addWeighted(overlay, BAND_ALPHA, img, 1 - BAND_ALPHA, 0, img)

        for idx, path in enumerate(self.files):

            basename = os.path.basename(path)

            t0 = time.time()
            if not self._is_running:
                break
            try:
                basename = os.path.basename(path)
                self.log.emit(f"Processing {basename} ({idx + 1}/{total})")
                self.status.emit(f"正在处理 {basename} ({idx + 1}/{total})")
                img_bgr = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img_bgr is None:
                    self.log.emit(f"无法读取图片: {path}")
                    self.progress.emit(int((idx + 1) / total * 100))
                    continue

                if self.mode == 'auto':
                    # =========== seg 推理并生成 union_mask ===========
                    try:
                        input_blob, padding_info, orig_shape, scaled_size = letterbox_preprocess(img_bgr, (640, 640))
                        outputs = self.seg.run({"images": input_blob}, output_names=["output0", "output1"])
                        boxes, masks = process_seg_outputs(outputs, orig_shape,
                                                           (padding_info[0], padding_info[1], padding_info[2],
                                                            scaled_size))
                    except Exception as e:
                        self.log.emit(f"Seg 推理或后处理失败: {e}")
                        boxes, masks = np.array([]), []

                    union_mask = None
                    if masks and len(masks) > 0:
                        union_mask = np.zeros_like(masks[0], dtype=np.uint8)
                        for m in masks:
                            union_mask = cv2.bitwise_or(union_mask, (m > 0).astype(np.uint8) * 255)

                    # =========== 旋转矩形：先 minAreaRect，再基于边界均值细化 ===========
                    bbox_union = None
                    rot_info = None
                    rot_rect_mask = None
                    if union_mask is not None and union_mask.sum() > 0:
                        contours, _ = cv2.findContours(union_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        if contours:
                            c = max(contours, key=cv2.contourArea)
                            rect = cv2.minAreaRect(c)
                            rot_cx, rot_cy, rot_w, rot_h, rot_ang, box_points = _refine_rect_with_avg_edges(union_mask, rect)

                            rot_ang = _normalize_angle_to_global_right(rot_ang)
                            box_points = cv2.boxPoints(((rot_cx, rot_cy), (rot_w, rot_h), rot_ang)).astype(np.int32)

                            xs = box_points[:, 0]
                            ys = box_points[:, 1]
                            left = int(max(0, xs.min()))
                            top = int(max(0, ys.min()))
                            right = int(min(union_mask.shape[1], xs.max()))
                            bottom = int(min(union_mask.shape[0], ys.max()))
                            bbox_union = (left, top, right, bottom)

                            rot_info = (float(rot_cx), float(rot_cy), float(rot_w), float(rot_h), float(rot_ang), box_points)

                            # 使用旋转矩形作为检测区域掩膜

                            try:

                                rot_rect_mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)

                                cv2.fillPoly(rot_rect_mask, [box_points.astype(np.int32)], 255)

                            except Exception:

                                rot_rect_mask = None
                    # =========== crop ===========
                    if bbox_union is None:
                        crop = img_bgr.copy()
                        crop_mask = None
                        crop_left = 0
                        crop_top = 0
                    else:
                        left, top, right, bottom = bbox_union
                        left = max(0, min(left, img_bgr.shape[1] - 1))
                        top = max(0, min(top, img_bgr.shape[0] - 1))
                        right = max(left + 1, min(right, img_bgr.shape[1]))
                        bottom = max(top + 1, min(bottom, img_bgr.shape[0]))
                        crop = img_bgr[top:bottom, left:right].copy()
                        crop_mask = (rot_rect_mask[top:bottom, left:right].copy() if rot_rect_mask is not None else (union_mask[top:bottom, left:right].copy() if union_mask is not None else None))
                        crop_left = left
                        crop_top = top

                    if crop_mask is not None and crop_mask.sum() > 0:
                        masked_crop = cv2.bitwise_and(crop, crop, mask=crop_mask)
                    else:
                        masked_crop = crop

                    # det 推理
                    boxes_det = self._run_det_on_crop(masked_crop)

                    # 绘制结果基底
                    result_img = img_bgr.copy()
                    if union_mask is not None and union_mask.sum() > 0:
                        overlay = result_img.copy()
                        colored_mask = np.zeros_like(result_img)
                        colored_mask[union_mask > 0] = MASK_COLOR
                        cv2.addWeighted(colored_mask, ALPHA, overlay, 1 - ALPHA, 0, overlay)
                        result_img = cv2.addWeighted(result_img, 1 - ALPHA, overlay, ALPHA, 0)

                    if bbox_union is not None and len(boxes_det) > 0 and rot_info is not None:
                        # ============ 稳定“右下”排除 ============
                        def side_indices_from_box(center, box_pts):
                            Cx, Cy = center
                            mids = []
                            for i in range(4):
                                p0 = box_pts[i]
                                p1 = box_pts[(i + 1) % 4]
                                mx = 0.5 * (p0[0] + p1[0])
                                my = 0.5 * (p0[1] + p1[1])
                                mids.append((mx, my))
                            dirs = [(mx - Cx, my - Cy) for (mx, my) in mids]
                            xs = [d[0] for d in dirs]
                            ys = [d[1] for d in dirs]
                            right_i = int(xs.index(max(xs)))
                            left_i = int(xs.index(min(xs)))
                            bottom_i = int(ys.index(max(ys)))
                            top_i = int(ys.index(min(ys)))
                            return left_i, right_i, top_i, bottom_i

                        def band_polygon_for_side(box_pts, side_index, band_w):
                            i = side_index
                            p0 = box_pts[i].astype(float)
                            p1 = box_pts[(i + 1) % 4].astype(float)
                            C = np.mean(box_pts, axis=0).astype(float)
                            mid = 0.5 * (p0 + p1)
                            u = C - mid
                            un = np.linalg.norm(u) + 1e-6
                            u = u / un
                            p1_in = p1 + u * band_w
                            p0_in = p0 + u * band_w
                            poly = np.array([p0, p1, p1_in, p0_in], dtype=np.int32)
                            return poly

                        def draw_band_mask(img, poly, color, alpha, out_mask=None):
                            overlay = img.copy()
                            cv2.fillPoly(overlay, [poly], color)
                            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
                            if out_mask is not None:
                                cv2.fillPoly(out_mask, [poly], 255)

                        rot_cx, rot_cy, rot_w, rot_h, rot_ang, rot_box_points = rot_info
                        bw = max(1, int(BAND_WIDTH_RATIO * max(rot_w, rot_h)))
                        li, ri, ti, bi = side_indices_from_box((rot_cx, rot_cy), rot_box_points)

                        band_mask_right = np.zeros(result_img.shape[:2], dtype=np.uint8)
                        band_mask_bottom = np.zeros_like(band_mask_right)

                        poly_right = band_polygon_for_side(rot_box_points, ri, bw)
                        poly_bottom = band_polygon_for_side(rot_box_points, bi, bw)
                        poly_left = band_polygon_for_side(rot_box_points, li, bw)
                        poly_top = band_polygon_for_side(rot_box_points, ti, bw)

                        draw_band_mask(result_img, poly_left, BAND_COLOR_ALLOW, BAND_ALPHA)
                        draw_band_mask(result_img, poly_top, BAND_COLOR_ALLOW, BAND_ALPHA)
                        draw_band_mask(result_img, poly_right, BAND_COLOR_BLOCK, BAND_ALPHA, out_mask=band_mask_right)
                        draw_band_mask(result_img, poly_bottom, BAND_COLOR_BLOCK, BAND_ALPHA, out_mask=band_mask_bottom)

                        count_included = 0
                        count_excluded = 0
                        for (x1b, y1b, x2b, y2b) in boxes_det:
                            gx1 = int(round(x1b + crop_left))
                            gy1 = int(round(y1b + crop_top))
                            gx2 = int(round(x2b + crop_left))
                            gy2 = int(round(y2b + crop_top))

                            gx1c = max(0, min(gx1, result_img.shape[1] - 1))
                            gx2c = max(0, min(gx2, result_img.shape[1]))
                            gy1c = max(0, min(gy1, result_img.shape[0] - 1))
                            gy2c = max(0, min(gy2, result_img.shape[0]))

                            if gx2c <= gx1c or gy2c <= gy1c:
                                continue

                            roi_r = band_mask_right[gy1c:gy2c, gx1c:gx2c]
                            roi_b = band_mask_bottom[gy1c:gy2c, gx1c:gx2c]
                            touch_right = int(np.any(roi_r))
                            touch_bottom = int(np.any(roi_b))
                            if touch_right or touch_bottom:
                                color = EXCLUDED_COLOR
                                count_excluded += 1
                            else:
                                color = DET_COLOR
                                count_included += 1
                            cv2.rectangle(result_img, (gx1, gy1), (gx2, gy2), color, 2)

                        cv2.polylines(result_img, [rot_box_points], True, BBOX_COLOR, 2)
                    else:
                        for (x1b, y1b, x2b, y2b) in boxes_det:
                            gx1 = int(round(x1b + crop_left))
                            gy1 = int(round(y1b + crop_top))
                            gx2 = int(round(x2b + crop_left))
                            gy2 = int(round(y2b + crop_top))
                            cv2.rectangle(result_img, (gx1, gy1), (gx2, gy2), DET_COLOR, 2)
                        count_included = len(boxes_det)
                        count_excluded = 0  # <<< 修复：确保变量已定义

                    self.log.emit(f"{basename} -> included {count_included}, excluded {count_excluded}")

                    os.makedirs(self.processed_dir, exist_ok=True)
                    save_path = os.path.join(self.processed_dir, basename)
                    try:
                        cv2.imencode(os.path.splitext(save_path)[1] or '.png', result_img)[1].tofile(save_path)
                    except Exception:
                        png_path = os.path.splitext(save_path)[0] + '.png'
                        cv2.imencode('.png', result_img)[1].tofile(png_path)
                        save_path = png_path

                    height, width, channel = result_img.shape
                    bytes_per_line = 3 * width
                    qimg = QImage(result_img.data.tobytes(), width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

                    elapsed = time.time() - t0

                    self.log.emit(f"{basename} 处理时间：{elapsed:.2f}s")

                    self.single_result.emit(path, qimg, count_included)

                else:
                    boxes = np.array([])
                    masks = []
                    bbox_union = None
                    union_mask = None

                    crop = img_bgr.copy()
                    crop_mask = None

                    if crop_mask is not None and crop_mask.sum() > 0:
                        mask_resized = crop_mask
                        masked_crop = cv2.bitwise_and(crop, crop, mask=mask_resized)
                    else:
                        masked_crop = crop

                    boxes_det = self._run_det_on_crop(masked_crop)

                    result_img = img_bgr.copy()
                    if union_mask is not None and union_mask.sum() > 0:
                        overlay = result_img.copy()
                        colored_mask = np.zeros_like(result_img)
                        colored_mask[union_mask > 0] = MASK_COLOR
                        cv2.addWeighted(colored_mask, ALPHA, overlay, 1 - ALPHA, 0, overlay)
                        result_img = cv2.addWeighted(result_img, 1 - ALPHA, overlay, ALPHA, 0)

                    for (x1b, y1b, x2b, y2b) in boxes_det:
                        cv2.rectangle(result_img, (int(x1b), int(y1b)), (int(x2b), int(y2b)), DET_COLOR, 2)

                    count = len(boxes_det)
                    self.log.emit(f"{basename} -> detected {count} boxes")

                    os.makedirs(self.processed_dir, exist_ok=True)
                    save_path = os.path.join(self.processed_dir, basename)
                    try:
                        cv2.imencode(os.path.splitext(save_path)[1] or '.png', result_img)[1].tofile(save_path)
                    except Exception:
                        png_path = os.path.splitext(save_path)[0] + '.png'
                        cv2.imencode('.png', result_img)[1].tofile(png_path)
                        save_path = png_path

                    height, width, channel = result_img.shape
                    bytes_per_line = 3 * width
                    qimg = QImage(result_img.data.tobytes(), width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()

                    elapsed = time.time() - t0

                    self.log.emit(f"{basename} 处理时间：{elapsed:.2f}s")

                    self.single_result.emit(path, qimg, count)

            except Exception as e:
                elapsed = time.time() - t0
                self.log.emit(f"{basename} 处理时间：{elapsed:.2f}s")
                self.log.emit(f"Error processing {path}: {e}\n" + traceback.format_exc())

            self.progress.emit(int((idx + 1) / total * 100))
        self.status.emit("空闲")
        self.finished_all.emit()

    def _run_det_on_crop(self, crop_bgr: np.ndarray) -> List[Tuple[float, float, float, float]]:
        if crop_bgr.size == 0:
            return []
        h0, w0 = crop_bgr.shape[:2]
        inp = self._preprocess_for_det(crop_bgr, (640, 640))
        try:
            outs = self.det.run({self.det.input_names[0]: inp})
        except Exception:
            outs = self.det.run({"images": inp})
        det_out = np.array(outs[0])
        if det_out.ndim == 3:
            det_arr = det_out[0]
        else:
            det_arr = det_out
        if det_arr.shape[0] == 5 and det_arr.shape[1] != 5:
            det_arr = det_arr.T

        boxes = []
        scores = []
        for row in det_arr:
            if row.shape[0] < 5:
                continue
            cx, cy, w, h, conf = float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])
            if conf < self.det_conf_thresh:
                continue
            if max(cx, cy, w, h) <= 1.01:
                scale = 640.0
                cx *= scale
                cy *= scale
                w *= scale
                h *= scale
            x1, y1, x2, y2 = ((cx - w / 2), (cy - h / 2), (cx + w / 2), (cy + h / 2))
            x1 = x1 / 640.0 * w0
            x2 = x2 / 640.0 * w0
            y1 = y1 / 640.0 * h0
            y2 = y2 / 640.0 * h0
            boxes.append([x1, y1, x2, y2])
            scores.append(conf)
        if len(boxes) == 0:
            return []
        boxes = np.array(boxes)
        scores = np.array(scores)
        keep = nms(boxes, scores, iou_threshold=self.iou_thresh)
        boxes = boxes[keep]
        return boxes.astype(np.int32).tolist()

    def _preprocess_for_det(self, img: np.ndarray, target_size=(640, 640)) -> np.ndarray:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_LINEAR)
        arr = img_resized.astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))[None, ...]
        return arr

# ----------------------------- UI 组件 -----------------------------
class ImagePreviewDialog(QDialog):
    def __init__(self, qimg: QImage, title="预览", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.orig_qimg = qimg
        self.scale = 1.0

        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setPixmap(QPixmap.fromImage(self.orig_qimg))

        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidget(self.label)
        self.scroll.setWidgetResizable(True)

        layout = QVBoxLayout()
        layout.addWidget(self.scroll)

        btn_close = QPushButton("关闭")
        btn_zoom_in = QPushButton("放大")
        btn_zoom_out = QPushButton("缩小")
        h = QHBoxLayout()
        h.addStretch(1)
        h.addWidget(btn_zoom_out)
        h.addWidget(btn_zoom_in)
        h.addWidget(btn_close)
        layout.addLayout(h)

        btn_close.clicked.connect(self.accept)
        btn_zoom_in.clicked.connect(self.zoom_in)
        btn_zoom_out.clicked.connect(self.zoom_out)

        self.setLayout(layout)

        try:
            screen = QApplication.primaryScreen()
            if screen is not None:
                geom = screen.availableGeometry()
                max_w = int(geom.width() * 0.8)
                max_h = int(geom.height() * 0.8)
            else:
                max_w, max_h = 1200, 900
        except Exception:
            max_w, max_h = 1200, 900

        img_w = self.orig_qimg.width()
        img_h = self.orig_qimg.height()

        win_w = min(max_w, img_w + 50)
        win_h = min(max_h, img_h + 100)
        self.resize(max(400, win_w), max(300, win_h))

        self._fit_to_window()

    def _fit_to_window(self):
        container_w = max(1, self.width() - 40)
        container_h = max(1, self.height() - 120)
        pix = QPixmap.fromImage(self.orig_qimg).scaled(container_w, container_h, Qt.KeepAspectRatio)
        self.label.setPixmap(pix)

    def zoom_in(self):
        self.scale *= 1.25
        self._apply_scale()

    def zoom_out(self):
        self.scale /= 1.25
        self._apply_scale()

    def _apply_scale(self):
        s = self.scale
        w = max(1, int(self.orig_qimg.width() * s))
        h = max(1, int(self.orig_qimg.height() * s))
        pix = QPixmap.fromImage(self.orig_qimg).scaled(w, h, Qt.KeepAspectRatio)
        self.label.setPixmap(pix)

    def resizeEvent(self, event):
        if abs(self.scale - 1.0) < 1e-6:
            self._fit_to_window()
        super().resizeEvent(event)

class ImageListItemWidget(QWidget):
    def __init__(self, image_path: str, batch_id: int, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.batch_id = batch_id
        self.thumbnail_label = QLabel()
        self.name_label = QLabel(os.path.basename(image_path))
        self.btn_delete = QPushButton('删除')
        self.btn_view_original = QPushButton('原图')
        self.btn_view_result = QPushButton('查看结果')
        self.count_label = QLabel('Count: -')

        h = QHBoxLayout()
        h.setContentsMargins(6, 6, 6, 6)
        h.setSpacing(8)
        self.thumbnail_label.setFixedSize(120, 90)
        h.addWidget(self.thumbnail_label)
        v = QVBoxLayout()
        v.addWidget(self.name_label)
        v.addWidget(self.count_label)
        h.addLayout(v)
        btns = QVBoxLayout()
        btns.addWidget(self.btn_view_original)
        btns.addWidget(self.btn_view_result)
        btns.addWidget(self.btn_delete)
        h.addLayout(btns)
        self.setLayout(h)

        self._load_thumbnail()

    def _load_thumbnail(self):
        try:
            img = cv2.imdecode(np.fromfile(self.image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                return
            h, w = img.shape[:2]
            max_h, max_w = 90, 120
            scale = min(max_w / w, max_h / h, 1.0)
            nw, nh = int(w * scale), int(h * scale)
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
            qimg = QImage(img.data.tobytes(), img.shape[1], img.shape[0], img.shape[1] * 3, QImage.Format_RGB888).rgbSwapped()
            self.thumbnail_label.setPixmap(QPixmap.fromImage(qimg))
        except Exception as e:
            print('thumbnail error', e)

    def set_count(self, c: int):
        self.count_label.setText(f'Count: {c}')

class ModeDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('选择模式')
        self.mode = None
        self.radio_auto = QRadioButton('自动中方格识别裁剪')
        self.radio_manual = QRadioButton('人工裁剪')
        self.radio_auto.setChecked(True)

        btn_cancel = QPushButton('取消')
        btn_ok = QPushButton('确定')

        v = QVBoxLayout()
        v.addWidget(self.radio_auto)
        v.addWidget(self.radio_manual)
        h = QHBoxLayout()
        h.addStretch(1)
        h.addWidget(btn_cancel)
        h.addWidget(btn_ok)
        v.addLayout(h)
        self.setLayout(v)

        btn_ok.clicked.connect(self.on_ok)
        btn_cancel.clicked.connect(self.reject)

    def on_ok(self):
        self.mode = 'auto' if self.radio_auto.isChecked() else 'manual'
        self.accept()

class SettingsDialog(QDialog):
    """提供界面修改全局变量（阈值/颜色等），保存生效并写入 config.json"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("设置")
        self.setModal(True)
        self.resize(560, 560)

        # threshold widgets
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.0, 1.0)
        self.conf_spin.setSingleStep(0.01)
        self.conf_spin.setValue(float(CONF_THRESHOLD))

        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.0, 1.0)
        self.iou_spin.setSingleStep(0.01)
        self.iou_spin.setValue(float(IOU_THRESHOLD))

        self.mask_spin = QDoubleSpinBox()
        self.mask_spin.setRange(0.0, 1.0)
        self.mask_spin.setSingleStep(0.01)
        self.mask_spin.setValue(float(MASK_THRESHOLD))

        # band width ratio
        self.band_ratio_spin = QDoubleSpinBox()
        self.band_ratio_spin.setRange(0.0, 1.0)
        self.band_ratio_spin.setSingleStep(0.01)
        self.band_ratio_spin.setValue(float(BAND_WIDTH_RATIO))

        # mask alpha
        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.0, 1.0)
        self.alpha_spin.setSingleStep(0.05)
        self.alpha_spin.setValue(float(ALPHA))

        # band alpha
        self.band_alpha_spin = QDoubleSpinBox()
        self.band_alpha_spin.setRange(0.0, 1.0)
        self.band_alpha_spin.setSingleStep(0.05)
        self.band_alpha_spin.setValue(float(BAND_ALPHA))

        # color pickers
        self.mask_color_btn = QPushButton("选择")
        self.det_color_btn = QPushButton("选择")
        self.excl_color_btn = QPushButton("选择")
        self.bbox_color_btn = QPushButton("选择")

        # hex labels
        self.mask_color_label = QLabel(_bgr_to_hex(MASK_COLOR))
        self.det_color_label = QLabel(_bgr_to_hex(DET_COLOR))
        self.excl_color_label = QLabel(_bgr_to_hex(EXCLUDED_COLOR))
        self.bbox_color_label = QLabel(_bgr_to_hex(BBOX_COLOR))

        self.mask_color_btn.clicked.connect(lambda: self._pick_color(self.mask_color_label))
        self.det_color_btn.clicked.connect(lambda: self._pick_color(self.det_color_label))
        self.excl_color_btn.clicked.connect(lambda: self._pick_color(self.excl_color_label))
        self.bbox_color_btn.clicked.connect(lambda: self._pick_color(self.bbox_color_label))

        # band colors
        self.band_allow_btn = QPushButton("选择")
        self.band_block_btn = QPushButton("选择")
        self.band_allow_label = QLabel(_bgr_to_hex(BAND_COLOR_ALLOW))
        self.band_block_label = QLabel(_bgr_to_hex(BAND_COLOR_BLOCK))
        self.band_allow_btn.clicked.connect(lambda: self._pick_color(self.band_allow_label))
        self.band_block_btn.clicked.connect(lambda: self._pick_color(self.band_block_label))

        for lab in [self.mask_color_label, self.det_color_label, self.excl_color_label, self.bbox_color_label,
                    self.band_allow_label, self.band_block_label]:
            lab.setMinimumWidth(90)

        for btn in [self.mask_color_btn, self.det_color_btn, self.excl_color_btn, self.bbox_color_btn,
                    self.band_allow_btn, self.band_block_btn]:
            btn.setMinimumWidth(60)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        form.addRow("检测置信阈值:", self.conf_spin)
        form.addRow("NMS IOU 阈值:", self.iou_spin)
        form.addRow("掩膜阈值:", self.mask_spin)
        form.addRow("带宽比例:", self.band_ratio_spin)
        form.addRow("掩膜透明度:", self.alpha_spin)
        form.addRow("计数带透明度:", self.band_alpha_spin)

        color_group = QGroupBox("颜色设置")
        color_grid = QGridLayout()
        color_grid.setHorizontalSpacing(12)
        color_grid.setVerticalSpacing(6)
        def add_color_row(row, name, hex_label, btn):
            name_lab = QLabel(name)
            name_lab.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            color_grid.addWidget(name_lab, row, 0)
            color_grid.addWidget(hex_label, row, 1)
            color_grid.addWidget(btn, row, 2, alignment=Qt.AlignRight)

        add_color_row(0, "中方格颜色:", self.mask_color_label, self.mask_color_btn)
        add_color_row(1, "检测框颜色:", self.bbox_color_label, self.bbox_color_btn)
        add_color_row(2, "正常计数颜色:", self.det_color_label, self.det_color_btn)
        add_color_row(3, "排除计数颜色:", self.excl_color_label, self.excl_color_btn)
        color_group.setLayout(color_grid)

        band_group = QGroupBox("计数带颜色")
        band_grid = QGridLayout()
        band_grid.setHorizontalSpacing(12)
        band_grid.setVerticalSpacing(6)
        def add_band_row(row, name, hex_label, btn):
            name_lab = QLabel(name)
            name_lab.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            band_grid.addWidget(name_lab, row, 0)
            band_grid.addWidget(hex_label, row, 1)
            band_grid.addWidget(btn, row, 2, alignment=Qt.AlignRight)

        add_band_row(0, "允许计数带:", self.band_allow_label, self.band_allow_btn)
        add_band_row(1, "排除计数带:", self.band_block_label, self.band_block_btn)
        band_group.setLayout(band_grid)

        btn_save = QPushButton("保存")
        btn_cancel = QPushButton("取消")
        btn_save.clicked.connect(self.save)
        btn_cancel.clicked.connect(self.reject)
        h = QHBoxLayout()
        h.addStretch(1)
        h.addWidget(btn_cancel)
        h.addWidget(btn_save)

        v = QVBoxLayout()
        v.addLayout(form)
        v.addWidget(color_group)
        v.addWidget(band_group)
        v.addStretch(1)
        v.addLayout(h)
        self.setLayout(v)

    def _pick_color(self, label: QLabel):
        current = label.text()
        c = QColor(current)
        col = QColorDialog.getColor(initial=c, parent=self, options=QColorDialog.ShowAlphaChannel)
        if col.isValid():
            label.setText(col.name())

    def save(self):
        global CONF_THRESHOLD, IOU_THRESHOLD, MASK_THRESHOLD, BAND_WIDTH_RATIO, ALPHA, BAND_ALPHA
        global MASK_COLOR, BBOX_COLOR, DET_COLOR, EXCLUDED_COLOR, BAND_COLOR_ALLOW, BAND_COLOR_BLOCK
        global SHOW_SPLASH, SPLASH_MIN_MS, SPLASH_IMAGE

        CONF_THRESHOLD = float(self.conf_spin.value())
        IOU_THRESHOLD = float(self.iou_spin.value())
        MASK_THRESHOLD = float(self.mask_spin.value())
        BAND_WIDTH_RATIO = float(self.band_ratio_spin.value())
        ALPHA = float(self.alpha_spin.value())
        BAND_ALPHA = float(self.band_alpha_spin.value())

        def hex_to_bgr(hexstr):
            c = QColor(hexstr)
            r, g, b, _ = c.getRgb()
            return (b, g, r)

        MASK_COLOR = hex_to_bgr(self.mask_color_label.text())
        BBOX_COLOR = hex_to_bgr(self.bbox_color_label.text())
        DET_COLOR = hex_to_bgr(self.det_color_label.text())
        EXCLUDED_COLOR = hex_to_bgr(self.excl_color_label.text())
        BAND_COLOR_ALLOW = hex_to_bgr(self.band_allow_label.text())
        BAND_COLOR_BLOCK = hex_to_bgr(self.band_block_label.text())

        # Splash 配置保持原样（可在 config.json 直接编辑）
        save_config()
        self.accept()

class MainWindow(QMainWindow):
    def __init__(self, splash: SplashManager = None):
        super().__init__()
        self._splash = splash
        self._smsg("初始化界面 ...")
        self.setWindowTitle('Yemina')
        self.resize(1200, 820)

        self.batch_counter = 0

        try:
            self._smsg("清理临时结果目录 ...")
            if os.path.exists(PROCESSED_DIR):
                shutil.rmtree(PROCESSED_DIR)
            os.makedirs(PROCESSED_DIR, exist_ok=True)
        except Exception:
            pass

        if not os.path.exists(DET_MODEL_PATH):
            QMessageBox.critical(self, '错误', f'检测模型文件不存在:\n{DET_MODEL_PATH}')
            sys.exit(1)
        if not os.path.exists(SEG_MODEL_PATH):
            QMessageBox.critical(self, '错误', f'分割模型文件不存在:\n{SEG_MODEL_PATH}')
            sys.exit(1)

        try:
            self._smsg("加载分割模型 ...")
            self.seg = ONNXModelWrapper(SEG_MODEL_PATH, providers=None)
            self._smsg("加载检测模型 ...")
            self.det = ONNXModelWrapper(DET_MODEL_PATH, providers=None)
        except Exception as e:
            QMessageBox.critical(self, '模型加载失败', str(e))
            raise

        self.mode = None
        self.worker = None

        self.upload_btn = QPushButton('上传图片 (先选择模式)')
        self.start_btn = QPushButton('开始处理')
        self.start_btn.setEnabled(False)
        self.clear_btn = QPushButton('清空已上传')
        self.settings_btn = QPushButton('设置')
        self.help_btn = QPushButton('帮助')
        self.about_btn = QPushButton('关于')

        top_layout = QHBoxLayout()
        top_layout.addWidget(self.upload_btn)
        top_layout.addWidget(self.start_btn)
        top_layout.addWidget(self.clear_btn)
        top_layout.addWidget(self.settings_btn)
        top_layout.addWidget(self.help_btn)
        top_layout.addWidget(self.about_btn)
        top_layout.addStretch(1)

        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)

        self.preview_label = QLabel('预览区')
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(400, 400)
        self.progress_bar = QProgressBar()
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)

        right_layout = QVBoxLayout()
        right_layout.addWidget(self.preview_label)
        right_layout.addWidget(self.progress_bar)
        right_layout.addWidget(self.log_text)

        main_layout = QHBoxLayout()
        main_layout.addWidget(self.list_widget, 3)
        main_layout.addLayout(right_layout, 5)

        central = QWidget()
        v_layout = QVBoxLayout()
        v_layout.addLayout(top_layout)
        v_layout.addLayout(main_layout)
        central.setLayout(v_layout)
        self.setCentralWidget(central)

        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        self.statusbar.showMessage("空闲")

        self._apply_styles()

        self.upload_btn.clicked.connect(self.on_upload_clicked)
        self.start_btn.clicked.connect(self.on_start_processing)
        self.clear_btn.clicked.connect(self.on_clear_uploaded)
        self.settings_btn.clicked.connect(self.open_settings)
        self.help_btn.clicked.connect(self.show_help)
        self.about_btn.clicked.connect(self.show_about)
        self.list_widget.itemClicked.connect(self.on_item_clicked)

        self.setAcceptDrops(True)
        self._smsg("就绪")

    def _smsg(self, text: str):
        if isinstance(self._splash, SplashManager):
            self._splash.message(text)

    def _apply_styles(self):
        self.setStyleSheet("""
            QMainWindow { background: #f7fbf9; }
            QPushButton {
                background: #e8f5ee;
                border: 1px solid #cfece0;
                padding: 6px 10px;
                border-radius: 6px;
            }
            QPushButton:hover { background: #d7efe3; }
            QLabel { color: #2d3b36; }
            QListWidget { background: white; border: 1px solid #e8e8e8; }
            QTextEdit { background: #ffffff; border: 1px solid #e8e8e8; }
            QProgressBar { border: 1px solid #cfcfcf; height: 14px; border-radius: 6px; }
        """)
        font = self.font()
        font.setPointSize(10)
        self.setFont(font)

    def show_help(self):
        if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
            base_dir = sys._MEIPASS
        else:
            base_dir = os.path.dirname(os.path.abspath(__file__))
        pdf_path = os.path.join(base_dir, "docs", "readme.pdf")
        if os.path.exists(pdf_path):
            try:
                if sys.platform.startswith('win'):
                    os.startfile(pdf_path)
                elif sys.platform.startswith('darwin'):
                    subprocess.run(["open", pdf_path])
                else:
                    subprocess.run(["xdg-open", pdf_path])
            except Exception as e:
                QMessageBox.warning(self, "错误", f"无法打开帮助文档：{e}")
        else:
            QMessageBox.warning(self, "错误", "帮助文档不存在！")

    def show_about(self):
        QMessageBox.information(self, "关于 Yemina",
                                "Yemina\n版本: 2.6\n作者: Deckmiy\n")

    def open_settings(self):
        dlg = SettingsDialog(self)
        if dlg.exec_() == QDialog.Accepted:
            self.statusbar.showMessage(
                f"已更新设置：CONF={CONF_THRESHOLD:.2f} IOU={IOU_THRESHOLD:.2f} MASK={MASK_THRESHOLD:.2f}"
            )

    def _new_batch(self) -> int:
        self.batch_counter += 1
        return self.batch_counter

    def on_upload_clicked(self):
        dlg = ModeDialog(self)
        if dlg.exec_() != QDialog.Accepted:
            return
        mode = dlg.mode
        self.mode = mode
        if mode == 'auto':
            batch_id = self._new_batch()
            paths, _ = QFileDialog.getOpenFileNames(self, '选择图片（可多选）', '', 'Images (*.png *.jpg *.jpeg *.bmp)')
            if not paths:
                return
            for p in paths:
                self._add_image_item(p, batch_id)
            self.start_btn.setEnabled(True)
            self.log(f'已添加 {len(paths)} 张图片（自动模式），设为第 {batch_id} 批')
        else:
            batch_id = self._new_batch()
            added = 0
            while True:
                p, _ = QFileDialog.getOpenFileName(self, '选择单张图片（人工裁剪）', '', 'Images (*.png *.jpg *.jpeg *.bmp)')
                if not p:
                    break
                img_bgr = cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)
                if img_bgr is None:
                    QMessageBox.warning(self, '读取失败', '无法读取图片')
                    break
                cropper = PolygonCropper(img_bgr, self)
                if cropper.exec_() == QDialog.Accepted:
                    res_img = cropper.get_cropped_image()
                    suffix = os.path.splitext(p)[1] if os.path.splitext(p)[1] else '.png'
                    tmpf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                    tmpf.close()
                    res_path = tmpf.name
                    cv2.imencode(suffix, res_img)[1].tofile(res_path)
                    self._add_image_item(res_path, batch_id)
                    added += 1
                    self.log('人工多边形裁剪完成，已加入列表')
                qm = QMessageBox(self)
                qm.setWindowTitle('下一步')
                qm.setText('继续下一张还是开始推理？')
                btn_continue = qm.addButton('继续下一张', QMessageBox.AcceptRole)
                btn_start = qm.addButton('结束裁剪', QMessageBox.AcceptRole)
                qm.exec_()
                if qm.clickedButton() == btn_continue:
                    continue
                else:
                    break
            if added > 0:
                self.start_btn.setEnabled(True)
                self.log(f'人工裁剪加入 {added} 张，设为第 {batch_id} 批')

    def _add_image_item(self, path: str, batch_id: int):
        item = QListWidgetItem(self.list_widget)
        widget = ImageListItemWidget(path, batch_id=batch_id)
        item.setSizeHint(widget.sizeHint())
        self.list_widget.addItem(item)
        self.list_widget.setItemWidget(item, widget)

        widget.btn_delete.clicked.connect(lambda: self._remove_item(item))
        widget.btn_view_original.clicked.connect(lambda: self._view_original(path))
        widget.btn_view_result.clicked.connect(lambda: self._view_result(path))

    def _remove_item(self, item: QListWidgetItem):
        row = self.list_widget.row(item)
        widget = self.list_widget.itemWidget(item)
        try:
            if os.path.dirname(widget.image_path) == tempfile.gettempdir():
                if os.path.exists(widget.image_path):
                    os.remove(widget.image_path)
        except Exception:
            pass
        self.list_widget.takeItem(row)
        if self.list_widget.count() == 0:
            self.start_btn.setEnabled(False)

    def _view_original(self, path):
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            QMessageBox.warning(self, '查看失败', '无法读取图片')
            return
        h, w = img.shape[:2]
        qimg = QImage(img.data.tobytes(), w, h, w * 3, QImage.Format_RGB888).rgbSwapped()
        dlg = ImagePreviewDialog(qimg, title=os.path.basename(path), parent=self)
        dlg.exec_()

    def _view_result(self, path):
        basename = os.path.basename(path)
        candidates = []
        p1 = os.path.join(PROCESSED_DIR, basename)
        candidates.append(p1)
        p2 = os.path.join(PROCESSED_DIR, os.path.splitext(basename)[0] + '.png')
        candidates.append(p2)
        found = None
        for c in candidates:
            if os.path.exists(c):
                found = c
                break
        if not found:
            QMessageBox.information(self, "提示", "尚未生成该图片的处理结果，请先运行处理。")
            return
        img = cv2.imdecode(np.fromfile(found, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            QMessageBox.warning(self, '查看失败', '无法读取处理结果')
            return
        h, w = img.shape[:2]
        qimg = QImage(img.data.tobytes(), w, h, w * 3, QImage.Format_RGB888).rgbSwapped()
        dlg = ImagePreviewDialog(qimg, title=f"{os.path.basename(found)} (处理结果)", parent=self)
        dlg.exec_()

    def on_item_clicked(self, item: QListWidgetItem):
        widget = self.list_widget.itemWidget(item)
        path = widget.image_path
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return
        h, w = img.shape[:2]
        qimg = QImage(img.data.tobytes(), w, h, w * 3, QImage.Format_RGB888).rgbSwapped()
        self.preview_label.setPixmap(QPixmap.fromImage(qimg).scaled(self.preview_label.size(), Qt.KeepAspectRatio))

    def on_start_processing(self):
        if self.list_widget.count() == 0:
            QMessageBox.information(self, '提示', '请先添加图片')
            return

        batch_ids = []
        widgets = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            widget = self.list_widget.itemWidget(item)
            batch_ids.append(getattr(widget, 'batch_id', 0))
            widgets.append(widget)
        if not batch_ids:
            QMessageBox.information(self, '提示', '请先添加图片')
            return
        latest_batch = max(batch_ids)

        files = [w.image_path for w in widgets if getattr(w, 'batch_id', 0) == latest_batch]
        if len(files) == 0:
            QMessageBox.information(self, '提示', '最新一批没有图片，请先添加')
            return

        self.worker = ProcessWorker(files, self.seg, self.det, mode=self.mode, processed_dir=PROCESSED_DIR)
        self.worker.progress.connect(lambda v: self.progress_bar.setValue(v))
        self.worker.single_result.connect(self.on_single_result)
        self.worker.log.connect(self.log)
        self.worker.finished_all.connect(lambda: self.log('全部处理完成'))
        self.worker.status.connect(lambda s: self.statusbar.showMessage(s))
        self.worker.start()
        self.log(f'开始批量处理：第 {latest_batch} 批，共 {len(files)} 张')
        self.statusbar.showMessage("开始批量处理...")

    def on_single_result(self, path: str, qimg: QImage, count: int):
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            widget = self.list_widget.itemWidget(item)
            if widget.image_path == path:
                widget.set_count(count)
                break
        self.preview_label.setPixmap(QPixmap.fromImage(qimg).scaled(self.preview_label.size(), Qt.KeepAspectRatio))
        self.statusbar.showMessage(f"处理完成: {os.path.basename(path)}")

    def on_clear_uploaded(self):
        removed = 0
        for i in reversed(range(self.list_widget.count())):
            item = self.list_widget.item(i)
            widget = self.list_widget.itemWidget(item)
            path = widget.image_path
            try:
                if os.path.dirname(path) == tempfile.gettempdir():
                    if os.path.exists(path):
                        os.remove(path)
            except Exception:
                pass
            self.list_widget.takeItem(i)
            removed += 1
        self.start_btn.setEnabled(False)
        self.preview_label.setText('预览区')
        self.progress_bar.setValue(0)
        self.log(f'已清空列表（删除 {removed} 项；其中人工裁剪临时文件已清理）')

    def log(self, text: str):
        ts = time.strftime('%H:%M:%S')
        self.log_text.append(f"[{ts}] {text}")

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        paths = [u.toLocalFile() for u in urls]
        img_paths = [p for p in paths if os.path.splitext(p)[1].lower() in ('.png', '.jpg', '.jpeg', '.bmp')]
        if not img_paths:
            return
        if self.mode is None:
            dlg = ModeDialog(self)
            if dlg.exec_() != QDialog.Accepted:
                return
            self.mode = dlg.mode
        if self.mode != 'auto':
            QMessageBox.information(self, '提示', '拖拽只在自动模式下支持批量添加。')
            return
        batch_id = self._new_batch()
        for p in img_paths:
            self._add_image_item(p, batch_id)
        if img_paths:
            self.start_btn.setEnabled(True)
            self.log(f'拖拽添加 {len(img_paths)} 张图片，设为第 {batch_id} 批')

# ----------------------------- 启动 -----------------------------
def main():
    app = QApplication(sys.argv)

    splash_mgr = None
    if SHOW_SPLASH:
        splash_mgr = SplashManager(app)
        splash_mgr.start()
        splash_mgr.message("启动中 ...")

    win = MainWindow(splash=splash_mgr)
    if splash_mgr:
        splash_mgr.finish(win)

    win.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
