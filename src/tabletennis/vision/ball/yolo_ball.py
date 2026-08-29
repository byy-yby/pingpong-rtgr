"""YOLO 球检测器（onnxruntime 推理 + 亚像素精修）。

加载 ``train_ball.py`` 导出的 ONNX 模型，onnxruntime 推理（与 RTMPose 一致，运行时无需
torch），bbox 后用 :func:`refine_ball_center` 精修出亚像素球心。

单类检测（训练时 ``single_cls=True``）：ONNX 输出 ``(1, 5, N)``，5 = 4 个 box
（cx, cy, w, h，letterbox 后的输入坐标系）+ 1 个 class 置信度；需 unletterbox 映射回原图。
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from ...core.types import Ball2D, Frame
from ..detector import BallDetector
from .refine import refine_ball_center

__all__ = ["YoloBallDetector"]


def _letterbox(gray: np.ndarray, new_shape: Tuple[int, int], color: int = 114):
    """等比缩放 + 补灰边到 ``new_shape``。返回 ``(canvas, (r, dw, dh))``，r 为缩放比。"""
    h, w = gray.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = (new_shape[1] - new_unpad[0]) / 2.0
    dh = (new_shape[0] - new_unpad[1]) / 2.0
    resized = cv2.resize(gray, new_unpad, interpolation=cv2.INTER_LINEAR)
    canvas = np.full(new_shape, color, dtype=np.uint8)
    top, left = int(round(dh)), int(round(dw))
    canvas[top:top + new_unpad[1], left:left + new_unpad[0]] = resized
    return canvas, (r, dw, dh)


def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """box (4,) 与 boxes (M,4) 的 IoU（都是 xyxy）。"""
    x0 = np.maximum(box[0], boxes[:, 0])
    y0 = np.maximum(box[1], boxes[:, 1])
    x1 = np.minimum(box[2], boxes[:, 2])
    y1 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x1 - x0, 0.0, None) * np.clip(y1 - y0, 0.0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area + areas - inter + 1e-9)


def _nms(xyxy: np.ndarray, confs: np.ndarray, iou_thresh: float) -> List[int]:
    """手写 NMS（版本无关），返回保留的框索引。"""
    order = np.argsort(confs)[::-1].tolist()
    keep: List[int] = []
    while order:
        i = order.pop(0)
        keep.append(i)
        if not order:
            break
        rest = np.array(order)
        ious = _iou(xyxy[i], xyxy[rest])
        order = [int(j) for j, v in zip(rest, ious) if v < iou_thresh]
    return keep


class YoloBallDetector(BallDetector):
    """ONNX YOLO 球检测器（无状态，可跨相机复用）。"""

    def __init__(
        self,
        model_path: str,
        imgsz: int = 1280,
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45,
        refine: bool = True,
    ) -> None:
        import onnxruntime as ort

        # GPU：先预加载 pip 装的 CUDA 运行库（onnxruntime 靠 dlopen 按 soname 找库；
        # 不预加载会报 libcublasLt.so.12 not found 并静默回退 CPU）。与 rtmpose 一致。
        if "CUDAExecutionProvider" in ort.get_available_providers():
            from ..gpu_env import preload_nvidia_libs
            preload_nvidia_libs()

        self.imgsz = int(imgsz)
        self.conf_thresh = float(conf_thresh)
        self.iou_thresh = float(iou_thresh)
        self.refine_enabled = bool(refine)

        providers = ort.get_available_providers()
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def detect(self, frame: Frame) -> List[Ball2D]:
        gray = frame.image
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape[:2]

        lb, (r, dw, dh) = _letterbox(gray, (self.imgsz, self.imgsz))
        inp = cv2.cvtColor(lb, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
        inp = inp.transpose(2, 0, 1)[None].astype(np.float32)  # (1,3,H,W)

        out = self.session.run(None, {self.input_name: inp})[0]  # (1, 5, N)
        boxes = out[0, :4, :]  # (4, N) cx,cy,w,h（letterbox 空间）
        confs = out[0, 4, :]   # (N,)

        keep_mask = confs > self.conf_thresh
        if not keep_mask.any():
            return []
        boxes = boxes[:, keep_mask].T  # (K,4) cxcywh
        confs = confs[keep_mask]

        xyxy = np.empty_like(boxes)
        xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
        xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
        xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
        xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0

        balls: List[Ball2D] = []
        for i in _nms(xyxy, confs, self.iou_thresh):
            x0, y0, x1, y1 = xyxy[i]
            # unletterbox 回原图
            x0 = (x0 - dw) / r
            x1 = (x1 - dw) / r
            y0 = (y0 - dh) / r
            y1 = (y1 - dh) / r
            x0 = float(np.clip(x0, 0.0, W - 1.0))
            x1 = float(np.clip(x1, 0.0, W - 1.0))
            y0 = float(np.clip(y0, 0.0, H - 1.0))
            y1 = float(np.clip(y1, 0.0, H - 1.0))

            cx = (x0 + x1) / 2.0
            cy = (y0 + y1) / 2.0
            radius = ((x1 - x0) + (y1 - y0)) / 4.0
            conf = float(confs[i])

            if self.refine_enabled:
                x, y, r2, c2 = refine_ball_center(gray, cx, cy, radius, contrast_floor=6.0)
                if c2 > 0.0:
                    cx, cy, radius = x, y, r2

            balls.append(Ball2D(
                camera_id=frame.camera_id,
                center=np.array([cx, cy], dtype=np.float32),
                radius=float(radius),
                confidence=conf,
            ))
        return balls
