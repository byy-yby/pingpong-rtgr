"""YOLO11n 灰度原生人体检测器（单通道输入，onnxruntime + TensorRT FP16 + 批量）。

替换 rtmlib 的 YOLOX（humanart）做人检测：灰度原生（ch=1）模型直接吃单通道 Mono8，
省掉 ``gray→3ch`` 复制；NMS/解码在 numpy 上**只对 person 一类**做（vs YOLOX 对 80 类
逐类 NMS），CPU 开销从 ~9ms 降到 ~0.1ms 量级。

模型由 ``scripts/export_yolo11_person.py`` 从 ``data/weights/gray/yolo11n-grayscale.pt``
导出：输入 ``(batch, 1, 640, 640)`` [0,1] float、输出 ``(batch, 84, 8400)`` raw
（84 = 4 box(cxcywh) + 80 类，person=类 0，分数在 ``out[:, 4, :]``）。

用法：与 rtmlib YOLOX 输出同一语义的 xyxy 框列表，RTMPose 直接吃。
"""
from __future__ import annotations

import hashlib
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np

from ...core.types import Frame
from ..ball.yolo_ball import _letterbox, _nms  # 纯函数复用

__all__ = ["Yolo11PersonDetector"]

_TRT_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "tabletennis",
                              "trt_engines")
_SCALE = 1.0 / 255.0  # uint8 -> [0,1] float
_PERSON_CLS = 0       # COCO person


class Yolo11PersonDetector:
    """ONNX 单通道人体检测器（无状态，可跨相机复用）。

    Args:
        onnx_path: 导出的单通道动态 batch ONNX（``export_yolo11_person.py`` 产物）。
        imgsz: 输入边长（模型在该尺寸训练/导出），默认 640。
        conf_thresh: person 置信度阈值，默认 0.35。
        iou_thresh: NMS IoU 阈值，默认 0.45。
        roi: 可选 Fixed ROI ``(x0, y0, x1, y1)``（全图像素坐标）。设了就先裁剪该区域
            再 letterbox，检测框再平移回全图坐标；默认 None=全图。
        backend: ``"auto"``（有 TRT 走 TRT FP16，否则 CUDA/CPU）/ ``"tensorrt"`` /
            ``"cuda"`` / ``"cpu"``。
    """

    def __init__(
        self,
        onnx_path: str,
        imgsz: int = 416,
        conf_thresh: float = 0.35,
        iou_thresh: float = 0.45,
        roi: Optional[Tuple[int, int, int, int]] = None,
        backend: str = "auto",
    ) -> None:
        import onnxruntime as ort

        if "CUDAExecutionProvider" in ort.get_available_providers():
            from ..gpu_env import preload_nvidia_libs
            preload_nvidia_libs()

        self.imgsz = int(imgsz)
        self.conf_thresh = float(conf_thresh)
        self.iou_thresh = float(iou_thresh)
        self.roi = tuple(int(v) for v in roi) if roi is not None else None
        self.backend = backend

        available = ort.get_available_providers()
        use_trt = (
            "TensorrtExecutionProvider" in available
            and backend in ("tensorrt", "auto")
        )
        if backend == "cpu":
            self.session = ort.InferenceSession(onnx_path,
                                                providers=["CPUExecutionProvider"])
        elif use_trt:
            # engine 缓存按 onnx 内容哈希分目录（换权重必换缓存，杜绝跨权重复用旧引擎）。
            onnx_hash = hashlib.md5(open(onnx_path, "rb").read()).hexdigest()[:16]
            self.engine_cache_path = os.path.join(_TRT_CACHE_DIR, onnx_hash)
            os.makedirs(self.engine_cache_path, exist_ok=True)
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            providers = [
                ("TensorrtExecutionProvider", {
                    "device_id": 0,
                    "trt_fp16_enable": True,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": self.engine_cache_path,
                }),
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]
            self.session = ort.InferenceSession(onnx_path, sess_options=so,
                                                providers=providers)
        else:
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if "CUDAExecutionProvider" in available
                else ["CPUExecutionProvider"]
            )
            self.session = ort.InferenceSession(onnx_path, providers=providers)

        in_shape = self.session.get_inputs()[0].shape
        self.input_name = self.session.get_inputs()[0].name
        self._batch_axis = in_shape[0] if len(in_shape) == 4 else None
        self._batch_supported = isinstance(self._batch_axis, str)  # 动态 batch
        self.actual_provider = self.session.get_providers()[0]

        # 预处理 buffer 复用：单通道，避免每帧临时大数组分配
        self._inp = np.empty((1, 1, self.imgsz, self.imgsz), dtype=np.float32)
        self._inp_batch = np.empty((4, 1, self.imgsz, self.imgsz), dtype=np.float32)

    # -- 预处理 / 后处理 --

    def _fill_input(self, gray: np.ndarray, dst: np.ndarray,
                    roi: Optional[Tuple[int, int, int, int]] = None
                    ) -> Tuple[int, int, float, float, float, int, int]:
        """letterbox + 归一化直接写入 dst（(1,H,W) float32）。

        返回 ``(H, W, r, dw, dh, ox, oy)``：H/W 为裁剪后尺寸（无 ROI 时=原图），
        ox/oy 为 ROI 左上角在**全图**坐标（无 ROI 时=0），用于把检测框平移回全图。

        ``roi`` 为**本次调用**的临时 ROI（None = 用构造时的 ``self.roi``；显式传
        ``(-1, -1, -1, -1)`` 可强制全图）。ROI 引导重检测靠它。
        """
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        if roi is None:
            roi = self.roi
        elif roi[0] < 0:
            roi = None
        ox = oy = 0
        if roi is not None:
            x0, y0, x1, y1 = (int(v) for v in roi)
            x0 = max(0, x0); y0 = max(0, y0)
            x1 = min(gray.shape[1], x1); y1 = min(gray.shape[0], y1)
            if x1 - x0 >= 8 and y1 - y0 >= 8:
                gray = gray[y0:y1, x0:x1]
                ox, oy = x0, y0
        H, W = gray.shape[:2]
        lb, (r, dw, dh) = _letterbox(gray, (self.imgsz, self.imgsz))
        np.multiply(lb, _SCALE, out=dst[0])  # uint8 -> float32，单遍
        return H, W, r, dw, dh, ox, oy

    def _postprocess(self, det: np.ndarray, H: int, W: int,
                     r: float, dw: float, dh: float,
                     ox: int, oy: int,
                     conf_thresh: Optional[float] = None,
                     return_scores: bool = False):
        """单帧 ``(84, N)`` raw 输出 -> 全图坐标 xyxy 框 ``(M, 4)``（仅 person 类）。

        ``conf_thresh`` 为本次调用的临时阈值（None = ``self.conf_thresh``）；
        ``return_scores`` 为 True 时返回 ``(boxes, scores)``（ROI 重检测要用框置信度）。
        """
        boxes = det[:4, :]       # (4, N) cx,cy,w,h（letterbox 输入坐标）
        confs = det[4, :]        # (N,) person 类 0 分数

        thr = self.conf_thresh if conf_thresh is None else float(conf_thresh)
        keep_mask = confs > thr
        if not keep_mask.any():
            empty = np.zeros((0, 4), dtype=np.float32)
            return (empty, np.zeros((0,), np.float32)) if return_scores else empty
        boxes = boxes[:, keep_mask].T  # (K,4) cxcywh
        confs = confs[keep_mask]

        xyxy = np.empty_like(boxes)
        xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
        xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
        xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
        xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0

        out: List[np.ndarray] = []
        out_scores: List[float] = []
        for i in _nms(xyxy, confs, self.iou_thresh):
            x0, y0, x1, y1 = xyxy[i]
            # unletterbox 回裁剪坐标，再平移到全图坐标（含 ROI 偏移）
            x0 = float(np.clip((x0 - dw) / r, 0.0, W - 1.0)) + ox
            x1 = float(np.clip((x1 - dw) / r, 0.0, W - 1.0)) + ox
            y0 = float(np.clip((y0 - dh) / r, 0.0, H - 1.0)) + oy
            y1 = float(np.clip((y1 - dh) / r, 0.0, H - 1.0)) + oy
            out.append([x0, y0, x1, y1])
            out_scores.append(float(confs[i]))
        boxes = np.asarray(out, dtype=np.float32) if out else np.zeros((0, 4), np.float32)
        if return_scores:
            return boxes, np.asarray(out_scores, dtype=np.float32)
        return boxes

    def detect(self, frame: Frame, *, conf_thresh: Optional[float] = None,
               roi: Optional[Tuple[int, int, int, int]] = None,
               return_scores: bool = False):
        """单帧检测，返回 person 框 ``(M, 4)`` xyxy（全图坐标）。

        ``conf_thresh`` / ``roi`` 是**本次调用**的临时覆盖（None = 构造值），
        供 ROI 引导重检测用低阈值 + 小窗口再检测一次；``return_scores=True`` 时
        额外返回逐框置信度 ``(M,)``（跨相机关联的权重要用）。
        """
        gray = frame.image
        H, W, r, dw, dh, ox, oy = self._fill_input(gray, self._inp[0], roi)
        out = self.session.run(None, {self.input_name: self._inp})[0]  # (1,84,N)
        return self._postprocess(out[0], H, W, r, dw, dh, ox, oy,
                                 conf_thresh, return_scores)

    def detect_batch(self, frames: List[Frame], *,
                     conf_thresh: Optional[float] = None) -> List[np.ndarray]:
        """多帧一次推理（batch=N），返回逐帧 person 框列表。

        batch 维非动态、或帧数异常时自动回退逐帧 ``detect``。
        """
        n = len(frames)
        if n <= 1 or not self._batch_supported or n > self._inp_batch.shape[0]:
            return [self.detect(f, conf_thresh=conf_thresh) for f in frames]
        buf = self._inp_batch[:n]
        metas = [self._fill_input(f.image, buf[i]) for i, f in enumerate(frames)]
        out = self.session.run(None, {self.input_name: buf})[0]  # (n,84,N)
        return [
            self._postprocess(out[i], metas[i][0], metas[i][1], metas[i][2],
                              metas[i][3], metas[i][4], metas[i][5], metas[i][6],
                              conf_thresh)
            for i in range(n)
        ]
