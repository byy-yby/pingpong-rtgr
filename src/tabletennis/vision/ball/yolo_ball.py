"""YOLO 球检测器（onnxruntime 推理 + 亚像素精修）。

加载 ``train_ball.py`` 导出的 ONNX 模型，onnxruntime 推理（与 RTMPose 一致，运行时无需
torch），bbox 后用 :func:`refine_ball_center` 精修出亚像素球心。

单类检测（训练时 ``single_cls=True``）：ONNX 输出 ``(1, 5, N)``，5 = 4 个 box
（cx, cy, w, h，letterbox 后的输入坐标系）+ 1 个 class 置信度；需 unletterbox 映射回原图。
"""
from __future__ import annotations

import hashlib
import os
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


_TRT_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "tabletennis", "trt_engines")
_SCALE = 1.0 / 255.0  # uint8 → [0,1] float


def _read_onnx_input(model_path: str) -> Tuple[str, list]:
    """读 ONNX 第一个输入的 ``(name, dims)``；dims 里 int=固定、str=动态。

    用于在创建 session **之前**确定输入通道数（1=灰度 / 3=彩色）与 batch 是否动态，
    从而给 TensorRT 配正确的动态 batch profile。
    """
    import onnx

    m = onnx.load(model_path)
    inp = m.graph.input[0]
    dims = []
    for d in inp.type.tensor_type.shape.dim:
        dims.append(d.dim_value if d.dim_value else d.dim_param)
    return inp.name, dims


class YoloBallDetector(BallDetector):
    """ONNX YOLO 球检测器（无状态，可跨相机复用）。

    ``backend`` 取值：
    - ``"auto"``（默认）：有 TensorRT 则走 TRT FP16，否则 CUDA，再否则 CPU。
    - ``"tensorrt"``：显式 TRT（engine 缓存到 ``~/.cache/tabletennis/trt_engines``，
      首次构建 ~30-60s，之后复用；构建失败自动回退 CUDA/CPU）。
    - ``"cuda"`` / ``"cpu"``：固定后端。

    注意：不能把 ``get_available_providers()`` 全量列表传给 InferenceSession——
    本机装了 tensorrt-cu12-libs，全量传会让 CUDA 时也先去建 TRT 引擎（实测 ~52s 阻塞）。
    """

    def __init__(
        self,
        model_path: str,
        imgsz: int = 1280,
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45,
        refine: bool = True,
        backend: str = "auto",
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
        self.backend = backend

        # 创建 session 之前读输入 meta：通道数（1=灰度 / 3=彩色）与 batch 是否动态。
        self.input_name, in_dims = _read_onnx_input(model_path)
        self._channels = (
            int(in_dims[1]) if len(in_dims) >= 2 and isinstance(in_dims[1], int) else 3
        )
        self._batch_supported = isinstance(in_dims[0], str)  # 动态 batch（如 'batch'）

        available = ort.get_available_providers()
        use_trt = (
            "TensorrtExecutionProvider" in available
            and (backend == "tensorrt" or backend == "auto")
        )

        if use_trt:
            # 显式 TRT EP（FP16 + engine 缓存，与 rtmpose 的 _trt_session 一致）。
            # 首次构建引擎较慢，之后从缓存加载；构建失败 onnxruntime 自动回退 CUDA。
            # 关键：缓存目录按 onnx 内容哈希分档。实测 onnxruntime 的 TRT 引擎缓存 key
            # 只按图结构（不含权重）算——换权重不换 key 会静默复用旧引擎（推理白跑旧模型）。
            # 按内容哈希分目录后，换权重必然换目录 → 必然重建，杜绝跨权重复用。
            onnx_hash = hashlib.md5(open(model_path, "rb").read()).hexdigest()[:16]
            self.engine_cache_path = os.path.join(_TRT_CACHE_DIR, onnx_hash)
            os.makedirs(self.engine_cache_path, exist_ok=True)
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            trt_opts = {
                "device_id": 0,
                "trt_fp16_enable": True,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": self.engine_cache_path,
            }
            # 动态 batch 需配 profile（min/opt/max batch，H/W 固定到 imgsz）；灰度与彩色通用。
            if self._batch_supported:
                c, sz = self._channels, self.imgsz
                trt_opts.update({
                    "trt_profile_min_shapes": f"{self.input_name}:1x{c}x{sz}x{sz}",
                    "trt_profile_opt_shapes": f"{self.input_name}:4x{c}x{sz}x{sz}",
                    "trt_profile_max_shapes": f"{self.input_name}:8x{c}x{sz}x{sz}",
                })
            providers = [
                ("TensorrtExecutionProvider", trt_opts),
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]
            self.session = ort.InferenceSession(
                model_path, sess_options=so, providers=providers)
        else:
            # 只选 CUDA（有则用，无则 CPU），显式限定，避免误入 TRT 构建。
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if "CUDAExecutionProvider" in available
                else ["CPUExecutionProvider"]
            )
            self.session = ort.InferenceSession(model_path, providers=providers)

        self.actual_provider = self.session.get_providers()[0]  # 供上层打印实际后端

        # 预处理 buffer 复用：避免每帧 4 次 1280x1280 的临时数组分配（实测 ~27ms → ~5ms）。
        # 通道数随模型自适应：灰度模型 (B,1,H,W)，彩色模型 (B,3,H,W)。
        self._inp = np.empty((1, self._channels, self.imgsz, self.imgsz), dtype=np.float32)
        self._inp_batch = np.empty((4, self._channels, self.imgsz, self.imgsz), dtype=np.float32)

    # -- 预处理 / 后处理（buffer 复用，GPU 不再是瓶颈时才轮到 TRT 生效） --

    def _fill_input(self, gray: np.ndarray, dst: np.ndarray) -> Tuple[int, int, float, float, float]:
        """letterbox + 归一化直接写入 dst（(C,H,W) float32），返回 (H, W, r, dw, dh)。

        灰度模型 C=1 只写通道 0；彩色模型 C=3 复制三份（无需 cvtColor，直接广播）。
        """
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape[:2]
        lb, (r, dw, dh) = _letterbox(gray, (self.imgsz, self.imgsz))
        np.multiply(lb, _SCALE, out=dst[0])  # uint8 → float32，单遍、无中间数组
        for c in range(1, self._channels):
            dst[c] = dst[0]
        return H, W, r, dw, dh

    def _postprocess(self, det: np.ndarray, gray: np.ndarray, camera_id: int,
                     r: float, dw: float, dh: float) -> List[Ball2D]:
        """单帧 (5,N) 原始输出 → NMS + unletterbox + 亚像素精修 → Ball2D 列表。"""
        H, W = gray.shape[:2]
        boxes = det[:4, :]  # (4, N) cx,cy,w,h（letterbox 空间）
        confs = det[4, :]   # (N,)

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
            x0 = float(np.clip((x0 - dw) / r, 0.0, W - 1.0))
            x1 = float(np.clip((x1 - dw) / r, 0.0, W - 1.0))
            y0 = float(np.clip((y0 - dh) / r, 0.0, H - 1.0))
            y1 = float(np.clip((y1 - dh) / r, 0.0, H - 1.0))

            cx = (x0 + x1) / 2.0
            cy = (y0 + y1) / 2.0
            radius = ((x1 - x0) + (y1 - y0)) / 4.0
            conf = float(confs[i])

            if self.refine_enabled:
                x, y, r2, c2 = refine_ball_center(gray, cx, cy, radius, contrast_floor=6.0)
                if c2 > 0.0:
                    cx, cy, radius = x, y, r2

            balls.append(Ball2D(
                camera_id=camera_id,
                center=np.array([cx, cy], dtype=np.float32),
                radius=float(radius),
                confidence=conf,
            ))
        return balls

    def detect(self, frame: Frame) -> List[Ball2D]:
        gray = frame.image
        _h, _w, r, dw, dh = self._fill_input(gray, self._inp[0])
        out = self.session.run(None, {self.input_name: self._inp})[0]  # (1, 5, N)
        return self._postprocess(out[0], gray, frame.camera_id, r, dw, dh)

    def detect_batch(self, frames: List[Frame]) -> List[List[Ball2D]]:
        """多帧一次推理（batch=N），返回逐帧 Ball2D 列表。

        live_control 4 路相机每帧合成一个 batch，GPU 一次跑完，省去 3 次额外
        session.run 的 kernel 启动 / Python 开销（4 路顺序 ~225ms → batch ~几十 ms）。
        模型 batch 维度非动态、或帧数异常时自动回退逐帧 ``detect``。
        """
        n = len(frames)
        if n <= 1 or not self._batch_supported or n > self._inp_batch.shape[0]:
            return [self.detect(f) for f in frames]
        buf = self._inp_batch[:n]
        metas = [self._fill_input(f.image, buf[i]) for i, f in enumerate(frames)]
        out = self.session.run(None, {self.input_name: buf})[0]  # (n, 5, N)
        return [
            self._postprocess(out[i], frames[i].image, frames[i].camera_id,
                              metas[i][2], metas[i][3], metas[i][4])
            for i in range(n)
        ]
