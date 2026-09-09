"""ViTPose 2D 姿态检测器（top-down：yolo11n 灰度人检测 + ViTPose 热力图关键点）。

RTMPose（SimCC 回归）换 SOTA top-down **ViTPose** 试试是否提升重建质量。权重用
JunkyByte/easy_ViTPose 的 ``coco_25``（COCO-17 + neck/骨盆 + 脚趾/脚跟，共 25 点）
ONNX，与 halpe26 布局只差「halpe 17=头顶」，因此**输出重排成 halpe26(26,3)** 后，
下游三角化 / SMPL 拟合 / pose2d.json / 回放叠加全都无需改动（见 ``COCO25_TO_HALPE26``）。

对比 RTMPoseDetector 的差异与取舍：
- **不依赖 mmpose**：rtmlib 没有 ViTPose 也能跑——这里自建 onnxruntime session，
  只复用 rtmlib 的 top-down 仿射预处理（``bbox_xyxy2cs`` + ``top_down_affine``）与
  DARK 亚像素解码（``post_dark_udp``），数值约定照抄 ``rtmlib/tools/pose_estimation/vitpose.py``。
- **灰度输入**：相机是 Mono8。ViTPose 训练在 RGB 上，这里照旧 gray→3ch 复制（domain
  gap 与 RTMPose 相同）。归一化对灰度三通道相等，故 ImageNet(0-255) 与 torchvision([0,1])
  两套约定**数值等价**，此处用 rtmlib 的 (123.675, 116.28, 103.53)/(58.395, 57.12, 57.375)。
- **逐人推理**：ViTPose ONNX 静态 batch=1，每框一次 ``session.run``（离线重建人物少，够用）。

权重：
- 来源 ``https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/onnx/coco_25/``
- 名称 ``vitpose-s/b/l-coco_25.onnx``（s 97MB / b 360MB / l 1.2GB）
- 本地缓存 ``VITPOSE_DIR``（默认 ``/mnt/newdisk1/vitpose``），``VITPOSE_ONNX`` 可覆盖路径。
  缺失时自动下载到该目录。ViTPose 官方 Apache-2.0（非商用自用没问题）。

关节序（coco_25 = easy_ViTPose 的 BodyWithFeet，25 点）：
  0 nose, 1/2 L/R eye, 3/4 L/R ear, 5 neck, 6/7 L/R shoulder, 8/9 L/R elbow,
  10/11 L/R wrist, 12/13 L/R hip, 14 hip(中), 15/16 L/R knee, 17/18 L/R ankle,
  19 LBigToe, 20 LSmallToe, 21 LHeel, 22 RBigToe, 23 RSmallToe, 24 RHeel
"""
from __future__ import annotations

import logging
import os
import urllib.request
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ...core.types import Frame, Pose2D
from ..detector import PoseDetector
from ..gpu_env import preload_nvidia_libs

logger = logging.getLogger(__name__)

__all__ = ["ViTPosePoseDetector", "COCO25_TO_HALPE26", "coco25_to_halpe26",
           "resolve_vitpose_onnx"]

# easy_ViTPose HF 上的 coco_25（BodyWithFeet，含脚趾/脚跟）ONNX。
VITPOSE_URLS: Dict[str, str] = {
    "vitpose-s-coco_25": ("https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/"
                          "onnx/coco_25/vitpose-s-coco_25.onnx"),
    "vitpose-b-coco_25": ("https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/"
                          "onnx/coco_25/vitpose-b-coco_25.onnx"),
    "vitpose-l-coco_25": ("https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/"
                          "onnx/coco_25/vitpose-l-coco_25.onnx"),
}
DEFAULT_VITPOSE_MODEL = "vitpose-b-coco_25"

# 已知模型的权威文件大小（HF x-linked-size）。本地缓存文件若大小不匹配即视为
# 下载被截断，删除重下——否则 onnxruntime 会拿坏文件报很绕的解析错。
VITPOSE_SIZES: Dict[str, int] = {
    "vitpose-s-coco_25": 97_248_083,
    "vitpose-b-coco_25": 360_063_878,
    "vitpose-l-coco_25": 1_234_336_007,
}

# ImageNet 归一化（0-255 尺度，rtmlib ViTPose 约定）。灰度三通道相等 ⇒ 与 torchvision
# [0,1] 归一化数值等价（(x-123.675)/58.395 == (x/255-0.485)/0.229）。
_VP_MEAN = np.asarray((123.675, 116.28, 103.53), dtype=np.float32)
_VP_STD = np.asarray((58.395, 57.12, 57.375), dtype=np.float32)


def _default_vitpose_dir() -> str:
    """权重目录（新盘 /mnt/newdisk1；VITPOSE_DIR 覆盖）。"""
    return os.environ.get("VITPOSE_DIR") or "/mnt/newdisk1/vitpose"


def resolve_vitpose_onnx(model: str) -> str:
    """把模型名 / URL / 本地路径解析成本地 onnx 路径（缺则下载到缓存目录）。

    Args:
        model: ``vitpose-s/b/l-coco_25`` 之一，或 http(s) URL / 本地 .onnx 路径。

    Returns:
        本地文件路径。``VITPOSE_ONNX`` 环境变量优先级最高（直接当路径用）。
    """
    env = os.environ.get("VITPOSE_ONNX")
    if env:
        return env
    if model in VITPOSE_URLS:
        url = VITPOSE_URLS[model]
        local = os.path.join(_default_vitpose_dir(), f"{model}.onnx")
        expected = VITPOSE_SIZES.get(model)
        # 缓存文件若大小与权威值不符 → 半截文件，删掉重下（保留 .part 续传逻辑）。
        if os.path.exists(local) and expected is not None \
                and os.path.getsize(local) != expected:
            print(f"[ViTPose] 缓存 {local} 大小不符（{os.path.getsize(local)}"
                  f" != {expected}），判定为截断下载，删除重下", flush=True)
            os.remove(local)
        if os.path.exists(local):
            return local
    elif model.startswith(("http://", "https://")):
        url = model
        local = os.path.join(_default_vitpose_dir(), os.path.basename(model))
    else:
        # 本地路径：不存在时报错（不猜测下载）。
        if not os.path.exists(model):
            raise FileNotFoundError(
                f"ViTPose onnx 不存在：{model}。可传 vitpose-s/b/l-coco_25 名（自动下载），"
                f"或设 VITPOSE_ONNX 指向本地文件。")
        return model
    _download(url, local)
    return local


def _download(url: str, dest: str) -> None:
    """分块下载 onnx（避免整块进内存），断点可续传（本地半截文件自动续）。"""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    resume = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    req = urllib.request.Request(url, headers={"User-Agent": "tt-tabletennis"})
    if resume:
        req.add_header("Range", f"bytes={resume}-")
    try:
        print(f"[ViTPose] 下载 {os.path.basename(dest)}（{url}）...", flush=True)
        with urllib.request.urlopen(req, timeout=120) as src, open(tmp, "ab") as fh:
            while True:
                chunk = src.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
        os.replace(tmp, dest)
        print(f"[ViTPose] 下载完成：{dest}", flush=True)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"下载 ViTPose 权重失败：{exc}") from exc


# ----------------------------------------------------------------------
# coco_25 -> halpe26 重排
# ----------------------------------------------------------------------
# coco_25 关节序见模块 docstring；halpe26 序（rtmlib/下游）：0 nose, 1/2 L/R eye,
# 3/4 L/R ear, 5/6 L/R shoulder, 7/8 L/R elbow, 9/10 L/R wrist, 11/12 L/R hip,
# 13/14 L/R knee, 15/16 L/R ankle, 17 head(头顶), 18 neck, 19 hip, 20/21 L/R big toe,
# 22/23 L/R small toe, 24/25 L/R heel。
# coco_25 只差 halpe 的 17(头顶)：其余纯重排。halpe 17 下游本就不映射到 body25，
# 故填 (0,0,0) 即可。
COCO25_TO_HALPE26: List[Tuple[int, int]] = [
    (0, 0),                # nose
    (1, 1), (2, 2), (3, 3), (4, 4),     # 眼 / 耳
    (5, 18),               # neck -> halpe 18
    (6, 5), (7, 6), (8, 7), (9, 8),     # L/R shoulder/elbow
    (10, 9), (11, 10),                  # L/R wrist
    (12, 11), (13, 12),                 # L/R hip
    (14, 19),              # hip(中) -> halpe 19
    (15, 13), (16, 14), (17, 15), (18, 16),  # L/R knee/ankle
    (19, 20), (20, 22), (21, 24),       # L 大趾/小趾/跟
    (22, 21), (23, 23), (24, 25),       # R 大趾/小趾/跟
]
# coco_25 各关节重排后应落在的 body25 关节（测试锚点，防布局悄悄改）。
COCO25_TO_BODY25: Dict[int, int] = {
    0: 0, 1: 16, 2: 15, 3: 18, 4: 17, 5: 1,
    6: 5, 7: 2, 8: 6, 9: 3, 10: 7, 11: 4,
    12: 12, 13: 9, 14: 8, 15: 13, 16: 10, 17: 14, 18: 11,
    19: 19, 20: 20, 21: 21, 22: 22, 23: 23, 24: 24,
}
_HALPE_N = 26


def coco25_to_halpe26(kpts25: np.ndarray) -> np.ndarray:
    """coco_25 (25,3) [x,y,conf] -> halpe26 (26,3)；halpe 17(头顶) 填 0。

    纯重排 + 头部留空，保证下游 ``HALPE26_TO_BODY25`` 映射表无需改动。
    """
    kpts25 = np.asarray(kpts25, dtype=np.float32)
    out = np.zeros((_HALPE_N, 3), dtype=np.float32)
    if kpts25.ndim == 2 and kpts25.shape[0] >= 25:
        for c, h in COCO25_TO_HALPE26:
            out[h] = kpts25[c]
    return out


# ----------------------------------------------------------------------
# 检测器
# ----------------------------------------------------------------------
class ViTPosePoseDetector(PoseDetector):
    """ViTPose top-down 2D 姿态检测器（yolo11n 灰度人检测 + ViTPose 热力图）。

    输出与 :class:`RTMPoseDetector` 同构：``Pose2D.keypoints`` 是 halpe26 布局
    ``(26, 3)``，``skeleton="halpe26"``，可直接替换下游。逐人推理（ONNX batch=1）。

    Args:
        model: 模型名 ``vitpose-s/b/l-coco_25`` / onnx 路径 / URL。
            默认 ``vitpose-b-coco_25``（s 97MB 最快、b 360MB 精度甜点、l 1.2GB 最强）。
        input_size: ViTPose 输入尺寸 ``(H, W)``；None=从 onnx 静态输入读取。
            默认 easy_ViTPose 导出为 256×192 ⇒ (192, 256)。
        det: 人检测器，默认 ``yolo11n-gray``（灰度原生 1ch）。
        det_imgsz: 人检测输入边长，默认 416。
        device: "cuda"/"cpu"。
        backend: "cuda"（onnxruntime CUDA EP）/ "tensorrt"（TRT FP16+缓存）/"cpu"。
        score_thr: 人检测置信度阈值（默认 0.5，同 RTMPoseDetector）。
        nms_thr: 人检测 NMS IoU 阈值。
        det_onnx: 人检测 onnx 路径覆盖（默认取 yolo11n 灰度导出的缓存文件）。
        padding: bbox→crop 的仿射 padding（mmpose 1.25 惯例），默认 1.25。
    """

    def __init__(
        self,
        model: str = DEFAULT_VITPOSE_MODEL,
        input_size: Optional[Tuple[int, int]] = None,
        det: str = "yolo11n-gray",
        det_imgsz: int = 416,
        device: str = "cuda",
        backend: str = "cuda",
        score_thr: float = 0.5,
        nms_thr: float = 0.45,
        det_onnx: Optional[str] = None,
        padding: float = 1.25,
    ) -> None:
        import onnxruntime as ort

        if device != "cpu":
            preload_nvidia_libs()

        onnx_path = resolve_vitpose_onnx(model)

        # 后端：CUDA EP / TRT FP16（复用 rtmpose_pose 的 _trt_session 缓存约定）。
        if backend == "tensorrt":
            from .rtmpose_pose import _default_trt_cache_dir, _trt_session
            self.session = _trt_session(onnx_path, _default_trt_cache_dir())
            use_trt = True
        else:
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if backend != "cpu" and "CUDAExecutionProvider"
                in ort.get_available_providers()
                else ["CPUExecutionProvider"]
            )
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(onnx_path, sess_options=so,
                                                providers=providers)
            use_trt = False
        self._actual_provider = self.session.get_providers()[0]
        if device == "cuda" and not use_trt and \
                self._actual_provider != "CUDAExecutionProvider":
            logger.warning("ViTPose CUDA EP 加载失败（实际 %s），已在 CPU 推理",
                           self._actual_provider)

        # 输入尺寸：优先显式参数，否则读 onnx 静态 H/W（动态则报错提示传参）。
        in_shape = self.session.get_inputs()[0].shape
        self.input_name = self.session.get_inputs()[0].name
        h, w = (in_shape[2], in_shape[3]) if len(in_shape) == 4 else (None, None)
        if input_size is not None:
            self._ih, self._iw = int(input_size[0]), int(input_size[1])
        elif isinstance(h, int) and isinstance(w, int):
            self._ih, self._iw = h, w
        else:
            raise RuntimeError(
                f"ViTPose onnx 输入为动态 {in_shape}，请显式传 input_size=(H, W)")
        # 关节数从输出热力图通道读（勿用输入 in_shape[1]=3=通道）。预期 25（coco_25）。
        self._k = int(self.session.get_outputs()[0].shape[1])
        if self._k != 25:
            raise ValueError(
                f"ViTPose onnx 输出关节数 {self._k} != 25（coco_25 预期）——"
                f"请确认权重是 easy_ViTPose 的 coco_25 变体，而非 coco_17 / mpii 等。")

        # 人检测器（yolo11n 灰度原生，与 RTMPoseDetector 默认一致）。
        from .rtmpose_pose import _default_person_onnx
        self._use_yolo11_det = det == "yolo11n-gray"
        self._det_person = None
        self._det_model = None
        if self._use_yolo11_det:
            from .yolo11_person import Yolo11PersonDetector
            person_backend = ("tensorrt" if backend == "tensorrt"
                              else ("cpu" if device == "cpu" else "cuda"))
            self._det_person = Yolo11PersonDetector(
                det_onnx or _default_person_onnx(det_imgsz),
                imgsz=det_imgsz, conf_thresh=score_thr,
                backend=person_backend)
        else:
            raise ValueError(f"ViTPosePoseDetector 目前只支持 det='yolo11n-gray'，got {det!r}")

        self._padding = float(padding)
        self._skeleton = "halpe26"

        if use_trt:
            # 预热触发 TRT 引擎构建，避免首帧卡顿。
            side = max(self._ih, self._iw)
            dummy = np.zeros((side, side), np.uint8)
            self._det_person.detect(Frame(
                camera_id=0, serial="warmup", frame_num=0, device_timestamp=0,
                host_timestamp=0, image=dummy, pixel_format=17301505,
                width=side, height=side))
            crop, _, _ = self._preprocess_pose(
                np.zeros((side, side, 3), np.uint8), [0, 0, side, side])
            self._forward(crop)
            print("[TensorRT] ViTPose 引擎构建完成 ✓", flush=True)

    # -- 预处理 / 后处理 --

    def _preprocess_pose(self, bgr: np.ndarray, bbox):
        """单框 top-down 仿射预处理。

        返回 ``(crop (H,W,3) 归一化 float32, center, scale)``。注意
        ``top_down_affine`` 的 ``input_size`` 是 ``(w, h)`` 顺序，且会按模型宽高比
        修正 scale 并返回——解码必须用这个修正后的 scale。
        """
        from rtmlib.tools.pose_estimation.pre_processings import (
            bbox_xyxy2cs, top_down_affine)
        center, scale = bbox_xyxy2cs(np.asarray(bbox), padding=self._padding)
        img, scale = top_down_affine((self._iw, self._ih), scale, center, bgr)
        img = np.asarray(img, dtype=np.float32)
        img -= self._mean()
        img /= self._std()
        crop = np.ascontiguousarray(img, dtype=np.float32)  # (H, W, 3)
        return crop, center, scale

    def _mean(self) -> np.ndarray:
        return _VP_MEAN

    def _std(self) -> np.ndarray:
        return _VP_STD

    def _forward(self, crop: np.ndarray) -> np.ndarray:
        """归一化 crop (H,W,3) -> heatmaps (1,K,h,w)。"""
        x = crop.transpose(2, 0, 1)[None, ...]
        return self.session.run(None, {self.input_name: x})[0]

    def _decode(self, heatmaps: np.ndarray, center: np.ndarray,
                scale: np.ndarray, dark_kernel: int = 11) -> Tuple[np.ndarray, np.ndarray]:
        """heatmaps(1,K,h,w) -> (kpts (K,2) 原图坐标, conf (K,))。照抄 rtmlib ViTPose.postprocess。"""
        from rtmlib.tools.pose_estimation.post_processings import post_dark_udp
        N, K, H, W = heatmaps.shape
        flat = heatmaps.reshape(N, K, -1)
        idx = np.argmax(flat, 2).reshape(N, K, 1)
        scores = np.max(flat, 2).reshape(N, K, 1)
        kpts = np.tile(idx, (1, 1, 2)).astype(np.float32)
        kpts[..., 0] %= W
        kpts[..., 1] //= W
        # DARK 亚像素精修（就地改 heatmaps，需正值——sigmoid 输出天然 (0,1)）
        kpts = post_dark_udp(kpts, heatmaps, kernel=dark_kernel)
        kpts = kpts / (np.array([W, H], np.float32) - 1.0) * scale
        kpts = kpts + center - scale / 2.0
        return np.asarray(kpts[0], np.float32), np.asarray(scores[0, :, 0], np.float32)

    # -- 检测接口 --

    def detect(self, frame: Frame) -> List[Pose2D]:
        """单帧 top-down 姿态检测，返回 halpe26 ``Pose2D`` 列表。"""
        if frame.image is None or frame.image.size == 0:
            return []
        bgr = (cv2.cvtColor(frame.image, cv2.COLOR_GRAY2BGR)
               if frame.image.ndim == 2 else frame.image)
        boxes = self._det_person.detect(frame)
        if boxes is None or len(boxes) == 0:
            return []
        return self._detect_bgr_boxes(frame, bgr, boxes)

    def detect_person_boxes(self, frame: Frame, *, conf_thresh: Optional[float] = None,
                            roi=None, return_scores: bool = True):
        """只跑人检测（不跑 ViTPose），返回 ``(boxes (M,4), scores (M,))`` 全图坐标。

        ROI 引导重检测用：小窗口 + 低阈值（如 0.15）再检测一次，命中才值得跑姿态。
        """
        return self._det_person.detect(frame, conf_thresh=conf_thresh, roi=roi,
                                       return_scores=return_scores)

    def detect_on_boxes(self, frame: Frame, boxes) -> List[Pose2D]:
        """对**指定的人框**跑 ViTPose（不重复做人检测）；框为全图像素坐标。"""
        if frame.image is None or frame.image.size == 0:
            return []
        boxes = np.asarray(boxes, dtype=np.float32)
        if boxes.ndim != 2 or len(boxes) == 0:
            return []
        bgr = (cv2.cvtColor(frame.image, cv2.COLOR_GRAY2BGR)
               if frame.image.ndim == 2 else frame.image)
        return self._detect_bgr_boxes(frame, bgr, boxes)

    def detect_batch(self, frames: List[Frame]) -> List[List[Pose2D]]:
        """批处理多帧：人检测一次 batch，ViTPose 逐人逐帧。"""
        n = len(frames)
        results: List[List[Pose2D]] = [[] for _ in range(n)]
        if n == 0:
            return results
        if self._use_yolo11_det:
            boxes_list = self._det_person.detect_batch(frames)
        else:
            boxes_list = [self._det_person.detect(f) for f in frames]
        for i, frame in enumerate(frames):
            if frame.image is None or frame.image.size == 0:
                continue
            boxes = boxes_list[i]
            if boxes is None or len(boxes) == 0:
                continue
            bgr = (cv2.cvtColor(frame.image, cv2.COLOR_GRAY2BGR)
                   if frame.image.ndim == 2 else frame.image)
            results[i] = self._detect_bgr_boxes(frame, bgr, boxes)
        return results

    def _detect_bgr_boxes(self, frame: Frame, bgr: np.ndarray,
                          boxes: np.ndarray) -> List[Pose2D]:
        """对一批人框跑 ViTPose，组装 halpe26 Pose2D 列表。"""
        poses: List[Pose2D] = []
        for bbox in boxes:
            bbox = np.asarray(bbox, dtype=np.float32)[:4]
            crop, center, scale = self._preprocess_pose(bgr, bbox)
            heat = self._forward(crop)
            kpts25, confs25 = self._decode(heat, center, scale)
            kpts26 = coco25_to_halpe26(
                np.concatenate([kpts25, confs25[:, None]], axis=1))
            poses.append(Pose2D(
                camera_id=frame.camera_id,
                keypoints=kpts26,
                score=float(np.max(confs25)) if len(confs25) else 0.0,
                bbox=bbox,
                skeleton=self._skeleton,
            ))
        return poses

    def close(self) -> None:
        self.session = None
        self._det_person = None
        self._det_model = None
