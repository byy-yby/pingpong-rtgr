"""RTMPose 2D 姿态检测器（top-down：YOLOX 检测人 + RTMPose 估计关键点）。

RTMPose 是 top-down 方案：先跑一个轻量人体检测器（YOLOX）拿到每个人的
bounding box，再对每个框跑 RTMPose 回归关键点。相比 one-stage 方案，
关键点定位更准（对后续三角化的 3D 精度更重要），代价是耗时随人数线性增长。

默认模型：
- 姿态：**rtmpose-l-halpe26**（Halpe-26，256×192），26 个关键点（COCO-17 + 头顶/颈/骨盆 + 足趾/脚跟），
  适合后续动作重建与三角化。
- 检测：yolox-m（humanart 人体检测）。

权重自动下载到 ``~/.cache/rtmlib/hub/checkpoints``（rtmlib 内置下载，openmmlab
失效时自动回退 HuggingFace 镜像）。当前跑 CPU（GT 1030 太弱），换好 GPU 后把
``device`` 改成 ``cuda`` 即可。
"""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from ...core.types import Frame, Pose2D
from ..detector import PoseDetector
from ..gpu_env import preload_nvidia_libs

logger = logging.getLogger(__name__)

# RTMPose 官方 ONNX SDK 权重（COCO-17 / body7 与 Halpe-26 两套关键点集）
RTMPOSE_MODEL_URLS = {
    # COCO-17（17 点）
    "rtmpose-l": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
                 "rtmpose-l_simcc-body7_pt-body7_420e-256x192-4dba18fc_20230504.zip",
    "rtmpose-m": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
                 "rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip",
    "rtmpose-s": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
                 "rtmpose-s_simcc-body7_pt-body7_420e-256x192-acd4a1ef_20230504.zip",
    "rtmpose-x": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
                 "rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.zip",
    # Halpe-26（26 点）
    "rtmpose-l-halpe26": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
                         "rtmpose-l_simcc-body7_pt-body7-halpe26_700e-256x192-2abb7558_20230605.zip",
    "rtmpose-m-halpe26": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
                         "rtmpose-m_simcc-body7_pt-body7-halpe26_700e-256x192-4d3e73dd_20230605.zip",
    "rtmpose-s-halpe26": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
                         "rtmpose-s_simcc-body7_pt-body7-halpe26_700e-256x192-7f134165_20230605.zip",
    "rtmpose-x-halpe26": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
                         "rtmpose-x_simcc-body7_pt-body7-halpe26_700e-384x288-7fb6e239_20230606.zip",
}

# 人体检测器（YOLOX humanart）权重
YOLOX_MODEL_URLS = {
    "yolox-m": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
               "yolox_m_8xb8-300e_humanart-c2c7a14a.zip",
    "yolox-x": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
               "yolox_x_8xb8-300e_humanart-a39d44ed.zip",
    "yolox-tiny": "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
                  "yolox_tiny_8xb8-300e_humanart-6f3252f9.zip",
}


class RTMPoseDetector(PoseDetector):
    """RTMPose top-down 2D 姿态检测器（YOLOX 检测 + RTMPose 关键点）。

    Args:
        model: 姿态模型标识，带 ``-halpe26`` 后缀为 26 点模型、不带为 COCO-17：
            "rtmpose-l"/"rtmpose-l-halpe26"/"rtmpose-m"/"rtmpose-m-halpe26"/
            "rtmpose-s"/"rtmpose-s-halpe26"/"rtmpose-x"/"rtmpose-x-halpe26"，
            或本地 onnx 路径 / 下载 URL。
        input_size: 姿态模型输入尺寸 (H, W)，默认 (192, 256)（对应 256×192）。
        det: 人体检测器标识（"yolox-m"/"yolox-x"/"yolox-tiny"）或本地/URL。
        det_input_size: 检测器输入尺寸 (H, W)，默认 (640, 640)。
        device: "cpu" 或 "cuda"。GT 1030 建议 cpu，换好 GPU 后改 cuda。
        backend: 推理后端，默认 "onnxruntime"。
        score_thr: 人体检测置信度阈值（YOLOX），默认 0.5。
        nms_thr: 检测 NMS IoU 阈值，默认 0.45。
        to_openpose: 是否转 OpenPose 输出（默认 False，保持模型自身关键点集）。
    """

    def __init__(
        self,
        model: str = "rtmpose-l-halpe26",
        input_size: tuple = (192, 256),
        det: str = "yolox-m",
        det_input_size: tuple = (640, 640),
        device: str = "cuda",
        backend: str = "onnxruntime",
        score_thr: float = 0.5,
        nms_thr: float = 0.45,
        to_openpose: bool = False,
    ) -> None:
        try:
            from rtmlib import RTMPose, YOLOX
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "缺少 rtmlib 依赖。请先：conda run -n tt pip install rtmlib onnxruntime"
            ) from exc

        # GPU：先预加载 pip 装的 CUDA 运行库（onnxruntime 靠 dlopen 按 soname 找库）。
        if device != "cpu":
            preload_nvidia_libs()

        # 骨架类型由模型名推断：-halpe26 -> 26 点；否则 COCO-17。
        if to_openpose:
            self._skeleton = "openpose18"
        elif "halpe26" in model:
            self._skeleton = "halpe26"
        else:
            self._skeleton = "coco17"

        self._det_model = YOLOX(
            YOLOX_MODEL_URLS.get(det, det),
            model_input_size=det_input_size,
            backend=backend,
            device=device,
            score_thr=score_thr,
            nms_thr=nms_thr,
        )
        self._pose_model = RTMPose(
            RTMPOSE_MODEL_URLS.get(model, model),
            model_input_size=input_size,
            backend=backend,
            device=device,
            to_openpose=to_openpose,
        )

        # 校验 CUDA 是否真正生效：onnxruntime 缺 CUDA 库时会静默回退 CPU
        # （get_available_providers 仍列出 CUDAExecutionProvider，但 session 实际用 CPU）。
        if device == "cuda":
            actual = self._det_model.session.get_providers()
            if not actual or actual[0] != "CUDAExecutionProvider":
                logger.warning(
                    "CUDA EP 加载失败（实际 providers=%s），已在 CPU 上推理；"
                    "请确认已装 onnxruntime-gpu 及匹配的 CUDA 运行库", actual
                )

    def detect(self, frame: Frame) -> List[Pose2D]:
        """对一帧做 top-down 2D 姿态检测。

        本机是黑白相机（单通道灰度），RTMPose 训练在 RGB 上，这里把灰度复制成
        3 通道再送模型（存在 domain gap，靠固定短曝光 + 补光缓解）。
        """
        if frame.image is None or frame.image.size == 0:
            return []

        if frame.image.ndim == 2:
            bgr = np.stack([frame.image] * 3, axis=-1)
        else:
            bgr = frame.image

        # 1) 检测人（返回 xyxy 框，已按 score_thr/nms 过滤）
        bboxes = self._det_model(bgr)
        if bboxes is None or len(bboxes) == 0:
            return []

        # 2) 对每个人框跑关键点估计
        keypoints, scores = self._pose_model(bgr, bboxes=bboxes)
        if keypoints is None or len(keypoints) == 0:
            return []

        poses: List[Pose2D] = []
        for i, (kpts, scs) in enumerate(zip(keypoints, scores)):
            kpts = np.asarray(kpts, dtype=np.float32)
            scs = np.asarray(scs, dtype=np.float32)
            if kpts.ndim == 2:
                kpts = np.concatenate([kpts, scs[:, None]], axis=1)  # (N, 3) [x, y, conf]

            bbox = None
            if i < len(bboxes):
                bbox = np.asarray(bboxes[i], dtype=np.float32)[:4]

            poses.append(
                Pose2D(
                    camera_id=frame.camera_id,
                    keypoints=kpts,
                    score=float(np.max(scs)) if len(scs) else 0.0,
                    bbox=bbox,
                    skeleton=self._skeleton,
                )
            )
        return poses

    def close(self) -> None:
        self._det_model = None
        self._pose_model = None
