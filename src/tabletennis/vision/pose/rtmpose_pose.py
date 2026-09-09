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
import os
from typing import List, Optional

import cv2
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


def _default_trt_cache_dir() -> str:
    """TensorRT engine 缓存目录（构建一次、之后复用，避免每次启动重建引擎）。"""
    return os.path.join(os.path.expanduser("~"), ".cache", "tabletennis", "trt_engines")


def _trt_session(onnx_path: str, cache_dir: str, fp16: bool = True):
    """用 TensorrtExecutionProvider 创建 onnxruntime 会话（FP16 + engine 缓存）。

    复用 rtmlib 的 pre/postprocess（其 ``inference()`` 只调用 ``self.session.run``），
    这里仅把 session 换成 TRT EP。TRT 对不支持的算子自动回退 CUDA/CPU EP。
    """
    import onnxruntime as ort

    os.makedirs(cache_dir, exist_ok=True)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = [
        ("TensorrtExecutionProvider", {
            "device_id": 0,
            "trt_fp16_enable": fp16,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": cache_dir,
        }),
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    return ort.InferenceSession(onnx_path, sess_options=so, providers=providers)


def _patch_yolox_for_trt(onnx_path: str) -> str:
    """把 YOLOX ONNX 里预 NMS 的 TopK 的 K 降到 ≤3840（TensorRT 上限），返回可用模型路径。

    mmpose SDK 导出的 YOLOX 烤入了 EfficientNMS，其预 NMS 的 ``TopK`` 用 K=5000，
    超过 TensorRT ``ITopKLayer`` 的 3840 上限，导致 TRT 转换报
    ``K exceeds the maximum value allowed (3840)``。这里把 K 改成 3000——对 1~4 人的
    场景完全无损（真实目标都在 top 几百内），结果缓存到本地只 patch 一次。
    模型若没有超限 TopK 则原样返回原路径。
    """
    import onnx
    from onnx import numpy_helper

    out_path = os.path.join(
        _default_trt_cache_dir(),
        os.path.basename(onnx_path).replace(".onnx", "_topk3000.onnx"),
    )
    if os.path.exists(out_path):
        return out_path

    model = onnx.load(onnx_path)
    inits = {i.name: i for i in model.graph.initializer}
    changed = False
    for node in model.graph.node:
        if node.op_type != "TopK":
            continue
        for kname in node.input[1:]:  # TopK 的 K 输入（通常第二个）
            init = inits.get(kname)
            if init is None:
                continue
            arr = numpy_helper.to_array(init)
            if arr.size == 1 and int(arr[0]) > 3840:
                new = numpy_helper.from_array(
                    np.array([3000], dtype=arr.dtype), name=init.name
                )
                model.graph.initializer.remove(init)
                model.graph.initializer.append(new)
                changed = True
    if not changed:
        return onnx_path
    os.makedirs(_default_trt_cache_dir(), exist_ok=True)
    onnx.save(model, out_path)
    logger.info("YOLOX 预 NMS TopK 已 patch（K→3000）保存到 %s", out_path)
    return out_path


def _default_det_batch_onnx() -> str:
    """动态 batch YOLOX ONNX 路径（scripts/export_yolox_dynamic_batch.py 导出）。"""
    return os.path.join(os.path.expanduser("~"), ".cache", "tabletennis",
                        "yolox_tiny_dynamic_416.onnx")


def _default_person_onnx(imgsz: int = 416) -> str:
    """yolo11n 灰度人检测 ONNX 路径（scripts/export_yolo11_person.py 导出）。

    按输入尺寸分文件（h/w 固定进 ONNX，换 imgsz 需重新导出）。可用环境变量
    ``PERSON_ONNX`` 覆盖（换权重 / 不同路径时用）。
    """
    return os.environ.get("PERSON_ONNX") or os.path.join(
        os.path.expanduser("~"), ".cache", "tabletennis",
        f"yolo11n_grayscale_person_{imgsz}.onnx")


def _nms(boxes: np.ndarray, scores: np.ndarray, nms_thr: float) -> List[int]:
    """单类 NMS（numpy）。boxes: (N,4) xyxy。"""
    x1, y1 = boxes[:, 0], boxes[:, 1]
    x2, y2 = boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(ovr <= nms_thr)[0] + 1]
    return keep


def _multiclass_nms(boxes: np.ndarray, scores: np.ndarray,
                    nms_thr: float, score_thr: float) -> Optional[np.ndarray]:
    """逐类 NMS，返回 (M,6) [x1,y1,x2,y2,score,cls] 或 None。与 rtmlib multiclass_nms 等价。"""
    final = []
    n = boxes.shape[0]
    all_idx = np.arange(n)
    for c in range(scores.shape[1]):
        cs = scores[:, c]
        m = cs > score_thr
        if not m.any():
            continue
        keep = _nms(boxes[m], cs[m], nms_thr)
        if keep:
            final.append(np.concatenate([
                boxes[m][keep], cs[m][keep, None],
                np.full((len(keep), 1), c, np.float32)], 1))
    return None if not final else np.concatenate(final, 0).astype(np.float32)


def _yolox_decode_batch(preds: np.ndarray, ratios: List[float],
                        score_thr: float = 0.3, nms_thr: float = 0.45) -> List[np.ndarray]:
    """动态 batch YOLOX 输出 (B, 3549, 85) → 每帧 xyxy 框列表（原图坐标）。

    decode 用 rtmlib 约定 **center=(delta + grid)×stride**（不加 0.5）——实测该 humanart
    ONNX 的 decode 就是 cell 左上角约定，与 rtmlib ``YOLOX.postprocess`` 的 no-NMS 分支
    完全一致（mmdet ``predict_by_feat`` 的 grid+0.5 会差约半格，框系统性偏大 ~24px）。
    wh=exp(delta)×stride，obj×cls 逐类 NMS。3549 = 52²+26²+13²（stride 8/16/32 依序
    y-major 展平）。score_thr=0.3 对齐原 ONNX（内置 EfficientNMS）＋ rtmlib 的过滤阈值。
    """
    centers, strides = [], []
    for s in (8, 16, 32):
        hs = ws = 416 // s
        gy, gx = np.meshgrid(np.arange(hs), np.arange(ws), indexing="ij")
        centers.append(np.stack([gx * s, gy * s], -1).reshape(-1, 2))
        strides.append(np.full((hs * ws, 1), s, np.float32))
    pc = np.concatenate(centers).astype(np.float32)
    ps = np.concatenate(strides)
    results = []
    for b in range(len(preds)):
        p = preds[b]
        xy = p[:, :2] * ps + pc
        wh = np.exp(p[:, 2:4]) * ps
        boxes = np.concatenate([xy - wh / 2, xy + wh / 2], 1) / ratios[b]
        scores = p[:, 4:5] * p[:, 5:]  # (N, 80) = obj × cls
        dets = _multiclass_nms(boxes, scores, nms_thr, score_thr)
        results.append([] if dets is None else dets[:, :4])
    return results


def _build_det_batch_session(onnx_path: str, backend: str,
                             cache_dir: str) -> Optional[object]:
    """为动态 batch YOLOX ONNX 建 onnxruntime 会话（CUDA EP；tensorrt 加 TRT EP+profile）。

    找不到 ONNX 文件返回 None（调用方回退逐帧检测）。TRT 会话用动态 batch profile
    （min=1 / opt=4 / max=8，匹配 4 相机实时场景），输入名固定 "input"（导出脚本指定）。
    """
    if not os.path.exists(onnx_path):
        return None
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if backend == "tensorrt":
        providers = [
            ("TensorrtExecutionProvider", {
                "device_id": 0,
                "trt_fp16_enable": True,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": cache_dir,
                "trt_profile_min_shapes": "input:1x3x416x416",
                "trt_profile_opt_shapes": "input:4x3x416x416",
                "trt_profile_max_shapes": "input:8x3x416x416",
            }),
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
    else:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ort.InferenceSession(onnx_path, sess_options=so, providers=providers)


class RTMPoseDetector(PoseDetector):
    """RTMPose top-down 2D 姿态检测器（YOLOX 检测 + RTMPose 关键点）。

    Args:
        model: 姿态模型标识，带 ``-halpe26`` 后缀为 26 点模型、不带为 COCO-17：
            "rtmpose-l"/"rtmpose-l-halpe26"/"rtmpose-m"/"rtmpose-m-halpe26"/
            "rtmpose-s"/"rtmpose-s-halpe26"/"rtmpose-x"/"rtmpose-x-halpe26"，
            或本地 onnx 路径 / 下载 URL。
        input_size: 姿态模型输入尺寸 (H, W)，默认 (192, 256)（对应 256×192）。
        det: 人体检测器标识，默认 "yolo11n-gray"（灰度原生 yolo11n，1ch）；
            也可用 "yolox-tiny"/"yolox-m"/"yolox-x"（rtmlib humanart）或本地/URL。
        det_input_size: 检测器输入尺寸 (H, W)，默认 (416, 416)（须与 yolox-tiny 匹配）。
        device: "cpu" 或 "cuda"。GT 1030 建议 cpu，换好 GPU 后改 cuda。
        backend: 推理后端，默认 "onnxruntime"；"tensorrt" 走 TensorrtExecutionProvider
            （FP16 + engine 缓存，需已装 tensorrt 运行库）。
        score_thr: 人体检测置信度阈值（YOLOX），默认 0.5。
        nms_thr: 检测 NMS IoU 阈值，默认 0.45。
        to_openpose: 是否转 OpenPose 输出（默认 False，保持模型自身关键点集）。
    """

    def __init__(
        self,
        model: str = "rtmpose-l-halpe26",
        input_size: tuple = (192, 256),
        det: str = "yolo11n-gray",
        det_input_size: tuple = (416, 416),
        det_onnx: Optional[str] = None,
        det_imgsz: int = 416,
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

        # rtmlib 只认 onnxruntime 后端；backend="tensorrt" 时先用 CUDA EP 建好，
        # 再把两个模型的 session 换成 TensorrtExecutionProvider（复用其 pre/postprocess）。
        use_trt = backend == "tensorrt"
        rtmlib_backend = "onnxruntime" if use_trt else backend

        self._use_yolo11_det = det == "yolo11n-gray"
        if self._use_yolo11_det:
            # 灰度原生 yolo11n 人检测（1ch + TRT FP16 + batch），替换 rtmlib YOLOX：
            # 省掉 gray→3ch 复制与 CPU 上 80 类逐类 NMS。
            from .yolo11_person import Yolo11PersonDetector
            person_backend = ("tensorrt" if use_trt
                              else ("cpu" if device == "cpu" else "cuda"))
            self._det_model = None
            self._det_person = Yolo11PersonDetector(
                det_onnx or _default_person_onnx(det_imgsz), imgsz=det_imgsz,
                conf_thresh=score_thr, backend=person_backend)
        else:
            self._det_model = YOLOX(
                YOLOX_MODEL_URLS.get(det, det),
                model_input_size=det_input_size,
                backend=rtmlib_backend,
                device=device,
                score_thr=score_thr,
                nms_thr=nms_thr,
            )
            self._det_person = None
        self._pose_model = RTMPose(
            RTMPOSE_MODEL_URLS.get(model, model),
            model_input_size=input_size,
            backend=rtmlib_backend,
            device=device,
            to_openpose=to_openpose,
        )
        # 预计算 float32 归一化参数（复用 rtmlib 的 mean/std）：rtmlib 每次 preprocess
        # 都做 `(uint8 - float_mean)/float_std`，numpy 会把 uint8 提升成 float64，
        # 实测归一化占 RTMPose 预处理 ~60% 耗时；这里固定成 float32 就地算。
        self._pose_mean = np.asarray(self._pose_model.mean, dtype=np.float32)
        self._pose_std = np.asarray(self._pose_model.std, dtype=np.float32)

        if use_trt:
            cache_dir = _default_trt_cache_dir()
            print("[TensorRT] 首次构建引擎（约 30~40 秒，之后走缓存秒开），请稍候...", flush=True)
            # 逐模型尝试 TRT：YOLOX 需先 patch 掉预 NMS 的 TopK(5000→3000)；
            # yolo11n 人检测器自带 TRT（backend="tensorrt"），无需此处 session 替换。
            trt_models = [("RTMPose", self._pose_model)]
            if not self._use_yolo11_det:
                trt_models.insert(0, ("YOLOX", self._det_model))
            for name, model in trt_models:
                print(f"[TensorRT] 构建 {name} 引擎...", flush=True)
                try:
                    onnx_path = model.onnx_model
                    if name == "YOLOX":
                        onnx_path = _patch_yolox_for_trt(onnx_path)
                    model.session = _trt_session(onnx_path, cache_dir)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("%s 走 TensorRT 失败（%s），回退 CUDA EP", name, exc)
            # 预热触发 TRT engine 构建（含 FP16），避免首帧卡顿
            side = max(det_input_size)
            dummy = np.zeros((side, side, 3), dtype=np.uint8)
            if self._use_yolo11_det:
                # 人检测器首次 forward 触发其 TRT 引擎构建（1ch 灰度）
                self._det_person.detect(Frame(
                    camera_id=0, serial="warmup", frame_num=0, device_timestamp=0,
                    host_timestamp=0, image=np.zeros((side, side), np.uint8),
                    pixel_format=17301505, width=side, height=side))
            else:
                self._det_model(dummy)
            self._pose_model(dummy, bboxes=[[0, 0, side, side]])
            print("[TensorRT] 全部引擎构建完成 ✓", flush=True)

        # 校验 CUDA 是否真正生效：onnxruntime 缺 CUDA 库时会静默回退 CPU
        # （get_available_providers 仍列出 CUDAExecutionProvider，但 session 实际用 CPU）。
        elif device == "cuda":
            probe = self._pose_model if self._use_yolo11_det else self._det_model
            actual = probe.session.get_providers()
            if not actual or actual[0] != "CUDAExecutionProvider":
                logger.warning(
                    "CUDA EP 加载失败（实际 providers=%s），已在 CPU 上推理；"
                    "请确认已装 onnxruntime-gpu 及匹配的 CUDA 运行库", actual
                )

        # 动态 batch YOLOX 会话：4 相机一次 forward（省掉 3 次 session.run 固定开销）。
        # 仅当检测输入恰为 416×416（与导出的 ONNX 一致）且用 GPU 时启用；导出文件缺失
        # 则回退逐帧检测（detect_batch 走原路径）。yolo11n 人检测走自己的 detect_batch。
        self._det_batch_session = None
        if not self._use_yolo11_det and device != "cpu" and tuple(det_input_size) == (416, 416):
            batch_onnx = _default_det_batch_onnx()
            if not os.path.exists(batch_onnx):
                logger.warning(
                    "未找到动态 batch YOLOX ONNX（%s），4 相机将逐帧检测；"
                    "可运行 scripts/export_yolox_dynamic_batch.py 导出后提速",
                    batch_onnx)
            elif use_trt:
                print("[TensorRT] 构建动态 batch YOLOX 引擎（约 30 秒）...", flush=True)
                try:
                    self._det_batch_session = _build_det_batch_session(
                        batch_onnx, "tensorrt", _default_trt_cache_dir())
                    # 预热触发引擎构建（batch=4 形状）
                    self._det_batch_session.run(
                        [self._det_batch_session.get_outputs()[0].name],
                        {self._det_batch_session.get_inputs()[0].name:
                         np.zeros((4, 3, 416, 416), np.float32)})
                    print("[TensorRT] 动态 batch YOLOX 引擎完成 ✓", flush=True)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "动态 batch YOLOX 走 TensorRT 失败（%s），回退 CUDA EP", exc)
                    self._det_batch_session = _build_det_batch_session(
                        batch_onnx, "onnxruntime", _default_trt_cache_dir())
            else:
                self._det_batch_session = _build_det_batch_session(
                    batch_onnx, "onnxruntime", _default_trt_cache_dir())

    def _preprocess_pose(self, bgr: np.ndarray, bbox) -> tuple:
        """RTMPose 预处理（与 rtmlib 完全等价，但归一化用 float32 就地算）。

        复用 rtmlib 的 ``bbox_xyxy2cs`` / ``top_down_affine``（中心/尺度/仿射 warp
        逻辑一致），仅把归一化从 ``(uint8 - mean)/std``（numpy 提升成 float64）改成
        float32，省掉 float64 中间量与后续 stack 时的 dtype 转换。
        返回 ``(normalized_img_float32, center, adjusted_scale)``，语义同 rtmlib。
        """
        from rtmlib.tools.pose_estimation.pre_processings import (
            bbox_xyxy2cs, top_down_affine)
        center, scale = bbox_xyxy2cs(np.asarray(bbox), padding=1.25)
        img, adj_scale = top_down_affine(
            self._pose_model.model_input_size, scale, center, bgr)
        img = np.asarray(img, dtype=np.float32)
        img -= self._pose_mean
        img /= self._pose_std
        return img, center, adj_scale

    def detect(self, frame: Frame) -> List[Pose2D]:
        """对一帧做 top-down 2D 姿态检测。

        本机是黑白相机（单通道灰度），RTMPose 训练在 RGB 上，这里把灰度复制成
        3 通道再送模型（存在 domain gap，靠固定短曝光 + 补光缓解）。
        """
        if frame.image is None or frame.image.size == 0:
            return []

        if frame.image.ndim == 2:
            bgr = cv2.cvtColor(frame.image, cv2.COLOR_GRAY2BGR)
        else:
            bgr = frame.image

        # 1) 检测人（返回 xyxy 框，已按 score_thr/nms 过滤）
        if self._use_yolo11_det:
            bboxes = self._det_person.detect(frame)
        else:
            bboxes = self._det_model(bgr)
        if bboxes is None or len(bboxes) == 0:
            return []
        return self.detect_on_boxes(frame, bboxes)

    def detect_person_boxes(self, frame: Frame, *, conf_thresh: Optional[float] = None,
                            roi=None, return_scores: bool = True):
        """只跑人检测（不跑姿态），返回 ``(boxes (M,4), scores (M,))`` 全图坐标。

        ROI 引导重检测用：小窗口 + 低阈值再检测一次，找不到人就不必再跑 RTMPose。
        非 yolo11n-gray 人检测器（YOLOX 回退分支）不支持临时阈值/ROI，按原样返回。
        """
        if not self._use_yolo11_det:
            bgr = (cv2.cvtColor(frame.image, cv2.COLOR_GRAY2BGR)
                   if frame.image.ndim == 2 else frame.image)
            boxes = self._det_model(bgr)
            boxes = np.zeros((0, 4), np.float32) if boxes is None else np.asarray(boxes)
            return boxes[:, :4], np.ones(len(boxes), np.float32)
        return self._det_person.detect(frame, conf_thresh=conf_thresh, roi=roi,
                                       return_scores=return_scores)

    def detect_on_boxes(self, frame: Frame, bboxes) -> List[Pose2D]:
        """对**指定的人框**跑关键点估计（不重复做人检测）。

        跟踪阶段 A 的 ROI 重检测拿到框后直接喂这里，省掉一次人检测；框坐标须是
        全图像素坐标（与 ``detect`` 的输出同坐标系）。
        """
        if frame.image is None or frame.image.size == 0:
            return []
        bboxes = np.asarray(bboxes, dtype=np.float32)
        if bboxes.ndim != 2 or len(bboxes) == 0:
            return []
        bgr = (cv2.cvtColor(frame.image, cv2.COLOR_GRAY2BGR)
               if frame.image.ndim == 2 else frame.image)

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

    def detect_batch(self, frames: List[Frame]) -> List[List[Pose2D]]:
        """批处理多帧姿态检测，返回与 ``frames`` 等长的 ``List[List[Pose2D]]``。

        多相机实时用：把 4 台相机的人框收集起来，RTMPose 一次 forward 处理所有
        裁剪（RTMPose ONNX 的 batch 维是动态的），相比逐人逐帧调用少掉大量
        kernel 启动与传输开销。YOLOX 走 ``scripts/export_yolox_dynamic_batch.py``
        重导出的动态 batch ONNX，4 帧一次 forward（省 session.run 固定开销，这是
        多相机的主要耗时），numpy 解码 + 逐类 NMS；导出缺失或非 416 输入时回退
        逐帧 rtmlib。
        """
        n = len(frames)
        results: List[List[Pose2D]] = [[] for _ in range(n)]
        if n == 0:
            return results

        # 1) 检测人。yolo11n 灰度走自己的 detect_batch（1ch，4 帧一次 forward）；否则走
        #    YOLOX 动态 batch 会话（4 帧一次 forward + numpy 解码/NMS）或逐帧 rtmlib。
        dets: List = []  # 每帧: (bgr 或 None, bboxes)
        if self._use_yolo11_det:
            boxes_list = self._det_person.detect_batch(frames)  # List[np.ndarray (M,4)]
            for frame, boxes in zip(frames, boxes_list):
                if frame.image is None or frame.image.size == 0:
                    dets.append((None, []))
                    continue
                bgr = cv2.cvtColor(frame.image, cv2.COLOR_GRAY2BGR) if frame.image.ndim == 2 else frame.image
                dets.append((bgr, boxes))
        elif self._det_batch_session is not None:
            bgr_list, padded, ratios, valid = [], [], [], []
            for frame in frames:
                if frame.image is None or frame.image.size == 0:
                    valid.append(False)
                    continue
                bgr = cv2.cvtColor(frame.image, cv2.COLOR_GRAY2BGR) if frame.image.ndim == 2 else frame.image
                bgr_list.append(bgr)
                p, ratio = self._det_model.preprocess(bgr)  # letterbox 到 416，与原路径一致
                padded.append(p)
                ratios.append(ratio)
                valid.append(True)
            boxes_list: List = []
            if padded:
                batch = np.stack(padded).transpose(0, 3, 1, 2).astype(np.float32)  # (B,3,416,416)
                sess = self._det_batch_session
                out = sess.run([sess.get_outputs()[0].name],
                               {sess.get_inputs()[0].name: batch})[0]  # (B,3549,85)
                boxes_list = _yolox_decode_batch(out, ratios)
            vi = 0
            for fi in range(len(frames)):
                if valid[fi]:
                    dets.append((bgr_list[vi], boxes_list[vi]))
                    vi += 1
                else:
                    dets.append((None, []))
        else:
            for frame in frames:
                if frame.image is None or frame.image.size == 0:
                    dets.append((None, []))
                    continue
                bgr = cv2.cvtColor(frame.image, cv2.COLOR_GRAY2BGR) if frame.image.ndim == 2 else frame.image
                bboxes = self._det_model(bgr)
                dets.append((bgr, [] if bboxes is None or len(bboxes) == 0 else bboxes))

        # 2) 收集所有 (帧, 人) 的裁剪与 center/scale
        crops: List[np.ndarray] = []
        meta: List = []  # (frame_idx, person_idx, center, scale)
        for fi, (bgr, bboxes) in enumerate(dets):
            if bgr is None:
                continue
            for pi, bbox in enumerate(bboxes):
                img, center, scale = self._preprocess_pose(bgr, bbox)
                crops.append(img)
                meta.append((fi, pi, center, scale))

        # 3) RTMPose 批量 forward（一次 session.run 处理所有裁剪）
        if crops:
            batch = np.ascontiguousarray(
                np.stack(crops).transpose(0, 3, 1, 2), dtype=np.float32
            )  # (N, 3, H, W)
            sess = self._pose_model.session
            sess_input = {sess.get_inputs()[0].name: batch}
            sess_output = [o.name for o in sess.get_outputs()]
            outputs = sess.run(sess_output, sess_input)
            simcc_x, simcc_y = outputs[0], outputs[1]  # (N, K, Wx) / (N, K, Wy)

            for i, (fi, pi, center, scale) in enumerate(meta):
                # 逐人 postprocess（复用 rtmlib 逻辑，保持单人的 center/scale 语义）
                kpts, scs = self._pose_model.postprocess(
                    [simcc_x[i:i + 1], simcc_y[i:i + 1]], center, scale
                )
                kpts = np.asarray(kpts[0], dtype=np.float32)  # (K, 2)
                scs = np.asarray(scs[0], dtype=np.float32)    # (K,)
                kpts3 = np.concatenate([kpts, scs[:, None]], axis=1)  # (K, 3)
                bbox = np.asarray(dets[fi][1][pi], dtype=np.float32)[:4]
                results[fi].append(
                    Pose2D(
                        camera_id=frames[fi].camera_id,
                        keypoints=kpts3,
                        score=float(np.max(scs)) if len(scs) else 0.0,
                        bbox=bbox,
                        skeleton=self._skeleton,
                    )
                )
        return results

    def close(self) -> None:
        self._det_model = None
        self._det_person = None
        self._pose_model = None
        self._det_batch_session = None
