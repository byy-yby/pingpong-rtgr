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

        # rtmlib 只认 onnxruntime 后端；backend="tensorrt" 时先用 CUDA EP 建好，
        # 再把两个模型的 session 换成 TensorrtExecutionProvider（复用其 pre/postprocess）。
        use_trt = backend == "tensorrt"
        rtmlib_backend = "onnxruntime" if use_trt else backend

        self._det_model = YOLOX(
            YOLOX_MODEL_URLS.get(det, det),
            model_input_size=det_input_size,
            backend=rtmlib_backend,
            device=device,
            score_thr=score_thr,
            nms_thr=nms_thr,
        )
        self._pose_model = RTMPose(
            RTMPOSE_MODEL_URLS.get(model, model),
            model_input_size=input_size,
            backend=rtmlib_backend,
            device=device,
            to_openpose=to_openpose,
        )

        if use_trt:
            cache_dir = _default_trt_cache_dir()
            print("[TensorRT] 首次构建引擎（约 30~40 秒，之后走缓存秒开），请稍候...", flush=True)
            # 逐模型尝试 TRT：YOLOX 需先 patch 掉预 NMS 的 TopK(5000→3000)，
            # 某模型转换失败则回退 CUDA EP。
            for name, model in (("YOLOX", self._det_model), ("RTMPose", self._pose_model)):
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
            self._det_model(dummy)
            print("[TensorRT] YOLOX 引擎完成，构建 RTMPose 引擎（约 30 秒）...", flush=True)
            self._pose_model(dummy, bboxes=[[0, 0, side, side]])
            print("[TensorRT] 全部引擎构建完成 ✓", flush=True)

        # 校验 CUDA 是否真正生效：onnxruntime 缺 CUDA 库时会静默回退 CPU
        # （get_available_providers 仍列出 CUDAExecutionProvider，但 session 实际用 CPU）。
        elif device == "cuda":
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

    def detect_batch(self, frames: List[Frame]) -> List[List[Pose2D]]:
        """批处理多帧姿态检测，返回与 ``frames`` 等长的 ``List[List[Pose2D]]``。

        多相机实时用：把 4 台相机的人框收集起来，RTMPose 一次 forward 处理所有
        裁剪（RTMPose ONNX 的 batch 维是动态的），相比逐人逐帧调用少掉大量
        kernel 启动与传输开销。YOLOX 的 ONNX 是固定 batch=1（且已内置 NMS），
        无法批，仍逐帧跑——但它经 TensorRT 后单帧 ~1-2ms，4 路也就几 ms。
        """
        n = len(frames)
        results: List[List[Pose2D]] = [[] for _ in range(n)]
        if n == 0:
            return results

        # 1) 逐帧 YOLOX 检测（模型固定 batch=1）
        dets: List = []  # 每帧: (bgr 或 None, bboxes)
        for frame in frames:
            if frame.image is None or frame.image.size == 0:
                dets.append((None, []))
                continue
            bgr = np.stack([frame.image] * 3, axis=-1) if frame.image.ndim == 2 else frame.image
            bboxes = self._det_model(bgr)
            dets.append((bgr, [] if bboxes is None or len(bboxes) == 0 else bboxes))

        # 2) 收集所有 (帧, 人) 的裁剪与 center/scale
        crops: List[np.ndarray] = []
        meta: List = []  # (frame_idx, person_idx, center, scale)
        for fi, (bgr, bboxes) in enumerate(dets):
            if bgr is None:
                continue
            for pi, bbox in enumerate(bboxes):
                img, center, scale = self._pose_model.preprocess(bgr, bbox)
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
        self._pose_model = None
