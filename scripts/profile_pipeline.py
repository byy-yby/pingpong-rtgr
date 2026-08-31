#!/usr/bin/env python3
"""全管线 profiling：人检测（YOLOX vs yolo11n-gray）+ 姿态 + DLT 三角化 + 球检测。

对照 **100fps = 10ms/轮**（一轮 = 4 相机一帧同步数据，重建 1 人 + 1 球）。

用法::

    conda run -n tt python scripts/profile_pipeline.py                     # CUDA onnxruntime（启动快）
    conda run -n tt python scripts/profile_pipeline.py --backend tensorrt  # TRT FP16（首轮构建引擎 ~30s）
    conda run -n tt python scripts/profile_pipeline.py --device cpu        # CPU 对比

输出各阶段耗时（实测均值/中位）与端到端折算 FPS，标注每项是实测还是跳过（模型/标定缺失）。
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np

from tabletennis.core.types import Frame


def make_person_frame(camera_id: int = 0, h: int = 1080, w: int = 1440) -> Frame:
    """合成一台相机的灰度图（亮矩形当"人"，含纹理让卷积有实际计算量）。"""
    img = np.full((h, w), 30, dtype=np.uint8)
    img[200:900, 600:840] = 180
    img[200:900:4, 600:840] = 60
    return Frame(
        camera_id=camera_id, serial=f"s{camera_id}", frame_num=0,
        device_timestamp=0, host_timestamp=0, image=img,
        pixel_format=17301505, width=w, height=h,
    )


def bench_person(det_variant: str, frames, device: str, backend: str,
                 n: int = 30, warmup: int = 5):
    """完整人管线（检测 + RTMPose）一轮耗时。返回 (均值_ms, 中位_ms, 每帧人数)。"""
    from tabletennis.vision.pose.rtmpose_pose import RTMPoseDetector

    det = RTMPoseDetector(det=det_variant, device=device, backend=backend)
    for _ in range(warmup):
        det.detect_batch(frames)
    ts = []
    counts = None
    for _ in range(n):
        t = time.perf_counter()
        counts = det.detect_batch(frames)
        ts.append(time.perf_counter() - t)
    det.close()
    ts = np.asarray(ts)
    return ts.mean() * 1000, np.median(ts) * 1000, [len(r) for r in counts]


def bench_dlt(n: int = 500):
    """DLT 批量三角化（26 关节 × 4 相机）单次耗时。标定缺失返回 None。"""
    try:
        from tabletennis.reconstruction.triangulate import (
            MultiViewTriangulator, load_camera_rig)
        intrinsics, extrinsics = load_camera_rig()
        tri = MultiViewTriangulator(intrinsics, extrinsics)
        n_cams = len(tri.cameras)
        if n_cams < 2:
            return None
    except Exception:
        return None
    # 合成 26 关节观测（去畸变后的无畸变像素坐标 + 置信度）
    uv = np.random.rand(26, n_cams, 2) * 1000
    conf = np.random.rand(26, n_cams).clip(0.3, 1.0)
    for _ in range(10):
        tri.triangulate_batch(uv, conf)
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        tri.triangulate_batch(uv, conf)
        ts.append(time.perf_counter() - t)
    return np.median(ts) * 1000


def bench_ball(frames, n: int = 30, warmup: int = 5):
    """球检测：经典路线 vs YOLO（模型存在才测）。返回 dict。"""
    out = {}

    # 经典路线（背景减除 + 帧差 + 尺寸先验）
    from tabletennis.vision.ball.classical_ball import ClassicalBallDetector
    cdet = ClassicalBallDetector()
    try:
        for _ in range(warmup):
            cdet.detect(frames[0])
        ts = []
        for _ in range(n):
            t = time.perf_counter()
            cdet.detect(frames[0])
            ts.append(time.perf_counter() - t)
        out["classical"] = np.median(ts) * 1000
    except Exception as exc:
        out["classical"] = f"跳过({type(exc).__name__})"

    # YOLO 球（onnx 存在才测）
    from tabletennis.core.config import project_root
    onnx = os.path.join(project_root(), "runs", "detect", "ball", "weights", "best.onnx")
    if not os.path.exists(onnx):
        out["yolo"] = "跳过(无 best.onnx)"
        return out
    from tabletennis.vision.ball.yolo_ball import YoloBallDetector
    ydet = YoloBallDetector(onnx)
    try:
        for _ in range(warmup):
            ydet.detect(frames[0])
        ts = []
        for _ in range(n):
            t = time.perf_counter()
            ydet.detect(frames[0])
            ts.append(time.perf_counter() - t)
        out["yolo"] = np.median(ts) * 1000
    except Exception as exc:
        out["yolo"] = f"跳过({type(exc).__name__})"
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--backend", default="onnxruntime")
    ap.add_argument("--frames", type=int, default=30)
    args = ap.parse_args()

    frames = [make_person_frame(i) for i in range(4)]
    print(f"== 人管线（4 相机一轮，{args.device}/{args.backend}）==\n")

    # yolo11n-gray（默认）
    print("构建 yolo11n-gray 人检测 + RTMPose ...")
    t0 = time.time()
    m11, med11, c11 = bench_person("yolo11n-gray", frames, args.device, args.backend, args.frames)
    print(f"  yolo11n-gray: 均值 {m11:.1f}ms | 中位 {med11:.1f}ms  检出 {c11} 人/帧  "
          f"(加载 {time.time()-t0:.1f}s)")

    # YOLOX（回退对比）
    print("构建 yolox-tiny 人检测 + RTMPose ...")
    t0 = time.time()
    mox, medox, cox = bench_person("yolox-tiny", frames, args.device, args.backend, args.frames)
    print(f"  yolox-tiny:   均值 {mox:.1f}ms | 中位 {medox:.1f}ms  检出 {cox} 人/帧  "
          f"(加载 {time.time()-t0:.1f}s)")

    print(f"\n== DLT 三角化 ==\n")
    dlt = bench_dlt()
    if dlt is None:
        print("  跳过（无标定数据）")
    else:
        print(f"  26 关节 × 4 相机批量 SVD: {dlt:.2f}ms/人")

    print(f"\n== 球检测 ==\n")
    ball = bench_ball(frames, args.frames)
    for k, v in ball.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}ms/相机")
        else:
            print(f"  {k}: {v}")

    print(f"\n== 端到端估算（4 相机一轮，1 人 + 1 球）==\n")
    person = med11
    dlt_ms = dlt if dlt else 0.0
    ball_ms = ball.get("classical", 0.0)
    if isinstance(ball_ms, str):
        ball_ms = 0.0
    total = person + dlt_ms + ball_ms * 4
    print(f"  人管线(yolo11n-gray): {person:.1f}ms")
    print(f"  DLT:                   {dlt_ms:.2f}ms")
    print(f"  球(经典, 4 相机):       {ball_ms*4:.1f}ms" if isinstance(ball.get('classical'), float) else "  球: 跳过")
    print(f"  合计:                  {total:.1f}ms ≈ {1000/total:.0f} FPS  "
          f"(预算 10ms/100fps)")
    print("\n注：球走 YOLO@1280 会额外 +20~28ms（见 docs/optimization_report.md），"
          "那是 100fps 的主要障碍。")


if __name__ == "__main__":
    main()
