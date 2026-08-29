#!/usr/bin/env python3
"""姿态检测器 GPU 延迟基准：测 YOLOX + RTMPose 在单相机上的推理耗时。

用法：
  conda activate tt
  python scripts/benchmark_pose.py                 # 默认 cuda
  python scripts/benchmark_pose.py --device cpu    # 对比 CPU
  python scripts/benchmark_pose.py --frames 100 --people 1
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np

from tabletennis.core.types import Frame
from tabletennis.vision.pose.rtmpose_pose import RTMPoseDetector


def make_frame(camera_id: int = 0, h: int = 1080, w: int = 1440) -> Frame:
    # 模拟一台相机的灰度图（放一个亮矩形当"人"，含纹理让卷积有实际计算量）
    img = np.full((h, w), 30, dtype=np.uint8)
    img[200:900, 600:840] = 180
    img[200:900:4, 600:840] = 60  # 纹理
    return Frame(
        camera_id=camera_id, serial=f"s{camera_id}", frame_num=0,
        device_timestamp=0, host_timestamp=0, image=img,
        pixel_format=17301505, width=w, height=h,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="RTMPose 检测延迟基准")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--model", default="rtmpose-l-halpe26")
    ap.add_argument("--frames", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    print(f"加载 RTMPose（{args.model}, {args.device}）...")
    t0 = time.time()
    det = RTMPoseDetector(model=args.model, device=args.device)
    print(f"  模型加载耗时 {time.time() - t0:.2f}s")

    frame = make_frame()

    # 预热
    for _ in range(args.warmup):
        det.detect(frame)

    # 端到端（YOLOX 检测 + RTMPose 关键点）
    ts = []
    for _ in range(args.frames):
        t = time.perf_counter()
        poses = det.detect(frame)
        ts.append(time.perf_counter() - t)
    ts = np.array(ts)
    det_ms = np.mean(ts) * 1000
    n_people = len(poses)

    # 单独测 RTMPose 关键点（喂全图当 bbox，模拟一个人；模型需 3 通道）
    bgr = np.stack([frame.image] * 3, axis=-1)
    t = time.perf_counter()
    for _ in range(30):
        det._pose_model(bgr, bboxes=[[0, 0, frame.width, frame.height]])
    pose_ms = (time.perf_counter() - t) / 30 * 1000

    print(f"\n===== 单相机延迟（{args.device}） =====")
    print(f"  detect 端到端: 均值 {det_ms:.1f} ms | 中位 {np.median(ts)*1000:.1f} ms"
          f" | 检测到 {n_people} 人")
    print(f"  RTMPose 关键点(1人): {pose_ms:.1f} ms")
    print(f"  估计 4 相机串行: {det_ms*4:.1f} ms ≈ {1000/(det_ms*4):.1f} FPS")
    print(f"  估计 4 相机并行: {det_ms:.1f} ms ≈ {1000/det_ms:.1f} FPS")


if __name__ == "__main__":
    main()
