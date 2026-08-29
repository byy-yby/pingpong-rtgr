#!/usr/bin/env python3
"""采集乒乓球检测训练数据：抓同步帧存 PNG，可选经典检测器预标注（YOLO 格式）。

用途：为训练单类「ball」NN 采集数据。每台相机的帧单独存成一张图，全部喂给
同一个检测器训练（多相机视角 = 更多样本多样性）。

用法：
  conda activate tt
  python scripts/capture_ball.py                        # 外部触发，只存图
  python scripts/capture_ball.py --trigger continuous   # 自由采集（无信号发生器）
  python scripts/capture_ball.py --prelabel             # 用经典检测器预标注（需人工校正）
  python scripts/capture_ball.py --n 500 --stride 2     # 采集 500 帧，隔 1 帧取
  python scripts/capture_ball.py --exposure 100         # 冻结运动模糊

采集建议：球在桌面各处 / 不同速度 / 不同模糊度移动，同时**用手、球拍、反光、阴影
制造干扰样本**（这些是 NN 要学的负样本），否则 NN 也会把没见过的干扰当球。

输出结构（``data/ball_dataset/``）：
  images/f000000_c0.png …  灰度 PNG
  labels/f000000_c0.txt …  YOLO 标签：``0 cx cy w h``（归一化），无球则为空文件
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import cv2
import numpy as np

from tabletennis.core.config import load_yaml, project_root, resolve_camera_settings
from tabletennis.vision.ball import ClassicalBallDetector


def ball_to_yolo(ball, W: int, H: int) -> str:
    """把 :class:`Ball2D` 转成 YOLO 标签行 ``0 cx cy w h``（归一化）。"""
    cx = float(ball.center[0]) / W
    cy = float(ball.center[1]) / H
    w = float(2.0 * ball.radius) / W
    h = float(2.0 * ball.radius) / H
    cx = min(max(cx, 0.0), 1.0)
    cy = min(max(cy, 0.0), 1.0)
    w = min(max(w, 0.0), 1.0)
    h = min(max(h, 0.0), 1.0)
    return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="采集乒乓球检测训练数据")
    ap.add_argument("--trigger", choices=["external", "software", "continuous"], default="external")
    ap.add_argument("--exposure", type=float, default=None, help="曝光时间(us)；冻结球需 ≤100")
    ap.add_argument("--n", type=int, default=500, help="采集帧数（同步 bundle 数）")
    ap.add_argument("--stride", type=int, default=1, help="隔 N 个 bundle 采一帧")
    ap.add_argument("--out", default=None, help="数据集根目录（默认 data/ball_dataset）")
    ap.add_argument("--prelabel", action="store_true", help="用经典检测器预标注（YOLO 格式）")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()

    from tabletennis.camera import CameraManager

    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    cfg_path = os.path.join(root, "config", "cameras.yaml")
    config = load_yaml(cfg_path) if os.path.exists(cfg_path) else {}
    trigger = config.get("trigger", {}) or {}
    image = config.get("image", {}) or {}

    out_root = args.out or os.path.join(project_root(), "data", "ball_dataset")
    img_dir = os.path.join(out_root, "images")
    lbl_dir = os.path.join(out_root, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    cs = resolve_camera_settings(args.exposure, None)
    det = ClassicalBallDetector() if args.prelabel else None

    saved = 0
    frame_idx = 0
    try:
        with CameraManager(
            trigger_mode=args.trigger,
            trigger_source=trigger.get("source", "Line0"),
            pixel_format=image.get("pixel_format", "Mono8"),
            exposure_us=cs["exposure_us"],
            gain_db=cs["gain_db"],
        ) as mgr:
            print(f"已连接 {len(mgr.cameras)} 台相机（触发 {args.trigger}），开始采集 {args.n} 帧到 {out_root}")
            if args.prelabel:
                print("  预标注已开启：经典检测器输出 YOLO 标签，记得后续用 label_ball.py 校正误检。")
            mgr.start()
            while saved < args.n:
                bundle = mgr.get_synchronized_bundle(block=True, timeout=1.0)
                frame_idx += 1
                if frame_idx % max(args.stride, 1) != 0:
                    # 跳过仍喂检测器，保持背景模型新鲜
                    if det is not None:
                        for frame in bundle.frames.values():
                            det.detect(frame)
                    continue

                for cid in sorted(bundle.frames):
                    frame = bundle.frames[cid]
                    name = f"f{saved:06d}_c{cid}"
                    cv2.imwrite(os.path.join(img_dir, name + ".png"), frame.image)

                    if det is not None:
                        balls = det.detect(frame)
                        line = ball_to_yolo(balls[0], frame.width, frame.height) if balls else ""
                        with open(os.path.join(lbl_dir, name + ".txt"), "w") as f:
                            f.write(line + ("\n" if line else ""))
                saved += 1
                if saved % 50 == 0:
                    print(f"  已采 {saved}/{args.n} 帧", flush=True)
    except KeyboardInterrupt:
        pass

    print(f"完成：{saved} 帧 → {img_dir}（标签在 {lbl_dir}）")
    if saved == 0:
        print("⚠ 未采到任何帧，检查触发信号 / Line0 接线。")


if __name__ == "__main__":
    main()
