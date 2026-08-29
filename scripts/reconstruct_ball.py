#!/usr/bin/env python3
"""多相机 2D 球检测 → 3D 球心重建（经典检测 + 置信度加权 DLT + 卡尔曼）。

流程：读标定（内参 data/calibration/cam_N.yaml、外参 data/extrinsics/table_extrinsics.yaml）
→ 四机取同步帧 → 每机经典球检测（背景减除 + 帧差 + 尺寸先验）+ 亚像素质心
→ 置信度加权 DLT 三角化 → 卡尔曼平滑 → 2D 叠加 + 打印 3D 轨迹。

用法：
  conda activate tt
  python scripts/reconstruct_ball.py                       # 外部触发
  python scripts/reconstruct_ball.py --trigger continuous  # 自由采集
  python scripts/reconstruct_ball.py --synthetic           # 无硬件自检：合成轨迹重建
  python scripts/reconstruct_ball.py --no-display          # 只算不弹窗，打印 FPS 与误差

注意：球检测是有状态背景模型，每台相机须用独立实例（各视角背景不同）。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import cv2
import numpy as np

from tabletennis.core.types import Ball2D, CameraExtrinsics, Frame, Table3D
from tabletennis.reconstruction import (
    BallTracker,
    MultiViewTriangulator,
    load_camera_rig,
    triangulate_ball,
)
from tabletennis.vision.ball import ClassicalBallDetector
from tabletennis.visualization.overlay2d import draw_ball, gray_to_bgr, tile_images


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="多相机乒乓球 3D 重建（经典检测 + 三角化 + 卡尔曼）")
    ap.add_argument("--trigger", choices=["continuous", "external", "software"], default="external")
    ap.add_argument("--exposure", type=float, default=None, help="曝光时间(us)；冻结球需 ≤100")

    ap.add_argument("--synthetic", action="store_true", help="无硬件自检：合成轨迹投影重建")
    ap.add_argument("--no-display", action="store_true", help="不弹 2D 窗口，只打印统计")
    ap.add_argument("--n-frames", type=int, default=120, help="合成自检帧数")
    ap.add_argument("--noise-px", type=float, default=0.3, help="合成自检 2D 噪声(px)")

    ap.add_argument("--min-conf", type=float, default=0.3, help="球检测最低置信度")
    ap.add_argument("--radius-min", type=float, default=5.0, help="球半径像素下限")
    ap.add_argument("--radius-max", type=float, default=15.0, help="球半径像素上限")
    ap.add_argument("--ball-model", default=None, help="YOLO ONNX 模型路径（提供则用 YOLO 代替经典检测）")
    ap.add_argument("--imgsz", type=int, default=1280, help="YOLO 输入分辨率")
    return ap


def make_synthetic_trajectory(n_frames: int) -> np.ndarray:
    """生成一条桌面系内的抛物线轨迹（世界系=桌面系，X 短边 / Y 长边 / Z 向上）。

    Returns:
        ``(n_frames, 3)`` 球心坐标（米）。
    """
    pts = []
    for i in range(n_frames):
        t = i / max(n_frames - 1, 1)
        x = 0.7625                       # 桌面短边中点
        y = 0.3 + 2.14 * t               # 沿长边从近到远
        z = 0.15 + 4 * 0.35 * t * (1 - t)  # 抛物线，峰值约 0.5m
        pts.append([x, y, z])
    return np.array(pts, dtype=np.float64)


class ReconstructBall:
    """多相机球 3D 重建主循环。"""

    def __init__(self, args) -> None:
        self.args = args
        self.intrinsics, self.extrinsics = load_camera_rig()
        self.triangulator = MultiViewTriangulator(self.intrinsics, self.extrinsics)
        self.tracker = BallTracker()
        self.table = Table3D()
        self.viewer3d = None
        # 检测器：--ball-model 给定则用 YOLO（无状态共享）；否则每相机一个经典检测器
        self.detector = None
        self.detectors: Dict[int, ClassicalBallDetector] = {}
        if args.ball_model:
            from tabletennis.vision.ball import YoloBallDetector
            self.detector = YoloBallDetector(args.ball_model, imgsz=args.imgsz)
        else:
            self.detectors = {
                cid: ClassicalBallDetector(radius_px=(args.radius_min, args.radius_max))
                for cid in self.triangulator.cameras
            }

    # ------------------------------------------------------------------
    def _project_trajectory(self, X3: np.ndarray, rng) -> Dict[int, Ball2D]:
        """把 3D 球心投影到各相机（走完整畸变模型）+ 加噪，构造 Ball2D 检测。"""
        balls: Dict[int, Ball2D] = {}
        for cid in self.triangulator.cameras:
            Kobj, ext = self.intrinsics[cid], self.extrinsics[cid]
            rvec, _ = cv2.Rodrigues(np.asarray(ext.R, dtype=np.float64))
            proj, _ = cv2.projectPoints(
                X3.astype(np.float64).reshape(1, 3), rvec,
                np.asarray(ext.t, dtype=np.float64).reshape(3, 1),
                Kobj.K, Kobj.dist,
            )
            u, v = proj.reshape(2)
            u += rng.normal(0.0, self.args.noise_px)
            v += rng.normal(0.0, self.args.noise_px)
            if not (0 <= u < Kobj.width and 0 <= v < Kobj.height):
                continue
            balls[cid] = Ball2D(
                camera_id=cid,
                center=np.array([u, v], dtype=np.float32),
                radius=8.0,
                confidence=0.9,
            )
        return balls

    # ------------------------------------------------------------------
    # 核心：一帧 2D 检测 -> 3D 球心
    # ------------------------------------------------------------------
    def reconstruct_frame(self, balls_per_cam: Dict[int, Ball2D]):
        return triangulate_ball(balls_per_cam, self.triangulator, min_conf=self.args.min_conf)

    def _detect(self, frame: Frame) -> List[Ball2D]:
        """对单帧做球检测：YOLO（共享）或该相机的经典检测器。"""
        if self.detector is not None:
            return self.detector.detect(frame)
        det = self.detectors.get(frame.camera_id)
        return det.detect(frame) if det is not None else []

    def _start_viewer(self) -> None:
        """启动 Open3D 3D 场景（球桌 + 相机 + 球轨迹），--no-display 时跳过。"""
        if self.args.no_display:
            return
        from tabletennis.visualization.viewer3d import SceneViewer3D

        camera_poses = {
            cid: CameraExtrinsics(R=e.R, t=e.t) for cid, e in self.extrinsics.items()
        }
        self.viewer3d = SceneViewer3D()
        self.viewer3d.build_scene(self.table, camera_poses, self.intrinsics)
        self.viewer3d.add_ball_layer()
        self.viewer3d.start()

    def _stop_viewer(self) -> None:
        if self.viewer3d is not None:
            self.viewer3d.close()
            self.viewer3d = None

    # ------------------------------------------------------------------
    # 真实相机循环
    # ------------------------------------------------------------------
    def run_live(self) -> None:
        from tabletennis.camera import CameraManager
        from tabletennis.core.config import load_yaml, resolve_camera_settings

        root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        cfg_path = os.path.join(root, "config", "cameras.yaml")
        config = load_yaml(cfg_path) if os.path.exists(cfg_path) else {}
        trigger = config.get("trigger", {}) or {}
        image = config.get("image", {}) or {}

        cs = resolve_camera_settings(self.args.exposure, None)
        frame_idx = 0
        t0, n = time.time(), 0
        try:
            with CameraManager(
                trigger_mode=self.args.trigger,
                trigger_source=trigger.get("source", "Line0"),
                pixel_format=image.get("pixel_format", "Mono8"),
                exposure_us=cs["exposure_us"],
                gain_db=cs["gain_db"],
            ) as mgr:
                print(f"已连接 {len(mgr.cameras)} 台相机（触发 {self.args.trigger}），按 ESC/q 退出")
                mgr.start()
                self._start_viewer()
                while True:
                    bundle = mgr.get_synchronized_bundle(block=True, timeout=1.0)
                    balls_per_cam: Dict[int, Ball2D] = {}
                    for cid, frame in bundle.frames.items():
                        dets = self._detect(frame)
                        if dets:
                            balls_per_cam[cid] = dets[0]  # 单球，取最高置信者

                    res = self.reconstruct_frame(balls_per_cam)
                    X3 = None
                    if res is not None:
                        X3, conf, err, nv, ang = res
                    X3 = self.tracker.update(X3, conf if res else 0.0)
                    if self.viewer3d is not None:
                        self.viewer3d.set_ball(X3)

                    if not self.args.no_display:
                        key = self._show_2d(bundle, balls_per_cam)
                        if key & 0xFF in (27, ord("q")):
                            break

                    frame_idx += 1
                    n += 1
                    if n % 30 == 0:
                        fps = n / (time.time() - t0)
                        pos = "" if X3 is None else f" 球≈({X3[0]:.3f},{X3[1]:.3f},{X3[2]:.3f})m"
                        print(f"  {fps:6.1f} fps{pos}", flush=True)
        except KeyboardInterrupt:
            pass
        finally:
            self._stop_viewer()

    def _show_2d(self, bundle, balls_per_cam: Dict[int, Ball2D]) -> int:
        images = []
        for cid, frame in sorted(bundle.frames.items()):
            img = gray_to_bgr(frame.image)
            ball = balls_per_cam.get(cid)
            if ball is not None:
                draw_ball(img, ball)
            cv2.putText(img, f"cam{cid}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2, cv2.LINE_AA)
            images.append(img)
        if images:
            cv2.imshow("reconstruct-ball", tile_images(images, cols=2))
            return cv2.waitKey(1)
        return -1

    # ------------------------------------------------------------------
    # 合成自检循环
    # ------------------------------------------------------------------
    def run_synthetic(self) -> None:
        print(f"合成自检：投影已知 3D 轨迹 → 三角化（噪声 {self.args.noise_px}px，无需相机）")
        self._start_viewer()
        traj = make_synthetic_trajectory(self.args.n_frames)
        rng = np.random.default_rng(0)
        errs: List[float] = []

        for i, X_gt in enumerate(traj):
            balls = self._project_trajectory(X_gt, rng)
            res = self.reconstruct_frame(balls)
            if res is None:
                continue
            X, _c, _e, nv, _a = res
            errs.append(float(np.linalg.norm(X - X_gt)))
            if self.viewer3d is not None:
                self.viewer3d.set_ball(X)

        if not errs:
            print("[错误] 合成自检没有成功三角化任何一帧，检查标定是否加载。")
            return
        med = float(np.median(errs))
        print(f"合成自检完成：成功 {len(errs)}/{self.args.n_frames} 帧")
        print(f"  3D 球心中位误差 ≈ {med * 1000:.2f} mm（噪声 {self.args.noise_px}px）")
        if self.viewer3d is not None:
            time.sleep(0.5)  # 让渲染线程多跑一会再关
        self._stop_viewer()


def main() -> None:
    args = build_arg_parser().parse_args()
    app = ReconstructBall(args)

    if not app.triangulator.cameras:
        print("[错误] 未加载到任何相机的内外参，请先完成标定。")
        sys.exit(1)

    print(f"已加载标定：{len(app.triangulator.cameras)} 台相机（世界系 = 桌面系）")

    if args.synthetic:
        app.run_synthetic()
    else:
        app.run_live()


if __name__ == "__main__":
    main()
