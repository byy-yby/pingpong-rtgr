#!/usr/bin/env python3
"""多相机 2D 姿态 → 3D 骨架实时重建（跨视角匹配 + 置信度加权 DLT + Open3D）。

流程：读标定（内参 data/calibration/cam_N.yaml、外参 data/extrinsics/table_extrinsics.yaml）
→ 四机取同步帧 → RTMPose-l-halpe26 2D 关键点 → 跨视角球员匹配（几何锚点）→ 逐关节
置信度加权三角化 → Open3D 3D 窗口实时渲染 3D 骨架（相机视锥 + 球桌 + 骨架）。

用法：
  conda activate tt
  python scripts/reconstruct_pose.py                      # 外部触发
  python scripts/reconstruct_pose.py --trigger continuous # 自由采集
  python scripts/reconstruct_pose.py --synthetic          # 无硬件自检：合成 2 人投影重建
  python scripts/reconstruct_pose.py --no-display         # 只算不弹窗，打印 FPS 与误差
  python scripts/reconstruct_pose.py --stride 3 --max-side 720   # 降采样提速（CPU 慢）

提示：本机 GT 1030 弱，RTMPose 跑 CPU；多路实时会卡，先单路 / 降采样 / 隔帧。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from dataclasses import replace

import cv2
import numpy as np

from tabletennis.core.types import (
    CameraExtrinsics,
    CameraIntrinsics,
    Frame,
    Pose2D,
    Skeleton3D,
    Table3D,
)
from tabletennis.reconstruction import (
    AssociationConfig,
    MultiViewTriangulator,
    load_camera_rig,
    match_people,
)
from tabletennis.visualization.overlay2d import annotate_frame, draw_pose, tile_images

# halpe26 关键点索引（与 vision/skeleton.py 一致），供合成数据生成
_K = {
    "nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6, "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10, "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14, "left_ankle": 15, "right_ankle": 16,
    "head": 17, "neck": 18, "hip": 19, "left_big_toe": 20, "right_big_toe": 21,
    "left_small_toe": 22, "right_small_toe": 23, "left_heel": 24, "right_heel": 25,
}


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="多相机姿态 3D 重建（三角化 + Open3D）")
    ap.add_argument("--trigger", choices=["continuous", "external", "software"], default="external")
    ap.add_argument("--exposure", type=float, default=None, help="曝光时间(us)")
    ap.add_argument("--gain", type=float, default=None, help="增益(dB)")

    ap.add_argument("--synthetic", action="store_true", help="无硬件自检：合成 2 人投影重建")
    ap.add_argument("--no-display", action="store_true", help="不弹 2D/3D 窗口，只打印统计")

    ap.add_argument("--model", default="rtmpose-l-halpe26", help="RTMPose 模型标识")
    ap.add_argument("--device", default="cuda", help="cpu / cuda（默认 cuda，缺 CUDA 自动回退）")
    ap.add_argument("--input-size", type=int, nargs=2, default=(192, 256), metavar=("H", "W"))
    ap.add_argument("--score-thr", type=float, default=0.5, help="人体检测置信度阈值")
    ap.add_argument("--backend", default="tensorrt", help="onnxruntime / tensorrt（默认 tensorrt）")
    ap.add_argument("--stride", type=int, default=1, help="隔 N 帧检测一次（复用上次结果提速）")
    ap.add_argument("--max-side", type=int, default=None, help="检测前长边缩到该像素（提速）")

    ap.add_argument("--min-conf", type=float, default=0.3, help="关键点/锚点最低置信度")
    ap.add_argument("--anchor-max-reproj", type=float, default=30.0, help="锚点匹配重投影门限(px)")
    return ap


def make_synthetic_skeleton(x: float, y: float, height: float = 1.75) -> np.ndarray:
    """生成一个站立的 halpe26 3D 骨架（世界系 = 桌面系，Z 向上，米），返回 (26, 3)。"""
    h = float(height)
    k = np.zeros((26, 3), dtype=np.float64)
    k[_K["hip"]] = [x, y, 0.53 * h]
    k[_K["neck"]] = [x, y, 0.82 * h]
    k[_K["head"]] = [x, y, 0.90 * h]
    k[_K["nose"]] = [x, y + 0.03, 0.93 * h]
    k[_K["left_eye"]] = [x - 0.03, y + 0.05, 0.91 * h]
    k[_K["right_eye"]] = [x + 0.03, y + 0.05, 0.91 * h]
    k[_K["left_ear"]] = [x - 0.06, y + 0.03, 0.91 * h]
    k[_K["right_ear"]] = [x + 0.06, y + 0.03, 0.91 * h]
    k[_K["left_shoulder"]] = [x - 0.18, y, 0.81 * h]
    k[_K["right_shoulder"]] = [x + 0.18, y, 0.81 * h]
    k[_K["left_elbow"]] = [x - 0.22, y, 0.62 * h]
    k[_K["right_elbow"]] = [x + 0.22, y, 0.62 * h]
    k[_K["left_wrist"]] = [x - 0.24, y, 0.44 * h]
    k[_K["right_wrist"]] = [x + 0.24, y, 0.44 * h]
    k[_K["left_hip"]] = [x - 0.09, y, 0.52 * h]
    k[_K["right_hip"]] = [x + 0.09, y, 0.52 * h]
    k[_K["left_knee"]] = [x - 0.10, y, 0.26 * h]
    k[_K["right_knee"]] = [x + 0.10, y, 0.26 * h]
    k[_K["left_ankle"]] = [x - 0.10, y, 0.055 * h]
    k[_K["right_ankle"]] = [x + 0.10, y, 0.055 * h]
    k[_K["left_big_toe"]] = [x - 0.10, y + 0.12, 0.02]
    k[_K["right_big_toe"]] = [x + 0.10, y + 0.12, 0.02]
    k[_K["left_small_toe"]] = [x - 0.14, y - 0.02, 0.02]
    k[_K["right_small_toe"]] = [x + 0.14, y - 0.02, 0.02]
    k[_K["left_heel"]] = [x - 0.10, y - 0.10, 0.02]
    k[_K["right_heel"]] = [x + 0.10, y - 0.10, 0.02]
    return k


def project_people_to_cams(
    people_gt: List[np.ndarray],
    view_map: List[List[int]],
    intrinsics: Dict[int, CameraIntrinsics],
    extrinsics: Dict[int, CameraExtrinsics],
    noise_px: float = 1.5,
    seed: int = 0,
) -> Dict[int, List[Pose2D]]:
    """把地面真值 3D 骨架投影到指定相机（走完整畸变模型）+ 加噪，生成 2D 检测。"""
    rng = np.random.default_rng(seed)
    out: Dict[int, List[Pose2D]] = {}
    for pid, kp3 in enumerate(people_gt):
        for cid in view_map[pid]:
            Kobj, ext = intrinsics[cid], extrinsics[cid]
            rvec, _ = cv2.Rodrigues(np.asarray(ext.R, dtype=np.float64))
            proj, _ = cv2.projectPoints(
                kp3.astype(np.float64), rvec,
                np.asarray(ext.t, dtype=np.float64).reshape(3, 1),
                Kobj.K, Kobj.dist,
            )
            uv = proj.reshape(-1, 2) + rng.normal(0.0, noise_px, size=(len(kp3), 2))
            conf = np.clip(0.95 - rng.uniform(0.0, 0.1, size=(len(kp3),)), 0.0, 1.0)
            kpts = np.concatenate([uv, conf[:, None]], axis=1).astype(np.float32)
            out.setdefault(cid, []).append(
                Pose2D(camera_id=cid, keypoints=kpts, score=float(conf.mean()),
                       skeleton="halpe26")
            )
    return out


class ReconstructPose:
    """多相机姿态 3D 重建主循环。"""

    def __init__(self, args) -> None:
        self.args = args
        self.intrinsics, self.extrinsics = load_camera_rig()
        self.triangulator = MultiViewTriangulator(self.intrinsics, self.extrinsics)
        self.table = Table3D()
        self.viewer3d = None
        self._last_poses: Dict[int, List[Pose2D]] = {}
        self._pose_tracker = None

    def _assoc_config(self) -> AssociationConfig:
        return AssociationConfig(
            anchor_min_conf=self.args.min_conf,
            anchor_max_reproj_px=self.args.anchor_max_reproj,
            min_views=2,
        )

    # ------------------------------------------------------------------
    # 场景
    # ------------------------------------------------------------------
    def _start_viewer(self) -> None:
        if self.args.no_display:
            return
        from tabletennis.visualization.viewer3d import SceneViewer3D

        camera_poses = {
            cid: CameraExtrinsics(R=e.R, t=e.t) for cid, e in self.extrinsics.items()
        }
        self.viewer3d = SceneViewer3D()
        self.viewer3d.build_scene(self.table, camera_poses, self.intrinsics)
        self.viewer3d.add_skeleton_layer(skeleton="halpe26", max_people=8)
        self.viewer3d.start()

    def _stop_viewer(self) -> None:
        if self.viewer3d is not None:
            self.viewer3d.close()
            self.viewer3d = None

    # ------------------------------------------------------------------
    # 核心：一帧 2D 检测 -> 3D 骨架
    # ------------------------------------------------------------------
    def reconstruct_frame(self, poses_per_cam: Dict[int, List[Pose2D]]) -> List[Skeleton3D]:
        people = match_people(poses_per_cam, self.triangulator, self._assoc_config())
        skeletons = [
            self.triangulator.triangulate_pose(obs, min_conf=self.args.min_conf)
            for obs in people
        ]
        # 时序跟踪稳定身份（否则 match_people 逐帧独立，多人顺序会闪变）
        if self._pose_tracker is None:
            from tabletennis.reconstruction import PoseTracker
            self._pose_tracker = PoseTracker()
        return self._pose_tracker.update(skeletons)

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

        cs = resolve_camera_settings(self.args.exposure, self.args.gain)
        detector = self._make_detector()
        if detector is None:
            print("[错误] 姿态检测器加载失败（rtmlib/onnxruntime 未安装？），无法重建。")
            return

        self._start_viewer()
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
                while True:
                    bundle = mgr.get_synchronized_bundle(block=True, timeout=1.0)
                    # 批处理检测：YOLOX 逐帧、RTMPose 把 4 路所有人框拼一批
                    poses_per_cam = self._detect_batch_frame(
                        detector, sorted(bundle.frames.items()), frame_idx
                    )

                    skeletons = self.reconstruct_frame(poses_per_cam)
                    if self.viewer3d is not None:
                        self.viewer3d.set_skeletons(skeletons)

                    if not self.args.no_display:
                        key = self._show_2d(bundle, poses_per_cam)
                        if key & 0xFF in (27, ord("q")):
                            break

                    frame_idx += 1
                    n += 1
                    if n % 30 == 0:
                        fps = n / (time.time() - t0)
                        print(f"  {fps:6.1f} fps | 重建 {len(skeletons)} 人", flush=True)
        except KeyboardInterrupt:
            pass
        finally:
            self._stop_viewer()

    def _make_detector(self):
        """按命令行参数构造 RTMPose 检测器（失败返回 None，不中断启动）。"""
        try:
            from tabletennis.vision.pose.rtmpose_pose import RTMPoseDetector
            return RTMPoseDetector(
                model=self.args.model,
                input_size=tuple(self.args.input_size),
                device=self.args.device,
                backend=self.args.backend,
                score_thr=self.args.score_thr,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[警告] RTMPose 加载失败：{exc}")
            return None

    def _detect(self, detector, frame: Frame) -> List[Pose2D]:
        det_frame = frame
        scale = 1.0
        if self.args.max_side and max(frame.height, frame.width) > self.args.max_side:
            scale = self.args.max_side / max(frame.height, frame.width)
            small = cv2.resize(frame.image, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA)
            det_frame = replace(frame, image=small)
        poses = detector.detect(det_frame)
        if scale != 1.0:
            for p in poses:
                p.keypoints[:, :2] /= scale
        return poses

    def _detect_batch_frame(self, detector, frames_items, frame_idx: int) -> Dict[int, List[Pose2D]]:
        """批处理一帧：stride 隔帧复用 + max_side 降采样 + ``detector.detect_batch``。

        Args:
            frames_items: ``[(cid, Frame), ...]`` 同一同步时刻的各相机帧。
        Returns:
            ``{cid: [Pose2D, ...]}``。
        """
        poses_per_cam: Dict[int, List[Pose2D]] = {}
        if frame_idx % max(self.args.stride, 1) != 0:
            for cid, _ in frames_items:
                poses_per_cam[cid] = self._last_poses.get(cid, [])
            return poses_per_cam

        # 需要检测的相机：应用 max_side 降采样
        detect_cids, detect_frames, scales = [], [], {}
        for cid, frame in frames_items:
            scale = 1.0
            det_frame = frame
            if self.args.max_side and max(frame.height, frame.width) > self.args.max_side:
                scale = self.args.max_side / max(frame.height, frame.width)
                small = cv2.resize(frame.image, None, fx=scale, fy=scale,
                                   interpolation=cv2.INTER_AREA)
                det_frame = replace(frame, image=small)
            detect_cids.append(cid)
            detect_frames.append(det_frame)
            scales[cid] = scale

        results = detector.detect_batch(detect_frames)
        for cid, poses in zip(detect_cids, results):
            if scales[cid] != 1.0:
                for p in poses:
                    p.keypoints[:, :2] /= scales[cid]
            self._last_poses[cid] = poses
            poses_per_cam[cid] = poses
        return poses_per_cam

    def _show_2d(self, bundle, poses_per_cam: Dict[int, List[Pose2D]]) -> int:
        images = []
        for cid, frame in sorted(bundle.frames.items()):
            images.append(annotate_frame(
                frame.image, poses_per_cam.get(cid, []), title=f"cam{cid}"
            ))
        if images:
            cv2.imshow("reconstruct-2d", tile_images(images, cols=2))
            return cv2.waitKey(1)
        return -1

    # ------------------------------------------------------------------
    # 合成自检循环
    # ------------------------------------------------------------------
    def run_synthetic(self, n_frames: int = 120) -> None:
        print("合成自检：投影 2 人 → 匹配 → 三角化 → 渲染（无需相机）")
        people_gt = [
            make_synthetic_skeleton(0.5, 0.8),
            make_synthetic_skeleton(1.1, 1.9),
        ]
        view_map = [[0, 1], [2, 3]]  # 每人只被 2 台相机看到
        self._start_viewer()

        errs: List[float] = []
        n_people: List[int] = []
        t0, n = time.time(), 0
        for f in range(n_frames):
            poses_per_cam = project_people_to_cams(
                people_gt, view_map, self.intrinsics, self.extrinsics, seed=f
            )
            skeletons = self.reconstruct_frame(poses_per_cam)
            n_people.append(len(skeletons))
            errs.append(self._eval_synthetic(skeletons, people_gt))

            if self.viewer3d is not None:
                self.viewer3d.set_skeletons(skeletons)
            if self.args.no_display:
                time.sleep(0.02)
            else:
                cv2.imshow("reconstruct-2d", self._synthetic_2d(poses_per_cam))
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
            n += 1

        t1 = time.time()
        if self.viewer3d is not None:
            time.sleep(0.5)  # 让渲染线程多跑一会再关
        print(f"合成自检完成：{n} 帧 / {t1 - t0:.2f}s，平均重建 {np.mean(n_people):.2f} 人")
        valid = [e for e in errs if np.isfinite(e)]
        if valid:
            print(f"  3D 关节中位误差 ≈ {np.median(valid) * 1000:.1f} mm")
        self._stop_viewer()

    def _eval_synthetic(self, skeletons: List[Skeleton3D], people_gt: List[np.ndarray]) -> float:
        """重建骨架与地面真值最近邻配对，返回中位关节误差（米）。"""
        medians: List[float] = []
        for skel in skeletons:
            kp = skel.keypoints
            valid = np.isfinite(kp).all(axis=1)
            if valid.sum() < 2:
                continue
            best = min(
                float(np.median(np.linalg.norm(kp[valid] - gt[valid], axis=1)))
                for gt in people_gt
            )
            medians.append(best)
        return float(np.median(medians)) if medians else float("nan")

    def _synthetic_2d(self, poses_per_cam: Dict[int, List[Pose2D]]) -> np.ndarray:
        images = []
        for cid in sorted(poses_per_cam):
            img = np.zeros((1080, 1440, 3), np.uint8)
            for pose in poses_per_cam[cid]:
                draw_pose(img, pose)
            cv2.putText(img, f"cam{cid}", (8, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (255, 255, 255), 2, cv2.LINE_AA)
            images.append(img)
        return tile_images(images, cols=2)


def main() -> None:
    args = build_arg_parser().parse_args()
    app = ReconstructPose(args)

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
