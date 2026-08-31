#!/usr/bin/env python3
"""交互式相机控制：四机实时预览 + 参数调节 + 检测开关。

单窗口布局（自上而下）：
  1. 2×2 相机画面（含检测叠加 + 触发信号自检提示）
  2. 控制面板：曝光 / 增益 / 伽马 三条等宽滑块（带名称与当前值标注，鼠标拖拽调节）
  3. 提示条：检测开关状态 + 操作说明

用法：
  conda activate tt
  python scripts/live_control.py                        # 外部触发（默认）
  python scripts/live_control.py --trigger continuous   # 自由采集（无信号发生器）
  python scripts/live_control.py --exposure 5000 --gain 5 --gamma 1.0
  python scripts/live_control.py --max-width 2400       # 调窗口大小
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from typing import Dict, Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import cv2
import numpy as np

from tabletennis.calibration.extrinsics import save_extrinsics
from tabletennis.camera import CameraManager
from tabletennis.core.config import load_yaml, project_root, resolve_camera_settings, save_camera_settings
from tabletennis.core.types import CameraExtrinsics, Frame
from tabletennis.visualization.overlay2d import (
    draw_ball,
    draw_pose,
    draw_table_model,
    gray_to_bgr,
    tile_images,
)
from tabletennis.vision.ball import ClassicalBallDetector
from tabletennis.vision.detector import create_detector

MAIN_WIN = "Cameras"

# 检测开关：kind -> (中文名, 键盘键)
DETECTION_TOGGLES = {
    "pose": ("人体姿态", "p"),
    "ball": ("球", "b"),
    "table": ("球桌", "t"),
}

# 三个可调参数：名称 / 范围 / 单位 / 显示格式
PARAM_SPECS = [
    {"key": "exposure", "label": "曝光", "min": 15.0, "max": 20000.0, "unit": " us", "fmt": "{:>6.0f}"},
    {"key": "gain", "label": "增益", "min": 0.0, "max": 17.0, "unit": " dB", "fmt": "{:>4.1f}"},
    {"key": "gamma", "label": "伽马", "min": 0.0, "max": 4.0, "unit": "", "fmt": "{:>5.2f}"},
]
PARAM_BY_KEY = {s["key"]: s for s in PARAM_SPECS}

GRID_COLS = 2
DEFAULT_MAX_WIDTH = 1920          # 视频网格目标宽度（越大窗口越大）
PANEL_H = 180                     # 控制面板高度
HINT_H = 46                       # 提示条高度
DEFAULT_SIZE = (1080, 1440)       # (H, W)：MV-CS016-10UM 全分辨率 1440x1080


def camera_kwargs_from_config(config: dict) -> dict:
    """把 cameras.yaml 的 trigger/image 段转成 CameraManager 的 kwargs。

    曝光 / 增益 / 伽马不走 cameras.yaml，改由 config/camera_settings.json 统一管理。
    """
    trigger = config.get("trigger", {}) or {}
    image = config.get("image", {}) or {}
    return {
        "trigger_mode": trigger.get("mode", "external"),
        "trigger_source": trigger.get("source", "Line0"),
        "pixel_format": image.get("pixel_format", "Mono8"),
    }


class LiveControl:
    """交互式控制：单窗口，视频在上、自绘滑块在中、提示在下。"""

    def __init__(self, mgr: CameraManager, *, exposure_us, gain_db, gamma, trigger_mode, max_width):
        self.mgr = mgr
        self.trigger_mode = trigger_mode
        self.max_width = max_width

        self.values = {"exposure": float(exposure_us), "gain": float(gain_db), "gamma": float(gamma)}
        self.enable: Dict[str, bool] = {k: False for k in DETECTION_TOGGLES}
        # 惰性创建：首次开启某项检测时才实例化（避免启动时下载/加载模型）
        self.detectors = {k: None for k in DETECTION_TOGGLES}

        self._latest: Dict[int, Frame] = {}
        self._ref_size = DEFAULT_SIZE
        self.trigger_error = False
        self._slider_rects: Dict[str, tuple] = {}   # key -> (x0, x1, y0, y1) 画布坐标
        self._save_rect = (0, 0, 0, 0)              # 保存按钮命中区域（画布坐标）
        self._save_flash_until = 0.0                # 「已保存」提示显示到的时间点
        self._canvas_shape = (0, 0)

        # 球桌识别（按 T）：检测器 + 一次性缓存的外参 + Open3D 3D 场景
        self._table_detector = None
        self._table_poses: Dict[int, tuple] = {}   # cid -> (R, t)（桌面系 -> 相机系）
        self.viewer3d = None

        # 数据录制（按 r）：倒计时 3s → 存图 + 经典检测器预标注
        self._recording = False
        self._record_countdown_until = 0.0
        self._record_count = 0
        self._record_base = 0
        self._record_detector = None
        self._record_saved_framenum: Dict[int, int] = {}

        # 3D 姿态重建（按 P）：标定三角化器 + 各相机最近一帧的 2D 姿态
        self._triangulator = None
        self._recon_extrinsics: Dict[int, object] = {}
        self._last_poses: Dict[int, list] = {}
        self._pose_tracker = None
        self._frame_idx = 0
        self._pose_fps = 0.0        # 3D 姿态重建实时帧率（EMA，右上角叠加显示）

        # 3D 球重建（按 B）：各相机最近一帧球检测 + 3D 球心 + 卡尔曼平滑
        self._last_balls: Dict[int, list] = {}
        self._ball3d = None            # 最近一帧 3D 球心（桌面系，米）
        self._ball_tracker = None      # 卡尔曼平滑（重建前才创建）
        self._ball_ready = False       # 球模型+三角化器已就绪（后台线程置位）
        self._ball_pending_viewer = False  # 模型就绪但 3D 窗口待主线程打开
        self._ball_load_thread: Optional[threading.Thread] = None
        # 球重建实际帧率（EMA 平滑，用于画面右上角显示）
        self._ball_fps: Optional[float] = None
        self._ball_fps_t: Optional[float] = None
        self._ball_diag_t: Optional[float] = None   # 三角化失败诊断限频

    # ------------------------------------------------------------------
    # 参数应用
    # ------------------------------------------------------------------
    def _apply_param(self, key: str, value: float) -> None:
        for cam in self.mgr.cameras:
            c = cam.controls
            if key == "exposure":
                c.set_exposure_time_us(value)
            elif key == "gain":
                c.set_gain_db(value)
            elif key == "gamma":
                c.set_gamma(value)

    def apply_all_params(self) -> None:
        """启动时把初始参数一次性应用到所有相机（伽马也在这里统一设）。"""
        for key, value in self.values.items():
            self._apply_param(key, value)

    def save_settings(self) -> None:
        """把当前曝光 / 增益 / 伽马保存到 config/camera_settings.json。"""
        save_camera_settings(self.values["exposure"], self.values["gain"], self.values["gamma"])
        self._save_flash_until = time.time() + 1.5
        print("✓ 已保存曝光/增益/伽马到 config/camera_settings.json")

    # ------------------------------------------------------------------
    # 数据录制（按 r）：倒计时 3s → 存图 + 预标注，再按 r 停止
    # ------------------------------------------------------------------
    def toggle_record(self) -> None:
        """切换录制：录制中按 r 立即停止；空闲按 r 开始 3s 倒计时。"""
        if self._recording:
            self._recording = False
            print(f"[录制] ■ 停止，本次共保存 {self._record_count} 组帧 → data/ball_dataset/")
            return
        if self._record_countdown_until > 0:
            self._record_countdown_until = 0.0  # 倒计时中再按 → 取消
            print("[录制] 已取消")
            return
        self._record_countdown_until = time.time() + 3.0
        self._record_count = 0
        self._record_base = self._next_record_index()
        self._record_saved_framenum = {}
        print("[录制] 3 秒后开始录制（再按 r 停止）...")
        print(f"        保存到 data/ball_dataset/（从 f{self._record_base:06d} 起）")

    def _next_record_index(self) -> int:
        """扫描已有数据，返回下一个可用编号，避免覆盖之前的录制。"""
        img_dir = os.path.join(project_root(), "data", "ball_dataset", "images")
        if not os.path.isdir(img_dir):
            return 0
        max_idx = -1
        for name in os.listdir(img_dir):
            base = name.split("_c")[0]
            if base.startswith("f") and base[1:].isdigit():
                max_idx = max(max_idx, int(base[1:]))
        return max_idx + 1

    def _tick_record(self, latest: Dict[int, Frame]) -> None:
        """每帧驱动录制状态机：倒计时（预热背景）→ 录制（存图 + 预标注）。"""
        if self._record_countdown_until <= 0 and not self._recording:
            return

        if self._record_detector is None:
            self._record_detector = ClassicalBallDetector()

        # 倒计时阶段：喂帧预热背景，到点转录制
        if not self._recording:
            for f in latest.values():
                if f is not None:
                    self._record_detector.detect(f)
            if time.time() >= self._record_countdown_until:
                self._recording = True
                self._record_countdown_until = 0.0
                print("[录制] ▶ 开始录制（再按 r 停止）")
            return

        # 录制阶段：按帧号去重，保存新帧 + YOLO 预标注
        img_dir = os.path.join(project_root(), "data", "ball_dataset", "images")
        lbl_dir = os.path.join(project_root(), "data", "ball_dataset", "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

        saved_any = False
        for cid, f in latest.items():
            if f is None:
                continue
            if self._record_saved_framenum.get(cid) == f.frame_num:
                continue
            self._record_saved_framenum[cid] = f.frame_num
            name = f"f{self._record_base + self._record_count:06d}_c{cid}"
            cv2.imwrite(os.path.join(img_dir, name + ".png"), f.image)
            balls = self._record_detector.detect(f)
            line = self._ball_to_yolo(balls[0], f.width, f.height) if balls else ""
            with open(os.path.join(lbl_dir, name + ".txt"), "w") as fh:
                fh.write(line + ("\n" if line else ""))
            saved_any = True
        if saved_any:
            self._record_count += 1

    @staticmethod
    def _ball_to_yolo(ball, W: int, H: int) -> str:
        """Ball2D → YOLO 标签行 ``0 cx cy w h``（归一化）。"""
        cx = min(max(float(ball.center[0]) / W, 0.0), 1.0)
        cy = min(max(float(ball.center[1]) / H, 0.0), 1.0)
        w = min(max(float(2.0 * ball.radius) / W, 0.0), 1.0)
        h = min(max(float(2.0 * ball.radius) / H, 0.0), 1.0)
        return f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"

    # ------------------------------------------------------------------
    # 触发信号检查
    # ------------------------------------------------------------------
    def wait_for_trigger(self, timeout_s: float = 3.0) -> None:
        """外部触发模式下等待第一帧，超时说明没时钟/触发信号。"""
        print(f"等待触发/时钟信号（最多 {timeout_s}s）...")
        deadline = time.time() + timeout_s
        got = set()
        while time.time() < deadline:
            frames = self.mgr.get_latest_frames(block=False)
            for cid, f in frames.items():
                if f is not None:
                    got.add(cid)
            if got:
                break
            time.sleep(0.05)

        if got:
            self.trigger_error = False
            print(f"✓ 已收到 {len(got)}/{len(self.mgr.cameras)} 台相机的帧，触发信号正常。")
        else:
            self.trigger_error = True
            print(f"❌ 外部触发模式下 {timeout_s}s 内未收到任何帧——请检查信号发生器 / Line0 接线。")

    # ------------------------------------------------------------------
    # 检测开关
    # ------------------------------------------------------------------
    def toggle(self, kind: str) -> None:
        label, _ = DETECTION_TOGGLES[kind]
        self.enable[kind] = not self.enable[kind]
        state = "ON" if self.enable[kind] else "OFF"

        # 球桌识别需要标定数据 + 3D 场景，走专用路径（一次性识别 + 缓存外参）
        if kind == "table":
            if self.enable[kind]:
                self._enable_table()
            else:
                self._disable_table()
            return

        # 姿态：GPU 实时 3D 重建（按 P）
        if kind == "pose":
            if self.enable[kind]:
                self._enable_pose_recon()
            else:
                self._disable_pose_recon()
            return

        # 球：2D 检测 + 3D DLT 三角化 + Open3D 渲染（按 B）
        if kind == "ball":
            if self.enable[kind]:
                self._enable_ball_recon()
            else:
                self._disable_ball_recon()
            return

        if self.enable[kind] and self.detectors[kind] is None:
            self.detectors[kind] = create_detector(kind)   # 首次开启才加载模型
            if self.detectors[kind] is None:
                print(f"[检测] {label}: ON（接口已定义，算法待实现）")
                return
        print(f"[检测] {label}: {state}")

    def handle_key(self, key: int) -> None:
        if key == ord("s"):
            self.save_settings()
            return
        if key == ord("r"):
            self.toggle_record()
            return
        for kind, (_, k) in DETECTION_TOGGLES.items():
            if key == ord(k):
                self.toggle(kind)
                return

    # ------------------------------------------------------------------
    # 球桌识别 + 3D 场景
    # ------------------------------------------------------------------
    def _enable_table(self) -> None:
        """识别球桌：跨相机融合检测大标记得到各相机桌面系外参，缓存、保存并启动 3D 场景。

        球桌静止不动，识别一次即可；之后每帧用缓存外参把标准尺寸球桌（桌面边框 +
        桌腿 + 球网）投影到各相机，3D 场景也用同一组外参构建。融合算法与
        calibrate_extrinsics.py 的 T 键一致，并把结果写到 table_extrinsics.yaml，
        供球/姿态重建直接读取。
        """
        if self._table_detector is None:
            self._table_detector = create_detector("table")
            if self._table_detector is None:
                print("[检测] 球桌: ON（接口已定义，算法待实现）")
                return
        self.detectors["table"] = self._table_detector

        # 用一组同步帧做一次性跨相机融合识别（外部触发下四机来自同一时钟周期）
        bundle = self.mgr.get_synchronized_bundle(block=True, timeout=1.0)
        if not bundle.frames:
            print("[检测] 球桌: ON——但未收到同步帧（检查触发信号 / Line0）。")
            return
        gray_frames = {cid: f.image for cid, f in bundle.frames.items()}
        table_extrinsics = self._table_detector.localize_bundle(gray_frames)

        poses: Dict[int, tuple] = {}
        n_live = n_fallback = 0
        if table_extrinsics:
            poses.update(table_extrinsics)
            n_live = len(table_extrinsics)

        # 融合失败或某相机缺失时，回退到已保存外参（没有则该相机不画）
        for cid, ext in self._table_detector.fallback.items():
            if cid not in poses and cid in self._table_detector.intrinsics:
                poses[cid] = (ext.R, ext.t)
                n_fallback += 1

        if not poses:
            print("[检测] 球桌: ON——但未识别到球桌（缺画面 / 内参 / 外参），请先完成标定。")
            return

        self._table_poses = poses
        print(f"[检测] 球桌: ON（跨相机融合识别 {n_live} 台，回退已保存外参 {n_fallback} 台）")
        if table_extrinsics:
            self._save_table_extrinsics(table_extrinsics)
        elif n_live == 0:
            print("  ⚠ 融合失败，正在使用 data/extrinsics/table_extrinsics.yaml 里的已存外参。")
            print("    请确认四角大标记可见，或先跑 calibrate_extrinsics.py 按 T 重新定原点。")
        self._start_viewer()

    def _save_table_extrinsics(self, table_extrinsics: Dict[int, tuple]) -> None:
        """把融合得到的桌面系外参写到 table_extrinsics.yaml（供球/姿态重建读取）。"""
        out_dir = os.path.join(project_root(), "data", "extrinsics")
        path = os.path.join(out_dir, "table_extrinsics.yaml")
        save_extrinsics(path, table_extrinsics, world_frame="table")
        print(f"✓ 已保存桌面坐标系 -> {path}")

    def _disable_table(self) -> None:
        self._table_poses = {}
        self._close_viewer()
        print("[检测] 球桌: OFF")

    def _annotate_table(self, bgr: np.ndarray, cid: int) -> None:
        """把缓存外参下的标准尺寸球桌线框画到该相机画面。"""
        if self._table_detector is None:
            return
        pose = self._table_poses.get(cid)
        K = self._table_detector.intrinsics.get(cid)
        if pose is None or K is None:
            return
        R, t = pose
        draw_table_model(bgr, self._table_detector.table, R, t, K.K, K.dist)

    def _start_viewer(self) -> None:
        """启动（或复用）Open3D 3D 场景窗口。"""
        if self.viewer3d is not None and self.viewer3d.is_running():
            return
        if self._table_detector is None:
            return
        from tabletennis.visualization.viewer3d import SceneViewer3D

        camera_poses = {
            cid: CameraExtrinsics(R=R, t=t) for cid, (R, t) in self._table_poses.items()
        }
        intrinsics = {
            cid: K
            for cid, K in self._table_detector.intrinsics.items()
            if cid in self._table_poses
        }
        self.viewer3d = SceneViewer3D()
        self.viewer3d.build_scene(self._table_detector.table, camera_poses, intrinsics)
        self.viewer3d.add_skeleton_layer(skeleton="halpe26", max_people=8)
        self.viewer3d.add_ball_layer(trail_len=200)
        self.viewer3d.start()
        print("✓ 已生成 3D 场景窗口（Open3D，可鼠标旋转 / 缩放）。")

    def _close_viewer(self) -> None:
        if self.viewer3d is not None:
            self.viewer3d.close()
            self.viewer3d = None

    # ------------------------------------------------------------------
    # 3D 姿态重建（按 P）
    # ------------------------------------------------------------------
    def _enable_pose_recon(self) -> None:
        """按 P 开启 GPU 实时 3D 姿态重建：加载标定三角化器 + 启动 3D 场景。"""
        if self.detectors["pose"] is None:
            self.detectors["pose"] = create_detector("pose")
            if self.detectors["pose"] is None:
                print("[检测] 姿态: ON（接口已定义，算法待实现）")
                return
        if self._triangulator is None:
            try:
                from tabletennis.reconstruction import (
                    MultiViewTriangulator,
                    load_camera_rig,
                )
                intrinsics, extrinsics = load_camera_rig()
                if not extrinsics:
                    print("[检测] 姿态: 未找到标定外参（table_extrinsics.yaml），仅 2D 显示")
                    return
                self._recon_extrinsics = extrinsics
                self._triangulator = MultiViewTriangulator(intrinsics, extrinsics)
                print(f"[检测] 姿态: 三角化器已加载 {len(self._triangulator.cameras)} 台相机")
                # 用标定外参当桌面系相机位姿（与 table 检测 fallback 一致）
                self._table_poses = {cid: (e.R, e.t) for cid, e in extrinsics.items()}
                if self._table_detector is None:
                    self._table_detector = create_detector("table")
            except Exception as exc:  # noqa: BLE001
                print(f"[检测] 姿态: 标定加载失败（{exc}），仅 2D 显示")
                return
        self._start_viewer()
        print("[检测] 姿态: ON（GPU 实时 3D 重建 + Open3D）")

    def _disable_pose_recon(self) -> None:
        self._last_poses = {}
        self._pose_fps = 0.0
        print("[检测] 姿态: OFF")

    def _reconstruct_frame(self) -> None:
        """批处理检测所有相机 + 跨视角匹配 + 三角化，更新 Open3D 骨架与 2D 姿态。"""
        from tabletennis.reconstruction import match_people

        t0 = time.perf_counter()
        detector = self.detectors["pose"]
        frames_items = [
            (cid, f) for cid, f in sorted(self._latest.items()) if f is not None
        ]
        if not frames_items:
            return
        cids = [c for c, _ in frames_items]
        frames = [f for _, f in frames_items]
        t_det0 = time.perf_counter()
        try:
            poses_list = detector.detect_batch(frames)
        except AttributeError:  # 检测器无批处理接口则逐帧回退
            poses_list = [detector.detect(f) for f in frames]
        t_detect = time.perf_counter() - t_det0
        poses_per_cam = dict(zip(cids, poses_list))
        self._last_poses = poses_per_cam

        t_recon = 0.0
        if self._triangulator is not None:
            t_r0 = time.perf_counter()
            people = match_people(poses_per_cam, self._triangulator)
            skeletons = [self._triangulator.triangulate_pose(obs) for obs in people]
            # 硬编码身份：按球桌长边(Y)两侧分 ID（Y 中点 2.74/2=1.37m），绝对稳定
            if self._pose_tracker is None:
                from tabletennis.reconstruction import PoseTracker
                self._pose_tracker = PoseTracker(partition_axis=1, partition_threshold=1.37)
            skeletons = self._pose_tracker.update(skeletons)
            if self.viewer3d is not None:
                self.viewer3d.set_skeletons(skeletons)
            t_recon = time.perf_counter() - t_r0
            # 诊断日志：前 5 帧 + 每 60 帧打印一次，定位骨架不出现的环节
            self._frame_idx += 1
            if self._frame_idx <= 5 or self._frame_idx % 60 == 0:
                n_det = {c: len(poses_per_cam.get(c, [])) for c in cids}
                n_valid = sum(
                    int(np.isfinite(s.keypoints).all(axis=1).sum()) for s in skeletons
                )
                print(f"[3D重建] 帧{self._frame_idx}: 检测 {n_det} | "
                      f"匹配 {len(people)} 人 | 有效关节 {n_valid} | "
                      f"检测 {t_detect*1000:.1f}ms + 重建 {t_recon*1000:.2f}ms")

        # 实时帧率（EMA）：本轮检测+三角化耗时换算成 FPS，供右上角叠加显示
        dt = time.perf_counter() - t0
        if dt > 0:
            fps = 1.0 / dt
            self._pose_fps = fps if self._pose_fps <= 0 else 0.9 * self._pose_fps + 0.1 * fps

    # ------------------------------------------------------------------
    # 球 2D 检测 + 3D 重建（按 B）
    # ------------------------------------------------------------------
    def _enable_ball_recon(self) -> None:
        """按 B 开启球追踪：UI 不阻塞，模型在后台线程加载。

        模型创建（onnxruntime CUDA session ~1s）+ 首帧热启动（~6s）都较重，
        若在主线程同步做会卡死窗口；改由 ``_ball_load_worker`` 后台加载，
        就绪后置 ``_ball_ready`` / ``_ball_pending_viewer``，主循环再开 3D 窗口并开始重建。
        """
        if self.detectors["ball"] is not None:
            self._ball_ready = True
            print(f"[检测] 球: ON（{type(self.detectors['ball']).__name__} + DLT + Open3D）")
            return
        if self._ball_load_thread is not None and self._ball_load_thread.is_alive():
            return  # 正在后台加载中
        self._ball_ready = False
        self._ball_load_thread = threading.Thread(
            target=self._ball_load_worker, name="ball-loader", daemon=True)
        self._ball_load_thread.start()
        print("[检测] 球: 模型后台加载中（首次约几秒，窗口不卡）…")

    def _ball_load_worker(self) -> None:
        """后台线程：创建球检测器 + 热启动首帧 + 加载三角化器；就绪后置标志。"""
        try:
            # YOLO（onnxruntime，无状态可跨相机共享）优先；ONNX 缺失回退经典
            self.detectors["ball"] = create_detector("ball_yolo") or create_detector("ball")
            if self.detectors["ball"] is None:
                print("[检测] 球: ON（接口已定义，算法待实现）")
                return
            det = self.detectors["ball"]

            # 热启动：喂一帧，把首次推理的惰性初始化（TRT 首建 batch-1 引擎
            # ~30-60s / CUDA EP 首次 ~6s）从主循环挪到后台线程。
            probe = next((f for f in self._latest.values() if f is not None), None)
            if probe is None:
                probe = Frame(camera_id=0, serial="probe", frame_num=0,
                              device_timestamp=0, host_timestamp=0,
                              image=np.zeros((1080, 1440), np.uint8),
                              pixel_format=0, width=1440, height=1080)
            det.detect(probe)

            # 三角化器：与姿态共用一套标定，只加载一次
            if self._triangulator is None:
                from tabletennis.reconstruction import (
                    MultiViewTriangulator,
                    load_camera_rig,
                )
                intrinsics, extrinsics = load_camera_rig()
                if not extrinsics:
                    print("[检测] 球: 未找到标定外参（table_extrinsics.yaml），仅 2D 显示")
                else:
                    self._triangulator = MultiViewTriangulator(intrinsics, extrinsics)
                    self._table_poses = {cid: (e.R, e.t) for cid, e in extrinsics.items()}
                    if self._table_detector is None:
                        self._table_detector = create_detector("table")
                    print(f"[检测] 球: 三角化器已加载 {len(self._triangulator.cameras)} 台相机")

            self._ball_ready = True
            self._ball_pending_viewer = True  # 主循环检测到后从主线程开 3D 窗口
            ep = getattr(det, "actual_provider", type(det).__name__)
            print(f"[检测] 球: ON（{type(det).__name__} @ {ep} + DLT + Open3D）")
        except Exception as exc:  # noqa: BLE001
            self._ball_ready = False
            print(f"[检测] 球: 加载失败（{exc}）——按 B 关闭后再按 B 重试")

    def _disable_ball_recon(self) -> None:
        self._last_balls = {}
        self._ball3d = None
        self._ball_tracker = None
        self._ball_ready = False
        self._ball_pending_viewer = False
        if self.viewer3d is not None:
            self.viewer3d.set_ball(None)
        print("[检测] 球: OFF")

    def _reconstruct_ball_frame(self) -> None:
        """各相机球检测（每帧一次）→ 置信度加权 DLT 三角化 → 卡尔曼平滑 → Open3D 球层。"""
        detector = self.detectors["ball"]
        frames_items = [
            (cid, f) for cid, f in sorted(self._latest.items()) if f is not None
        ]
        if not frames_items:
            return

        # 球重建实际帧率（EMA 平滑，显示在画面右上角）
        _now = time.time()
        if self._ball_fps_t is not None:
            _dt = _now - self._ball_fps_t
            if _dt > 0:
                _inst = 1.0 / _dt
                self._ball_fps = _inst if self._ball_fps is None else 0.9 * self._ball_fps + 0.1 * _inst
        self._ball_fps_t = _now

        # 4 相机 batch 一次推理（灰度模型 1 通道更轻，batch 省 3 次 session.run 固定开销；
        # 模型 batch 非动态 / 经典检测器无批处理接口时自动回退逐帧）。
        frames = [f for _, f in frames_items]
        try:
            balls_list = detector.detect_batch(frames)
        except Exception as exc:  # noqa: BLE001 —— 批处理失败回退逐帧
            balls_list = []
            for f in frames:
                try:
                    balls_list.append(detector.detect(f))
                except Exception as exc2:  # noqa: BLE001 —— 单路失败不影响其余相机
                    balls_list.append([])
        balls_per_cam = {cid: balls_list[i] for i, (cid, _f) in enumerate(frames_items)}
        self._last_balls = balls_per_cam

        if self._triangulator is None:
            return
        from tabletennis.reconstruction import triangulate_ball

        single = {cid: balls[0] for cid, balls in balls_per_cam.items() if balls}
        # min_conf 降到 0.15：检测器 conf_thresh=0.25，三角化若用默认 0.3 会把
        # conf 0.25~0.3 的球（2D 已显示）滤掉 → 3D 不渲染。降到低于检测器阈值，
        # 由几何校验（重投影误差/交会角）兜底。
        res = triangulate_ball(single, self._triangulator, min_conf=0.15) if len(single) >= 2 else None
        if res is not None:
            X, conf, err, n_views, ang = res
            if self._ball_tracker is None:
                from tabletennis.reconstruction import BallTracker
                self._ball_tracker = BallTracker()
            X = self._ball_tracker.update(X, float(conf))
            self._ball3d = X
            if self.viewer3d is not None:
                self.viewer3d.set_ball(X)
        else:
            if self._ball_tracker is not None:
                self._ball_tracker.update(None, 0.0)   # 本帧无观测，内部 coast
            self._ball3d = None
            if self.viewer3d is not None:
                self.viewer3d.set_ball(None)
            # 诊断：检测到球但三角化失败（限频 1s，定位 3D 不显示原因）
            if single and (self._ball_diag_t is None or time.time() - self._ball_diag_t >= 1.0):
                self._ball_diag_t = time.time()
                confs = {cid: f"{b[0].confidence:.2f}" for cid, b in balls_per_cam.items() if b}
                print(f"[球诊断] {len(single)} 视角检出 conf={confs}，三角化失败（视角不足/交会角过小）")

    # ------------------------------------------------------------------
    # 鼠标：拖拽滑块
    # ------------------------------------------------------------------
    def on_mouse(self, event, x, y, flags, param) -> None:
        # 窗口用 WINDOW_AUTOSIZE，画布与窗口 1:1，鼠标坐标即图像坐标，无需再映射
        if event != cv2.EVENT_LBUTTONDOWN and not (
            event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON
        ):
            return
        for key, (x0, x1, y0, y1) in self._slider_rects.items():
            if x0 <= x <= x1 and y0 <= y <= y1:
                spec = PARAM_BY_KEY[key]
                frac = (x - x0) / (x1 - x0)
                frac = max(0.0, min(1.0, frac))
                self.values[key] = spec["min"] + frac * (spec["max"] - spec["min"])
                self._apply_param(key, self.values[key])
                return

        # 保存按钮
        bx0, bx1, by0, by1 = self._save_rect
        if bx0 <= x <= bx1 and by0 <= y <= by1:
            self.save_settings()

    # ------------------------------------------------------------------
    # 画面合成
    # ------------------------------------------------------------------
    def _annotate(self, cid: int, frame: Optional[Frame]) -> np.ndarray:
        if frame is None:
            h, w = self._ref_size
            gray = np.zeros((h, w), np.uint8)
        else:
            gray = frame.image
            self._ref_size = gray.shape[:2]

        bgr = gray_to_bgr(gray)

        if frame is not None:
            if self.enable["pose"]:
                # 姿态由 _reconstruct_frame 批处理检测，这里只画缓存结果
                for pose in self._last_poses.get(cid, []):
                    draw_pose(bgr, pose)
            if self.enable["ball"]:
                # 球检测由 _reconstruct_ball_frame 每帧统一做，这里只画缓存结果
                for ball in self._last_balls.get(cid, []):
                    draw_ball(bgr, ball)
            if self.enable["table"]:
                self._annotate_table(bgr, cid)

        cv2.putText(bgr, f"cam{cid}", (8, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 255, 255), 2, cv2.LINE_AA)
        return bgr

    def _compose_grid(self) -> np.ndarray:
        images = [
            self._annotate(cam.logical_id, self._latest.get(cam.logical_id))
            for cam in self.mgr.cameras
        ]
        grid = tile_images(images, cols=GRID_COLS)
        h, w = grid.shape[:2]
        if w > self.max_width:
            s = self.max_width / w
            grid = cv2.resize(grid, (int(w * s), int(h * s)))

        # 录制状态指示（左上角红点 + 计数 / 倒计时）
        if self._recording:
            cv2.circle(grid, (16, 16), 9, (0, 0, 255), -1)
            cv2.putText(grid, f"REC {self._record_count}", (32, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 255), 2, cv2.LINE_AA)
        elif self._record_countdown_until > 0:
            n = max(1, int(self._record_countdown_until - time.time()) + 1)
            cv2.circle(grid, (16, 16), 9, (0, 165, 255), -1)
            cv2.putText(grid, f"{n}", (32, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 165, 255), 2, cv2.LINE_AA)

        # 球重建实际帧率（右上角，按 B 开启后显示）
        if self.enable["ball"] and self._ball_fps is not None:
            txt = f"BALL {self._ball_fps:4.0f} FPS"
            gw = grid.shape[1]
            (tw, _th), _baseline = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cv2.putText(grid, txt, (gw - tw - 12, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 0), 2, cv2.LINE_AA)

        # 触发信号报错叠加在画面上
        if self.trigger_error:
            cv2.putText(grid, "NO TRIGGER SIGNAL", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.putText(grid, "check signal generator / Line0", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

        # 3D 姿态重建实时帧率（右上角，按 P 开启后显示）
        if self.enable["pose"] and self._pose_fps > 0:
            fps_txt = f"3D {self._pose_fps:4.1f} FPS"
            (tw, _th), _ = cv2.getTextSize(fps_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            # 注意用 resize 后的实际宽度：上面的 w 是缩放前的（网格 2880→max_width 1920），
            # 沿用旧 w 会把文字画到画布右缘之外而被裁剪
            gw = grid.shape[1]
            cv2.putText(grid, fps_txt, (gw - tw - 12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2, cv2.LINE_AA)
        return grid

    def _draw_panel(self, w: int, y_offset: int = 0) -> np.ndarray:
        """控制面板：三条等宽滑块，带名称与当前值标注。

        Args:
            w: 面板宽度。
            y_offset: 面板在整张画布中的 y 偏移（= 视频网格高度），用于把滑块命中区域
                记录成画布坐标，供鼠标回调命中检测。
        """
        panel = np.full((PANEL_H, w, 3), 38, np.uint8)
        cv2.putText(panel, "Control", (12, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (200, 200, 200), 1, cv2.LINE_AA)

        x0 = 160                 # 滑块起点（给名称留空间）
        x1 = w - 210             # 滑块终点（给数值留空间），三条等宽
        self._slider_rects = {}

        for i, spec in enumerate(PARAM_SPECS):
            cy = 64 + i * 48
            # 名称
            cv2.putText(panel, spec["label"], (18, cy + 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.62, (255, 255, 255), 1, cv2.LINE_AA)
            # 轨道（灰底 + 亮色填充）
            cv2.line(panel, (x0, cy), (x1, cy), (95, 95, 95), 5, cv2.LINE_AA)
            val = self.values[spec["key"]]
            frac = (val - spec["min"]) / (spec["max"] - spec["min"])
            frac = max(0.0, min(1.0, frac))
            fx = int(x0 + frac * (x1 - x0))
            cv2.line(panel, (x0, cy), (fx, cy), (0, 200, 255), 5, cv2.LINE_AA)
            cv2.circle(panel, (fx, cy), 9, (0, 255, 255), -1, cv2.LINE_AA)
            # 当前值
            txt = spec["fmt"].format(val) + spec["unit"]
            cv2.putText(panel, txt, (x1 + 12, cy + 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.62, (0, 255, 255), 1, cv2.LINE_AA)
            # 记录命中区域供鼠标回调（画布坐标 = 面板坐标 + y 偏移）
            self._slider_rects[spec["key"]] = (x0, x1, y_offset + cy - 9, y_offset + cy + 9)

        # 保存按钮（面板右上角）
        bx0, bx1 = w - 150, w - 20
        by0, by1 = 6, 38
        cv2.rectangle(panel, (bx0, by0), (bx1, by1), (60, 120, 60), -1, cv2.LINE_AA)
        cv2.rectangle(panel, (bx0, by0), (bx1, by1), (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(panel, "保存", (bx0 + 38, by0 + 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1, cv2.LINE_AA)
        self._save_rect = (bx0, bx1, y_offset + by0, y_offset + by1)
        return panel

    def _draw_hint(self, w: int) -> np.ndarray:
        """底部提示条：检测开关状态 + 操作说明。"""
        hint = np.full((HINT_H, w, 3), 28, np.uint8)
        toggles = "  ".join(
            f"{k.upper()} {'ON' if self.enable[k] else 'off'}" for k in DETECTION_TOGGLES
        )
        cv2.putText(hint, f"检测: {toggles}", (12, 18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 220, 255), 1, cv2.LINE_AA)
        if self._recording:
            cv2.putText(hint, f"● REC {self._record_count} 帧", (w - 200, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1, cv2.LINE_AA)
        elif self._record_countdown_until > 0:
            n = max(1, int(self._record_countdown_until - time.time()) + 1)
            cv2.putText(hint, f"录制倒计时 {n}s", (w - 200, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1, cv2.LINE_AA)
        elif time.time() < self._save_flash_until:
            cv2.putText(hint, "已保存 ✓", (w - 110, 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(hint, "拖滑块调参  [p]姿态 [b]球 [t]球桌+3D [r]录制 [s]保存  退出:[q]/ESC/X",
                    (12, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        return hint

    def _compose_canvas(self) -> np.ndarray:
        grid = self._compose_grid()
        gh, gw = grid.shape[:2]
        panel = self._draw_panel(gw, y_offset=gh)
        hint = self._draw_hint(gw)
        canvas = np.vstack([grid, panel, hint])
        self._canvas_shape = canvas.shape[:2]
        return canvas

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------
    def run(self) -> None:
        self.mgr.start()
        self.apply_all_params()
        if self.trigger_mode == "external":
            self.wait_for_trigger()

        cv2.namedWindow(MAIN_WIN, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(MAIN_WIN, self.on_mouse)
        cv2.imshow(MAIN_WIN, self._compose_canvas())  # 首次显示，确保窗口已存在

        try:
            while True:
                # 先判断窗口是否被右上角 X 关闭：imshow 会重建已关闭的窗口，
                # 所以必须在 imshow 之前检查，否则会出现「消失一下又回来」。
                if self._window_closed():
                    break

                # 取帧：外触发下按触发锁帧（清积压 + 阻塞等下一触发周期的同步帧），
                # 稳定在触发频率、不重复处理同一帧；连续/软触发下自由运行。
                if self.trigger_mode == "external":
                    bundle = self.mgr.get_synchronized_bundle(block=True, timeout=1.0)
                    latest = bundle.frames
                else:
                    latest = self.mgr.get_latest_frames(block=False)
                for cid, f in latest.items():
                    if f is not None:
                        self._latest[cid] = f
                        if self.trigger_error:
                            self.trigger_error = False
                            print("✓ 触发信号已恢复，开始出图。")

                self._tick_record(latest)

                # 3D 姿态重建（按 P 开启后每帧批处理检测 + 三角化 + Open3D 骨架）
                if self.enable["pose"] and self.detectors["pose"] is not None:
                    self._reconstruct_frame()

                # 球模型后台加载完成：主线程开 3D 窗口（Open3D 需保持主线程创建/轮询）
                if self._ball_pending_viewer:
                    if (self.viewer3d is not None and self.viewer3d.is_running()
                            and not self.viewer3d.has_ball_layer()):
                        self._close_viewer()   # 先按 T/P 开的窗口没球层，重建带上
                    self._start_viewer()
                    self._ball_pending_viewer = False

                # 3D 球重建（按 B 开启后每帧检测 + 三角化 + Open3D 球层）
                if self.enable["ball"] and self._ball_ready:
                    self._reconstruct_ball_frame()

                canvas = self._compose_canvas()
                cv2.imshow(MAIN_WIN, canvas)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):  # ESC / q
                    break
                self.handle_key(key)
        finally:
            self._close_viewer()
            cv2.destroyAllWindows()

    @staticmethod
    def _window_closed() -> bool:
        """窗口是否已被关闭（右上角 X / 外部销毁）。"""
        try:
            return cv2.getWindowProperty(MAIN_WIN, cv2.WND_PROP_VISIBLE) < 1
        except cv2.error:
            return True


def main() -> None:
    ap = argparse.ArgumentParser(description="交互式相机控制 + 四机实时预览")
    ap.add_argument("--config", default=None, help="cameras.yaml 路径（默认 config/cameras.yaml）")
    ap.add_argument("--trigger", choices=["external", "software", "continuous"], default=None)
    ap.add_argument("--exposure", type=float, default=None, help="初始曝光时间(us)")
    ap.add_argument("--gain", type=float, default=None, help="初始增益(dB)")
    ap.add_argument("--gamma", type=float, default=None, help="初始伽马")
    ap.add_argument("--max-width", type=int, default=DEFAULT_MAX_WIDTH, help="视频网格目标宽度（越大窗口越大）")
    args = ap.parse_args()

    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    cfg_path = args.config or os.path.join(root, "config", "cameras.yaml")
    config = load_yaml(cfg_path) if os.path.exists(cfg_path) else {}

    kwargs = camera_kwargs_from_config(config)
    if args.trigger:
        kwargs["trigger_mode"] = args.trigger

    # 曝光 / 增益 / 伽马：命令行 > 保存的设置文件 > 默认值
    cs = resolve_camera_settings(args.exposure, args.gain, args.gamma)
    kwargs["exposure_us"] = cs["exposure_us"]
    kwargs["gain_db"] = cs["gain_db"]

    trigger_mode = kwargs["trigger_mode"]
    exposure_us = cs["exposure_us"]
    gain_db = cs["gain_db"]
    gamma = cs["gamma"]

    serials = [c.get("serial") for c in config.get("cameras", []) if c.get("serial")] or None

    try:
        with CameraManager(serials=serials, **kwargs) as mgr:
            print(f"已连接 {len(mgr.cameras)} 台相机（触发模式 {trigger_mode}）")
            ui = LiveControl(
                mgr,
                exposure_us=exposure_us,
                gain_db=gain_db,
                gamma=gamma,
                trigger_mode=trigger_mode,
                max_width=args.max_width,
            )
            ui.run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
