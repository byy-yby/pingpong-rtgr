#!/usr/bin/env python3
"""多相机外参标定 · ChArUco 板 · 采集 + 求解 GUI。

外参标定与桌面定位**完全分开**：

- **外参（相机间相对位姿）**：按 Enter 在**不同位置**拍 ChArUco 板（每拍一次可
  移动板），按 C 求「参考相机系 -> 各相机」的相对外参，保存到
  ``data/extrinsics/extrinsics.yaml``，并用 Open3D 画出四台相机的相对位置帮助核对。
- **桌面定位（世界系建立）**：点「桌面定原点 (T)」识别球桌对角线两角的两个大
  ArUco 标记，**跨相机融合**（无需任何单台相机同时看到两个标记——用相对外参把
  各相机看到的标记统一到参考相机系），建立桌面世界系（X 短边 / Y 长边 / Z 向上），
  保存 ``data/extrinsics/table_extrinsics.yaml``，Open3D 显示「球桌 + 相机」，并在
  四路画面按标准尺寸持续画桌面边框 + 球网。

用法：
  conda activate tt
  python scripts/calibrate_extrinsics.py                 # 外部触发（默认）
  python scripts/calibrate_extrinsics.py --trigger continuous   # 无信号发生器时自由采集
  python scripts/calibrate_extrinsics.py --reference-camera 0    # 参考相机（默认 0）
  python scripts/calibrate_extrinsics.py --exposure 5000 --gain 0
  python scripts/calibrate_extrinsics.py --generate-board board.png   # 只生成打印板
  python scripts/calibrate_extrinsics.py --generate-markers markers/  # 只生成四个大标记

交互：
  - Enter：清空各机队列后各取「下一帧」（外触发下四机同一时钟周期），拍下当前
    位置的标定板并解「板 -> 相机」位姿，累积一个位姿（板可移动）。
  - C 或「计算并保存外参」：由多组板位姿求相机间相对外参（世界系=参考相机），
    保存 extrinsics.yaml 并弹出 Open3D 相机相对位置窗口。
  - T 或「桌面定原点」：识别两个大标记建立桌面系，保存 table_extrinsics.yaml，
    弹出 Open3D 球桌+相机窗口，并在四路画面持续画桌面边框 + 球网。
  - Q / ESC：退出。
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QFrame, QGridLayout, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
    QPushButton, QTextEdit, QVBoxLayout, QWidget,
)

import cv2
import numpy as np

from tabletennis.calibration.extrinsics import (
    CharucoConfig, compute_relative_extrinsics, create_board, detect_charuco,
    draw_charuco_overlay, estimate_board_pose, generate_board_image,
    generate_marker_image, load_extrinsics, localize_table_bundle,
    reprojection_error, resolve_dictionary, save_extrinsics,
)
from tabletennis.calibration.intrinsics import load_intrinsics
from tabletennis.camera import CameraManager
from tabletennis.core.config import load_yaml, project_root, resolve_camera_settings
from tabletennis.core.types import CameraExtrinsics, Table3D
from tabletennis.visualization.overlay2d import draw_table_model, gray_to_bgr

def _to_qimage(img: np.ndarray) -> QImage:
    """numpy 灰度(H,W) 或 BGR(H,W,3) -> QImage。"""
    if img.ndim == 2:
        h, w = img.shape
        return QImage(img.data, w, h, w, QImage.Format_Grayscale8).copy()
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()


class CameraView(QFrame):
    """单个相机画面面板：显示实时/标注画面 + 采集状态。"""

    def __init__(self, cam_id: int):
        super().__init__()
        self.cam_id = cam_id
        self.setObjectName(f"camview{cam_id}")
        self.setStyleSheet(
            f"#{self.objectName()} {{ border: 2px solid #555; background: #1e1e1e; }}"
        )

        self._title = QLabel(f"相机 {cam_id}")
        self._title.setStyleSheet("color:#eee; font-weight:bold; padding:2px;")

        self._img = QLabel("无信号")
        self._img.setAlignment(Qt.AlignCenter)
        self._img.setMinimumSize(320, 240)
        self._img.setStyleSheet("color:#888;")

        self._status = QLabel("—")
        self._status.setStyleSheet("color:#8f8; padding:2px;")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.addWidget(self._title)
        lay.addWidget(self._img, 1)
        lay.addWidget(self._status)

    def set_frame(self, img: np.ndarray) -> None:
        if img is None:
            return
        # 用完整分辨率（不降采样），再缩放到面板当前尺寸，保持完整视野不裁剪
        pix = QPixmap.fromImage(_to_qimage(img))
        pix = pix.scaled(self._img.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._img.setPixmap(pix)

    def set_status(self, text: str, color: str = "#8f8") -> None:
        self._status.setText(text)
        self._status.setStyleSheet(f"color:{color}; padding:2px;")


class MainWindow(QMainWindow):
    def __init__(self, mgr: CameraManager, cfg: CharucoConfig, board, detector,
                 intrinsics: dict, out_dir: str, min_snapshots: int, trigger_mode: str,
                 table_cfg: dict, reference_cam: int = 0):
        super().__init__()
        self.setWindowTitle("多相机外参标定 · ChArUco")
        self.mgr = mgr
        self.cfg = cfg
        self.board = board
        self.detector = detector
        self.intrinsics = intrinsics
        self.out_dir = out_dir
        self.min_snapshots = min_snapshots
        self.trigger_mode = trigger_mode
        self.reference_cam = reference_cam
        self.n_cam = len(mgr.cameras)

        # 桌面坐标系标定（两个大 ArUco 标记）
        self.table_cfg = table_cfg or {}
        self.marker_length_m = float(self.table_cfg.get("marker_length_m", 0.18))
        self.white_border_m = float(self.table_cfg.get("white_border_m", 0.0))
        # 四个大标记的 ID 与球桌四角的对应关系（ID0原点角/ID1短边角/ID2对角/ID3长边角）
        self.marker_ids = {
            "origin": int(self.table_cfg.get("marker_origin_id", 0)),
            "short": int(self.table_cfg.get("marker_short_id", 1)),
            "diagonal": int(self.table_cfg.get("marker_diagonal_id", 2)),
            "long": int(self.table_cfg.get("marker_long_id", 3)),
        }
        self.origin_offset = self.table_cfg.get("origin_offset_m", [0.0, 0.0, 0.0])
        # 几何校验：ID0↔ID2 应相距约桌面对角线，拒绝误检
        self._table_diag_m = float(self.table_cfg.get(
            "expected_marker_distance_m", np.sqrt(2.74 ** 2 + 1.525 ** 2)))
        self._marker_dist_tol = float(self.table_cfg.get("marker_distance_tol_m", 0.5))
        self.marker_detector = cv2.aruco.ArucoDetector(
            resolve_dictionary(self.table_cfg.get("marker_dict", "DICT_5X5_50"))
        )

        # 标准尺寸球桌（桌面定位后画边框 + 3D 场景用）
        self.table3d = Table3D()

        self.views = [CameraView(i) for i in range(self.n_cam)]
        grid = QGridLayout()
        for i, v in enumerate(self.views):
            grid.addWidget(v, i // 2, i % 2)

        # 外参采集：每个快照存一个 {cid: (R, t)}（板可移动）；errs 记重投影误差
        self.snapshots = []
        self.errs = {i: [] for i in range(self.n_cam)}
        self.n_shots = 0
        self._last_offsets = None  # 上次同步核对时的帧号偏移，用于判断偏移是否稳定

        # 桌面定位结果（桌面系外参）+ Open3D 查看器 + 相对外参（跨相机融合用）
        self._table_poses = {}
        self._marker_vis = {}   # {cid: {"corners","ids","origins"}}，用于持续画标记边缘/原点
        self.viewer3d = None
        self._relative_extrinsics = None  # {cam_id: (R, t)}，参考相机系 -> cam_id

        self.calc_btn = QPushButton("计算并保存外参 (C)")
        self.table_btn = QPushButton("桌面定原点 (T)")
        self.clear_btn = QPushButton("清空重来")
        self.exit_btn = QPushButton("退出 (Q)")
        self.calc_btn.clicked.connect(self._on_calc)
        self.table_btn.clicked.connect(self._register_table)
        self.clear_btn.clicked.connect(self._on_clear)
        self.exit_btn.clicked.connect(self.close)
        btns = QHBoxLayout()
        for b in (self.calc_btn, self.table_btn, self.clear_btn, self.exit_btn):
            b.setFocusPolicy(Qt.NoFocus)  # 不让按钮抢键盘焦点，避免 Enter 误触
            btns.addWidget(b)
        btns.addStretch(1)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setFocusPolicy(Qt.NoFocus)
        self.log.setMaximumHeight(130)

        central = QWidget()
        root = QVBoxLayout(central)
        root.addLayout(grid)
        root.addLayout(btns)
        root.addWidget(self.log)
        self.setCentralWidget(central)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self._refresh)
        self.timer.start(33)

        self._refresh_status()
        mode_txt = {"external": "外部触发（信号发生器时钟）", "software": "软件触发",
                    "continuous": "连续采集（无时钟对齐）"}.get(self.trigger_mode, self.trigger_mode)
        self._log(f"就绪。触发模式：{mode_txt}。参考相机：相机{self.reference_cam}。")
        self._log("Enter 拍不同位置的 ChArUco 板；C 求相对外参+3D相机；T 桌面定原点+3D球桌；Q 退出。")

    # ---- 交互 ----
    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self._capture()
        elif event.key() == Qt.Key_C:
            self._on_calc()
        elif event.key() == Qt.Key_T:
            self._register_table()
        elif event.key() in (Qt.Key_Q, Qt.Key_Escape):
            self.close()
        else:
            super().keyPressEvent(event)

    def _capture(self) -> None:
        """四机同一时钟周期拍一张：存盘 + ChArUco 检测 + solvePnP，累积一个板位姿快照。"""
        self.n_shots += 1
        shot = self.n_shots
        # 先清空各机队列、再各取下一帧：外触发下四机来自同一次触发边沿（同一时钟周期）
        bundle = self.mgr.get_synchronized_bundle(block=True, timeout=1.0)
        if not bundle.frames:
            self._log(f"第 {shot} 张：未收到任何帧（外部触发下请检查信号发生器 / Line0 接线）。")
            return
        self._report_sync(bundle.frames)

        snap = {}
        for cid in range(self.n_cam):
            frame = bundle.frames.get(cid)
            if frame is None:
                self.views[cid].set_status("无画面", "#e33")
                continue
            gray = frame.image
            idx = self._save_raw(cid, gray)

            corners, ids = detect_charuco(gray, self.detector, self.cfg.min_corners)
            if corners is None:
                self.views[cid].set_status(f"第{shot}张：未检测到标定板", "#e88")
                continue
            if cid not in self.intrinsics:
                self.views[cid].set_status(f"第{shot}张：缺内参，跳过", "#e88")
                continue

            res = estimate_board_pose(corners, ids, self.board, self.intrinsics[cid])
            if res is None:
                self.views[cid].set_status(f"第{shot}张：位姿求解失败", "#e88")
                continue
            rvec, tvec, R, t, n = res
            err = reprojection_error(corners, ids, self.board, self.intrinsics[cid], rvec, tvec)

            snap[cid] = (R, t)
            self.errs[cid].append(err)

            overlay = draw_charuco_overlay(gray, corners, ids, self.intrinsics[cid], rvec, tvec)
            self._save_annot(cid, overlay, idx)
            self.views[cid].set_frame(overlay)
            self.views[cid].set_status(f"第{shot}张：{n} 角点  重投影 {err:.2f}px", "#8f8")

        if snap:
            self.snapshots.append(snap)
        self._log(f"第 {shot} 张采集完成（{len(snap)} 台看到板，已累计 {len(self.snapshots)} 个位姿）。")

    def _report_sync(self, frames) -> None:
        """核对四机是否同一时钟周期：打印各机帧号相对参考相机的偏移。

        外触发下四机被同一信号发生器时钟驱动、frame_num 逐拍递增。若已同步，
        各机相对参考相机的帧号偏移是**固定常数**（启动先后差了几拍），每次采集
        都一样；偏移若在变，说明有丢帧 / 未真正同步。
        device_timestamp 是各机内部时钟、零点不同，绝对差值无同步意义，故不打印。
        """
        ref = min(frames)  # cam_id 最小的相机作参考
        ref_fn = frames[ref].frame_num
        offsets = {cid: frames[cid].frame_num - ref_fn for cid in sorted(frames)}
        off_txt = "  ".join(f"cam{cid}:{d:+d}" for cid, d in offsets.items())
        self._log(f"同步核对（帧号相对 cam{ref} 偏移）：{off_txt}")

        if self._last_offsets is not None and self._last_offsets != offsets:
            self._log("[警告] 帧号偏移与上次不同，可能丢帧或未严格同步，建议重拍。")
        else:
            self._log("  偏移固定 = 四机已锁定同一时钟（仅启动先后相差几拍），同步正常。")
        self._last_offsets = offsets

    def _on_calc(self) -> None:
        """由多组板位姿求相机间相对外参（世界系=参考相机），保存并弹 3D 相机窗口。"""
        if not self.snapshots:
            self._log("没有采集到任何板位姿（先按 Enter 在不同位置拍几张板）。")
            return

        results, counts = compute_relative_extrinsics(self.snapshots, self.reference_cam)
        if len(results) < 2:
            self._log("有效数据不足：至少需要参考相机和另一台相机同时看到板。")
            return
        self._relative_extrinsics = results  # 供桌面定位跨相机融合使用

        meta = {
            f"cam_{cid}": {
                "num_snapshots": counts.get(cid, 0),
                "mean_reproj_error_px": round(float(np.mean(self.errs[cid])), 3) if self.errs[cid] else None,
            }
            for cid in sorted(results)
        }
        path = os.path.join(self.out_dir, "extrinsics.yaml")
        save_extrinsics(path, results, world_frame=f"camera_{self.reference_cam}", extra=meta)
        self._log(f"相对外参已保存 -> {path}（世界系 = 相机{self.reference_cam}）")
        for cid in sorted(results):
            R, t = results[cid]
            me = meta[f"cam_{cid}"]
            self._log(
                f"[相机{cid}] {me['num_snapshots']} 帧 | 重投影 {me['mean_reproj_error_px']}px | "
                f"t=({t[0,0]:.3f},{t[1,0]:.3f},{t[2,0]:.3f})m"
            )
        self._open_cameras_3d(results)

    # ---- Open3D 3D 场景 ----
    def _close_viewer(self) -> None:
        if self.viewer3d is not None:
            self.viewer3d.close()
            self.viewer3d = None

    def _open_cameras_3d(self, results) -> None:
        """弹出只含四台相机相对位置的 3D 窗口（世界系=参考相机）。"""
        from tabletennis.visualization.viewer3d import SceneViewer3D

        self._close_viewer()
        camera_poses = {cid: CameraExtrinsics(R=R, t=t) for cid, (R, t) in results.items()}
        intrinsics = {cid: self.intrinsics[cid] for cid in results if cid in self.intrinsics}
        self.viewer3d = SceneViewer3D("相机相对位置（外参调试）")
        self.viewer3d.build_cameras_scene(camera_poses, intrinsics)
        self.viewer3d.start()
        self._log("已打开 3D 场景（相机相对位置，可鼠标旋转 / 缩放）。")

    def _open_table_3d(self) -> None:
        """弹出「球桌 + 相机」的 3D 窗口（世界系=桌面）。"""
        from tabletennis.visualization.viewer3d import SceneViewer3D

        self._close_viewer()
        camera_poses = {
            cid: CameraExtrinsics(R=R, t=t) for cid, (R, t) in self._table_poses.items()
        }
        intrinsics = {
            cid: self.intrinsics[cid] for cid in self._table_poses if cid in self.intrinsics
        }
        self.viewer3d = SceneViewer3D("球桌 + 相机（桌面世界系）")
        self.viewer3d.build_scene(self.table3d, camera_poses, intrinsics)
        self.viewer3d.start()
        self._log("已打开 3D 场景（球桌 + 相机，可鼠标旋转 / 缩放）。")

    def _draw_marker_origin(self, bgr, pose, marker_id, intrinsics) -> None:
        """把标记的原点角（白边角，已含 white_border 偏移）投影到画面，画成红点 + ID 标注。"""
        R, t = pose  # t = 白边角在相机系 (3,1)
        K = intrinsics
        # 相机系点直接投影（rvec/tvec=0），并应用畸变
        uv, _ = cv2.projectPoints(t.reshape(1, 3), np.zeros(3), np.zeros(3), K.K, K.dist)
        u, v = uv[0][0]
        cv2.circle(bgr, (int(u), int(v)), 9, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(bgr, (int(u), int(v)), 9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(bgr, f"ID{marker_id}", (int(u) + 12, int(v) - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    def _load_relative_extrinsics(self):
        """取「参考相机系 -> 各相机」的相对外参：优先本次会话已算，否则读 extrinsics.yaml。"""
        if self._relative_extrinsics is not None:
            return self._relative_extrinsics
        path = os.path.join(self.out_dir, "extrinsics.yaml")
        if os.path.exists(path):
            loaded = load_extrinsics(path)  # {cid: CameraExtrinsics}
            self._relative_extrinsics = {cid: (ext.R, ext.t) for cid, ext in loaded.items()}
        return self._relative_extrinsics

    def _register_table(self) -> None:
        """桌面定位：识别四角大标记，跨相机三角化后由四角点位置构造桌面系。

        四个标记分居球桌四角；每台相机看到任意几个标记都行，融合到参考相机系后
        用 origin/short/long 三个角点位置构造桌面系（diagonal 角用于校验）。
        """
        bundle = self.mgr.get_synchronized_bundle(block=True, timeout=1.0)
        if not bundle.frames:
            self._log("桌面定原点：未收到任何帧（检查信号发生器 / Line0）。")
            return
        self._report_sync(bundle.frames)

        rel = self._load_relative_extrinsics()
        if not rel:
            self._log("桌面定原点：缺少相对外参（先按 C 做相机外参标定，得到参考相机系外参）。")
            return

        gray_frames = {cid: f.image for cid, f in bundle.frames.items()}
        table_extrinsics, info = localize_table_bundle(
            gray_frames, self.intrinsics, rel, self.marker_detector,
            self.marker_ids, self.marker_length_m, self.white_border_m,
            expected_diag_m=self._table_diag_m, diag_tol_m=self._marker_dist_tol,
            reference_cam=self.reference_cam,
        )

        # 回填每台相机状态 + 标记叠加（供 _annotate_live 持续画）
        self._marker_vis = {}
        for cid in range(self.n_cam):
            pc = info["per_cam"].get(cid)
            if cid not in bundle.frames or cid not in self.intrinsics:
                self.views[cid].set_status("无画面/缺内参", "#e33")
            elif pc is None or pc["ids"] is None:
                self.views[cid].set_status("未检测到大标记", "#e88")
            else:
                self._marker_vis[cid] = {
                    "corners": pc["corners"], "ids": pc["ids"], "origins": pc["origins"],
                }
                got = pc["got"]
                self.views[cid].set_status(f"看到标记 {sorted(got)}", "#8f8" if got else "#e88")

        marker_obs = info["marker_obs"]
        fused = info["fused"]
        if table_extrinsics is None:
            if not marker_obs:
                self._log("桌面定原点失败：没有相机看到任何大标记。")
            else:
                seen = {m: sorted(obs) for m, obs in marker_obs.items()}
                self._log(f"桌面定原点失败：融合后缺必需标记 {info['missing']}"
                          f"（各标记被相机看到：{seen}）。")
            return

        # 校验：ID0↔ID2 距离应≈桌面对角线
        diag_id = self.marker_ids["diagonal"]
        dist = info["diag_m"]
        if dist is not None:
            if info.get("diag_warn"):
                self._log(f"对角校验偏大：ID0↔ID{diag_id} 距离 {dist:.2f}m ≠ 对角线 "
                          f"{self._table_diag_m:.2f}m，请核对标记摆放。")
            else:
                self._log(f"对角校验通过：ID0↔ID{diag_id} 距离 {dist:.3f}m"
                          f"（≈对角线 {self._table_diag_m:.3f}m）。")

        # 桌面尺寸用标准值（四角点位置已确定尺度）
        self.table3d = Table3D()

        results = table_extrinsics
        meta = {
            f"cam_{cid}": {"marker_distance_m": round(dist, 3) if dist is not None else None}
            for cid in sorted(results)
        }

        path = os.path.join(self.out_dir, "table_extrinsics.yaml")
        save_extrinsics(path, results, world_frame="table", extra=meta)
        self._log(f"桌面坐标系已保存 -> {path}")
        seen = {m: sorted(obs) for m, obs in marker_obs.items()}
        seen_summary = ", ".join(
            f"ID{v}←相机{seen.get(v, [])}" for v in self.marker_ids.values() if v in fused
        )
        self._log(f"融合建立：{seen_summary}")
        for cid in sorted(results):
            R, t = results[cid]
            self._log(f"[相机{cid}] 桌面系 t=({t[0,0]:.3f},{t[1,0]:.3f},{t[2,0]:.3f})m")
        self._log("提示：世界系=桌面（原点=ID%d 原点角，X 短边 / Y 长边 / Z 向上）。" % self.marker_ids["origin"])

        # 缓存桌面系外参：四路画面持续画桌面边框 + 球网，并弹 3D 场景
        self._table_poses = results
        self._open_table_3d()

    def _on_clear(self) -> None:
        ret = QMessageBox.question(self, "确认", "清空已采集的板位姿数据（不动磁盘照片）?")
        if ret != QMessageBox.Yes:
            return
        self.snapshots = []
        self.errs = {i: [] for i in range(self.n_cam)}
        self.n_shots = 0
        self._refresh_status()
        self._log("已清空累积数据。")

    # ---- 刷新 / 存盘 ----
    def _refresh(self) -> None:
        frames = self.mgr.get_latest_frames(block=False)
        for cid, f in frames.items():
            if f is not None:
                self.views[cid].set_frame(self._annotate_live(cid, f.image))

    def _annotate_live(self, cid: int, gray: np.ndarray) -> np.ndarray:
        """持续叠加：标记边缘 + 原点（若有）+ 桌面边框/球网（若已定桌面）。"""
        bgr = gray_to_bgr(gray)

        vis = self._marker_vis.get(cid)
        if vis is not None and cid in self.intrinsics:
            # 标记边缘用品红色，与球桌边框(绿)/球网(青)区分
            cv2.aruco.drawDetectedMarkers(bgr, vis["corners"], vis["ids"], (255, 0, 255))
            for i in range(len(np.asarray(vis["ids"]).ravel())):
                for pt in np.asarray(vis["corners"][i]).reshape(-1, 2):
                    cv2.circle(bgr, tuple(pt.astype(int)), 4, (255, 0, 255), -1, cv2.LINE_AA)
            for mid, pose in vis["origins"].items():
                self._draw_marker_origin(bgr, pose, mid, self.intrinsics[cid])

        if cid in self._table_poses and cid in self.intrinsics:
            R, t = self._table_poses[cid]
            K = self.intrinsics[cid]
            draw_table_model(bgr, self.table3d, R, t, K.K, K.dist)
        return bgr

    def _refresh_status(self) -> None:
        n_per_cam = {cid: sum(1 for s in self.snapshots if cid in s) for cid in range(self.n_cam)}
        for cid in range(self.n_cam):
            n = n_per_cam[cid]
            flag = " ✔" if n >= self.min_snapshots else ""
            self.views[cid].set_status(f"累计 {n} 次{flag}", "#8f8" if n else "#888")

    def _next_index(self, cid: int) -> int:
        """当前相机已存 raw 照片数（用作下一个序号）。"""
        d = os.path.join(self.out_dir, f"cam_{cid}")
        if not os.path.isdir(d):
            return 0
        return len([f for f in os.listdir(d) if f.endswith(".png") and "_annot" not in f])

    def _save_raw(self, cid: int, gray: np.ndarray) -> int:
        idx = self._next_index(cid)
        d = os.path.join(self.out_dir, f"cam_{cid}")
        os.makedirs(d, exist_ok=True)
        cv2.imwrite(os.path.join(d, f"{idx:03d}.png"), gray)
        return idx

    def _save_annot(self, cid: int, img: np.ndarray, idx: int) -> None:
        d = os.path.join(self.out_dir, f"cam_{cid}")
        os.makedirs(d, exist_ok=True)
        cv2.imwrite(os.path.join(d, f"{idx:03d}_annot.png"), img)

    # ---- 收尾 ----
    def _log(self, msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self.log.append(f"[{ts}] {msg}")

    def closeEvent(self, event) -> None:
        self.timer.stop()
        self._close_viewer()
        super().closeEvent(event)


def _load_intrinsics(intrinsics_dir: str, n_cam: int) -> dict:
    """读每台相机的内参 cam_N.yaml；缺失则 None。"""
    out = {}
    for cid in range(n_cam):
        path = os.path.join(intrinsics_dir, f"cam_{cid}.yaml")
        out[cid] = load_intrinsics(path) if os.path.exists(path) else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="多相机外参标定 · ChArUco")
    ap.add_argument("--config", default=None, help="extrinsics.yaml 路径")
    ap.add_argument("--cameras", default=None, help="cameras.yaml 路径")
    ap.add_argument("--exposure", type=float, default=None, help="曝光时间(us)")
    ap.add_argument("--gain", type=float, default=None, help="增益(dB)")
    ap.add_argument("--square-length-m", type=float, default=None, help="覆盖每格边长(m)")
    ap.add_argument("--marker-length-m", type=float, default=None, help="覆盖标记边长(m)")
    ap.add_argument("--dictionary", default=None, help="覆盖 ArUco 字典名")
    ap.add_argument("--trigger", choices=["external", "software", "continuous"], default=None,
                    help="触发模式，默认读 cameras.yaml 的 trigger.mode")
    ap.add_argument("--reference-camera", type=int, default=None,
                    help="参考相机（相对外参的世界系原点，默认读 extrinsics.yaml 的 reference_camera）")
    ap.add_argument("--generate-board", metavar="PATH", default=None,
                    help="只生成与配置一致的打印板 PNG，然后退出（不开相机）")
    ap.add_argument("--generate-markers", metavar="DIR", default=None,
                    help="只生成桌面定原点用的两个大标记 PNG（ID0/ID1），存到 DIR 后退出")
    args = ap.parse_args()

    root = project_root()
    cfg_path = args.config or os.path.join(root, "config", "extrinsics.yaml")
    cam_path = args.cameras or os.path.join(root, "config", "cameras.yaml")

    cal_cfg = load_yaml(cfg_path) if os.path.exists(cfg_path) else {}
    cam_cfg = load_yaml(cam_path) if os.path.exists(cam_path) else {}

    cfg = CharucoConfig.from_dict(cal_cfg)
    if args.square_length_m is not None:
        cfg.square_length_m = args.square_length_m
    if args.marker_length_m is not None:
        cfg.marker_length_m = args.marker_length_m
    if args.dictionary is not None:
        cfg.dictionary = args.dictionary

    if args.generate_board:
        img = generate_board_image(cfg, px_per_square=60, margin_px=40)
        cv2.imwrite(args.generate_board, img)
        print(f"已生成打印板 -> {args.generate_board}（按 1:1 原尺寸打印）")
        sys.exit(0)

    table_cfg = cal_cfg.get("table", {}) or {}

    if args.generate_markers:
        os.makedirs(args.generate_markers, exist_ok=True)
        ids = sorted({
            int(table_cfg.get(k, d)) for k, d in (
                ("marker_origin_id", 0), ("marker_short_id", 1),
                ("marker_diagonal_id", 2), ("marker_long_id", 3),
            )
        })
        for mid in ids:
            img = generate_marker_image(table_cfg.get("marker_dict", "DICT_5X5_50"), mid)
            p = os.path.join(args.generate_markers, f"marker_{mid}.png")
            cv2.imwrite(p, img)
            print(f"已生成标记 -> {p}")
        print(f"标记边长 = {table_cfg.get('marker_length_m', 0.18)}m，请按 1:1 原尺寸打印（不要缩放）。")
        sys.exit(0)

    board = create_board(cfg)
    detector = cv2.aruco.CharucoDetector(board)

    out_dir = os.path.join(root, cal_cfg.get("output_dir", "data/extrinsics"))
    intr_dir = os.path.join(root, cal_cfg.get("intrinsics_dir", "data/calibration"))
    min_snapshots = int(cal_cfg.get("min_snapshots", 1))
    reference_cam = (
        args.reference_camera if args.reference_camera is not None
        else int(cal_cfg.get("reference_camera", 0))
    )

    image = cam_cfg.get("image", {}) or {}
    cs = resolve_camera_settings(args.exposure, args.gain)

    # 触发模式：命令行 > cameras.yaml 的 trigger.mode > external（默认走信号发生器时钟）
    trigger_cfg = cam_cfg.get("trigger", {}) or {}
    trigger_mode = args.trigger or trigger_cfg.get("mode", "external")
    trigger_source = trigger_cfg.get("source", "Line0")

    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)

    try:
        mgr = CameraManager(
            serials=[c.get("serial") for c in cam_cfg.get("cameras", []) if c.get("serial")] or None,
            trigger_mode=trigger_mode,
            trigger_source=trigger_source,
            exposure_us=cs["exposure_us"],
            gain_db=cs["gain_db"],
            pixel_format=image.get("pixel_format", "Mono8"),
        )
        mgr.setup()
        mgr.start()
    except Exception as e:  # noqa: BLE001
        QMessageBox.critical(None, "打开相机失败", str(e))
        sys.exit(1)

    intrinsics = _load_intrinsics(intr_dir, len(mgr.cameras))
    missing = [cid for cid, v in intrinsics.items() if v is None]
    if missing:
        QMessageBox.warning(
            None, "缺少内参",
            f"相机 {missing} 没有内参结果（{intr_dir}/cam_N.yaml）。\n"
            f"请先跑 scripts/calibrate_intrinsics.py 标定内参，否则这些相机无法解外参。"
        )

    win = MainWindow(mgr, cfg, board, detector, intrinsics, out_dir, min_snapshots, trigger_mode, table_cfg, reference_cam)
    win.show()
    code = app.exec()

    mgr.close()
    sys.exit(code)


if __name__ == "__main__":
    main()
