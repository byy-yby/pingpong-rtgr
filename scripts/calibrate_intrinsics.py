#!/usr/bin/env python3
"""相机内参标定 · ChArUco 板 · 采集 + 标定 + 精度检验 GUI。

四路实时预览 + 点击选中 + Enter 纯拍照（只存盘，不做检测），
采集够多张后点「计算并检验」统一识别（逐张检测 ChArUco）并对每台相机做内参标定，
输出精度报告（重投影误差 RMS、主点偏移、fx/fy 合理性），结果存 ``data/calibration/cam_N.yaml``。

用法：
  conda activate tt
  python scripts/calibrate_intrinsics.py                 # 连续采集（无信号发生器）
  python scripts/calibrate_intrinsics.py --exposure 5000 --gain 0

交互：
  - 点击某个画面选中该相机（红框高亮）
  - 移动/倾斜标定板，按 Enter 拍一张（纯拍照，只存盘计数）
  - 每路采集 ≥15 张后点「计算并检验 (C)」：统一识别 + 标定 + 精度报告
  - 板子摆放见 README / 脚本底部提示：覆盖画面九宫格 + 每张不同倾斜/距离

棋盘摆放要点（内参标定，板要**动**，与外参标定「板不动」相反）：
  - 所有照片合起来覆盖整个画面（四角 + 边缘 + 中心），因为畸变在边缘最强。
  - 每张给不同 3D 倾斜（绕板长/短轴转 20~45°），别一直正对相机。
  - 近（板占画面 60~70%）和远（20~30%）各来几张。
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QFrame, QGridLayout, QHBoxLayout, QLabel, QMainWindow, QMessageBox,
    QPushButton, QTextEdit, QVBoxLayout, QWidget,
)

import cv2
import numpy as np

from tabletennis.calibration.extrinsics import (
    CharucoConfig, create_board,
)
from tabletennis.calibration.intrinsics import (
    calibrate_charuco, capture_dir, count_captures, save_capture,
    save_intrinsics, scan_charuco_images, validate_intrinsics,
)
from tabletennis.camera import CameraManager
from tabletennis.core.config import load_yaml, project_root, resolve_camera_settings

def _annotate(gray: np.ndarray, corners: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """把检测到的 ChArUco 角点画到 BGR 图上，便于人工核对。"""
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if corners is not None:
        cv2.aruco.drawDetectedCornersCharuco(
            img, corners.reshape(-1, 1, 2), ids.reshape(-1, 1), (0, 255, 0)
        )
    return img


class CameraView(QFrame):
    """单个相机画面面板：点击选中，显示实时画面 + 采集计数 + 检测状态。"""

    clicked = Signal(int)

    def __init__(self, cam_id: int):
        super().__init__()
        self.cam_id = cam_id
        self._selected = False
        self.setObjectName(f"camview{cam_id}")

        self._title = QLabel(f"相机 {cam_id}  |  0 张")
        self._title.setStyleSheet("color:#eee; font-weight:bold; padding:2px;")

        self._img = QLabel("无信号")
        self._img.setAlignment(Qt.AlignCenter)
        self._img.setMinimumSize(320, 240)
        self._img.setStyleSheet("color:#888;")

        self._status = QLabel("—")
        self._status.setStyleSheet("color:#888; padding:2px;")

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.addWidget(self._title)
        lay.addWidget(self._img, 1)
        lay.addWidget(self._status)
        self._apply_border()

    def _apply_border(self) -> None:
        color = "#e33" if self._selected else "#555"
        width = "3px" if self._selected else "2px"
        self.setStyleSheet(
            f"#{self.objectName()} {{ border: {width} solid {color}; "
            f"background: #1e1e1e; }}"
        )

    def set_frame(self, gray: np.ndarray) -> None:
        if gray is None:
            return
        h, w = gray.shape[:2]
        # 用完整分辨率（不降采样），再缩放到面板当前尺寸，保持完整视野不裁剪
        qimg = QImage(gray.data, w, h, w, QImage.Format_Grayscale8).copy()
        pix = QPixmap.fromImage(qimg)
        pix = pix.scaled(self._img.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._img.setPixmap(pix)

    def set_count(self, n: int, target: int) -> None:
        flag = " ✔" if n >= target else ""
        self._title.setText(f"相机 {self.cam_id}  |  {n} 张{flag}")

    def set_status(self, text: str, color: str = "#8f8") -> None:
        self._status.setText(text)
        self._status.setStyleSheet(f"color:{color}; padding:2px;")

    def set_selected(self, sel: bool) -> None:
        self._selected = sel
        self._apply_border()

    def mousePressEvent(self, event) -> None:
        self.clicked.emit(self.cam_id)
        super().mousePressEvent(event)


class MainWindow(QMainWindow):
    def __init__(self, mgr: CameraManager, cfg: CharucoConfig, board, detector,
                 out_dir: str, min_images: int, min_images_hard: int):
        super().__init__()
        self.setWindowTitle("相机内参标定 · ChArUco")
        self.mgr = mgr
        self.cfg = cfg
        self.board = board
        self.detector = detector
        self.out_dir = out_dir
        self.min_images = min_images
        self.min_images_hard = min_images_hard
        self.n_cam = len(mgr.cameras)
        self.selected = 0

        self.views = [CameraView(i) for i in range(self.n_cam)]
        self.views[0].set_selected(True)
        for v in self.views:
            v.clicked.connect(self._on_select)

        grid = QGridLayout()
        for i, v in enumerate(self.views):
            grid.addWidget(v, i // 2, i % 2)

        self.calc_btn = QPushButton("计算并检验 (C)")
        self.clear_btn = QPushButton("清空数据")
        self.exit_btn = QPushButton("退出 (Q)")
        self.calc_btn.clicked.connect(self._on_calc)
        self.clear_btn.clicked.connect(self._on_clear)
        self.exit_btn.clicked.connect(self.close)

        btns = QHBoxLayout()
        for b in (self.calc_btn, self.clear_btn, self.exit_btn):
            b.setFocusPolicy(Qt.NoFocus)  # 按钮不抢焦点，避免 Enter 误触
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

        self._refresh_counts()
        self._log("就绪。点击选中相机，移动/倾斜标定板后按 Enter 采集；C 计算并检验。")

    # ---- 交互 ----
    def _on_select(self, cam_id: int) -> None:
        self.selected = cam_id
        for i, v in enumerate(self.views):
            v.set_selected(i == cam_id)

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self._capture()
        elif event.key() == Qt.Key_C:
            self._on_calc()
        elif event.key() in (Qt.Key_Q, Qt.Key_Escape):
            self.close()
        else:
            super().keyPressEvent(event)

    def _capture(self) -> None:
        """纯拍照：把选中相机的当前帧直接存盘（不做检测，统一到「计算并检验」）。"""
        cam_id = self.selected
        frame = self.mgr.get_latest_frame(cam_id, block=True, timeout=1.0)
        if frame is None:
            self.views[cam_id].set_status("无画面", "#e33")
            return
        path = save_capture(self.out_dir, cam_id, frame.image)
        n = count_captures(self.out_dir, cam_id)
        self.views[cam_id].set_count(n, self.min_images)
        self.views[cam_id].set_status(f"已存第 {n} 张", "#8f8")
        self._log(f"[相机{cam_id}] 第 {n} 张已存 -> {os.path.basename(path)}。")

    # ---- 刷新 ----
    def _refresh(self) -> None:
        frames = self.mgr.get_latest_frames(block=False)
        for cid, f in frames.items():
            if f is not None:
                self.views[cid].set_frame(f.image)

    def _refresh_counts(self) -> None:
        for i in range(self.n_cam):
            self.views[i].set_count(count_captures(self.out_dir, i), self.min_images)

    # ---- 计算并检验 ----
    def _on_calc(self) -> None:
        self._log("开始统一识别并标定 ...")
        for cid in range(self.n_cam):
            n_imgs = count_captures(self.out_dir, cid)
            if n_imgs < self.min_images_hard:
                self.views[cid].set_status(f"张数不足 {n_imgs}", "#e88")
                self._log(f"[相机{cid}] 张数不足({n_imgs}<{self.min_images_hard})，跳过。")
                continue

            # 统一识别：逐张报告 标记数/角点数
            diag = scan_charuco_images(self.out_dir, cid, self.detector)
            n_ok = sum(1 for d in diag if d["n_corners"] >= self.cfg.min_corners)
            self._log(f"[相机{cid}] 统一识别：{n_ok}/{len(diag)} 张有效。")
            for d in diag:
                if d["n_corners"] < self.cfg.min_corners:
                    self._log(f"    跳过 {d['file']}: {d['n_markers']} 标记 -> {d['n_corners']} 角点")

            intr, rms, per_view, n = calibrate_charuco(
                self.out_dir, cid, self.detector, self.board,
                self.cfg.min_corners, self.min_images_hard,
            )
            if intr is None:
                self.views[cid].set_status(f"标定失败（有效{n}张）", "#e88")
                self._log(f"[相机{cid}] 标定失败：有效张数 {n} < {self.min_images_hard}。")
                continue

            report = validate_intrinsics(intr, rms, per_view)
            save_intrinsics(self.out_dir, cid, intr, rms, n, extra={
                "verdict": report["verdict"],
                "problems": report["problems"],
                "per_view_rms": [round(x, 3) for x in per_view],
            })

            self._log(f"[相机{cid}] {n} 张 | RMS={rms:.3f}px | 判定={report['verdict']}")
            self._log(f"    fx={report['fx']:.1f} fy={report['fy']:.1f} "
                      f"cx={report['cx']:.1f} cy={report['cy']:.1f}")
            self._log(f"    主点偏移=({report['principal_point_offset_px'][0]:.1f},"
                      f"{report['principal_point_offset_px'][1]:.1f})px  "
                      f"焦距比={report['focal_ratio']:.3f}")
            for prob in report["problems"]:
                self._log(f"    ⚠ {prob}")
            self.views[cid].set_status(
                f"RMS {rms:.2f}px [{report['verdict']}]",
                "#8f8" if report["verdict"] == "ok" else "#ee0",
            )
        self._log("完成。结果存 data/calibration/cam_N.yaml（rms<0.5 良好、<0.3 优秀）。")

    # ---- 清空 ----
    def _on_clear(self) -> None:
        ret = QMessageBox.question(self, "确认", "清空所有已采集的照片?")
        if ret != QMessageBox.Yes:
            return
        for i in range(self.n_cam):
            d = capture_dir(self.out_dir, i)
            for f in os.listdir(d):
                if f.endswith(".png"):
                    os.remove(os.path.join(d, f))
        self._refresh_counts()
        self._log("已清空采集数据。")

    # ---- 收尾 ----
    def _log(self, msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self.log.append(f"[{ts}] {msg}")

    def closeEvent(self, event) -> None:
        self.timer.stop()
        super().closeEvent(event)


def main() -> None:
    ap = argparse.ArgumentParser(description="相机内参标定 · ChArUco")
    ap.add_argument("--config", default=None, help="calibration.yaml 路径")
    ap.add_argument("--cameras", default=None, help="cameras.yaml 路径")
    ap.add_argument("--exposure", type=float, default=None, help="曝光时间(us)")
    ap.add_argument("--gain", type=float, default=None, help="增益(dB)")
    ap.add_argument("--square-length-m", type=float, default=None, help="覆盖每格边长(m)")
    ap.add_argument("--marker-length-m", type=float, default=None, help="覆盖标记边长(m)")
    ap.add_argument("--dictionary", default=None, help="覆盖 ArUco 字典名")
    args = ap.parse_args()

    root = project_root()
    cal_path = args.config or os.path.join(root, "config", "calibration.yaml")
    cam_path = args.cameras or os.path.join(root, "config", "cameras.yaml")

    cal_cfg = load_yaml(cal_path) if os.path.exists(cal_path) else {}
    cam_cfg = load_yaml(cam_path) if os.path.exists(cam_path) else {}

    cfg = CharucoConfig.from_dict(cal_cfg)
    if args.square_length_m is not None:
        cfg.square_length_m = args.square_length_m
    if args.marker_length_m is not None:
        cfg.marker_length_m = args.marker_length_m
    if args.dictionary is not None:
        cfg.dictionary = args.dictionary

    board = create_board(cfg)
    detector = cv2.aruco.CharucoDetector(board)

    out_dir = os.path.join(root, cal_cfg.get("output_dir", "data/calibration"))
    min_images = int(cal_cfg.get("min_images", 15))
    min_images_hard = int(cal_cfg.get("min_images_hard", 5))

    image = cam_cfg.get("image", {}) or {}
    cs = resolve_camera_settings(args.exposure, args.gain)

    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)

    try:
        mgr = CameraManager(
            serials=[c.get("serial") for c in cam_cfg.get("cameras", []) if c.get("serial")] or None,
            trigger_mode="continuous",
            exposure_us=cs["exposure_us"],
            gain_db=cs["gain_db"],
            pixel_format=image.get("pixel_format", "Mono8"),
        )
        mgr.setup()
        mgr.start()
    except Exception as e:  # noqa: BLE001
        QMessageBox.critical(None, "打开相机失败", str(e))
        sys.exit(1)

    win = MainWindow(mgr, cfg, board, detector, out_dir, min_images, min_images_hard)
    win.show()
    code = app.exec()

    mgr.close()
    sys.exit(code)


if __name__ == "__main__":
    main()
