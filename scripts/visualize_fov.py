#!/usr/bin/env python
"""3D 视场角 / 焦距可视化（matplotlib 版，无 Open3D 依赖）。

对着 4 台相机的**桌面系外参**（``data/extrinsics/table_extrinsics.yaml``）和**内参**
（``data/calibration/cam_N.yaml``）用 matplotlib 的 ``mplot3d`` 建 3D 场景，再用
``Slider`` / ``RadioButtons`` / ``CheckButtons`` 做交互：拖「焦距 / FOV 水平 /
FOV 垂直」三个滑动条实时看每台相机的覆盖范围（4 条角射线 + 桌面足迹四边形 + 半透明
锥体）怎么收窄/放大；勾选框选要渲染的相机；右侧单选选中一台相机后，用 X/Y/Z +
yaw/pitch/roll 滑动条**移动 / 转动**它，实时看覆盖范围随位姿变化。

改用 matplotlib 的原因：Open3D 的交互窗口在本机 GL 后端反复段错误（覆盖层用透明材质
/ 三角网格重挂触发，与 recon_player 记录的同一类问题）。matplotlib 纯 CPU 软件渲染，
无 GPU / Filament / OpenGL，几乎不可能段错误；代价是 3D 观感朴素、拖动稍显迟钝，
但足够看位姿与视场角。

物理约定（与重建管道一致）：
- 世界系 = 桌面系：原点在桌角标记，X 短边(宽 1.525m)、Y 长边(长 2.74m)、Z 竖直向上，桌面 z=0。
- 外参 ``(R, t)`` 是「桌面系 -> 相机系」：``X_cam = R @ X_world + t``。相机光心
  世界坐标 ``C = -R^T @ t``。
- 内参 ``K = [[fx,0,cx],[0,fy,cy],[0,0,1]]``，分辨率 1440×1080，像元 3.45µm（Sony IMX273）。
  焦距与视场角互推：``FOV_h = 2·atan(W/(2fx))``、``FOV_v = 2·atan(H/(2fy))``、
  ``f[mm] = fx · 像元``。
- 三个滑动条里**焦距是各向同性主控**（拖它 → fx=fy，两 FOV 跟随）；「FOV 水平 /
  垂直」各自独立改 fx / fy（可各向异性）。拖动任意一个，其余两个自动同步。

覆盖几何：4 条角射线打到桌面平面 z=0 的 4 个交点围成足迹四边形，4 条角射线 + 4 条
足迹边画成线，锥体 4 个侧面 + 足迹底面用 ``Poly3DCollection`` 半透明填充（alpha，
即「空间里能拍到的地方」）；角射线打不到桌面时退到远平面兜底点。

用法：
    python scripts/visualize_fov.py                      # 交互窗口（滑动条 + 相机勾选）
    python scripts/visualize_fov.py --offscreen out.png  # 无头渲染一帧 PNG（默认内参）

依赖：matplotlib + numpy（无 open3d）。交互需要 GUI 后端（本机默认 QtAgg）。
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

# 仅导入 3D 多边形类（不碰 pyplot/后端，交互与无头共用；pyplot 在各自入口惰性 import）
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

# 引导 import：本项目脚本统一把 src/ 加进 sys.path 找 tabletennis 包。
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if os.path.isdir(os.path.join(_ROOT, "src")):
    sys.path.insert(0, os.path.join(_ROOT, "src"))

from tabletennis.calibration.extrinsics import load_extrinsics  # noqa: E402
from tabletennis.calibration.intrinsics import load_intrinsics  # noqa: E402
from tabletennis.core.types import Table3D  # noqa: E402

# ---- 常量 ----
_SENSOR_PIXEL_MM = 0.00345          # Sony IMX273 像元 3.45µm
_CAM_COLORS = [
    "#e04848",   # 红
    "#4a8cf0",   # 蓝
    "#3ecf5e",   # 绿
    "#f0d43c",   # 黄
]
_CAM_COLOR_NAMES = ["red", "blue", "green", "yellow"]

_TABLE_TOP = "#2a8f4d"
_TABLE_EDGE = "#1c6b38"
_TABLE_STRUCT = "#777777"
_TABLE_NET = "#2ec8c8"

# 远平面深度（角射线打不到桌面时的兜底终点）：= clamp(因子 × 相机到球桌中心距离)。
_FRUSTUM_DEPTH_FACTOR = 1.3
_FRUSTUM_DEPTH_MIN_M = 2.5
_FRUSTUM_DEPTH_MAX_M = 6.0
# 覆盖锥半透明填充透明度
_FILL_ALPHA = 0.10
# 视图自动适配时，足迹点相对球桌中心的最大水平距离（米）——超广角足迹裁到此，避免视图爆炸
_MAX_FIT_R = 10.0

_AXIS_COLORS = ["#ff3b3b", "#2ecc40", "#3b6bff"]   # 坐标架 X/Y/Z


# =========================================================================
# 数据加载与几何计算（纯 numpy）
# =========================================================================
def load_rig(root: str):
    """读内参 + 桌面系外参，返回 ``(intrinsics, extrinsics)``（key=cam_id）。"""
    intr_dir = os.path.join(root, "data", "calibration")
    ext_path = os.path.join(root, "data", "extrinsics", "table_extrinsics.yaml")
    intrinsics: Dict[int, object] = {}
    for cid in range(8):
        p = os.path.join(intr_dir, f"cam_{cid}.yaml")
        if os.path.isfile(p):
            intrinsics[cid] = load_intrinsics(p)
    if not intrinsics:
        raise FileNotFoundError(f"未找到内参文件（{intr_dir}/cam_*.yaml）")
    extrinsics = load_extrinsics(ext_path)
    return intrinsics, extrinsics


def fov_from_fx_fy(fx: float, fy: float, W: int, H: int) -> Tuple[float, float, float]:
    """由焦距（像素）推水平/垂直视场角（度）与等效焦距（mm）。"""
    fov_h = math.degrees(2.0 * math.atan(W / (2.0 * fx)))
    fov_v = math.degrees(2.0 * math.atan(H / (2.0 * fy)))
    f_mm = fx * _SENSOR_PIXEL_MM
    return fov_h, fov_v, f_mm


def build_cameras(intrinsics, extrinsics, table: Table3D) -> List[dict]:
    """固化每台相机：光心 C、Rw(相机->世界)、cx/cy、图像尺寸、远平面深度、颜色。"""
    cams: List[dict] = []
    table_center = np.array([table.width / 2.0, table.length / 2.0, 0.0])
    for cid in sorted(extrinsics):
        K = intrinsics.get(cid)
        if K is None:
            continue
        ext = extrinsics[cid]
        R = np.asarray(ext.R, dtype=np.float64)
        t = np.asarray(ext.t, dtype=np.float64).reshape(3)
        Rw = R.T
        C = -Rw @ t
        yaw, pitch, roll = rot_to_ypr(Rw)
        depth = float(np.clip(
            _FRUSTUM_DEPTH_FACTOR * np.linalg.norm(C - table_center),
            _FRUSTUM_DEPTH_MIN_M, _FRUSTUM_DEPTH_MAX_M))
        cams.append({
            "cid": cid,
            "Rw": Rw,
            "C": C,
            "yaw": yaw, "pitch": pitch, "roll": roll,   # 当前朝向（可编辑）
            "C0": C.copy(), "Rw0": Rw.copy(),             # 标定基准（复位用）
            "yaw0": yaw, "pitch0": pitch, "roll0": roll,
            "cx": float(K.K[0, 2]),
            "cy": float(K.K[1, 2]),
            "W": int(K.width),
            "H": int(K.height),
            "depth": depth,
            "color": _CAM_COLORS[cid % len(_CAM_COLORS)],
        })
    return cams


def corner_dirs(cam: dict, fx: float, fy: float) -> np.ndarray:
    """4 个图像角的单位射线方向（世界系）(4,3)。"""
    W, H = cam["W"], cam["H"]
    K = np.array([[fx, 0, cam["cx"]], [0, fy, cam["cy"]], [0, 0, 1]], np.float64)
    uv = np.array([[0, 0, 1], [W, 0, 1], [W, H, 1], [0, H, 1]], np.float64)
    dirs = (np.linalg.inv(K) @ uv.T).T
    dirs = (cam["Rw"] @ dirs.T).T
    return dirs / np.linalg.norm(dirs, axis=1, keepdims=True)


def ray_ends(cam: dict, dirs: np.ndarray) -> np.ndarray:
    """每条单位射线 ``(N,3)`` 与桌面 z=0 的交点；打不到则退到远平面点。"""
    C = cam["C"]
    ends = np.empty_like(dirs)
    for i, d in enumerate(dirs):
        dz = d[2]
        if dz < -1e-9:
            s = (0.0 - C[2]) / dz
            if s > 0.0:
                ends[i] = C + s * d
                continue
        ends[i] = C + cam["depth"] * d
    return ends


def cone_corners(cam: dict, fx: float, fy: float) -> Tuple[np.ndarray, np.ndarray]:
    """光心 C 与 4 个足迹角点 corners(4,3)（世界系）。"""
    C = cam["C"]
    return C, ray_ends(cam, corner_dirs(cam, fx, fy))


def rot_to_ypr(R: np.ndarray) -> Tuple[float, float, float]:
    """旋转矩阵 → 世界系 ZYX 欧拉角（yaw/pitch/roll，度）。"""
    R = np.asarray(R, np.float64)
    yaw = math.degrees(math.atan2(R[1, 0], R[0, 0]))
    pitch = math.degrees(math.atan2(-R[2, 0], math.hypot(R[0, 0], R[1, 0])))
    roll = math.degrees(math.atan2(R[2, 1], R[2, 2]))
    return yaw, pitch, roll


def ypr_to_rot(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """世界系 ZYX 欧拉角（度）→ 旋转矩阵 ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``。"""
    y, p, r = math.radians(yaw_deg), math.radians(pitch_deg), math.radians(roll_deg)
    cy, sy = math.cos(y), math.sin(y)
    cp, sp = math.cos(p), math.sin(p)
    cr, sr = math.cos(r), math.sin(r)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ], np.float64)


# =========================================================================
# 绘图辅助（模块级，交互与无头共用）
# =========================================================================
def draw_table(ax, table: Table3D) -> None:
    """球桌：半透明桌面 + 桌面边/桌腿/地面框/球网线框 + 原点坐标架。"""
    segs = table.segments()
    c = table.top_corners
    ax.add_collection3d(
        Poly3DCollection([c.tolist()], alpha=0.35, facecolor=_TABLE_TOP, edgecolor="none"))
    for key, color, lw in [("top", _TABLE_EDGE, 1.8), ("legs", _TABLE_STRUCT, 1.1),
                           ("floor", _TABLE_STRUCT, 0.9), ("net", _TABLE_NET, 1.5)]:
        for seg in segs[key]:
            ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=color, linewidth=lw)
    _draw_axes(ax, np.zeros(3), np.eye(3), 0.25)


def draw_camera_body(ax, cam: dict):
    """相机位置点 + 朝向短坐标架。返回 ``(marker, [axis_line...])`` 供显隐。"""
    C, Rw, color = cam["C"], cam["Rw"], cam["color"]
    marker = ax.scatter([C[0]], [C[1]], [C[2]], color=color, s=48,
                        depthshade=False, edgecolors="k", linewidths=0.5)
    axes = _draw_axes(ax, C, Rw, 0.28)
    return marker, axes


def draw_cone(ax, cam: dict, fx: float, fy: float) -> List:
    """单相机覆盖锥：4 条角射线 + 4 条足迹边（线）+ 锥体/足迹半透明填充。"""
    C, corners = cone_corners(cam, fx, fy)
    color = cam["color"]
    arts: List = []
    for i in range(4):
        ln, = ax.plot([C[0], corners[i, 0]], [C[1], corners[i, 1]],
                      [C[2], corners[i, 2]], color=color, linewidth=1.1)
        arts.append(ln)
    for i in range(4):
        j = (i + 1) % 4
        ln, = ax.plot([corners[i, 0], corners[j, 0]], [corners[i, 1], corners[j, 1]],
                      [corners[i, 2], corners[j, 2]], color=color, linewidth=1.1)
        arts.append(ln)
    faces = [
        [C.tolist(), corners[0].tolist(), corners[1].tolist()],
        [C.tolist(), corners[1].tolist(), corners[2].tolist()],
        [C.tolist(), corners[2].tolist(), corners[3].tolist()],
        [C.tolist(), corners[3].tolist(), corners[0].tolist()],
        [corners[0].tolist(), corners[1].tolist(), corners[2].tolist(), corners[3].tolist()],
    ]
    poly = Poly3DCollection(faces, alpha=_FILL_ALPHA, facecolor=color, edgecolor="none")
    ax.add_collection3d(poly)
    arts.append(poly)
    return arts


def _draw_axes(ax, origin: np.ndarray, Rw: np.ndarray, size: float) -> List:
    """从 origin 沿 Rw 三列画 X/Y/Z 三短轴，返回 3 个 Line3D。"""
    lines = []
    for i in range(3):
        d = Rw[:, i] * size
        ln, = ax.plot([origin[0], origin[0] + d[0]], [origin[1], origin[1] + d[1]],
                      [origin[2], origin[2] + d[2]], color=_AXIS_COLORS[i], linewidth=1.6)
        lines.append(ln)
    return lines


def setup_view(ax, table: Table3D) -> None:
    """设坐标轴标签与默认斜俯视视角（限值由 :func:`fit_view` 按覆盖范围动态设）。"""
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m]")
    ax.view_init(elev=22.0, azim=-58.0)


def fit_view(ax, cams, table: Table3D, enabled: List[bool], fx: float, fy: float) -> None:
    """按相机 + 球桌 + 当前足迹范围自动适配坐标轴限与等比例。

    足迹随 FOV 变化（焦距越小足迹越大），所以每次 FOV 变化都重算一次限值，保证覆盖
    范围始终完整显示；超广角足迹水平方向裁到 ``_MAX_FIT_R``、竖直裁到合理区间，避免
    视图被极远点撑爆。
    """
    center2 = np.array([table.width / 2.0, table.length / 2.0])
    pts: List[List[float]] = []
    for cam in cams:
        pts.append([cam["C"][0], cam["C"][1], cam["C"][2]])
    for cam in cams:
        if not enabled[cam["cid"]]:
            continue
        _, corners = cone_corners(cam, fx, fy)
        for p in corners:
            dx, dy = p[0] - center2[0], p[1] - center2[1]
            r = math.hypot(dx, dy)
            if r > _MAX_FIT_R:
                dx *= _MAX_FIT_R / r
                dy *= _MAX_FIT_R / r
                p = [center2[0] + dx, center2[1] + dy, p[2]]
            pz = min(max(p[2], -table.height - 0.5), 4.5)
            pts.append([p[0], p[1], pz])
    for c in table.top_corners:
        pts.append([c[0], c[1], c[2]])
    pts.append([0.0, 0.0, -table.height])
    pts.append([table.width, table.length, -table.height])
    a = np.asarray(pts, np.float64)
    lo, hi = a.min(axis=0), a.max(axis=0)
    span = max(hi[0] - lo[0], hi[1] - lo[1], 1.0)
    m = 0.12 * span
    ax.set_xlim(lo[0] - m, hi[0] + m)
    ax.set_ylim(lo[1] - m, hi[1] + m)
    ax.set_zlim(lo[2] - 0.3, hi[2] + 0.6)
    ax.set_box_aspect((hi[0] - lo[0] + 2 * m, hi[1] - lo[1] + 2 * m, hi[2] - lo[2] + 0.9))


# =========================================================================
# 交互窗口
# =========================================================================
class FovViewer:
    """matplotlib 交互窗口：3D 场景 + 焦距/FOV 滑动条 + 相机位姿编辑 + 勾选 + 复位。"""

    def __init__(self, cams: List[dict], table: Table3D,
                 base_fx: float, base_fy: float, W: int, H: int):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider, CheckButtons, Button, RadioButtons
        self.plt = plt
        self.cams = cams
        self.table = table
        self.W, self.H = W, H
        self.fx = base_fx
        self.fy = base_fy
        self._base_fx = base_fx
        self._base_fy = base_fy
        self.enabled = [True] * len(cams)
        self._edit_idx = 0                       # 当前编辑（移动/转动）的相机索引
        self._syncing = False                    # 焦距/FOV 滑动条重入保护
        self._syncing_pose = False               # 位姿滑动条重入保护
        self._dyn_arts: List[List] = []          # 动态图层（相机本体 + 覆盖锥）

        self.fig = plt.figure(figsize=(13.5, 10))
        self.ax = self.fig.add_subplot(111, projection="3d")
        draw_table(self.ax, table)
        setup_view(self.ax, table)

        fov_h, fov_v, f_mm = fov_from_fx_fy(base_fx, base_fy, W, H)
        self.fig.subplots_adjust(left=0.02, right=0.98, top=0.97, bottom=0.42)
        self.ax.set_position([0.02, 0.42, 0.66, 0.55])

        # 列 1：镜头（焦距 / FOV）
        self._sl_focal = Slider(self.plt.axes([0.02, 0.34, 0.20, 0.025]),
                                "focal f [mm]", 1.0, 25.0, valinit=f_mm)
        self._sl_fov_h = Slider(self.plt.axes([0.02, 0.27, 0.20, 0.025]),
                                "FOV horiz [deg]", 5.0, 150.0, valinit=fov_h)
        self._sl_fov_v = Slider(self.plt.axes([0.02, 0.20, 0.20, 0.025]),
                                "FOV vert [deg]", 5.0, 150.0, valinit=fov_v)
        self._btn_reset = Button(self.plt.axes([0.02, 0.12, 0.20, 0.035]), "Reset")
        self._sl_focal.on_changed(self._on_focal)
        self._sl_fov_h.on_changed(self._on_fov_h)
        self._sl_fov_v.on_changed(self._on_fov_v)
        self._btn_reset.on_clicked(self._on_reset)

        # 列 2：选中相机的位置（世界系，米）
        self._sl_cx = Slider(self.plt.axes([0.25, 0.34, 0.20, 0.025]),
                             "X [m]", -4.0, 6.0, valinit=cams[0]["C"][0])
        self._sl_cy = Slider(self.plt.axes([0.25, 0.27, 0.20, 0.025]),
                             "Y [m]", -5.0, 8.0, valinit=cams[0]["C"][1])
        self._sl_cz = Slider(self.plt.axes([0.25, 0.20, 0.20, 0.025]),
                             "Z [m]", 0.5, 5.0, valinit=cams[0]["C"][2])
        self._sl_cx.on_changed(self._on_cx)
        self._sl_cy.on_changed(self._on_cy)
        self._sl_cz.on_changed(self._on_cz)

        # 列 3：选中相机的朝向（世界系 ZYX 欧拉角，度）
        self._sl_yaw = Slider(self.plt.axes([0.48, 0.34, 0.20, 0.025]),
                              "yaw [deg]", -180.0, 180.0, valinit=cams[0]["yaw"])
        self._sl_pitch = Slider(self.plt.axes([0.48, 0.27, 0.20, 0.025]),
                                "pitch [deg]", -90.0, 90.0, valinit=cams[0]["pitch"])
        self._sl_roll = Slider(self.plt.axes([0.48, 0.20, 0.20, 0.025]),
                               "roll [deg]", -180.0, 180.0, valinit=cams[0]["roll"])
        self._sl_yaw.on_changed(self._on_yaw)
        self._sl_pitch.on_changed(self._on_pitch)
        self._sl_roll.on_changed(self._on_roll)

        # 列 4：选相机（编辑对象）+ 勾选渲染
        self._radio_labels = [f"cam {c['cid']}" for c in cams]
        self._radio = RadioButtons(self.plt.axes([0.72, 0.26, 0.26, 0.15]),
                                   self._radio_labels, active=0)
        self._radio.on_clicked(self._on_radio)
        self._check_labels = [f"cam {cam['cid']} ({_CAM_COLOR_NAMES[cam['cid'] % 4]})"
                              for cam in cams]
        self._check = CheckButtons(self.plt.axes([0.72, 0.04, 0.26, 0.20]),
                                   self._check_labels, [True] * len(cams))
        self._check.on_clicked(self._on_check)

        # 分节标题
        self.fig.text(0.02, 0.385, "LENS", fontsize=9, color="#888")
        self.fig.text(0.25, 0.385, "MOVE (selected cam) [m]", fontsize=9, color="#888")
        self.fig.text(0.48, 0.385, "ROTATE (selected cam) [deg]", fontsize=9, color="#888")
        self.fig.text(0.72, 0.385, "EDIT / SHOW", fontsize=9, color="#888")

        self._title = self.fig.suptitle("", fontsize=11)
        self._sync_title()
        self._update()          # 初始绘制相机本体 + 覆盖锥 + 适配视图

    # ---- 焦距 / FOV 回调 --------------------------------------------
    def _on_focal(self, v: float) -> None:
        if self._syncing:
            return
        self.fx = self.fy = v / _SENSOR_PIXEL_MM
        self._sync_sliders()
        self._update()

    def _on_fov_h(self, v: float) -> None:
        if self._syncing:
            return
        self.fx = self.W / (2.0 * math.tan(math.radians(v) / 2.0))
        self._sync_sliders()
        self._update()

    def _on_fov_v(self, v: float) -> None:
        if self._syncing:
            return
        self.fy = self.H / (2.0 * math.tan(math.radians(v) / 2.0))
        self._sync_sliders()
        self._update()

    # ---- 相机位姿回调 ------------------------------------------------
    def _on_cx(self, v: float) -> None:
        if self._syncing_pose:
            return
        self.cams[self._edit_idx]["C"][0] = v
        self._update()

    def _on_cy(self, v: float) -> None:
        if self._syncing_pose:
            return
        self.cams[self._edit_idx]["C"][1] = v
        self._update()

    def _on_cz(self, v: float) -> None:
        if self._syncing_pose:
            return
        self.cams[self._edit_idx]["C"][2] = v
        self._update()

    def _on_yaw(self, v: float) -> None:
        if self._syncing_pose:
            return
        cam = self.cams[self._edit_idx]
        cam["yaw"] = v
        cam["Rw"] = ypr_to_rot(cam["yaw"], cam["pitch"], cam["roll"])
        self._update()

    def _on_pitch(self, v: float) -> None:
        if self._syncing_pose:
            return
        cam = self.cams[self._edit_idx]
        cam["pitch"] = v
        cam["Rw"] = ypr_to_rot(cam["yaw"], cam["pitch"], cam["roll"])
        self._update()

    def _on_roll(self, v: float) -> None:
        if self._syncing_pose:
            return
        cam = self.cams[self._edit_idx]
        cam["roll"] = v
        cam["Rw"] = ypr_to_rot(cam["yaw"], cam["pitch"], cam["roll"])
        self._update()

    def _on_radio(self, label: str) -> None:
        idx = self._radio_labels.index(label)
        if idx == self._edit_idx:
            return
        self._edit_idx = idx
        self._sync_pose_sliders()

    def _on_check(self, label: str) -> None:
        i = self._check_labels.index(label)
        self.enabled[i] = not self.enabled[i]
        self._update()

    def _on_reset(self, event=None) -> None:
        self.fx, self.fy = self._base_fx, self._base_fy
        for cam in self.cams:
            cam["C"][:] = cam["C0"]
            cam["yaw"], cam["pitch"], cam["roll"] = cam["yaw0"], cam["pitch0"], cam["roll0"]
            cam["Rw"] = cam["Rw0"].copy()
        self._sync_sliders()
        self._sync_pose_sliders()
        self._update()

    # ---- 同步 / 重绘 -------------------------------------------------
    def _sync_sliders(self) -> None:
        """把当前 fx/fy 反推的 f/FOV 写回三个滑动条（带重入保护）。"""
        fov_h, fov_v, f_mm = fov_from_fx_fy(self.fx, self.fy, self.W, self.H)
        self._syncing = True
        try:
            self._sl_focal.set_val(f_mm)
            self._sl_fov_h.set_val(fov_h)
            self._sl_fov_v.set_val(fov_v)
        finally:
            self._syncing = False
        self._sync_title()

    def _sync_pose_sliders(self) -> None:
        """把选中相机的位姿写回 6 个位姿滑动条（带重入保护）。"""
        cam = self.cams[self._edit_idx]
        self._syncing_pose = True
        try:
            self._sl_cx.set_val(cam["C"][0])
            self._sl_cy.set_val(cam["C"][1])
            self._sl_cz.set_val(cam["C"][2])
            self._sl_yaw.set_val(cam["yaw"])
            self._sl_pitch.set_val(cam["pitch"])
            self._sl_roll.set_val(cam["roll"])
        finally:
            self._syncing_pose = False

    def _sync_title(self) -> None:
        fov_h, fov_v, f_mm = fov_from_fx_fy(self.fx, self.fy, self.W, self.H)
        cid = self.cams[self._edit_idx]["cid"]
        self._title.set_text(
            f"f = {f_mm:.2f} mm   FOV_h = {fov_h:.1f} deg   FOV_v = {fov_v:.1f} deg"
            f"   |   editing cam {cid}")

    def _update(self) -> None:
        """重建相机本体 + 覆盖锥（两者都随位姿/焦距变化），并适配视图。"""
        for arts in self._dyn_arts:
            for a in arts:
                a.remove()
        self._dyn_arts = []
        for cam in self.cams:
            if not self.enabled[cam["cid"]]:
                continue
            marker, axes = draw_camera_body(self.ax, cam)
            cone = draw_cone(self.ax, cam, self.fx, self.fy)
            self._dyn_arts.append([marker] + axes + cone)
        fit_view(self.ax, self.cams, self.table, self.enabled, self.fx, self.fy)
        self.fig.canvas.draw_idle()

    def run(self) -> None:
        print("[视场角] 鼠标：左拖=旋转 / 右拖=缩放 / 中拖=平移。")
        print("        左列滑动条改焦距/FOV；右侧单选「EDIT」选要移动/转动的相机，")
        print("        中间两列 X/Y/Z + yaw/pitch/roll 移动/转动它；SHOW 勾选渲染。")
        self.plt.show()


# =========================================================================
# 无头渲染（--offscreen）
# =========================================================================
def render_offscreen(cams, table, base_fx, base_fy, W, H, out_path: str,
                     width: int = 1280, height: int = 720) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(width / 100.0, height / 100.0), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    draw_table(ax, table)
    for cam in cams:
        draw_camera_body(ax, cam)
        draw_cone(ax, cam, base_fx, base_fy)
    setup_view(ax, table)
    fit_view(ax, cams, table, [True] * len(cams), base_fx, base_fy)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return out_path


# =========================================================================
# 入口
# =========================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="3D 相机视场角 / 焦距可视化（matplotlib）")
    ap.add_argument("--root", default=None,
                    help="项目根目录（默认脚本所在仓库根；其下 data/ 需含标定结果）")
    ap.add_argument("--offscreen", default=None, metavar="OUT.png",
                    help="不弹交互窗口，无头渲染一帧 PNG（用默认内参）")
    ap.add_argument("--width", type=int, default=1280, help="无头渲染宽度（像素）")
    ap.add_argument("--height", type=int, default=720, help="无头渲染高度（像素）")
    args = ap.parse_args()

    root = args.root or _ROOT
    intrinsics, extrinsics = load_rig(root)
    if len(extrinsics) == 0:
        print(f"[视场角] 错误：{root}/data/extrinsics/table_extrinsics.yaml 无相机外参")
        return 1

    table = Table3D()
    cams = build_cameras(intrinsics, extrinsics, table)
    ref = intrinsics[min(intrinsics)]
    base_fx = float(ref.K[0, 0])
    base_fy = float(ref.K[1, 1])
    W, H = int(ref.width), int(ref.height)

    fov_h, fov_v, f_mm = fov_from_fx_fy(base_fx, base_fy, W, H)
    print(f"[视场角] 加载 {len(cams)} 台相机，标定焦距 f={f_mm:.2f}mm，"
          f"FOV_h={fov_h:.1f}°，FOV_v={fov_v:.1f}°（{W}x{H}）")
    for cam in cams:
        C = cam["C"]
        print(f"  cam_{cam['cid']}: 光心=({C[0]:.2f},{C[1]:.2f},{C[2]:.2f})m "
              f"远平面深度={cam['depth']:.2f}m")

    if args.offscreen:
        out = render_offscreen(cams, table, base_fx, base_fy, W, H, args.offscreen,
                               args.width, args.height)
        print(f"[视场角] 已渲染 → {out}")
        return 0

    FovViewer(cams, table, base_fx, base_fy, W, H).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
