#!/usr/bin/env python
"""3D 视场角 / 焦距可视化（matplotlib 版，无 Open3D 依赖）。

对着 4 台相机的**桌面系外参**（``data/extrinsics/table_extrinsics.yaml``）和**内参**
（``data/calibration/cam_N.yaml``）用 matplotlib 的 ``mplot3d`` 建 3D 场景，再用
``Slider`` / ``CheckButtons`` 做交互：拖「焦距 / FOV 水平 / FOV 垂直」三个滑动条
实时看每台相机的覆盖范围（4 条角射线 + 桌面足迹四边形 + 半透明锥体）怎么收窄/放大，
勾选框选要渲染的相机。

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
        depth = float(np.clip(
            _FRUSTUM_DEPTH_FACTOR * np.linalg.norm(C - table_center),
            _FRUSTUM_DEPTH_MIN_M, _FRUSTUM_DEPTH_MAX_M))
        cams.append({
            "cid": cid,
            "Rw": Rw,
            "C": C,
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


def setup_view(ax, cams, table: Table3D) -> None:
    """按相机 + 球桌范围设坐标轴限与等比例，取一个舒服的斜俯视视角。"""
    xs = [c["C"][0] for c in cams] + [0.0, table.width]
    ys = [c["C"][1] for c in cams] + [0.0, table.length]
    zs = [c["C"][2] for c in cams] + [0.0, -table.height]
    m = 1.4
    xlim = (min(xs) - m, max(xs) + m)
    ylim = (min(ys) - m, max(ys) + m)
    zlim = (min(zs) - 0.4, max(zs) + 0.6)
    ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_zlim(zlim)
    ax.set_box_aspect((xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]))
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m]")
    ax.view_init(elev=22.0, azim=-58.0)


# =========================================================================
# 交互窗口
# =========================================================================
class FovViewer:
    """matplotlib 交互窗口：3D 场景 + 滑动条（焦距/FOV）+ 相机勾选 + 复位按钮。"""

    def __init__(self, cams: List[dict], table: Table3D,
                 base_fx: float, base_fy: float, W: int, H: int):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider, CheckButtons, Button
        self.plt = plt
        self.cams = cams
        self.table = table
        self.W, self.H = W, H
        self.fx = base_fx
        self.fy = base_fy
        self._base_fx = base_fx
        self._base_fy = base_fy
        self.enabled = [True] * len(cams)
        self._syncing = False
        self._cone_arts: Dict[int, List] = {}

        self.fig = plt.figure(figsize=(11.5, 8.5))
        self.ax = self.fig.add_subplot(111, projection="3d")

        # 静态层：球桌 + 相机本体（位置点 + 坐标架）
        draw_table(self.ax, table)
        self._cam_bodies = {cam["cid"]: draw_camera_body(self.ax, cam) for cam in cams}
        setup_view(self.ax, cams, table)

        # 初始覆盖锥
        for cam in cams:
            self._cone_arts[cam["cid"]] = draw_cone(self.ax, cam, self.fx, self.fy)

        # 底部控件
        fov_h, fov_v, f_mm = fov_from_fx_fy(base_fx, base_fy, W, H)
        self.fig.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.30)
        self.ax.set_position([0.02, 0.32, 0.74, 0.66])

        self._sl_focal = Slider(self.plt.axes([0.10, 0.20, 0.52, 0.03]),
                                "focal f [mm]", 1.0, 25.0, valinit=f_mm)
        self._sl_fov_h = Slider(self.plt.axes([0.10, 0.14, 0.52, 0.03]),
                                "FOV horiz [deg]", 5.0, 150.0, valinit=fov_h)
        self._sl_fov_v = Slider(self.plt.axes([0.10, 0.08, 0.52, 0.03]),
                                "FOV vert [deg]", 5.0, 150.0, valinit=fov_v)
        self._sl_focal.on_changed(self._on_focal)
        self._sl_fov_h.on_changed(self._on_fov_h)
        self._sl_fov_v.on_changed(self._on_fov_v)

        self._check_labels = [f"cam {cam['cid']} ({_CAM_COLOR_NAMES[cam['cid'] % 4]})"
                              for cam in cams]
        self._check = CheckButtons(self.plt.axes([0.80, 0.06, 0.17, 0.20]),
                                   self._check_labels, [True] * len(cams))
        self._check.on_clicked(self._on_check)

        self._btn_reset = Button(self.plt.axes([0.10, 0.02, 0.14, 0.04]), "Reset")
        self._btn_reset.on_clicked(self._on_reset)

        self._title = self.fig.suptitle("", fontsize=11)
        self._sync_title()

    # ---- 回调 -------------------------------------------------------
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

    def _on_check(self, label: str) -> None:
        i = self._check_labels.index(label)
        self.enabled[i] = not self.enabled[i]
        self._update()

    def _on_reset(self, event=None) -> None:
        self.fx, self.fy = self._base_fx, self._base_fy
        self._sync_sliders()
        self._update()

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

    def _sync_title(self) -> None:
        fov_h, fov_v, f_mm = fov_from_fx_fy(self.fx, self.fy, self.W, self.H)
        self._title.set_text(
            f"f = {f_mm:.2f} mm   FOV_h = {fov_h:.1f} deg   FOV_v = {fov_v:.1f} deg")

    def _update(self) -> None:
        """重建覆盖锥 + 相机本体显隐。"""
        for arts in self._cone_arts.values():
            for a in arts:
                a.remove()
        self._cone_arts = {}
        for cam in self.cams:
            cid = cam["cid"]
            on = self.enabled[cid]
            marker, axes = self._cam_bodies[cid]
            marker.set_visible(on)
            for a in axes:
                a.set_visible(on)
            if on:
                self._cone_arts[cid] = draw_cone(self.ax, cam, self.fx, self.fy)
        self.fig.canvas.draw_idle()

    def run(self) -> None:
        print("[视场角] 鼠标：左拖=旋转 / 右拖=缩放 / 中拖=平移。"
              "底部滑动条改焦距/FOV，勾选框选要渲染的相机。")
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
    setup_view(ax, cams, table)
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
