#!/usr/bin/env python
"""3D 视场角 / 焦距可视化：相机朝向 + 射线覆盖区域 + 球桌，滑动条控制焦距与 FOV H/V。

对着 4 台相机的**桌面系外参**（``data/extrinsics/table_extrinsics.yaml``）和**内参**
（``data/calibration/cam_N.yaml``）把整场建到 Open3D 场景里，然后用两个「焦距 / FOV」
滑动条做 what-if：改镜头焦距（或直接改水平/垂直视场角），实时看每台相机的**射线覆盖
区域**（视锥 = 光心发出的射线束 + 远平面窗口）怎么收窄/放大、覆盖到球桌哪里。

物理约定（与重建管道一致）：
- 世界系 = 桌面系：原点在桌角标记，X 短边(宽 1.525m)、Y 长边(长 2.74m)、Z 竖直向上，桌面 z=0。
- 外参 ``(R, t)`` 是「桌面系 -> 相机系」：``X_cam = R @ X_world + t``。相机光心
  世界坐标 ``C = -R^T @ t``。
- 内参 ``K = [[fx,0,cx],[0,fy,cy],[0,0,1]]``，分辨率 1440×1080，像元 3.45µm（Sony IMX273）。
  焦距与视场角互推：``FOV_h = 2·atan(W/(2fx))``、``FOV_v = 2·atan(H/(2fy))``、
  ``f[mm] = fx · 像元``。
- 三个滑动条里**焦距是各向同性主控**（拖它 → fx=fy，两 FOV 跟随）；「FOV 水平 /
  垂直」各自独立改 fx / fy（可做成各向异性、非方形像素）。拖动任意一个，其余两个
  的显示值自动同步，无矛盾。

射线区域只画成 ``LineSet``（光心 → 远平面网格点的射线束 + 远平面窗口线框），滑动时
只 remove+add 这些线条、**不重挂三角网格**——绕开 Filament 里 TriangleMesh 反复
remove+add 会累积引擎级残留、~1 万次段错误的坑（见 recon_player.py 的教训）。

用法：
    python scripts/visualize_fov.py                      # 交互窗口（滑动条）
    python scripts/visualize_fov.py --offscreen out.png  # 无头渲染一帧 PNG（默认内参）

依赖：``open3d 0.19``（``tt`` 环境）。交互窗口需要显示；``--offscreen`` 走 EGL 无头。
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

# 引导 import：本项目脚本统一把 src/ 加进 sys.path 找 tabletennis 包。
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if os.path.isdir(os.path.join(_ROOT, "src")):
    sys.path.insert(0, os.path.join(_ROOT, "src"))

import open3d as o3d  # noqa: E402
import open3d.visualization.rendering  # noqa: F401,E402  注册 o3d.visualization.rendering

from tabletennis.calibration.extrinsics import load_extrinsics  # noqa: E402
from tabletennis.calibration.intrinsics import load_intrinsics  # noqa: E402
from tabletennis.core.types import CameraExtrinsics, CameraIntrinsics, Table3D  # noqa: E402

# ---- 常量 ----
_SENSOR_PIXEL_MM = 0.00345          # Sony IMX273 像元 3.45µm
_CAM_COLORS = [
    [1.00, 0.35, 0.35],   # 红
    [0.35, 0.65, 1.00],   # 蓝
    [0.45, 1.00, 0.45],   # 绿
    [1.00, 0.85, 0.25],   # 黄
]
_TABLE_TOP_COLOR = [0.10, 0.45, 0.30]
_TABLE_EDGE_COLOR = [0.05, 0.28, 0.20]
_TABLE_LINE_COLOR = [0.15, 0.85, 0.30]
_TABLE_STRUCT_COLOR = [0.55, 0.55, 0.55]
_TABLE_NET_COLOR = [0.20, 0.85, 0.90]
_FLOOR_COLOR = [0.24, 0.24, 0.26]
_GRID_COLOR = [0.35, 0.35, 0.42]

# 射线束远平面（成像窗口）距离：= clamp(因子 × 相机到球桌中心距离)。只依赖相机位置，
# 不随 FOV 变，保证 FOV 变化时窗口大小直接可比。
_FRUSTUM_DEPTH_FACTOR = 1.3
_FRUSTUM_DEPTH_MIN_M = 2.5
_FRUSTUM_DEPTH_MAX_M = 6.0
# 射线束在成像面上的网格密度（点行列数）
_RAY_GX = 7
_RAY_GY = 5


# =========================================================================
# 数据加载与几何计算（纯 numpy，无 GUI，可无头单测）
# =========================================================================
def load_rig(root: str) -> Tuple[Dict[int, CameraIntrinsics], Dict[int, CameraExtrinsics]]:
    """读内参 + 桌面系外参，返回 ``(intrinsics, extrinsics)``（key=cam_id）。"""
    intr_dir = os.path.join(root, "data", "calibration")
    ext_path = os.path.join(root, "data", "extrinsics", "table_extrinsics.yaml")
    intrinsics: Dict[int, CameraIntrinsics] = {}
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


def fx_fy_from_fov(fov_h_deg: float, fov_v_deg: float, W: int, H: int) -> Tuple[float, float]:
    """由视场角（度）推焦距（像素）。"""
    fx = W / (2.0 * math.tan(math.radians(fov_h_deg) / 2.0))
    fy = H / (2.0 * math.tan(math.radians(fov_v_deg) / 2.0))
    return fx, fy


def camera_center(ext: CameraExtrinsics) -> np.ndarray:
    """相机光心世界坐标（桌面系）``C = -R^T @ t``。"""
    R = np.asarray(ext.R, dtype=np.float64)
    t = np.asarray(ext.t, dtype=np.float64).reshape(3)
    return -(R.T @ t)


def image_plane_points(K: np.ndarray, W: int, H: int, depth: float,
                       gx: int = _RAY_GX, gy: int = _RAY_GY) -> np.ndarray:
    """相机系里、z=depth 成像面上的 ``gx×gy`` 网格点坐标（射线方向归一后乘 depth）。"""
    Kinv = np.linalg.inv(K)
    us = np.linspace(0, W, gx)
    vs = np.linspace(0, H, gy)
    uv = np.array([[u, v, 1.0] for v in vs for u in us], dtype=np.float64)
    pts = (Kinv @ uv.T).T        # (N,3)，z=1
    return pts * depth           # z=depth


class SceneModel:
    """把「桌面系外参 + 可调焦距」转成 Open3D 场景几何（静态 + 射线区域两层）。

    静态层（球桌/地面/相机光心小球/坐标架）只建一次；射线层随 fx/fy 变化，
    :meth:`set_frustums` remove+add 对应的 LineSet。
    """

    def __init__(self, intrinsics: Dict[int, CameraIntrinsics],
                 extrinsics: Dict[int, CameraExtrinsics], table: Table3D):
        self.table = table
        # 每台相机固化：光心、Rw(相机->世界)、cx/cy、图像尺寸、颜色、远平面深度
        self.cams: List[dict] = []
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
            self.cams.append({
                "cid": cid,
                "Rw": Rw,
                "C": C,
                "cx": float(K.K[0, 2]),
                "cy": float(K.K[1, 2]),
                "W": K.width,
                "H": K.height,
                "depth": depth,
                "color": np.asarray(_CAM_COLORS[cid % len(_CAM_COLORS)], np.float64),
            })
        # 默认焦距 = 相机 0 的标定值（各相机同型号、焦距几乎一致）
        ref = intrinsics[min(intrinsics)]
        self.base_fx = float(ref.K[0, 0])
        self.base_fy = float(ref.K[1, 1])
        self.W = ref.width
        self.H = ref.height

    def default_fov(self) -> Tuple[float, float, float]:
        return fov_from_fx_fy(self.base_fx, self.base_fy, self.W, self.H)

    # ---- 静态层 ------------------------------------------------------
    def add_static(self, scene) -> None:
        self._add_table(scene)
        self._add_ground_grid(scene)
        self._add_axes(scene, np.zeros(3), "world_axes", 0.25)
        for cam in self.cams:
            self._add_camera_body(scene, cam)

    def _add_table(self, scene) -> None:
        t = self.table
        W, L, H = t.width, t.length, t.height

        # 桌面（实心，接收光照）
        c = t.top_corners
        surf = o3d.geometry.TriangleMesh()
        surf.vertices = o3d.utility.Vector3dVector(c)
        surf.triangles = o3d.utility.Vector3iVector(np.array([[0, 1, 2], [0, 2, 3]], np.int32))
        surf.compute_vertex_normals()
        scene.add_geometry("table_top", surf, _lit_material(_TABLE_TOP_COLOR, 0.6))

        # 桌面下沿垂面（视觉厚度）
        edge = o3d.geometry.TriangleMesh()
        v = np.vstack([c, c + np.array([[0, 0, -0.03]], np.float64)])
        tri = np.array([[0, 4, 5], [0, 5, 1], [1, 5, 6], [1, 6, 2],
                        [2, 6, 7], [2, 7, 3], [3, 7, 4], [3, 4, 0]], np.int32)
        edge.vertices = o3d.utility.Vector3dVector(v)
        edge.triangles = o3d.utility.Vector3iVector(tri)
        edge.compute_vertex_normals()
        scene.add_geometry("table_edge", edge, _lit_material(_TABLE_EDGE_COLOR, 0.9))

        # 地面（z=-H，承接球桌与射线落点参考）
        m = 0.9
        fv = np.array([[-m, -m, -H], [W + m, -m, -H], [W + m, L + m, -H], [-m, L + m, -H]],
                      np.float64)
        floor = o3d.geometry.TriangleMesh()
        floor.vertices = o3d.utility.Vector3dVector(fv)
        floor.triangles = o3d.utility.Vector3iVector(np.array([[0, 1, 2], [0, 2, 3]], np.int32))
        floor.compute_vertex_normals()
        scene.add_geometry("floor", floor, _lit_material(_FLOOR_COLOR, 0.95))

        # 线框：桌面边 / 桌腿 / 地面框 / 球网
        segs = t.segments()
        for key, color, w in [
                ("top", _TABLE_LINE_COLOR, 3.0), ("legs", _TABLE_STRUCT_COLOR, 3.0),
                ("floor", _TABLE_STRUCT_COLOR, 2.0), ("net", _TABLE_NET_COLOR, 3.0)]:
            pts = np.asarray(segs[key], np.float64).reshape(-1, 3)
            ls = o3d.geometry.LineSet()
            ls.points = o3d.utility.Vector3dVector(pts)
            ls.lines = o3d.utility.Vector2iVector(
                np.arange(len(pts)).reshape(-1, 2).astype(np.int32))
            _add_line_geo(scene, f"line_{key}", ls, np.asarray(color, np.float64), w)

    def _add_ground_grid(self, scene) -> None:
        """地面网格（z=-height），辅助空间定位与射线落点判断。"""
        t = self.table
        z = -t.height
        step = 0.5
        x0, x1 = -0.3, t.width + 0.3
        y0, y1 = -0.3, t.length + 0.3
        pts: List[List[float]] = []
        lines: List[List[int]] = []
        for y in np.arange(np.floor(y0 / step) * step, y1 + 1e-6, step):
            pts.append([x0, y, z]); pts.append([x1, y, z])
            lines.append([len(pts) - 2, len(pts) - 1])
        for x in np.arange(np.floor(x0 / step) * step, x1 + 1e-6, step):
            pts.append([x, y0, z]); pts.append([x, y1, z])
            lines.append([len(pts) - 2, len(pts) - 1])
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(np.asarray(pts, np.float64))
        ls.lines = o3d.utility.Vector2iVector(np.asarray(lines, np.int32))
        _add_line_geo(scene, "ground_grid", ls, np.asarray(_GRID_COLOR, np.float64), 1.0)

    def _add_camera_body(self, scene, cam) -> None:
        cid = cam["cid"]
        C = cam["C"]
        color = cam["color"]
        sph = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
        sph.translate(C)
        scene.add_geometry(f"cam_{cid}_sph", sph, _lit_material(color, 0.5))
        self._add_axes(scene, C, f"cam_{cid}_axes", 0.18, cam["Rw"])

    def _add_axes(self, scene, C: np.ndarray, name: str, size: float,
                  Rw: Optional[np.ndarray] = None) -> None:
        """短坐标架：X 红 / Y 绿 / Z 蓝；``Rw`` 给定时按相机朝向旋转。"""
        pts = np.array([[0, 0, 0], [size, 0, 0], [0, 0, 0], [0, size, 0],
                        [0, 0, 0], [0, 0, size]], np.float64)
        if Rw is not None:
            pts = (Rw @ pts.T).T
        pts = pts + np.asarray(C, np.float64).reshape(1, 3)
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(pts)
        ls.lines = o3d.utility.Vector2iVector(np.array([[0, 1], [2, 3], [4, 5]], np.int32))
        ls.colors = o3d.utility.Vector3dVector(
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float64))
        mr = o3d.visualization.rendering.MaterialRecord()
        mr.shader = "unlitLine"
        mr.line_width = 2.0
        scene.add_geometry(name, ls, mr)

    # ---- 射线区域层 ---------------------------------------------------
    def set_frustums(self, scene, fx: float, fy: float) -> None:
        """按当前 fx/fy 重建所有相机的射线覆盖区域（remove+add 各 LineSet）。"""
        self.remove_frustums(scene)
        for cam in self.cams:
            ls = self._ray_region(cam, fx, fy)
            _add_line_geo(scene, f"cam_{cam['cid']}_rays", ls, cam["color"], 1.5)

    def remove_frustums(self, scene) -> None:
        for cam in self.cams:
            scene.remove_geometry(f"cam_{cam['cid']}_rays")

    def _ray_region(self, cam: dict, fx: float, fy: float) -> "object":
        """单相机射线覆盖区域 LineSet：光心发出的射线束 + 远平面窗口线框。

        世界系里：光心 ``C``，远平面网格点 ``X_w = Rw @ (p_cam - t)``。射线束覆盖的
        就是该相机在远平面上的可视窗口，FOV 变大 → 窗口变大 → 覆盖球桌范围变宽。
        """
        W, H, depth = cam["W"], cam["H"], cam["depth"]
        Rw = cam["Rw"]
        K = np.array([[fx, 0, cam["cx"]], [0, fy, cam["cy"]], [0, 0, 1]], np.float64)
        grid_cam = image_plane_points(K, W, H, depth, _RAY_GX, _RAY_GY)  # (gx*gy,3) 相机系
        # 相机系 -> 世界系：X_w = Rw @ (X_cam - t)，其中 C = -Rw @ t => t = -Rw^T @ C
        t = -(Rw.T @ cam["C"])
        grid_world = (Rw @ (grid_cam - t).T).T
        apex = cam["C"].reshape(1, 3)
        pts = np.vstack([apex, grid_world])     # 点 0 = 光心，1..N = 远平面网格

        gx, gy = _RAY_GX, _RAY_GY
        lines: List[List[int]] = []
        # 射线束：光心 -> 每个网格点
        for i in range(gx * gy):
            lines.append([0, 1 + i])
        # 远平面窗口线框：横向（每行） + 纵向（每列）填满成像面
        for j in range(gy):
            for i in range(gx - 1):
                lines.append([1 + j * gx + i, 1 + j * gx + i + 1])
        for i in range(gx):
            for j in range(gy - 1):
                lines.append([1 + j * gx + i, 1 + (j + 1) * gx + i])

        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(pts)
        ls.lines = o3d.utility.Vector2iVector(np.asarray(lines, np.int32))
        return ls


def _lit_material(color, roughness: float = 0.7):
    mr = o3d.visualization.rendering.MaterialRecord()
    mr.base_color = [float(color[0]), float(color[1]), float(color[2]), 1.0]
    mr.base_roughness = float(roughness)
    mr.base_metallic = 0.0
    return mr


def _add_line_geo(scene, name, ls, color, width: float) -> None:
    n = len(ls.lines)
    if n:
        ls.colors = o3d.utility.Vector3dVector(
            np.tile(np.asarray(color, np.float64), (n, 1)))
    mr = o3d.visualization.rendering.MaterialRecord()
    mr.shader = "unlitLine"
    mr.line_width = width
    scene.add_geometry(name, ls, mr)


def _scene_bounds_center(model: SceneModel):
    t = model.table
    lo = np.array([-0.6, -0.6, -t.height - 0.2], np.float64)
    hi = np.array([t.width + 0.6, t.length + 0.6, 2.2], np.float64)
    bounds = o3d.geometry.AxisAlignedBoundingBox(lo, hi)
    center = np.array([t.width / 2.0, t.length / 2.0, -t.height / 3.0], np.float64)
    return bounds, center


# =========================================================================
# 无头渲染（--offscreen，EGL）
# =========================================================================
def render_offscreen(model: SceneModel, out_path: str, fx: float, fy: float,
                     width: int = 1280, height: int = 720) -> str:
    """用 OffscreenRenderer 渲染一帧 PNG（默认焦距/可指定 fx,fy），返回输出路径。"""
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    scene = renderer.scene
    scene.set_background(np.array([0.10, 0.11, 0.14, 1.0], np.float32))
    model.add_static(scene)
    model.set_frustums(scene, fx, fy)

    _, center = _scene_bounds_center(model)
    t = model.table
    eye = center + np.array([t.width, t.length, 0.8], np.float64) * 0.55
    renderer.setup_camera(50.0, center, eye, np.array([0.0, 0.0, 1.0], np.float64))

    img = renderer.render_to_image()
    o3d.io.write_image(out_path, img)
    return out_path


# =========================================================================
# 交互窗口（gui.Application + SceneWidget + Slider）
# =========================================================================
class FovViewerApp:
    """主窗口 = 3D 场景；副窗口 = 焦距 / FOV 滑动条 + 读值 + 复位按钮。"""

    def __init__(self, model: SceneModel, width: int = 1100, height: int = 760):
        import open3d.visualization.gui as gui
        self.gui = gui
        self.model = model
        self.fx = model.base_fx
        self.fy = model.base_fy
        self._syncing = False
        self._done = False

        self.app = gui.Application.instance
        self.app.initialize()

        # 主窗口：3D 场景
        self.win = self.app.create_window("相机视场角 / 焦距可视化", width, height)
        self.widget = gui.SceneWidget()
        self.win.add_child(self.widget)
        self.scene = o3d.visualization.rendering.Open3DScene(self.win.renderer)
        self.scene.set_background([0.10, 0.11, 0.14, 1.0])
        self.widget.scene = self.scene

        model.add_static(self.scene)
        model.set_frustums(self.scene, self.fx, self.fy)

        bounds, center = _scene_bounds_center(model)
        self.widget.setup_camera(60.0, bounds, center)
        self.widget.set_view_controls(self.gui.SceneWidget.Controls.ROTATE_CAMERA)

        # 副窗口：控制面板
        self.panel_win = self.app.create_window("镜头参数", 360, 260)
        self.panel_win.set_on_close(lambda: setattr(self, "_done", True))
        self.win.set_on_close(lambda: setattr(self, "_done", True))
        self._build_panel()
        self._sync_sliders_from_state()

    def _build_panel(self) -> None:
        gui = self.gui
        em = gui.Margins(12, 10, 12, 10)
        panel = gui.Vert(4, em)

        # 焦距滑动条（各向同性主控）
        panel.add_child(gui.Label("焦距 f [mm]（拖 = fx=fy，两 FOV 跟随）"))
        self._sl_focal = gui.Slider(gui.Slider.DOUBLE)
        self._sl_focal.set_limits(1.0, 25.0)
        self._sl_focal.set_on_value_changed(self._on_focal)
        panel.add_child(self._sl_focal)

        # 水平 FOV
        panel.add_child(gui.Label("视场角 水平 FOV_h [deg]"))
        self._sl_fov_h = gui.Slider(gui.Slider.DOUBLE)
        self._sl_fov_h.set_limits(5.0, 150.0)
        self._sl_fov_h.set_on_value_changed(self._on_fov_h)
        panel.add_child(self._sl_fov_h)

        # 垂直 FOV
        panel.add_child(gui.Label("视场角 垂直 FOV_v [deg]"))
        self._sl_fov_v = gui.Slider(gui.Slider.DOUBLE)
        self._sl_fov_v.set_limits(5.0, 150.0)
        self._sl_fov_v.set_on_value_changed(self._on_fov_v)
        panel.add_child(self._sl_fov_v)

        # 读值标签
        self._readout = gui.Label("")
        panel.add_child(self._readout)

        # 复位按钮
        btn = gui.Button("复位到标定内参")
        btn.set_on_clicked(self._on_reset)
        btn.horizontal_padding_em = 0.5
        panel.add_child(btn)

        self.panel_win.add_child(panel)

    # ---- 滑动条回调 ------------------------------------------------
    def _on_focal(self, v: float) -> None:
        if self._syncing:
            return
        self.fx = self.fy = v / _SENSOR_PIXEL_MM
        self._sync_sliders_from_state()
        self._apply_fov()

    def _on_fov_h(self, v: float) -> None:
        if self._syncing:
            return
        self.fx = self.model.W / (2.0 * math.tan(math.radians(v) / 2.0))
        self._sync_sliders_from_state()
        self._apply_fov()

    def _on_fov_v(self, v: float) -> None:
        if self._syncing:
            return
        self.fy = self.model.H / (2.0 * math.tan(math.radians(v) / 2.0))
        self._sync_sliders_from_state()
        self._apply_fov()

    def _on_reset(self) -> None:
        self.fx = self.model.base_fx
        self.fy = self.model.base_fy
        self._sync_sliders_from_state()
        self._apply_fov()

    def _sync_sliders_from_state(self) -> None:
        """把当前 fx/fy 反推的 f/FOV 值写回三个滑动条（带重入保护）。"""
        fov_h, fov_v, f_mm = fov_from_fx_fy(self.fx, self.fy, self.model.W, self.model.H)
        self._syncing = True
        try:
            self._sl_focal.double_value = f_mm
            self._sl_fov_h.double_value = fov_h
            self._sl_fov_v.double_value = fov_v
        finally:
            self._syncing = False
        self._readout.text = (
            f"f = {f_mm:.2f} mm  (fx={self.fx:.0f} fy={self.fy:.0f} px)\n"
            f"FOV_h = {fov_h:.1f}°   FOV_v = {fov_v:.1f}°"
        )

    def _apply_fov(self) -> None:
        self.model.set_frustums(self.scene, self.fx, self.fy)
        self.widget.force_redraw()
        fov_h, fov_v, f_mm = fov_from_fx_fy(self.fx, self.fy, self.model.W, self.model.H)
        self.win.title = (f"相机视场角 / 焦距可视化  |  f={f_mm:.2f}mm  "
                          f"FOV_h={fov_h:.1f}°  FOV_v={fov_v:.1f}°")

    def run(self) -> None:
        print("[视场角] 鼠标：左拖=旋转 / 右拖=平移 / 滚轮=缩放。"
              "在「镜头参数」窗里拖滑动条改焦距/FOV。")
        try:
            while not self._done and self.app.run_one_tick():
                pass
        finally:
            try:
                self.win.close()
            except Exception:  # noqa: BLE001
                pass
            try:
                self.panel_win.close()
            except Exception:  # noqa: BLE001
                pass


# =========================================================================
# 入口
# =========================================================================
def main() -> int:
    ap = argparse.ArgumentParser(description="3D 相机视场角 / 焦距可视化")
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
    model = SceneModel(intrinsics, extrinsics, table)
    fov_h, fov_v, f_mm = model.default_fov()
    print(f"[视场角] 加载 {len(model.cams)} 台相机，标定焦距 f={f_mm:.2f}mm，"
          f"FOV_h={fov_h:.1f}°，FOV_v={fov_v:.1f}°（{model.W}x{model.H}）")
    for cam in model.cams:
        C = cam["C"]
        print(f"  cam_{cam['cid']}: 光心=({C[0]:.2f},{C[1]:.2f},{C[2]:.2f})m "
              f"远平面深度={cam['depth']:.2f}m")

    if args.offscreen:
        out = render_offscreen(model, args.offscreen, model.base_fx, model.base_fy,
                               args.width, args.height)
        print(f"[视场角] 已渲染 → {out}")
        return 0

    FovViewerApp(model).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
