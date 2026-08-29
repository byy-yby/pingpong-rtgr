"""3D 场景可视化（Open3D）：把整场（相机 + 标准尺寸球桌）建模到独立 3D 窗口。

``scripts/live_control.py`` 按 T 识别球桌后，用一个后台线程跑 Open3D 的
``Visualizer``（``poll_events`` / ``update_renderer``），与 OpenCV 预览窗口并存：

- 球桌：按桌面边框 + 标准尺寸（2.74 × 1.525 × 0.76 m）建模——桌面 + 4 条腿 + 球网。
- 相机：每台相机的位姿（世界系 = 桌面系）画成彩色视锥 + 局部坐标架 + 位置小球。

约定：Open3D 几何对象只在渲染线程内创建 / 使用；主线程在启动前一次性
``build_scene``，之后只跑渲染循环，避免跨线程操作 OpenGL 资源。open3d 延迟
import，未启用 3D 场景时不会强制依赖。
"""
from __future__ import annotations

import threading
import time
from typing import Dict, List, Optional

import numpy as np

from ..core.types import CameraExtrinsics, CameraIntrinsics, Skeleton3D, Table3D
from ..vision.skeleton import get_skeleton

# 每台相机的视锥 / 位置小球颜色（按逻辑索引取模）
_CAM_COLORS = [
    [1.00, 0.35, 0.35],  # 红
    [0.35, 0.65, 1.00],  # 蓝
    [0.45, 1.00, 0.45],  # 绿
    [1.00, 0.85, 0.25],  # 黄
]
_TABLE_TOP_COLOR = [0.15, 0.85, 0.30]
_TABLE_STRUCT_COLOR = [0.55, 0.55, 0.55]
_TABLE_NET_COLOR = [0.20, 0.85, 0.90]
_TABLE_SURFACE_COLOR = [0.10, 0.45, 0.30]
_GRID_COLOR = [0.35, 0.35, 0.42]

# 多个球员骨架的骨骼颜色（按人取模）
_SKELETON_COLORS = [
    [1.00, 0.45, 0.10],  # 橙
    [0.20, 0.85, 1.00],  # 青
    [1.00, 0.30, 0.85],  # 品红
    [0.85, 1.00, 0.20],  # 黄绿
    [0.60, 0.85, 1.00],  # 淡蓝
    [1.00, 0.80, 0.30],  # 金
]

# 默认视角与键盘控制步长
_DEFAULT_ZOOM = 0.65   # 初始缩放（配合 reset_view_point 的包围盒，给出舒服的取景）
_PAN_STEP = 60.0       # W/A/S/D 平移步长（像素）
_ROT_STEP = 8.0        # 方向键旋转步长（度）
_ZOOM_STEP = 1.12      # +/- 缩放倍率


def _o3d():
    """延迟 import open3d（仅 3D 场景启用时）。"""
    import open3d as o3d
    return o3d


def _line_set(points, lines, color):
    """构造一条 LineSet：``points`` (N,3)，``lines`` (M,2) 点索引，``color`` 单一颜色。"""
    o3d = _o3d()
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    ls.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    ls.colors = o3d.utility.Vector3dVector(
        np.tile(np.asarray(color, dtype=np.float64), (len(lines), 1))
    )
    return ls


def _segments_line_set(segs, color):
    """把 ``(M,2,3)`` 的线段组转成 LineSet（2M 个点、M 条线）。"""
    segs = np.asarray(segs, dtype=np.float64)
    pts = segs.reshape(-1, 3)
    lines = [[2 * i, 2 * i + 1] for i in range(len(segs))]
    return _line_set(pts, lines, color)


def _conf_color(conf: float) -> List[float]:
    """置信度 -> 颜色：低置信偏红、高置信偏绿（RGB）。"""
    c = float(np.clip(conf, 0.0, 1.0))
    return [1.0 - c, c, 0.20]


def _skeleton_geometry(skel: Skeleton3D, edges: List, bone_color: List[float]):
    """把 :class:`Skeleton3D` 转成骨架几何数据。

    Returns:
        ``(pts, lines, line_colors, joint_pts, joint_colors)``：
        - pts: (M,3) 有效关键点坐标
        - lines: (L,2) 骨骼连线的点索引（两端都有效才连）
        - line_colors: (L,3)
        - joint_pts / joint_colors: 有效关键点及其按置信度着色
    """
    kp = np.asarray(skel.keypoints, dtype=np.float64)
    n = len(kp)
    valid = np.isfinite(kp).all(axis=1)

    idx_map = np.full(n, -1, dtype=np.int32)
    pts: List[np.ndarray] = []
    joint_colors: List[List[float]] = []
    for j in range(n):
        if not valid[j]:
            continue
        idx_map[j] = len(pts)
        pts.append(kp[j])
        conf = float(skel.confidence[j]) if j < len(skel.confidence) else 0.0
        joint_colors.append(_conf_color(conf))

    lines: List[List[int]] = []
    line_colors: List[List[float]] = []
    for a, b in edges:
        if a < n and b < n and idx_map[a] >= 0 and idx_map[b] >= 0:
            lines.append([int(idx_map[a]), int(idx_map[b])])
            line_colors.append(bone_color)

    pts_arr = np.asarray(pts, dtype=np.float64).reshape(-1, 3) if pts else np.zeros((0, 3))
    lines_arr = np.asarray(lines, dtype=np.int32).reshape(-1, 2) if lines else np.zeros((0, 2), dtype=np.int32)
    line_colors_arr = np.asarray(line_colors, dtype=np.float64).reshape(-1, 3) if line_colors else np.zeros((0, 3))
    joint_colors_arr = np.asarray(joint_colors, dtype=np.float64).reshape(-1, 3) if joint_colors else np.zeros((0, 3))
    return pts_arr, lines_arr, line_colors_arr, pts_arr, joint_colors_arr


class SceneViewer3D:
    """Open3D 3D 场景查看器（后台线程渲染）。

    用法：``build_scene(...)`` 构建几何 → ``start()`` 弹窗渲染 → ``close()`` 关闭。
    """

    def __init__(self, title: str = "Scene 3D", width: int = 900, height: int = 700):
        self._title = title
        self._width = width
        self._height = height
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._geometries: List = []
        self._lookat = np.zeros(3)

        # 实时骨架层：预分配的 LineSet + PointCloud（每人一对），跨线程传递最新骨架
        self._skeleton_lock = threading.Lock()
        self._skeleton_edges: List = []
        self._skeleton_geoms: List = []      # [(lineset, pointcloud), ...]
        self._latest_skeletons: List = []
        self._skeletons_dirty = False

    # ------------------------------------------------------------------
    # 场景构建（主线程，start 前调用一次）
    # ------------------------------------------------------------------
    def build_scene(
        self,
        table: Table3D,
        camera_poses: Dict[int, CameraExtrinsics],
        intrinsics: Dict[int, CameraIntrinsics],
        frustum_scale: float = 1.0,
    ) -> None:
        """构建整场几何：球桌 + 相机视锥 + 地面网格 + 世界坐标架。"""
        self._lookat = np.array(
            [table.width / 2.0, table.length / 2.0, -table.height / 2.0]
        )
        self._geometries = []
        self._geometries += self._table_geometries(table)
        self._geometries += self._camera_geometries(camera_poses, intrinsics, frustum_scale)
        self._geometries += self._ground_geometries(table)

    def build_cameras_scene(
        self,
        camera_poses: Dict[int, CameraExtrinsics],
        intrinsics: Dict[int, CameraIntrinsics],
        frustum_scale: float = 1.0,
        ground_radius: float = 2.0,
    ) -> None:
        """只建相机视锥 + 地面网格 + 世界坐标架（不含球桌），供外参调试。

        世界系 = 参考相机，所以地面网格画在 z=0（参考相机光心平面），方便直接
        看清四台相机的相对位置与朝向是否合理。
        """
        self._lookat = np.zeros(3)
        self._geometries = []
        self._geometries += self._camera_geometries(camera_poses, intrinsics, frustum_scale)
        self._geometries += self._ground_grid(0.0, ground_radius)
        self._geometries.append(_o3d().geometry.TriangleMesh.create_coordinate_frame(size=0.3))

    def _table_geometries(self, table: Table3D) -> List:
        o3d = _o3d()
        segs = table.segments()
        geos = []

        # 桌面表面（填充三角形，双面渲染，直观）
        surf = o3d.geometry.TriangleMesh()
        surf.vertices = o3d.utility.Vector3dVector(table.top_corners)
        surf.triangles = o3d.utility.Vector3iVector(
            np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        )
        surf.paint_uniform_color(_TABLE_SURFACE_COLOR)
        surf.compute_vertex_normals()
        geos.append(surf)

        geos.append(_segments_line_set(segs["top"], _TABLE_TOP_COLOR))
        geos.append(_segments_line_set(segs["legs"], _TABLE_STRUCT_COLOR))
        geos.append(_segments_line_set(segs["floor"], _TABLE_STRUCT_COLOR))
        geos.append(_segments_line_set(segs["net"], _TABLE_NET_COLOR))

        # 世界原点坐标架（= 桌面原点角）
        geos.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25))
        return geos

    def _camera_geometries(
        self,
        camera_poses: Dict[int, CameraExtrinsics],
        intrinsics: Dict[int, CameraIntrinsics],
        frustum_scale: float,
    ) -> List:
        o3d = _o3d()
        geos = []
        for cid in sorted(camera_poses):
            K = intrinsics.get(cid)
            if K is None:
                continue
            ext = camera_poses[cid]
            color = np.asarray(_CAM_COLORS[cid % len(_CAM_COLORS)], dtype=np.float64)

            R = np.asarray(ext.R, dtype=np.float64)
            t = np.asarray(ext.t, dtype=np.float64).reshape(3)
            Rw = R.T  # 相机 -> 世界旋转
            C = -Rw @ t  # 相机光心世界坐标

            # 视锥：顶点 = 光心，底面 = 距离 frustum_scale 处的成像面（由内参反投影）
            W, H = K.width, K.height
            Kinv = np.linalg.inv(K.K)
            uv = np.array([[0, 0, 1], [W, 0, 1], [W, H, 1], [0, H, 1]], dtype=np.float64)
            base_cam = (Kinv @ uv.T).T * frustum_scale  # z = frustum_scale
            pts_cam = np.vstack([np.zeros(3), base_cam])  # (5, 3)
            pts_world = (Rw @ (pts_cam - t).T).T  # (5, 3)
            lines = [[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]]
            geos.append(_line_set(pts_world, lines, color))

            # 相机位置小球 + 局部坐标架（显示朝向）
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
            sphere.paint_uniform_color(color)
            sphere.translate(C)
            geos.append(sphere)

            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.18)
            frame.rotate(Rw, center=(0.0, 0.0, 0.0))
            frame.translate(C)
            geos.append(frame)
        return geos

    def _ground_grid(self, z: float, radius: float, step: float = 0.5) -> List:
        """以原点为中心、z 高度的地面网格（相机-only 调试用）。"""
        pts: List = []
        lines: List = []
        for v in np.arange(-radius, radius + 1e-6, step):
            pts.append([v, -radius, z])
            pts.append([v, radius, z])
            lines.append([len(pts) - 2, len(pts) - 1])
        for v in np.arange(-radius, radius + 1e-6, step):
            pts.append([-radius, v, z])
            pts.append([radius, v, z])
            lines.append([len(pts) - 2, len(pts) - 1])
        return [_line_set(pts, lines, _GRID_COLOR)]

    def _ground_geometries(self, table: Table3D) -> List:
        """地面网格（z = -height），覆盖球桌区域 + 少量外扩，辅助空间定位。"""
        z = -table.height
        x0, x1 = -0.3, table.width + 0.3
        y0, y1 = -0.3, table.length + 0.3
        step = 0.5
        pts: List = []
        lines: List = []

        for y in np.arange(np.floor(y0 / step) * step, y1 + 1e-6, step):
            pts.append([x0, y, z])
            pts.append([x1, y, z])
            lines.append([len(pts) - 2, len(pts) - 1])
        for x in np.arange(np.floor(x0 / step) * step, x1 + 1e-6, step):
            pts.append([x, y0, z])
            pts.append([x, y1, z])
            lines.append([len(pts) - 2, len(pts) - 1])
        return [_line_set(pts, lines, _GRID_COLOR)]

    # ------------------------------------------------------------------
    # 实时骨架层
    # ------------------------------------------------------------------
    def add_skeleton_layer(self, skeleton: str = "halpe26", max_people: int = 8) -> None:
        """预分配骨架几何（每人一个 LineSet + PointCloud），须在 ``start()`` 前调用。

        Open3D 的 ``update_geometry`` 只对已 ``add_geometry`` 的对象生效，所以这里
        一次性把 ``max_people`` 个空几何加进场景，之后逐帧用 ``set_skeletons`` 更新。
        """
        skel = get_skeleton(skeleton)
        self._skeleton_edges = skel["edges"]
        o3d = _o3d()
        for _ in range(max_people):
            ls = o3d.geometry.LineSet()
            pc = o3d.geometry.PointCloud()
            self._skeleton_geoms.append((ls, pc))
            self._geometries.append(ls)
            self._geometries.append(pc)

    def set_skeletons(self, skeletons: List[Skeleton3D]) -> None:
        """线程安全地写入最新一帧的 3D 骨架（主线程调用，渲染线程读取）。"""
        with self._skeleton_lock:
            self._latest_skeletons = list(skeletons or [])
            self._skeletons_dirty = True

    def _update_skeleton_geometry(self, vis) -> None:
        """渲染线程内调用：把最新骨架写进预分配几何并 ``update_geometry``。"""
        with self._skeleton_lock:
            if not self._skeletons_dirty:
                return
            skeletons = list(self._latest_skeletons)
            self._skeletons_dirty = False

        o3d = _o3d()
        n_show = min(len(skeletons), len(self._skeleton_geoms))
        for i, (ls, pc) in enumerate(self._skeleton_geoms):
            if i < n_show:
                color = _SKELETON_COLORS[i % len(_SKELETON_COLORS)]
                pts, lines, lcolors, jpts, jcolors = _skeleton_geometry(
                    skeletons[i], self._skeleton_edges, color
                )
            else:
                pts = np.zeros((0, 3))
                lines = np.zeros((0, 2), dtype=np.int32)
                lcolors = np.zeros((0, 3))
                jpts = np.zeros((0, 3))
                jcolors = np.zeros((0, 3))
            ls.points = o3d.utility.Vector3dVector(pts)
            ls.lines = o3d.utility.Vector2iVector(lines)
            ls.colors = o3d.utility.Vector3dVector(lcolors)
            pc.points = o3d.utility.Vector3dVector(jpts)
            pc.colors = o3d.utility.Vector3dVector(jcolors)
            vis.update_geometry(ls)
            vis.update_geometry(pc)

    # ------------------------------------------------------------------
    # 渲染线程
    # ------------------------------------------------------------------
    def start(self) -> None:
        """启动后台渲染线程（幂等）。"""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="scene-viewer3d", daemon=True
        )
        self._thread.start()

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def close(self) -> None:
        """停止渲染线程并销毁窗口。"""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run(self) -> None:
        o3d = _o3d()
        vis = o3d.visualization.VisualizerWithKeyCallback()
        try:
            vis.create_window(
                window_name=self._title, width=self._width, height=self._height, visible=True
            )
            opt = vis.get_render_option()
            opt.background_color = np.array([0.10, 0.11, 0.14])
            opt.line_width = 3.0
            opt.mesh_show_back_face = True
            opt.mesh_show_wireframe = False

            for geo in self._geometries:
                vis.add_geometry(geo, reset_bounding_box=False)
            vis.reset_view_point(True)
            self._register_keys(vis)
            self._set_viewpoint(vis)
            print("[3D] 鼠标：左键拖=旋转，右键拖=平移，中键拖=滚转，滚轮=缩放；"
                  "键盘：W/A/S/D=平移，方向键=旋转，+/−=缩放，R=复位视角。")

            while not self._stop.is_set():
                # 用户点 X 关闭 3D 窗口时 poll_events 返回 False，退出渲染循环
                if not vis.poll_events():
                    break
                self._update_skeleton_geometry(vis)
                vis.update_renderer()
                time.sleep(0.01)
        except Exception as exc:  # noqa: BLE001 —— 3D 窗口失败不影响 2D 主流程
            print(f"[3D] 场景窗口异常退出：{exc}")
        finally:
            try:
                vis.destroy_window()
            except Exception:  # noqa: BLE001
                pass

    def _set_viewpoint(self, vis) -> None:
        """初始视角：设为默认的 3/4 俯瞰（失败不影响渲染，用户仍可手动调整）。"""
        try:
            self._apply_viewpoint(vis.get_view_control())
        except Exception:  # noqa: BLE001 —— 视角只是默认值，出错不致命
            pass

    def _apply_viewpoint(self, vc) -> None:
        """把视角设成「建模软件风格」的默认 3/4 俯瞰：Z 轴竖直、从斜上方看球桌。

        世界系 = 桌面系（X 短边 / Y 长边 / Z 向上），所以 ``up=(0,0,1)`` 保证画面
        竖直方向就是世界 Z（球桌腿垂直、桌面不歪）；``front`` 取 -X/-Y/-Z 对角线
        （相机在 +X+Y+Z 角落）得到能同时看到桌面和两条边的舒服视角。R 键复位即回到此视角。
        """
        vc.set_lookat(self._lookat)
        d = np.array([-1.0, -1.0, -0.9], dtype=np.float64)
        d = d / np.linalg.norm(d)
        vc.set_front(d)
        vc.set_up([0.0, 0.0, 1.0])
        vc.set_zoom(_DEFAULT_ZOOM)

    def _register_keys(self, vis) -> None:
        """注册键盘视角控制：R 复位、W/A/S/D 平移、方向键旋转、+/− 缩放。"""
        vc = vis.get_view_control()

        def make_cb(kind: str, dx: float = 0.0, dy: float = 0.0, s: float = 1.0):
            def cb(_vis) -> bool:
                if kind == "pan":
                    vc.translate(dx, dy)
                elif kind == "rotate":
                    vc.rotate(dx, dy)
                elif kind == "zoom":
                    vc.scale(s)
                else:  # reset
                    self._apply_viewpoint(vc)
                return False
            return cb

        # 方向键用 GLFW 键码：上 265 / 下 264 / 左 263 / 右 262。
        bindings = {
            ord("R"): ("reset", 0.0, 0.0, 1.0),
            ord("W"): ("pan", 0.0, -_PAN_STEP, 1.0),
            ord("S"): ("pan", 0.0, _PAN_STEP, 1.0),
            ord("A"): ("pan", _PAN_STEP, 0.0, 1.0),
            ord("D"): ("pan", -_PAN_STEP, 0.0, 1.0),
            265: ("rotate", 0.0, -_ROT_STEP, 1.0),   # 上
            264: ("rotate", 0.0, _ROT_STEP, 1.0),    # 下
            263: ("rotate", -_ROT_STEP, 0.0, 1.0),   # 左
            262: ("rotate", _ROT_STEP, 0.0, 1.0),    # 右
            ord("="): ("zoom", 0.0, 0.0, _ZOOM_STEP),
            ord("+"): ("zoom", 0.0, 0.0, _ZOOM_STEP),
            ord("-"): ("zoom", 0.0, 0.0, 1.0 / _ZOOM_STEP),
        }
        for key, (kind, dx, dy, s) in bindings.items():
            vis.register_key_callback(key, make_cb(kind, dx, dy, s))
