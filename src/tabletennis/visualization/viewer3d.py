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

# SMPL 24 关节（原生顺序）的骨骼连线（由 kintree_table 推导，与 easymocap.py 一致）。
# 顺序：0 pelvis, 1/2 hip, 3 spine1, 4/5 knee, 6 spine2, 7/8 ankle, 9 spine3,
# 10/11 foot, 12 neck, 13/14 collar, 15 head, 16/17 shoulder, 18/19 elbow,
# 20/21 wrist, 22/23 hand。
_SMPL_EDGES = [
    (0, 1), (0, 2), (0, 3),
    (1, 4), (2, 5), (3, 6),
    (4, 7), (5, 8), (6, 9),
    (7, 10), (8, 11),
    (9, 12), (9, 13), (9, 14),
    (12, 15),
    (13, 16), (14, 17),
    (16, 18), (17, 19),
    (18, 20), (19, 21),
    (20, 22), (21, 23),
]
_SMPL_MESH_COLOR = [0.82, 0.71, 0.60]   # 肤色
_SMPL_BONE_COLOR = [0.95, 0.60, 0.25]   # 橙

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


def _paddle_mesh(blade_radius: float = 0.08, handle_len: float = 0.10):
    """构造球拍三角网格（局部系：拍面在 XY 平面、法线 +Z，手柄沿 +X）。

    Returns:
        ``(vertices (N,3), triangles (M,3))`` —— 拍面圆盘 + 手柄薄盒。
    """
    res = 40
    verts = [np.array([0.0, 0.0, 0.0])]
    for i in range(res):
        th = 2.0 * np.pi * i / res
        verts.append(np.array([blade_radius * np.cos(th), blade_radius * np.sin(th), 0.0]))
    tris = [[0, 1 + i, 1 + (i + 1) % res] for i in range(res)]

    hw, hh = 0.014, 0.006  # 手柄半宽(Y) / 半高(Z)
    x0, x1 = blade_radius - 0.005, blade_radius + handle_len
    box = np.array([
        [x0, -hw, -hh], [x1, -hw, -hh], [x1, hw, -hh], [x0, hw, -hh],
        [x0, -hw, hh], [x1, -hw, hh], [x1, hw, hh], [x0, hw, hh],
    ])
    base = len(verts)
    verts.extend(box)
    box_t = [
        [0, 1, 2], [0, 2, 3],
        [4, 6, 5], [4, 7, 6],
        [0, 4, 5], [0, 5, 1],
        [3, 2, 6], [3, 6, 7],
        [1, 5, 6], [1, 6, 2],
        [0, 3, 7], [0, 7, 4],
    ]
    tris.extend([[a + base, b + base, c + base] for a, b, c in box_t])
    return np.asarray(verts, dtype=np.float64), np.asarray(tris, dtype=np.int32)


def _imu_axes_geometry(size: float = 0.18):
    """IMU 坐标架几何：X/Y/Z 三段线（红/绿/蓝），起点都在原点。"""
    pts = np.array([
        [0.0, 0.0, 0.0], [size, 0.0, 0.0],
        [0.0, 0.0, 0.0], [0.0, size, 0.0],
        [0.0, 0.0, 0.0], [0.0, 0.0, size],
    ], dtype=np.float64)
    lines = np.array([[0, 1], [2, 3], [4, 5]], dtype=np.int32)
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    return pts, lines, colors


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

        # 实时球层：当前位置小球（无轨迹），跨线程传递最新 3D 球心
        self._ball_lock = threading.Lock()
        self._ball_sphere = None
        self._latest_ball = None
        self._ball_dirty = False

        # EasyMocap SMPL 层：人体网格 + 关节骨架，跨线程传递最新拟合结果
        self._smpl_lock = threading.Lock()
        self._smpl_mesh = None
        self._smpl_bones = None
        self._smpl_joints = None
        self._latest_smpl = None   # dict{vertices, faces, joints} 或 None
        self._smpl_dirty = False
        # 实时 IMU 层：球拍网格 + 坐标架，按最新旋转矩阵朝向（无绝对位置，锚点固定）
        self._imu_lock = threading.Lock()
        self._imu_anchor = None
        self._imu_axes = None
        self._imu_paddle = None
        self._imu_template_verts = None
        self._imu_template_tris = None
        self._imu_axes_pts = None
        self._imu_axes_lines = None
        self._imu_axes_colors = None
        self._latest_imu_R = None
        self._imu_dirty = False

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
    # 实时球层
    # ------------------------------------------------------------------
    def has_ball_layer(self) -> bool:
        """球层是否已加入场景（上层据此判断已运行的 3D 窗口是否缺球层、需重建）。"""
        return self._ball_sphere is not None

    def add_ball_layer(self, trail_len: int = 0) -> None:
        """预分配球几何（当前位置小球，无轨迹），须在 ``start()`` 前调用。

        ``trail_len`` 参数保留兼容（历史轨迹已废弃，恒为 0）。
        """
        o3d = _o3d()
        self._ball_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        self._ball_sphere.paint_uniform_color([1.0, 0.30, 0.20])
        self._geometries.append(self._ball_sphere)

    def set_ball(self, X) -> None:
        """线程安全写入最新 3D 球心（世界系=桌面系，米）；None 表示本帧无球。"""
        with self._ball_lock:
            self._latest_ball = (
                None if X is None else np.asarray(X, dtype=np.float64).reshape(3)
            )
            self._ball_dirty = True

    def _update_ball_geometry(self, vis) -> None:
        """渲染线程内调用：更新小球位置（无轨迹）。"""
        with self._ball_lock:
            if not self._ball_dirty:
                return
            X = None if self._latest_ball is None else self._latest_ball.copy()
            self._ball_dirty = False

        if self._ball_sphere is None or X is None:
            return
        cur = np.asarray(self._ball_sphere.get_center(), dtype=np.float64)
        self._ball_sphere.translate(X - cur)
        vis.update_geometry(self._ball_sphere)

    # ------------------------------------------------------------------
    # 实时 SMPL 层（EasyMocap 重建结果）
    # ------------------------------------------------------------------
    def has_smpl_layer(self) -> bool:
        """SMPL 层是否已加入场景（上层据此判断已运行的 3D 窗口是否需重建）。"""
        return self._smpl_mesh is not None

    def add_smpl_layer(self) -> None:
        """预分配 SMPL 网格 + 关节骨架几何，须在 ``start()`` 前调用。

        网格用空 TriangleMesh（逐帧更新顶点/面），关节骨架用 LineSet（点按
        ``_SMPL_EDGES`` 连线）+ PointCloud（关节点）。渲染线程内更新。
        """
        o3d = _o3d()
        self._smpl_mesh = o3d.geometry.TriangleMesh()
        self._smpl_mesh.paint_uniform_color(_SMPL_MESH_COLOR)
        self._smpl_mesh.compute_vertex_normals()
        self._geometries.append(self._smpl_mesh)

        self._smpl_bones = o3d.geometry.LineSet()
        self._smpl_bones.lines = o3d.utility.Vector2iVector(
            np.asarray(_SMPL_EDGES, dtype=np.int32)
        )
        self._smpl_bones.colors = o3d.utility.Vector3dVector(
            np.tile(np.asarray(_SMPL_BONE_COLOR, dtype=np.float64), (len(_SMPL_EDGES), 1))
        )
        self._geometries.append(self._smpl_bones)

        self._smpl_joints = o3d.geometry.PointCloud()
        self._smpl_joints.paint_uniform_color(_SMPL_BONE_COLOR)
        self._geometries.append(self._smpl_joints)

    def set_smpl(self, result) -> None:
        """线程安全写入最新一帧 SMPL 拟合结果；None 表示本帧无结果。

        Args:
            result: ``{vertices (N,3), faces (M,3), joints (24,3)}``（桌面系，米）。
        """
        with self._smpl_lock:
            self._latest_smpl = None if result is None else result
            self._smpl_dirty = True

    def _update_smpl_geometry(self, vis) -> None:
        """渲染线程内调用：把最新 SMPL 网格/关节写进几何并 ``update_geometry``。"""
        with self._smpl_lock:
            if not self._smpl_dirty:
                return
            result = self._latest_smpl
            self._smpl_dirty = False

        if self._smpl_mesh is None or self._smpl_bones is None:
            return

        o3d = _o3d()
        if result is not None:
            verts = np.asarray(result["vertices"], dtype=np.float64).reshape(-1, 3)
            faces = np.asarray(result["faces"], dtype=np.int64).reshape(-1, 3)
            joints = np.asarray(result["joints"], dtype=np.float64).reshape(-1, 3)
            self._smpl_mesh.vertices = o3d.utility.Vector3dVector(verts)
            self._smpl_mesh.triangles = o3d.utility.Vector3iVector(faces)
            if self._smpl_mesh.has_vertex_normals():
                self._smpl_mesh.compute_vertex_normals()
            self._smpl_bones.points = o3d.utility.Vector3dVector(joints[:24])
            self._smpl_joints.points = o3d.utility.Vector3dVector(joints[:24])
        else:
            self._smpl_mesh.vertices = o3d.utility.Vector3dVector(np.zeros((0, 3)))
            self._smpl_mesh.triangles = o3d.utility.Vector3iVector(np.zeros((0, 3), dtype=np.int32))
            self._smpl_bones.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
            self._smpl_joints.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))

        vis.update_geometry(self._smpl_mesh)
        vis.update_geometry(self._smpl_bones)
        vis.update_geometry(self._smpl_joints)
    # 实时 IMU 层（球拍朝向）
    # ------------------------------------------------------------------
    def has_imu_layer(self) -> bool:
        """IMU 球拍层是否已加入场景。"""
        return self._imu_paddle is not None

    def add_imu_layer(self, anchor=None) -> None:
        """预分配 IMU 球拍几何（拍面 + 手柄 + 坐标架），须在 ``start()`` 前调用。

        IMU 只有朝向没有绝对位置，球拍锚定在 ``anchor``（世界系=桌面系，米）处只做
        旋转；默认锚在原点上方 0.4m。初始为空（隐藏），``set_imu_orientation`` 写入
        朝向后才显示。
        """
        o3d = _o3d()
        self._imu_anchor = np.asarray(
            anchor if anchor is not None else [0.0, 0.0, 0.4], dtype=np.float64
        ).reshape(3)
        self._imu_template_verts, self._imu_template_tris = _paddle_mesh()
        (self._imu_axes_pts, self._imu_axes_lines,
         self._imu_axes_colors) = _imu_axes_geometry()

        self._imu_axes = o3d.geometry.LineSet()
        self._imu_axes.points = o3d.utility.Vector3dVector(np.zeros((0, 3)))
        self._imu_axes.lines = o3d.utility.Vector2iVector(np.zeros((0, 2), dtype=np.int32))
        self._imu_axes.colors = o3d.utility.Vector3dVector(np.zeros((0, 3)))

        self._imu_paddle = o3d.geometry.TriangleMesh()
        self._imu_paddle.vertices = o3d.utility.Vector3dVector(np.zeros((0, 3)))
        self._imu_paddle.triangles = o3d.utility.Vector3iVector(
            np.zeros((0, 3), dtype=np.int32))
        self._imu_paddle.paint_uniform_color([0.95, 0.25, 0.20])

        self._geometries.append(self._imu_axes)
        self._geometries.append(self._imu_paddle)

    def set_imu_orientation(self, R) -> None:
        """线程安全写入最新 IMU 朝向（3x3 旋转矩阵，body -> world）；None 隐藏。"""
        with self._imu_lock:
            self._latest_imu_R = (
                None if R is None else np.asarray(R, dtype=np.float64).reshape(3, 3)
            )
            self._imu_dirty = True

    def _update_imu_geometry(self, vis) -> None:
        """渲染线程内调用：按最新旋转矩阵更新球拍 + 坐标架朝向。"""
        with self._imu_lock:
            if not self._imu_dirty:
                return
            R = None if self._latest_imu_R is None else self._latest_imu_R.copy()
            self._imu_dirty = False

        if self._imu_paddle is None or self._imu_axes is None:
            return
        o3d = _o3d()
        a = self._imu_anchor
        if R is None:
            empty3 = np.zeros((0, 3))
            self._imu_axes.points = o3d.utility.Vector3dVector(empty3)
            self._imu_axes.lines = o3d.utility.Vector2iVector(np.zeros((0, 2), dtype=np.int32))
            self._imu_axes.colors = o3d.utility.Vector3dVector(empty3)
            self._imu_paddle.vertices = o3d.utility.Vector3dVector(empty3)
            self._imu_paddle.triangles = o3d.utility.Vector3iVector(
                np.zeros((0, 3), dtype=np.int32))
        else:
            self._imu_axes.points = o3d.utility.Vector3dVector(
                (R @ self._imu_axes_pts.T).T + a)
            self._imu_axes.lines = o3d.utility.Vector2iVector(self._imu_axes_lines)
            self._imu_axes.colors = o3d.utility.Vector3dVector(self._imu_axes_colors)
            self._imu_paddle.vertices = o3d.utility.Vector3dVector(
                (R @ self._imu_template_verts.T).T + a)
            self._imu_paddle.triangles = o3d.utility.Vector3iVector(self._imu_template_tris)
            self._imu_paddle.compute_vertex_normals()
        vis.update_geometry(self._imu_axes)
        vis.update_geometry(self._imu_paddle)

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
                self._update_ball_geometry(vis)
                self._update_smpl_geometry(vis)
                self._update_imu_geometry(vis)
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
        竖直方向就是世界 Z（球桌腿垂直、桌面不歪）。Open3D 的 ``set_front`` 传入的
        是「从 lookat 指向相机」的方向，所以 ``front=[1,1,0.9]``（+Z 朝上）表示相机
        位于 +X+Y+Z 斜上方、向下俯瞰球桌，得到能同时看到桌面和两条边的舒服视角。
        R 键复位即回到此视角。
        """
        vc.set_lookat(self._lookat)
        d = np.array([1.0, 1.0, 0.9], dtype=np.float64)
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
            ord("W"): ("pan", 0.0, _PAN_STEP, 1.0),    # 上（translate +y = 向上）
            ord("S"): ("pan", 0.0, -_PAN_STEP, 1.0),   # 下
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
