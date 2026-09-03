"""离线重建结果回放查看器（Open3D Filament 新渲染器，带真实太阳阴影）。

``scripts/reconstruct_video.py`` 每帧把 SMPL 拟合结果存成
``out/frame_NNNNNN.npz``（vertices + joints + SMPL 参数…），并把网格拓扑存一次
``out/recon_faces.npy``。本模块把这些结果放回一个 3D 场景里播放：

- 场景 = 桌面系（与重建/在线 3D 一致）：桌面（z=0）+ 地面（z=-height）+ 相机视锥。
- 人体 = 每帧一个 SMPL 网格（13776 面，肤色，PBR），**逐帧重建网格 + 计算法线按环境
  光着色**——不是一片白（旧 ``Visualizer`` 的坑是法线没重算，渲染成 unlit 平面白）。
- 阴影 = 本机 Filament 的**太阳定向光不生效**（实测 EGL 与屏幕窗口翻转太阳方向像素
  不变，只剩环境漫反射），故用程序化**接触软影**：按太阳水平方向在人物脚下画两层
  半透明椭圆（``defaultLitTransparency``）。Filament 窗口本身走 ``SOFT_SHADOWS``
  提供环境光；在太阳光可用的机器上会额外叠加真实阴影，二者方向一致。

两个入口共用同一套 ``ReconScene``（把几何加进任意 ``Open3DScene``）：

- 离线渲染单帧：``render_still(out_dir, t, ...)``（EGL headless，可出 PNG / 验证）；
- 交互窗口回放：``play_gui(out_dir, ...)``（``gui.Application`` + ``SceneWidget``，
  鼠标转视角，键盘控制播放），须在主线程调用。

参考 playback 语义：t 是主时钟（参考相机）脉冲号/帧号。某人帧期间显示该人网格；
``no_person``/失败帧清空人体；被 ``--stride`` 跳过的帧保持上一姿态。
"""
from __future__ import annotations

import os
import time
from typing import Dict, List, Optional

import numpy as np

from ..core.types import Table3D
from .viewer3d import (
    _CAM_COLORS,
    _SMPL_EDGES,
    _TABLE_NET_COLOR,
    _TABLE_STRUCT_COLOR,
    _TABLE_TOP_COLOR,
)

# 人物肤色（与 viewer3d._SMPL_MESH_COLOR 一致）
_SMPL_SKIN = np.array([0.82, 0.71, 0.60], np.float32)
_BONE_COLOR = np.array([0.95, 0.60, 0.25], np.float32)

_FLOOR_COLOR = np.array([0.36, 0.38, 0.46], np.float32)   # 地面（接阴影）
_FLOOR_ALPHA = 1.0

# 太阳方向：光传播方向（Filament 用「光走向」），选略偏左前上方，影子落到右后地面
_SUN_DIR = np.array([-0.42, 0.34, 0.84], np.float32)
_SUN_DIR = _SUN_DIR / np.linalg.norm(_SUN_DIR)

# 桌面 + 人物总体包围盒余量（用于默认取景）
_BOUNDS_MARGIN = 0.6

# 程序化接触阴影的两层软影参数（alpha 比例，核心深/外圈浅）
_SHADOW_LAYERS = (
    dict(alpha=0.30, off=0.30, rx=0.30, ry=0.20),   # 外圈软影
    dict(alpha=0.45, off=0.16, rx=0.15, ry=0.10),   # 脚下核心影
)


def _o3d():
    try:
        import open3d as o3d  # noqa: F401
        return o3d
    except ImportError as exc:  # pragma: no cover
        raise ImportError("需要 open3d（conda env tt）才能用 3D 回放查看器") from exc


# ----------------------------------------------------------------------
# 时间线：把 recon_index.npz / 输出目录里的逐帧 npz 变成「t → 该显示什么」
# ----------------------------------------------------------------------
class ReconTimeline:
    """把一次重建输出变成按主时钟 t 播放的时间线。

    - 有 ``recon_index.npz``（重建完成）时用它的 status 精确定义 no_person 空窗；
    - 重建进行中（只有逐帧 npz、index 还没写）则先从文件列表推断，
      ``reload()`` 可反复调用以追赶新写出的帧（watch 模式）。

    Attributes:
        n_ref: 主时钟帧总数（对齐后参考相机的帧数）。
        files: {t: frame_NNNNNN.npz 路径}，t 处有成功拟合结果。
        state: np.ndarray(n_ref) int，t 处应显示的 file 下标，-1 表示无人体
               （该显示时刻之前最近一次成功帧，中间 no_person 会清空）。
        index_ready: recon_index.npz 是否存在（决定 empty 空窗是否精确）。
        ref_rate_hz: 参考相机出帧率（由 meta 里 period_s 反推，回放真实速度用）。
    """

    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        self.meta = self._load_json("recon_meta.json")
        self.index_ready = os.path.exists(self._p("recon_index.npz"))
        self.files: Dict[int, str] = {}
        self.empty_ts: List[int] = []
        self.n_ref = 0
        self.state = np.zeros(0, dtype=np.int64)
        self.ref_rate_hz = 100.0
        self.reload()

    # ------------------------------------------------------------------
    def _p(self, name: str) -> str:
        return os.path.join(self.out_dir, name)

    def _load_json(self, name: str) -> Optional[dict]:
        path = self._p(name)
        if not os.path.exists(path):
            return None
        try:
            import json
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:  # noqa: BLE001 —— 可能正在写入/半截 json
            return None

    def n_ok(self) -> int:
        return len(self.files)

    # ------------------------------------------------------------------
    def reload(self) -> None:
        """重扫目录：把新出现的逐帧 npz 追加进时间线并重建 state 数组。"""
        self.index_ready = os.path.exists(self._p("recon_index.npz"))
        # 1) 逐帧文件（权威来源：帧号在文件名里）
        new_files: Dict[int, str] = {}
        try:
            names = sorted(os.listdir(self.out_dir))
        except OSError:
            names = []
        for name in names:
            if not (name.startswith("frame_") and name.endswith(".npz")):
                continue
            try:
                t = int(name[len("frame_"):-len(".npz")])
            except ValueError:
                continue
            if os.path.exists(self._p(name)):
                new_files[t] = self._p(name)
        self.files = new_files

        # 2) index 就绪 → 用 status 精确定 no_person 空窗（n_ref 全跨）
        if self.index_ready:
            try:
                idx = np.load(self._p("recon_index.npz"))
                n_ref = int(self.meta.get("n_ref_frames", 0)) if self.meta else 0
                if n_ref <= 0:
                    n_ref = int(idx["ref_frame"][-1]) + 1
                refs = idx["ref_frame"].astype(np.int64)
                status = idx["status"].astype(np.int64)
                # empty = 非 ok 的已处理帧（no_person/fit_failed/error）都算空窗
                self.empty_ts = [int(t) for t in refs[status != 0]]
                # 兜底：index 说 ok 但文件缺失 → 视为 empty
                missing = [int(t) for t in refs[status == 0] if int(t) not in self.files]
                self.empty_ts = sorted(set(self.empty_ts) | set(missing))
                self.n_ref = n_ref
            except Exception as exc:  # noqa: BLE001
                print(f"[recon_player] ⚠ 读 recon_index.npz 失败：{exc}；退回按文件推断")
                self.index_ready = False
                self.empty_ts = []
                self.n_ref = (max(self.files) + 1) if self.files else 0

        # 3) 主时钟总数
        if self.n_ref <= 0:
            if self.meta:
                self.n_ref = int(self.meta.get("n_ref_frames", 0))
            if self.n_ref <= 0 and self.files:
                self.n_ref = max(self.files) + 1

        # 4) 回放真实速率（录像是硬触发 100Hz，缺 ts 时回退 100）
        period = None
        if self.meta:
            ref = str(self.meta.get("ref_cam", ""))
            cams = (self.meta.get("source") or {}).get("cams") or {}
            if ref in cams:
                period = (cams[ref] or {}).get("period_s")
        if period:
            self.ref_rate_hz = 1.0 / float(period) if float(period) > 0 else 100.0
        else:
            self.ref_rate_hz = 100.0

        self._rebuild_state()

    # ------------------------------------------------------------------
    def _rebuild_state(self) -> None:
        """state[t] = 该显示的 file 下标（最近一次成功帧，遇 empty 清空）。"""
        state = np.full(max(1, self.n_ref), -1, dtype=np.int64)
        file_idx = {t: i for i, t in enumerate(sorted(self.files))}
        file_order = sorted(self.files)
        file_list = [self.files[t] for t in file_order]
        empty_set = set(self.empty_ts)
        latest = -1
        for t in range(len(state)):
            if t in file_idx:
                latest = file_idx[t]
            if t in empty_set:
                latest = -1
            state[t] = latest
        self.state = state
        self._file_order = file_order
        self._file_list = file_list

    # ------------------------------------------------------------------
    def person_path_at(self, t: int) -> Optional[str]:
        """t（主时钟帧号）处应显示的 npz 路径；无人体（空窗/未处理）返回 None。"""
        if t < 0 or t >= len(self.state):
            return None
        i = int(self.state[t])
        return None if i < 0 else self._file_list[i]

    def load_person(self, t: int) -> Optional[dict]:
        """读 t 处 npz：``{vertices (6890,3), joints (24,3)}``；读失败返回 None。"""
        path = self.person_path_at(t)
        if path is None:
            return None
        try:
            with np.load(path) as z:
                verts = np.asarray(z["vertices"], np.float32).reshape(-1, 3)
                joints = None
                if "joints" in z:
                    joints = np.asarray(z["joints"], np.float32).reshape(-1, 3)
            return {"vertices": verts, "joints": joints}
        except Exception as exc:  # noqa: BLE001 —— 重建正在写/文件半截
            print(f"[recon_player] ⚠ t={t} 读帧失败：{exc}")
            return None


# ----------------------------------------------------------------------
# 程序化接触阴影
# ----------------------------------------------------------------------
def contact_shadow_planes(verts: np.ndarray, floor_z: float,
                          sun_dir: Optional[np.ndarray] = None) -> List[dict]:
    """由 SMPL 顶点算脚下地面上的软影椭圆参数（纯 numpy，可单测）。

    为什么自己画影子：Open3D 新 Filament 渲染器在本机（EGL headless 与
    屏幕窗口都是 OpenGL 4.1）**太阳定向光不生效**——实测翻转太阳方向画面像素
    不变，只剩环境光漫反射。所以「真实投射阴影」做不到，改在人物脚下画两层
    半透明深色椭圆（``defaultLitTransparency`` 混合已验证可用），位置/朝向跟着
    太阳水平方向走，观感即接触阴影。

    Args:
        verts: (N,3) SMPL 顶点（桌面系，m）。
        floor_z: 地面高度（= -table.height）。
        sun_dir: 光传播方向（归一化）；缺省用模块 ``_SUN_DIR``。

    Returns:
        每层一个 dict：``center(3)`` 盘心、``e``(2) 影长方向单位向量、
        ``rx`` 沿影方向半轴、``ry`` 侧向半轴、``alpha``、``name``。
    """
    if verts is None or len(verts) < 3:
        return []
    verts = np.asarray(verts, np.float64)
    min_z = float(verts[:, 2].min())
    z = max(min_z, floor_z) + 0.004            # 略高于所在平面防 z-fight
    feet = verts[verts[:, 2] <= min_z + 0.06]  # 脚底附近顶点 → 立足点
    if len(feet) == 0:
        fc = verts[:1, :2].mean(axis=0)
    else:
        fc = feet[:, :2].mean(axis=0)
    height = float(verts[:, 2].max() - min_z)
    s = float(np.clip(height / 1.75, 0.7, 1.4))    # 按身高伸缩

    sd = np.asarray(_SUN_DIR if sun_dir is None else sun_dir, np.float64)
    h = sd[:2]
    nh = float(np.linalg.norm(h))
    e = np.array([1.0, 0.0]) if nh < 1e-6 else h / nh   # 影子伸向 +e

    out = []
    for i, lay in enumerate(_SHADOW_LAYERS):
        center = np.array([fc[0] + e[0] * lay["off"] * s,
                           fc[1] + e[1] * lay["off"] * s, z])
        out.append(dict(center=center, e=e, rx=lay["rx"] * s,
                        ry=lay["ry"] * s, alpha=lay["alpha"],
                        name=f"shadow{i}"))
    return out


# ----------------------------------------------------------------------
# 场景：把桌面/地面/相机 + 人体几何喂给任意 Open3DScene
# ----------------------------------------------------------------------
class ReconScene:
    """向一个 ``Open3DScene``（GUI widget 或 OffscreenRenderer）搭建重建回放场景。"""

    def __init__(self, tl: ReconTimeline, faces: Optional[np.ndarray],
                 table: Optional[Table3D] = None,
                 camera_rig: Optional[tuple] = None):
        o3d = _o3d()
        self.o3d = o3d
        self.tl = tl
        self.table = table or Table3D()
        self.floor_z = -self.table.height
        self.intrinsics, self.extrinsics = (camera_rig or (None, None))
        self.faces = faces                       # (13776,3) int；None 则只画关节
        self._mat_smpl = self._make_material(_SMPL_SKIN, roughness=0.62)
        self._mat_bones = self._make_material(_BONE_COLOR, roughness=0.8)
        self._last_t = None
        self.n_added = 0

    # ------------------------------------------------------------------
    @staticmethod
    def _make_material(color, roughness=0.7, alpha=1.0):
        mr = _o3d().visualization.rendering.MaterialRecord()
        mr.base_color = [float(color[0]), float(color[1]), float(color[2]), float(alpha)]
        mr.base_roughness = float(roughness)
        mr.base_metallic = 0.0
        return mr

    def bounds(self) -> "object":
        """覆盖桌面 + 站立人体高度的包围盒（供相机取景）。"""
        o3d = self.o3d
        W, L = self.table.width, self.table.length
        lo = np.array([-_BOUNDS_MARGIN, -_BOUNDS_MARGIN, -self.table.height - 0.1], np.float64)
        hi = np.array([W + _BOUNDS_MARGIN, L + _BOUNDS_MARGIN, 1.7], np.float64)
        box = o3d.geometry.AxisAlignedBoundingBox(lo, hi)
        return box

    def center(self) -> np.ndarray:
        W, L = self.table.width, self.table.length
        return np.array([W / 2.0, L / 2.0, -self.table.height / 3.0], np.float32)

    # ------------------------------------------------------------------
    def add_static(self, scene) -> None:
        """桌面 + 地面 + 相机视锥 + 原点坐标架（一次性）。"""
        self._add_table(scene)
        self._add_cameras(scene)
        self.n_added += 1

    def _add_line_geo(self, scene, name: str, ls, color, width: float = 2.5) -> None:
        o3d = self.o3d
        n = len(ls.lines)
        if n:
            ls.colors = o3d.utility.Vector3dVector(
                np.tile(np.asarray(color, np.float64), (n, 1))
            )
        mr = o3d.visualization.rendering.MaterialRecord()
        mr.shader = "unlitLine"
        mr.line_width = width
        scene.add_geometry(name, ls, mr)

    def _add_table(self, scene) -> None:
        o3d = self.o3d
        t = self.table
        W, L, H = t.width, t.length, t.height

        # 桌面（z=0 实心，接收阴影）
        c = t.top_corners
        surf = o3d.geometry.TriangleMesh()
        surf.vertices = o3d.utility.Vector3dVector(c)
        surf.triangles = o3d.utility.Vector3iVector(np.array([[0, 1, 2], [0, 2, 3]], np.int32))
        surf.compute_vertex_normals()
        scene.add_geometry("table_top", surf, self._make_material([0.10, 0.45, 0.30], 0.7))

        # 桌面下沿垂面（视觉厚度，可选 —— 从低视角不至于穿帮为空壳）
        edge = o3d.geometry.TriangleMesh()
        v = np.vstack([c, c + np.array([[0, 0, -0.03]], np.float64)])
        tri = np.array([[0, 4, 5], [0, 5, 1],
                        [1, 5, 6], [1, 6, 2],
                        [2, 6, 7], [2, 7, 3],
                        [3, 7, 4], [3, 4, 0]], np.int32)
        edge.vertices = o3d.utility.Vector3dVector(v)
        edge.triangles = o3d.utility.Vector3iVector(tri)
        edge.compute_vertex_normals()
        scene.add_geometry("table_edge", edge, self._make_material([0.05, 0.28, 0.20], 0.9))

        # 大地面（z=-H，接住人体阴影）
        floor = o3d.geometry.TriangleMesh()
        m = 0.9
        fv = np.array([[-m, -m, -H], [W + m, -m, -H], [W + m, L + m, -H], [-m, L + m, -H]],
                      np.float64)
        floor.vertices = o3d.utility.Vector3dVector(fv)
        floor.triangles = o3d.utility.Vector3iVector(np.array([[0, 1, 2], [0, 2, 3]], np.int32))
        floor.compute_vertex_normals()
        scene.add_geometry("floor", floor, self._make_material(_FLOOR_COLOR, 0.95))

        # 线框：桌面边 / 桌腿 / 地面框 / 球网
        segs = t.segments()
        for i, (key, color, w) in enumerate([
                ("top", _TABLE_TOP_COLOR, 3.0), ("legs", _TABLE_STRUCT_COLOR, 3.0),
                ("floor", _TABLE_STRUCT_COLOR, 2.0), ("net", _TABLE_NET_COLOR, 3.0)]):
            a = np.asarray(segs[key], np.float64).reshape(-1, 3)
            ls = o3d.geometry.LineSet()
            ls.points = o3d.utility.Vector3dVector(a)
            ls.lines = o3d.utility.Vector2iVector(
                np.arange(len(a)).reshape(-1, 2).astype(np.int32))
            self._add_line_geo(scene, f"line_{i}_{key}", ls, np.asarray(color, np.float64), w)

        # 原点坐标架（短三轴）
        self._add_axes(scene, np.zeros(3), "origin_axes", 0.25)

    def _add_shadow_disc(self, scene, name: str, center: np.ndarray, e: np.ndarray,
                         rx: float, ry: float, alpha: float, n: int = 48) -> None:
        """在平面上加一张半透明深色椭圆盘（程序化接触阴影的一层）。"""
        o3d = self.o3d
        nrm = float(np.hypot(e[0], e[1])) or 1.0
        ex = np.array([e[0] / nrm, e[1] / nrm, 0.0])     # 沿影方向（长半轴）
        ey = np.array([-ex[1], ex[0], 0.0])              # 侧向（短半轴）
        center3 = np.asarray(center, np.float64).reshape(3)
        ang = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
        ring = (center3[None, :]
                + np.cos(ang)[:, None] * (ex * rx)[None, :]
                + np.sin(ang)[:, None] * (ey * ry)[None, :])
        verts = np.vstack([center3, ring])
        tris = np.array([[0, i + 1, (i + 1) % n + 1] for i in range(n)], np.int32)
        m = o3d.geometry.TriangleMesh()
        m.vertices = o3d.utility.Vector3dVector(verts)
        m.triangles = o3d.utility.Vector3iVector(tris)
        m.compute_vertex_normals()
        mr = o3d.visualization.rendering.MaterialRecord()
        mr.shader = "defaultLitTransparency"          # 纯 alpha 不混合，必须显式选透明 shader
        mr.base_color = [0.0, 0.0, 0.0, float(alpha)]
        scene.add_geometry(name, m, mr)

    def _add_axes(self, scene, origin: np.ndarray, name: str, size: float) -> None:
        o3d = self.o3d
        for i, col in enumerate([[1, 0, 0], [0, 1, 0], [0, 0, 1]]):
            p = origin.astype(np.float64).copy()
            ls = o3d.geometry.LineSet()
            end = origin + np.eye(3)[i] * size
            ls.points = o3d.utility.Vector3dVector(np.vstack([p, end]))
            ls.lines = o3d.utility.Vector2iVector(np.array([[0, 1]], np.int32))
            self._add_line_geo(scene, f"{name}_{i}", ls, np.asarray(col, np.float64), 3.0)

    def _add_cameras(self, scene) -> None:
        if self.intrinsics is None or self.extrinsics is None:
            return
        o3d = self.o3d
        for cid in sorted(self.extrinsics):
            K = self.intrinsics.get(cid)
            if K is None:
                continue
            ext = self.extrinsics[cid]
            color = np.asarray(_CAM_COLORS[cid % len(_CAM_COLORS)], np.float64)

            R = np.asarray(ext.R, np.float64)
            t = np.asarray(ext.t, np.float64).reshape(3)
            Rw = R.T
            C = -Rw @ t

            # 视锥
            W, H = K.width, K.height
            Kinv = np.linalg.inv(K.K)
            scale = 1.6
            uv = np.array([[0, 0, 1], [W, 0, 1], [W, H, 1], [0, H, 1]], np.float64)
            base = (Kinv @ uv.T).T * scale
            pts = np.vstack([np.zeros(3), base])
            pts_w = (Rw @ (pts - t).T).T
            idx = [0, 1, 0, 2, 0, 3, 0, 4, 1, 2, 2, 3, 3, 4, 4, 1]
            ls = o3d.geometry.LineSet()
            ls.points = o3d.utility.Vector3dVector(pts_w)
            ls.lines = o3d.utility.Vector2iVector(np.asarray(idx, np.int32).reshape(-1, 2))
            self._add_line_geo(scene, f"cam_{cid}_frust", ls, color, 2.0)

            # 光心小球
            sph = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
            sph.translate(C)
            scene.add_geometry(f"cam_{cid}_sph", sph,
                               self._make_material(color, roughness=0.6))

            # 朝向短坐标架
            self._add_axes(scene, C, f"cam_{cid}_axes", 0.18)

    # ------------------------------------------------------------------
    # 人体层：逐帧移除+重建网格（Filament 无 TriangleMesh 的 update_geometry）
    # ------------------------------------------------------------------
    def set_lighting(self, scene) -> None:
        """环境光漫反射由 ``set_lighting`` 安装的 IBL 提供（本机已实测生效）。

        注意：Filament 太阳定向光在本机 OpenGL 4.1 下**不生效**（见模块 docstring），
        这里仍调用 SOFT_SHADOWS 仅为在太阳光可用的机器上叠加真实投影，不依赖它。
        """
        o3d = self.o3d
        scene.set_lighting(
            o3d.visualization.rendering.Open3DScene.LightingProfile.SOFT_SHADOWS,
            _SUN_DIR)
        s = scene.scene
        try:                                   # 弱环境光，避免阴影侧死黑
            s.enable_indirect_light(True)
            s.set_indirect_light_intensity(0.5)
        except Exception:  # noqa: BLE001
            pass
        try:                                   # 反方向弱填充光（不开阴影，避免双影）
            s.add_directional_light("fill", -_SUN_DIR, np.array([1, 1, 1], np.float32), 0.10)
        except Exception:  # noqa: BLE001
            pass

    def apply_person(self, scene, t: int) -> bool:
        """把 t 处的人体喂给场景。返回 True 表示本帧有人体被显示。"""
        person = self.tl.load_person(t)
        o3d = self.o3d
        # 总是先移除旧几何，再按当前状态重建（简单且无残留）
        for name in ("person", "bones", "shadow0", "shadow1"):
            try:
                scene.remove_geometry(name)
            except Exception:  # noqa: BLE001
                pass
        if person is None:
            self._last_t = None
            return False

        verts = person["vertices"]
        # 程序化接触阴影（画在人物网格之前，透明混合；见 contact_shadow_planes）
        for d in contact_shadow_planes(verts, self.floor_z):
            self._add_shadow_disc(scene, d["name"], d["center"], d["e"],
                                  d["rx"], d["ry"], d["alpha"])
        # 网格
        if self.faces is not None and len(verts):
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(verts.astype(np.float64))
            mesh.triangles = o3d.utility.Vector3iVector(self.faces.astype(np.int32))
            mesh.compute_vertex_normals()
            scene.add_geometry("person", mesh, self._mat_smpl)
        # 骨骼（若 npz 里有关节 24×3）
        joints = person.get("joints")
        if joints is not None and len(joints):
            ls = o3d.geometry.LineSet()
            ls.points = o3d.utility.Vector3dVector(joints[:24].astype(np.float64))
            ls.lines = o3d.utility.Vector2iVector(np.asarray(_SMPL_EDGES, np.int32))
            self._add_line_geo(scene, "bones", ls, _BONE_COLOR * 0.85, 2.0)
        self._last_t = t
        return True


def load_faces(out_dir: str, easymocap_root: str = "") -> Optional[np.ndarray]:
    """读 ``out_dir/recon_faces.npy``；缺则尝试用 EasyMocap SMPL 模型补一次。"""
    path = os.path.join(out_dir, "recon_faces.npy")
    if os.path.exists(path):
        return np.load(path).astype(np.int64)
    if not easymocap_root:
        return None
    import sys
    if easymocap_root not in sys.path:
        sys.path.insert(0, easymocap_root)
    try:
        from tabletennis.reconstruction.easymocap import EasymocapReconstructor  # noqa
        recon = EasymocapReconstructor(verbose=False)
        faces = np.asarray(recon.faces, np.int64)
        np.save(path, faces)
        print(f"[recon_player] 由模型补写 {path}")
        return faces
    except Exception as exc:  # noqa: BLE001
        print(f"[recon_player] ⚠ 补写 faces 失败：{exc}")
        return None


# ----------------------------------------------------------------------
# 离线单帧渲染（EGL headless）——验证 / 出 PNG
# ----------------------------------------------------------------------
def render_still(tl: ReconTimeline, t: int, width: int = 1280, height: int = 720,
                 out_png: str = "", root: Optional[str] = None) -> "np.ndarray":
    """渲染主时钟 t 处一帧到 RGB ndarray（可选写 PNG），供回放/验证。"""
    import open3d as o3d
    o3d.visualization.rendering  # noqa: F401
    from tabletennis.reconstruction.triangulate import load_camera_rig

    faces = load_faces(tl.out_dir)
    scene_b = ReconScene(tl, faces, camera_rig=load_camera_rig(root))

    r = o3d.visualization.rendering.OffscreenRenderer(width, height)
    sc = r.scene
    sc.set_background(np.array([0.09, 0.10, 0.13, 1.0], np.float32))
    scene_b.add_static(sc)
    scene_b.apply_person(sc, t)
    scene_b.set_lighting(sc)

    c = scene_b.center()
    eye = np.array([c[0] + 3.2, c[1] + 3.4, c[2] + 2.6], np.float32)
    r.setup_camera(50.0, c, eye, np.array([0.0, 0.0, 1.0]))
    img = r.render_to_image()
    arr = np.asarray(img).copy()
    if out_png:
        o3d.io.write_image(out_png, img)
    return arr


# ----------------------------------------------------------------------
# 交互回放窗口（gui.Application + SceneWidget，须主线程）
# ----------------------------------------------------------------------
class _PlayerApp:
    def __init__(self, tl: ReconTimeline, faces: Optional[np.ndarray],
                 scene_b: ReconScene, width: int, height: int, fps: float,
                 watch: bool = False):
        import open3d.visualization.gui as gui
        self.gui = gui
        self.tl = tl
        self.scene_b = scene_b
        self.watch = watch
        self.playing = not watch
        self.speed = 1.0
        self.t = 0.0                      # 当前播放位置（主时钟 float 帧号）
        self.fps = max(1.0, fps)
        self.last_render_t: Optional[int] = None
        self._need_show = True               # 首帧必画；watch 追到新帧时置 True
        self._watch_last = 0.0
        self._watch_prev_n = tl.n_ref
        self._watch_prev_ok = tl.n_ok()

        self.app = gui.Application.instance
        self.app.initialize()
        self.win = self.app.create_window(
            f"EasyMocap 重建回放 — {os.path.basename(tl.out_dir)}", width, height)
        self.widget = gui.SceneWidget()
        self.win.add_child(self.widget)

    def run(self) -> None:
        self._setup()
        last = time.perf_counter()
        while self.app.run_one_tick():
            now = time.perf_counter()
            dt = now - last
            last = now
            if self.watch:
                self._watch_poll(now)
            self._advance(dt)
            self._show()
            time.sleep(0.004)
        self.win.close()

    # -- watch：重建进行中，周期性重扫输出目录，追新帧 ----------------------
    def _watch_poll(self, now: float) -> None:
        if now - self._watch_last < 0.3:
            return
        self._watch_last = now
        prev_n, prev_ok = self._watch_prev_n, self._watch_prev_ok
        self.tl.reload()
        new_n, new_ok = self.tl.n_ref, self.tl.n_ok()
        if new_ok != prev_ok or new_n != prev_n:
            self._watch_prev_n, self._watch_prev_ok = new_n, new_ok
            print(f"[回放·追帧] 主时钟 {new_n} 帧（+{new_n - prev_n}），"
                  f"成功 {new_ok} 帧（+{new_ok - prev_ok}）"
                  + ("，重建已完成（index 就绪）" if self.tl.index_ready else ""))
            # 播到了旧末尾且追到新帧 → 接着播；n_ref 变长则允许继续前进
            if new_ok > prev_ok or new_n > prev_n:
                self._need_show = True
            if self.playing and new_n > prev_n:
                pass
            elif self.t >= max(0, prev_n - 1) and new_ok > prev_ok:
                self.playing = True
        if self.t >= max(0, new_n - 1):
            self.t = max(0, new_n - 1)
            if self.watch:
                self.playing = False   # 等下一批帧
        # 重建完成（index 就绪）：把整段播完
        if self.tl.index_ready and self.t < new_n - 1 and not self.playing:
            self.playing = True

    def _setup(self) -> None:
        w = self.widget
        o3d = _o3d()
        scene = o3d.visualization.rendering.Open3DScene(self.win.renderer)
        self.widget.scene = scene
        self.scene_b.add_static(scene)
        self.scene_b.set_lighting(scene)
        w.setup_camera(50.0, self.scene_b.bounds(), self.scene_b.center())
        w.set_on_key(self._on_key)
        print("[回放] 鼠标拖=旋转/平移/滚轮缩放；Space=暂停/继续，←/→=步进，"
              "Home/End=首/尾，R=复位视角，-/+=速度，Esc=退出")

    def _advance(self, dt: float) -> None:
        if not self.playing:
            return
        n = max(0, self.tl.n_ref - 1)
        if n <= 0:
            return
        self.t += dt * self.speed * self.fps
        if self.t > n:
            self.t = n
            self.playing = False

    def _show(self) -> None:
        t0 = int(self.t)
        if t0 == self.last_render_t and not self._need_show:
            return
        self._need_show = False
        self.last_render_t = t0
        person = self.scene_b.apply_person(self.widget.scene, t0)
        self.win.title = (f"EasyMocap 重建回放 — {os.path.basename(self.tl.out_dir)}"
                          f"  |  t {t0}/{max(0, self.tl.n_ref - 1)}"
                          f"  |  {'● 人物' if person else '— 无人'}"
                          f"  |  x{self.speed:.1f}")

    def _on_key(self, ev) -> bool:
        k = ev.key
        if ev.type != self.gui.KeyEvent.DOWN:
            return False
        KeyName = self.gui.KeyName
        handled = True
        if k == KeyName.SPACE:
            self.playing = not self.playing
        elif k == KeyName.LEFT:
            self.t = max(0.0, int(self.t) - 1)
        elif k == KeyName.RIGHT:
            self.t = min(self.tl.n_ref - 1, int(self.t) + 1)
        elif k == KeyName.HOME:
            self.t = 0.0
        elif k == KeyName.END:
            self.t = max(0, self.tl.n_ref - 1)
            self.playing = False
        elif k == ord("R"):
            self.widget.setup_camera(50.0, self.scene_b.bounds(), self.scene_b.center())
        elif k == ord("-"):
            self.speed = max(0.25, self.speed * 0.5)
        elif k == ord("=") or k == ord("+"):
            self.speed = min(16.0, self.speed * 2.0)
        elif k == KeyName.ESCAPE:
            self.playing = False
            self.win.close()
        else:
            handled = False
        if handled:
            print(f"[回放] t={int(self.t)} | {'暂停' if not self.playing else '播放'} | 速度 x{self.speed}")
        return handled


def play_gui(out_dir: str, easymocap_root: str = "", width: int = 1280,
             height: int = 720, fps: float = 0.0, watch: bool = False,
             root: Optional[str] = None) -> None:
    """在主线程弹出 Open3D 回放窗口并阻塞到关闭。``fps``<=0 用真实出帧率。"""
    _o3d()  # 尽早暴露缺依赖
    from tabletennis.reconstruction.triangulate import load_camera_rig

    tl = ReconTimeline(out_dir)
    faces = load_faces(out_dir, easymocap_root)
    scene_b = ReconScene(tl, faces, camera_rig=load_camera_rig(root))
    if fps <= 0:
        fps = tl.ref_rate_hz
    if tl.n_ref > 1 or not watch:
        print(f"[回放] {os.path.basename(out_dir)}：主时钟 {tl.n_ref} 帧，成功 {tl.n_ok()}，"
              f"faces={'有' if faces is not None else '缺'}"
              + ("，watch 追帧中" if watch else ""))
    if tl.n_ref <= 1 and not watch:
        print("[回放] 没有任何已重建帧，无事可播。")
        return
    _PlayerApp(tl, faces, scene_b, width, height, fps, watch=watch).run()
