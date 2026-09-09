"""离线重建结果回放查看器（Open3D Filament 新渲染器，带真实太阳阴影）。

``scripts/reconstruct_video.py`` 每帧把 SMPL 拟合结果存成
``out/frame_NNNNNN.npz``（vertices + joints + SMPL 参数…），并把网格拓扑存一次
``out/recon_faces.npy``。本模块把这些结果放回一个 3D 场景里播放：

- 场景 = 桌面系（与重建/在线 3D 一致）：桌面（z=0）+ 地面（z=-height）+ 相机视锥。
- 人体 = 每帧一个 SMPL 网格（13776 面，肤色），**凸凹明暗烘焙在顶点色里**
  （``bake_body_shading`` 逐顶点 Lambert，``defaultUnlit`` 显示）——朝光面亮、
  颌下/腋下/腹股沟等凹处法线不朝光自然变暗，一眼看出身体起伏。不再依赖 Filament
  实时光照（本机 EGL 实测 cast_shadows 不产影、fill 低强度无效；高 lux 方向光能出
  漫反射但真阴影不可靠，故明暗走烘焙、地上阴影走程序化假影，双保险）。
- 地上阴影 = 程序化假影（``defaultLitTransparency`` 透明混合，**两层都画在地板平面
  ``floor_z`` 上**——重建的 SMPL 脚底常悬空几 cm，贴脚画盘会悬在地板上方）：
  ① 两片脚下接触椭圆（核心深影，把足底到地板的空间感压实）；② **整身投影软影**：
  把所有 SMPL 顶点沿光水平方向投到地面 → 凸包填充盘，带身形且随姿态伸长——合起来
  才是肉眼看得出的影子。烘焙光源与假影太阳取同一侧，观感一致。
- 球轨迹 = 若有 ``ball_trajectory.npz``（reconstruct_video.py 的球重建输出），逐帧画
  当前位置红球 + 到 t 为止的轨迹线（``defaultUnlit`` 红球 + ``unlitLine`` 亮橙，
  按「缺测 >5 帧」断开，不把不同回合连成一条大线）。

动态层（人体/影子/骨骼/球/轨迹）在 Open3D 0.19 Filament 渲染器里有两条实现路径，
由 ``ReconScene(mode=...)`` 选择：

- ``mode="mesh"``：平滑三角网格（明暗烘焙进顶点色）。但 Filament 的
  ``Scene.update_geometry`` 此 build 只收 PointCloud，三角网格逐帧动画只能
  remove+add——每次重挂留不可回收的引擎级残留、约 1 万次即段错误（实测 8k~11.5k
  ops，RSS +0.52MB/op）。故 mesh 路径**只**用于 render_still：每次新建场景加一次、
  不累积，网格安全。GUI 长播绝不能用它。
- ``mode="points"``（GUI 回放用）：把人体表面（SMPL 顶点 + 每面质心 ≈ 20666 点）、
  地板影（顶点沿光水平投影）、骨骼（关节连线等分点）、球（单点）、轨迹（逐采样点）
  全部做成 ``t.geometry.PointCloud``，add 一次后每帧 ``Scene.update_geometry`` 原地
  改顶点缓冲（0 次 remove+add → 没有实体上限，播放/暂停全程 0 churn，可无限长播）。
  观感上点云人体替代平滑网格——用户拍板「只要点云就行」。mesh/points 两套逻辑共存：
  ``apply_people``/``apply_ball`` 按 ``self.mode`` 分派到 ``_apply_*_mesh`` 或
  ``_apply_*_cloud``。

两个入口共用同一套 ``ReconScene``（把几何加进任意 ``Open3DScene``）：

- 离线渲染单帧：``render_still(out_dir, t, ...)``（EGL headless，可出 PNG / 验证）；
- 交互窗口回放：``play_gui(out_dir, ...)``（``gui.Application`` + ``SceneWidget``，
  鼠标转视角，键盘控制播放），须在主线程调用。

参考 playback 语义：t 是主时钟（参考相机）脉冲号/帧号。某人帧期间显示该人网格；
``no_person``/失败帧清空人体；被 ``--stride`` 跳过的帧保持上一姿态。
"""
from __future__ import annotations

import gc
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

# 多人配色：person 0 用肤色，其余用区分度高的纯色（按身份稳定着色）
_PERSON_COLORS = [
    _SMPL_SKIN,
    np.array([0.36, 0.56, 0.86], np.float32),   # 蓝
    np.array([0.86, 0.36, 0.40], np.float32),   # 红
    np.array([0.42, 0.72, 0.46], np.float32),   # 绿
]

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

# 球轨迹渲染：当前位置球（红）+ 到 t 为止的轨迹线（亮橙，缺测 >GAP 帧断开，不跨回合连线）
_BALL_COLOR = np.array([0.95, 0.25, 0.18], np.float32)       # 当前位置球
_BALL_TRAIL_COLOR = np.array([1.00, 0.55, 0.28], np.float32)  # 轨迹线
_2D_SEEK_GAP = 5              # 2D 叠加目标帧与上次相差超过该帧数就 seek（顺序步进保持顺序解）
# mesh 路径专用（points 模式每帧原地 update，无 remove+add 残留、不需要降频）：
_AUX_CADENCE = 3             # mesh 播放中「次层几何」（投影软影/接触盘/骨骼/球轨迹）每 N 个
                             # 内容帧才重挂一次——Filament remove+add 每次都有不可回收残留，
                             # 主层（人体网格/球）每帧必换，次层掉到 ~1/3 肉眼无感但把每次
                             # 播放累积的换挂次数砍到 ~1/3（见 _apply_*_mesh）
_MESH_HZ = 60.0              # mesh 播放中人体网格最大重挂率：显示器 60Hz，内容 100fps 时屏上
                             # 分不到每帧一次刷新，把上限定在刷新率省 ~1/2 上传
_BALL_RADIUS = 0.02                                          # 乒乓球半径 20mm
_BALL_TRAIL_GAP = 5                                          # 断连阈值（帧）：缺测 >5 帧即断开

# ---- 点云回放模式（GUI 主路径；消除 remove+add 的实体上限 → 永不崩/可无限长播）----
# Filament 里 PointCloud 可用 Scene.update_geometry 原地改顶点缓冲（无新实体、无泄漏，
# 实测 8000 次更新 RSS 平、0.07ms/次）；三角网格不行——update_geometry 本 build 只收
# PointCloud，网格逐帧动画只能 remove+add（~1 万次后进程段错误，实测 8k~11.5k）。
# 故 GUI 播放把**所有动态层**都做成点云、每帧原地 update：
#   人体 = SMPL 表面顶点 + 每面质心（约 6890+13776 点，密度够实）；
#   地板影 = 人体顶点沿光水平方向投影到地板（不透明深色点，无重叠加深）；
#   骨骼 = 关节连线等分点；球 = 单点；轨迹 = 逐采样点。全部 add 一次、之后只
#   update / show/hide，播放+暂停全程 0 次 remove+add → 撞不到崩溃点。
# render_still（离线 PNG）仍用平滑三角网格：每次新建场景只加一次、无累积，网格安全。
_CLOUD_PT_BODY = 3.0        # 人体表面点大小（屏幕 px）
_CLOUD_PT_SHADOW = 5.0      # 地板影点大小（大点才能盖满成块）
_CLOUD_PT_BONE = 4.0        # 骨骼点大小
_CLOUD_PT_BALL = 7.0        # 球点大小（20mm 球在 3-6m 外本就亚像素，画个醒目红点）
_CLOUD_PT_TRAIL = 3.0       # 球轨迹点大小
_SHADOW_STRIDE = 2          # 地板影点抽稀步长（6890 → ~3445 点/人）
_SHADOW_Z_OFF = 0.004       # 影点离地板高度，防与地板平面 z-fight（同旧接触盘）
_TRAIL_CAP = 4096           # 球轨迹点云容量上限（watch 长时段也只保留最近这些）
_BONE_DOTS = 8              # 每根骨骼线段等分点数
_SHADOW_PT_COLOR = tuple(float(c * 0.5) for c in _FLOOR_COLOR)  # 不透明深地板色 = 影


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
        hold_gaps: 播放观感参数——连续 no_person ≤ hold_gaps 帧时不清空人体
                   （保持上一成功帧），超过才清空。100Hz 下拟合常单帧/两三帧抖动
                   掉点，直接清空会让人体高频闪没；默认 0 = 语义精确（与 index 一致）。
    """

    def __init__(self, out_dir: str, hold_gaps: int = 0):
        self.out_dir = out_dir
        self.meta = self._load_json("recon_meta.json")
        self.index_ready = os.path.exists(self._p("recon_index.npz"))
        self.files: Dict[int, str] = {}
        self.empty_ts: List[int] = []
        self.n_ref = 0
        self.state = np.zeros(0, dtype=np.int64)
        self.ref_rate_hz = 100.0
        self.hold_gaps = int(hold_gaps)
        self.ball_ok = False
        self.ball_refs = np.zeros(0, dtype=np.int64)
        self.ball_X = np.zeros((0, 3), dtype=np.float64)
        self.ball_valid = np.zeros(0, dtype=bool)
        self.ball_pos: Dict[int, np.ndarray] = {}
        self._people_cache: Dict[str, List[dict]] = {}
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

        # 3b) 只重建球（--ball-only）：无姿态帧 / 无 index 时，用 ball_trajectory 的帧号定 n_ref
        self._load_ball()
        if self.n_ref <= 0 and self.ball_ok and len(self.ball_refs):
            self.n_ref = int(self.ball_refs[-1]) + 1

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
        """state[t] = 该显示的 file 下标（最近一次成功帧，遇 empty 清空）。

        empty（no_person/fail）连续帧数 ≤ ``self.hold_gaps`` 时保持上一成功帧
        （短抖动不闪没）；超过才清空，且清空后持续到下一成功帧。hold_gaps=0 时
        与旧语义一致（一遇 empty 立刻清空）。非 empty 的 stride 跳过帧天然保持。
        """
        state = np.full(max(1, self.n_ref), -1, dtype=np.int64)
        file_idx = {t: i for i, t in enumerate(sorted(self.files))}
        file_order = sorted(self.files)
        file_list = [self.files[t] for t in file_order]
        empty_set = set(self.empty_ts)
        latest = -1
        empties = 0
        for t in range(len(state)):
            if t in empty_set:
                empties += 1
            else:
                empties = 0
                if t in file_idx:
                    latest = file_idx[t]
            if empties > self.hold_gaps:
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

    def load_people(self, t: int) -> List[dict]:
        """读 t 处 npz：返回 ``[{vertices (6890,3), joints (24,3)}, ...]``（多人）。

        旧单帧格式（无 ``n_people``）当作 1 人；新格式 person 0 无后缀、p≥1 用 ``_p``。
        读失败返回空列表。结果按文件路径缓存（同一帧被 hold_gaps/stride 复用时不再解压）。
        """
        path = self.person_path_at(t)
        if path is None:
            return []
        return self._load_people_path(path)

    def _load_people_path(self, path: str) -> List[dict]:
        """按路径读一帧（带内存缓存）；失败不缓存，下次可重试（watch 半截文件）。"""
        cached = self._people_cache.get(path)
        if cached is not None:
            return cached
        out: List[dict] = []
        try:
            with np.load(path) as z:
                n_people = int(z["n_people"]) if "n_people" in z else 1
                for p in range(n_people):
                    suf = "" if p == 0 else f"_{p}"
                    verts = np.asarray(z[f"vertices{suf}"], np.float32).reshape(-1, 3)
                    joints = None
                    if f"joints{suf}" in z:
                        joints = np.asarray(z[f"joints{suf}"], np.float32).reshape(-1, 3)
                    out.append({"vertices": verts, "joints": joints})
        except Exception as exc:  # noqa: BLE001 —— 重建正在写/文件半截
            print(f"[recon_player] ⚠ 读帧失败：{exc}")
            return []
        self._people_cache[path] = out
        return out

    def preload(self) -> None:
        """把已存在的成功帧全部读进内存（回放前预热）。

        逐帧 ``np.load`` + 解压 ~1.1ms，100fps 下占掉一帧预算的 1/10 且带磁盘 I/O
        抖动；预热后回放不再碰磁盘。内存 ≈ 帧数 × 人数 × (6890+24)×3×4B
        （本例 713 帧 2 人 ≈ 118MB），短视频完全可接受。watch 模式别用（帧还在写）。
        """
        for path in self.files.values():
            self._load_people_path(path)

    # ------------------------------------------------------------------
    # 球轨迹（ball_trajectory.npz）：ball_pos_at 取当前/coast 位置，
    # ball_trail_upto 取到 t 为止的连续轨迹段（缺测 >GAP 帧断开，不跨回合连线）
    # ------------------------------------------------------------------
    def _load_ball(self) -> None:
        """读 ``ball_trajectory.npz``（可能不存在 / 半截），失败则置 ball_ok=False。"""
        path = self._p("ball_trajectory.npz")
        if not os.path.exists(path):
            self.ball_ok = False
            self.ball_refs = np.zeros(0, dtype=np.int64)
            self.ball_X = np.zeros((0, 3), dtype=np.float64)
            self.ball_valid = np.zeros(0, dtype=bool)
            self.ball_pos = {}
            return
        try:
            z = np.load(path)
            refs = np.asarray(z["ref_frame"], np.int64)
            X = np.asarray(z["X"], np.float64).reshape(-1, 3)
            n = min(len(refs), len(X))
            refs, X = refs[:n], X[:n]
            valid = np.isfinite(X).all(axis=1)
            self.ball_refs = refs
            self.ball_X = X
            self.ball_valid = valid
            self.ball_pos = {int(t): X[i] for i, t in enumerate(refs) if valid[i]}
            self.ball_ok = bool(self.ball_pos)
        except Exception as exc:  # noqa: BLE001 —— 重建正在写 / 文件半截
            print(f"[recon_player] ⚠ 读 ball_trajectory.npz 失败：{exc}")
            self.ball_ok = False
            self.ball_refs = np.zeros(0, dtype=np.int64)
            self.ball_X = np.zeros((0, 3), dtype=np.float64)
            self.ball_valid = np.zeros(0, dtype=bool)
            self.ball_pos = {}

    def ball_pos_at(self, t: int) -> Optional[np.ndarray]:
        """t 处球心 ``(3,)``；当前帧无球则回看最近 ``hold_gaps`` 帧内的有效球（防闪没）。"""
        if not self.ball_ok:
            return None
        X = self.ball_pos.get(int(t))
        if X is not None:
            return X
        for dt in range(1, max(1, self.hold_gaps + 1)):
            X = self.ball_pos.get(int(t) - dt)
            if X is not None:
                return X
        return None

    def ball_trail_upto(self, t: int) -> List[np.ndarray]:
        """到 t 为止的球轨迹，按「连续缺测 ≤ _BALL_TRAIL_GAP 帧」切成若干段。

        断连处不连线（否则会把不同回合 / 长时间无球段错误连成一条大线）。每段 ``(M,3)``。
        """
        if not self.ball_ok:
            return []
        mask = self.ball_refs <= int(t)
        refs = self.ball_refs[mask]
        valid = self.ball_valid[mask]
        X = self.ball_X[mask]
        segs: List[np.ndarray] = []
        cur: List[np.ndarray] = []
        last_ref = None
        for i in range(len(refs)):
            if not valid[i]:
                last_ref = None
                if cur:
                    segs.append(np.asarray(cur, np.float64))
                    cur = []
                continue
            if last_ref is not None and (int(refs[i]) - last_ref) > _BALL_TRAIL_GAP:
                if cur:
                    segs.append(np.asarray(cur, np.float64))
                    cur = []
            cur.append(X[i])
            last_ref = int(refs[i])
        if cur:
            segs.append(np.asarray(cur, np.float64))
        return [s for s in segs if len(s) >= 2]


# ----------------------------------------------------------------------
# 程序化接触阴影
# ----------------------------------------------------------------------
def contact_shadow_planes(verts: np.ndarray, floor_z: float,
                          sun_dir: Optional[np.ndarray] = None) -> List[dict]:
    """由 SMPL 顶点算脚下地面上的软影椭圆参数（纯 numpy，可单测）。

    为什么自己画影子：本机 EGL/屏幕窗口实测 Filament **cast_shadows 不产真影**
    （高 lux 方向光能出漫反射明暗，但投射阴影在这条渲染路径不可靠）。所以「真实
    投射阴影」做不到，改在人物脚下画两层半透明深色椭圆
    （``defaultLitTransparency`` 混合已验证可用），位置/朝向跟着太阳水平方向走，
    观感即接触阴影。人体自身的凸凹明暗另走顶点色烘焙（``bake_body_shading``）。

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
    # 盘心 z 钉在地板平面上（与整身投影软影同层，见 _add_cast_shadow）：重建的
    # SMPL 脚底常悬在地板上方几 cm（median ~-0.71 vs floor -0.76），若盘子贴脚
    # 平面就会悬在地板上方，低视角看是一块脱开的深斑；画在地板才是真「地上
    # 影子」，脚底那几 cm 空隙远看不可辨。盘心 xy 仍是立足点 + 沿影方向偏移。
    z = floor_z + 0.004                 # 略高于地板防 z-fight
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
# 整身软影（cast shadow）：把整个人体顶点沿光水平方向投到地面，做一个
# 「带身形的拉长影子」——两片小椭圆只有 ~0.3m，真场景里几乎看不见（实测
# 1280×720 只影响 ~200px）；投影整身轮廓才看得出人在空间里有影子。
# ----------------------------------------------------------------------
_CAST_K = 0.5            # 每米高度沿影方向的伸长系数（太阳 ~63° 俯角的感觉）
_CAST_ALPHA = 0.26       # 软影透明度（比接触盘淡，覆盖更大范围）
_CAST_SUBSAMPLE = 3      # 顶点抽稀（6890→~2300，凸包足够；省建包时间）


# ----------------------------------------------------------------------
# 人体自身明暗（凸凹可读）——烘焙进顶点色
# ----------------------------------------------------------------------
# 方向性漫反射（Lambert）在纯 numpy 逐顶点算好写进顶点色，person 用
# ``defaultUnlit`` 直接显示，**不依赖** Filament 太阳/自阴影（EGL 实测
# cast_shadows 不产影、fill/IBL 低强度无效；高 lux 方向光虽能出漫反射，但
# 真阴影不可靠）。朝光面亮、背光/法线朝下的凹处（颌下/腋下/腹股沟）自然变暗
# → 一眼看出人体哪里凸哪里凹。光源方向与假影的太阳方向取同一侧，观感一致。
_KEY_LIGHT_FROM = np.array([-0.45, -0.25, 0.86], np.float64)  # 指向光源（光从上方略偏左前）
_KEY_LIGHT_FROM = _KEY_LIGHT_FROM / np.linalg.norm(_KEY_LIGHT_FROM)
_BAKE_AMBIENT = 0.52     # 环境光：背光/凹处最低亮度（≈0.52×肤色，保证全身可见）
_BAKE_KEY = 0.85         # 主光漫反射：朝光面 ≈ ambient + key（肤色顶格、强浮雕）


def bake_body_shading(normals, skin=_SMPL_SKIN, ambient=_BAKE_AMBIENT,
                      key=_BAKE_KEY, light_from=_KEY_LIGHT_FROM) -> np.ndarray:
    """把一束方向光漫反射烘焙成逐顶点颜色，让 SMPL 网格本身带明暗（凸凹可读）。

    纯 numpy、无渲染器依赖，返回 ``(V,3) float64 ∈ [0,1]`` 线性色，供
    ``defaultUnlit`` 顶点色使用。``light_from`` 是「指向光源」的单位向量；
    法线朝光源面 ≈ ``skin*(ambient+key)``，背光面 ≈ ``skin*ambient``。
    因为凹处（颌下/腋下/腹股沟/脐窝）法线不朝光源，自然比凸面暗一档。
    """
    n = np.asarray(normals, np.float64)
    lf = np.asarray(light_from, np.float64)
    lf = lf / np.linalg.norm(lf)
    lambert = np.clip(n @ lf, 0.0, 1.0)
    # 孤立顶点（不参与任何面）的 compute_vertex_normals 可能是 NaN → 抹成 0（环境光底）
    lambert = np.nan_to_num(lambert, nan=0.0)
    sk = np.asarray(skin, np.float64).reshape(3)
    col = sk[None, :] * (float(ambient) + float(key) * lambert[:, None])
    return np.clip(col, 0.0, 1.0)


def convex_hull2d(points) -> Optional[np.ndarray]:
    """二维凸包（CCW 有序）：优先 ``cv2.convexHull``（C 速度），缺 cv2 回退纯 numpy。

    为什么不用 open3d ``PointCloud.compute_convex_hull``：投影点全落在同一个
    ``z=const`` 平面上，3D qhull 会因退化输入抛精度错（QH6154）或每帧往 stderr
    打 QH7089 精度告警（100fps 回放会刷屏），还得 ``joggle_inputs=True`` 才能跑。
    直接对 ``(N,2)`` 做 2D 凸包没有这些坑，输出即 CCW 多边形。

    Returns:
        ``(M,2)`` 凸包顶点（CCW）；退化输入（<3 个不共线点 / 全共线）返回 None。
    """
    arr = np.asarray(points, np.float64).reshape(-1, 2)
    if len(arr) < 3:
        return None
    # 快路径：cv2.convexHull（Sklansky，C 实现）。回放每帧都要算整身投影软影的凸包，
    # 纯 numpy 的 monotone chain 在 ~2300 点上要 ~2ms/人（实测），cv2 只要 ~0.1ms
    # （~17×），是播放流畅的关键一环。clockwise=False 在数学 xy 坐标下恰好返回 CCW
    # （实测单位正方形 shoelace 面积 +1），可直接当三角扇用；全共线时返回 2 点，
    # 由下方 <3 判断兜成 None。
    try:
        import cv2
        hull = cv2.convexHull(arr.astype(np.float32),
                              clockwise=False, returnPoints=True)
        hull = np.asarray(hull, np.float64).reshape(-1, 2)
        if len(hull) < 3:
            return None                        # 全共线
        return hull
    except Exception:  # noqa: BLE001 —— cv2 不可用则回退纯 numpy（下）
        pass
    # 回退：纯 numpy monotone chain（无 cv2 依赖时仍可用；本段为旧实现原样保留）。
    # 一次 C 速度 lexsort 得到按 x/y 有序的 tuple 列表。别用 `for x,y in arr` 迭代
    # numpy 2D 行（每行造 view，2300 行 ~10ms，cast 每帧调用会拖垮播放）；
    # monotone chain 的 cross<=0 会把重复/共线点 pop 掉，无需显式去重。
    xs = arr[:, 0]
    ys = arr[:, 1]
    order = np.lexsort((ys, xs))
    pts = list(zip(xs[order].tolist(), ys[order].tolist()))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]              # 首尾不重复，CCW
    if len(hull) < 3:
        return None                             # 全共线
    return np.asarray(hull, np.float64)


def project_floor_shadow(verts, floor_z: float, sun_dir=None, k: float = _CAST_K):
    """把每个顶点沿光水平方向投影到地面，返回地面足迹 ``(N,2)``。

    ``sun_dir`` 语义与 :func:`contact_shadow_planes` 一致：归一化光传播方向，
    只用它的水平分量当「影子伸长方向」。头顶更高 → ``(z - floor_z)*k`` 越长的
    影尾；脚底 z≈floor → 基本贴脚。纯 numpy，可单测。
    """
    if verts is None:
        return np.empty((0, 2), np.float64)
    verts = np.asarray(verts, np.float64)
    n = len(verts)
    if n == 0:
        return np.empty((0, 2), np.float64)
    sd = np.asarray(_SUN_DIR if sun_dir is None else sun_dir, np.float64)
    h = sd[:2]
    nh = float(np.linalg.norm(h))
    e = np.array([1.0, 0.0]) if nh < 1e-6 else h / nh
    tail = np.clip(verts[:, 2] - floor_z, 0.0, None) * k
    return verts[:, :2] + tail[:, None] * e[None, :]


# ----------------------------------------------------------------------
# 场景：把桌面/地面/相机 + 人体几何喂给任意 Open3DScene
# ----------------------------------------------------------------------
class ReconScene:
    """向一个 ``Open3DScene``（GUI widget 或 OffscreenRenderer）搭建重建回放场景。"""

    def __init__(self, tl: ReconTimeline, faces: Optional[np.ndarray],
                 table: Optional[Table3D] = None,
                 camera_rig: Optional[tuple] = None,
                 cast_shadow: bool = True,
                 ball_trail: bool = True,
                 mode: str = "mesh"):
        """``mode="mesh"``（默认，离线 render_still 用）：平滑三角网格，remove+add 更新，
        仅适用于每次新建场景、调用次数少的离屏渲染；``mode="points"``（GUI 回放用）：
        全部动态层为点云、``Scene.update_geometry`` 原地更新，无 remove+add —— 唯一能
        扛住长时间/反复播放不段错误的路径（见模块 docstring「点云回放模式」）。
        """
        o3d = _o3d()
        self.o3d = o3d
        self.mode = mode
        self.tl = tl
        self.table = table or Table3D()
        self.floor_z = -self.table.height
        self.intrinsics, self.extrinsics = (camera_rig or (None, None))
        self.faces = faces                       # (13776,3) int；None 则只画关节
        self.cast_shadow = cast_shadow           # 脚下整身投影软影（默认开）
        self.ball_trail = ball_trail             # 球轨迹线（默认开）
        # person 用 defaultUnlit：肤色+明暗烘焙在顶点色里（bake_body_shading），
        # 不参与场景光照 —— 人体凸凹明暗由烘焙保证（Filament 真阴影在 EGL 不可靠）。
        # 多人身份区分靠烘焙时把对应 _PERSON_COLORS 的 skin 色传进去（见 apply_people）。
        self._mat_smpl = self._make_material(_SMPL_SKIN, roughness=0.62)
        self._mat_smpl.shader = "defaultUnlit"
        self._mat_smpl.base_color = [1.0, 1.0, 1.0, 1.0]
        self._mat_bones = self._make_material(_BONE_COLOR, roughness=0.8)
        self._mat_ball = self._make_material(_BALL_COLOR, roughness=0.35)
        self._mat_ball.shader = "defaultUnlit"   # 小球不参与光照，保证 2cm 红球始终醒目
        self._last_t = None
        self._last_n_people = 0
        self._applied_path: Optional[str] = None   # 上一帧显示的人体 npz 路径（复用则跳过重建）
        self._seq = 0                # 内容帧序号：次层几何按 _AUX_CADENCE 降频重挂
        self._last_mesh_up = 0.0     # 人体网格上一次重挂的墙钟（播放中按 _MESH_HZ 限频）
        self.n_added = 0
        # —— mesh 路径（render_still 离屏）的持久对象池。GUI 播放走点云路径
        #    （mode="points"，见下 _cl_* 状态），不需要这些槽；这里只服务网格模式。 ——
        # 每帧 NEW 一个 TriangleMesh 再 remove+add 会给 Open3D 场景/Filament 留下
        # ~1MB/帧引擎级残留、~1300 次全场景内容迭代即段错误。故按 (person 槽位) 复用
        # 同一对象就地改写顶点缓冲后 remove+add：视觉上仍是三角网格正确更新的唯一途径
        # （update_geometry 此 build 只收 PointCloud），但 remove+add 的引擎级残留
        # clear_geometry / 换新 Open3DScene 都清不掉，只能靠压低重挂频率推迟——mesh 模式
        # 因此**只**用于 render_still（每次新建场景调用少数次、不累积）。
        self._faces_i32 = (np.asarray(self.faces, np.int32)
                           if self.faces is not None else None)
        self._slots: List[Optional[dict]] = []   # 每槽：{mesh,bones,cast,discs,mat_bones}
        self._ball_mesh: Optional["object"] = None
        self._ball_base: Optional[np.ndarray] = None
        self._trail_ls: Optional["object"] = None
        # —— 点云回放模式（mode="points"）状态 ——
        self._cl_objs: Dict[str, dict] = {}     # 云名 -> {"pcd": t.PointCloud, "cap": int}
        self._cl_added: Dict[str, bool] = {}    # 云名是否已 add 进当前场景
        self._cl_vis: Dict[str, bool] = {}      # 云名当前可见性
        self._cl_nmesh: List[Optional["object"]] = []  # 每人物槽一个 CPU 法线网格（不进场景）
        self._cl_people_shown: int = -1         # 上帧显示的人物槽数（用于 hide 多余槽）
        self._cl_sample: Optional[tuple] = None # (idx(N,3)int, w(N,3)f64) 人体表面加密采样表
        self._cl_n_v = 0                        # SMPL 顶点数（6890，采样表按它建一次）

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

    def _fill_shadow_disc(self, mesh, center: np.ndarray, e: np.ndarray,
                          rx: float, ry: float, n: int = 48) -> None:
        """把一张半透明深色椭圆盘（接触阴影一层）的几何填进 ``mesh``（持久复用）。

        ``mesh`` 由调用方持有、跨帧复用：每帧只重写顶点环（三角拓扑固定，首帧建一次）。
        """
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
        if len(mesh.triangles) == 0:
            tris = np.array([[0, i + 1, (i + 1) % n + 1] for i in range(n)], np.int32)
            mesh.triangles = o3d.utility.Vector3iVector(tris)
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.compute_vertex_normals()

    def _fill_cast_shadow(self, mesh, verts: np.ndarray) -> bool:
        """把整身投影软影的凸包填充盘几何填进 ``mesh``；False = 本帧退化画不出。

        两片小接触椭圆（~0.3m）在真场景里小到几乎看不见；投影整身轮廓才能给出
        一个「带身形、随姿态伸长」的影子，把 SMPL 人明显锚定在地面上。
        """
        o3d = self.o3d
        v = np.asarray(verts, np.float64)
        if len(v) < 8:
            return False
        # 影平面固定在大地板上（不是脚平面）：重投影的人体可能因重建浮空几厘米，
        # 影盘若贴脚平面就会悬在地板上方；画在 floor_z 才是真「影子在地上」。
        z = self.floor_z + 0.004
        proj = project_floor_shadow(v[::_CAST_SUBSAMPLE], self.floor_z)
        hull = convex_hull2d(proj)               # 2D CCW 凸包（同平面，勿用 3D qhull）
        if hull is None or len(hull) < 3:
            return False
        # 质心 apex 三角扇填满凸多边形（凸包 CCW → 三角全 CCW → 法线 +Z 朝上）
        c = hull.mean(axis=0)
        ring = np.column_stack([hull, np.full(len(hull), z)])
        verts3 = np.vstack([np.append(c, z), ring])
        m0, m = 1, len(ring)
        tris = np.array([[0, i, i + 1] for i in range(m0, m)] + [[0, m, m0]],
                        np.int64)
        mesh.vertices = o3d.utility.Vector3dVector(verts3)
        mesh.triangles = o3d.utility.Vector3iVector(tris)
        mesh.compute_vertex_normals()
        return True

    def _transp_material(self, alpha: float):
        """半透明深色材质（接触/投影软影共用；材质非几何，无泄漏问题）。"""
        o3d = self.o3d
        mr = o3d.visualization.rendering.MaterialRecord()
        mr.shader = "defaultLitTransparency"      # 纯 alpha 不混合，必须显式选透明 shader
        mr.base_color = [0.0, 0.0, 0.0, float(alpha)]
        return mr

    def _slot(self, p: int) -> dict:
        """第 p 个人物的持久几何槽（按需增长；跨帧复用避免每帧 NEW 网格 → 泄漏）。"""
        while len(self._slots) <= p:
            self._slots.append({"mesh": None, "bones": None, "cast": None,
                                "mat_bones": None, "mat_cast": None,
                                "discs": [None, None], "mat_discs": [None, None]})
        return self._slots[p]

    def _hide(self, scene, name: str) -> None:
        """把某实体从场景摘掉（仅在它真在场时才 remove；不会每帧 remove → 不积 defer-free）。"""
        if scene.has_geometry(name):
            scene.remove_geometry(name)

    def _stage(self, scene, name: str, geo, mat) -> None:
        """remove+add 一个**持久复用**的几何对象，让画面真正换新几何。

        这是 Filament GUI 里唯一有效的逐帧更新途径（实验定论）：同名 ``add_geometry``
        是 no-op 画面永远停在第 1 帧；``Scene.update_geometry`` 本 build 只收 PointCloud，
        不收三角网格。remove+add 会留**引擎级**（GL 上下文内）不可回收的残留，只能靠把
        它压低来推迟段错误（clear_geometry / 换 Open3DScene 都不释放引擎缓存；关窗重启
        进程才清零）。压低手段：① 对象持久复用（不每帧 NEW）；② 次层几何降频
        （``_AUX_CADENCE``）；③ 人体网格播放中限速 ``_MESH_HZ``。综合后每遍播放残留
        只剩 ~1/4（714 帧实测 +58MB vs 原 +432MB），单遍 1000+ 帧安全、反复重播也能
        几十遍。
        """
        self._hide(scene, name)
        scene.add_geometry(name, geo, mat)

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
        """静态场景（桌/地/视锥）的环境光漫反射由 ``set_lighting`` 安装的 IBL 提供。

        人体不受此影响：person 走 ``defaultUnlit`` + 顶点色烘焙（``bake_body_shading``），
        明暗确定、不依赖渲染器实时光照。此处仅装饰静态物体，太阳/阴影细节见模块 docstring。
        """
        o3d = self.o3d
        scene.set_lighting(
            o3d.visualization.rendering.Open3DScene.LightingProfile.SOFT_SHADOWS,
            _SUN_DIR)
        s = scene.scene
        try:                                   # 弱环境光，静态物体阴影侧不黑死
            s.enable_indirect_light(True)
            s.set_indirect_light_intensity(0.5)
        except Exception:  # noqa: BLE001
            pass

    def apply_people(self, scene, t: int, full: bool = True) -> bool:
        """把 t 处的（可能多个）人体喂给场景。返回 True 表示本帧有人体被显示。

        ``mode="points"``（GUI 回放）→ :meth:`_apply_people_cloud`（点云原地更新，
        永不段错误）；``mode="mesh"``（render_still 离屏 PNG）→ :meth:`_apply_people_mesh`
        （平滑网格 remove+add，每次新建场景只调少数次、不累积，安全）。
        """
        if self.mode == "points":
            return self._apply_people_cloud(scene, t, full)
        return self._apply_people_mesh(scene, t, full)

    def _apply_people_mesh(self, scene, t: int, full: bool = True) -> bool:
        """平滑三角网格路径（render_still 离屏 PNG；每次新建场景仅调用少数次）。

        **更新语义 = remove+add（Filament 里三角网格画面真正换新的唯一途径）**：同名
        ``add_geometry`` 是 no-op，``Scene.update_geometry`` 本 build 只收 PointCloud。
        网格 remove+add 有引擎级残留、~1 万次后进程段错误——所以本路径**只能**用于
        「每次新建场景、调用次数很少」的离屏渲染；GUI 播放请用点云路径。
        """
        o3d = self.o3d
        path = self.tl.person_path_at(t)
        if path is not None and path == self._applied_path and self._last_n_people > 0:
            return True        # 与上一帧同一份结果（hold_gaps/stride 复用）→ 几何已就位
        people = self.tl.load_people(t)
        do_aux = full or (self._seq % _AUX_CADENCE == 0)
        self._seq += 1
        # 人数比上一帧少 → 把多出来的人体实体摘掉（槽仍在，之后可再复用）
        for p in range(len(people), self._last_n_people):
            for name in (f"person_{p}", f"bones_{p}", f"cast_{p}",
                         f"shadow0_{p}", f"shadow1_{p}"):
                self._hide(scene, name)
        self._last_n_people = len(people)
        self._applied_path = path
        if not people:
            self._last_t = None
            return False

        for p, person in enumerate(people):
            verts = person["vertices"]
            skin = _PERSON_COLORS[p % len(_PERSON_COLORS)]
            slot = self._slot(p)
            if do_aux:   # —— 次层：投影软影 + 接触盘 + 骨骼（降频，见 _AUX_CADENCE）——
                # 整身投影软影（把人体锚在地面）→ 再叠两片脚下接触盘（核心深影）
                if self.cast_shadow and len(verts) >= 8:
                    if slot["cast"] is None:
                        slot["cast"] = o3d.geometry.TriangleMesh()
                        slot["mat_cast"] = self._transp_material(_CAST_ALPHA)
                    if self._fill_cast_shadow(slot["cast"], verts):
                        self._stage(scene, f"cast_{p}", slot["cast"], slot["mat_cast"])
                    else:
                        self._hide(scene, f"cast_{p}")   # 本帧投影退化（如透视扁平）
                else:
                    self._hide(scene, f"cast_{p}")
                for k, d in enumerate(contact_shadow_planes(verts, self.floor_z)):
                    if slot["discs"][k] is None:
                        slot["discs"][k] = o3d.geometry.TriangleMesh()
                        slot["mat_discs"][k] = self._transp_material(d["alpha"])
                    self._fill_shadow_disc(slot["discs"][k], d["center"], d["e"],
                                           d["rx"], d["ry"])
                    self._stage(scene, f"{d['name']}_{p}", slot["discs"][k],
                                slot["mat_discs"][k])
                # 骨骼（若 npz 里有关节 24×3）
                joints = person.get("joints")
                if joints is not None and len(joints):
                    if slot["bones"] is None:
                        ls = o3d.geometry.LineSet()
                        ls.lines = o3d.utility.Vector2iVector(np.asarray(_SMPL_EDGES, np.int32))
                        slot["bones"] = ls
                        mr = o3d.visualization.rendering.MaterialRecord()
                        mr.shader = "unlitLine"
                        mr.line_width = 2.0
                        slot["mat_bones"] = mr
                    ls = slot["bones"]
                    ls.points = o3d.utility.Vector3dVector(joints[:24].astype(np.float64))
                    n = len(ls.lines)
                    if n:
                        ls.colors = o3d.utility.Vector3dVector(
                            np.tile(_BONE_COLOR * 0.85, (n, 1)))
                    self._stage(scene, f"bones_{p}", ls, slot["mat_bones"])
                else:
                    self._hide(scene, f"bones_{p}")
            # —— 主层：人体网格 ——
            # 播放中按 _MESH_HZ 限频（60Hz 屏分不到 >60 次/秒，内容 100fps 时跳过多余的
            # 中间帧，网格在下一允许时刻直接取最新姿势）；暂停/单帧/预热（full）必更新。
            now = time.perf_counter()
            if full or now - self._last_mesh_up >= 1.0 / _MESH_HZ or slot["mesh"] is None:
                self._last_mesh_up = now
                if self.faces is not None and len(verts):
                    if slot["mesh"] is None:
                        slot["mesh"] = o3d.geometry.TriangleMesh()
                        slot["mesh"].triangles = o3d.utility.Vector3iVector(self._faces_i32)
                    mesh = slot["mesh"]
                    mesh.vertices = o3d.utility.Vector3dVector(verts.astype(np.float64))
                    mesh.compute_vertex_normals()
                    mesh.vertex_colors = o3d.utility.Vector3dVector(
                        bake_body_shading(np.asarray(mesh.vertex_normals), skin=skin))
                    self._stage(scene, f"person_{p}", mesh, self._mat_smpl)
                else:
                    self._hide(scene, f"person_{p}")
        self._last_t = t
        return True

    # ------------------------------------------------------------------
    # 点云路径（mode="points"，GUI 回放）：全部动态层 = 点云，Scene.update_geometry
    # 原地更新顶点缓冲 —— 无 remove+add → 没有 ~1 万次段错误的实体上限，可无限长播。
    # 每片云 add 一次（占位），之后每帧只 update + show/hide。
    # ------------------------------------------------------------------
    def _cl_mat(self, pt_size: float):
        """点云材质：defaultUnlit + 顶点色，只带点大小（颜色全走逐点色）。"""
        o3d = self.o3d
        mr = o3d.visualization.rendering.MaterialRecord()
        mr.shader = "defaultUnlit"
        mr.base_color = [1.0, 1.0, 1.0, 1.0]
        mr.point_size = float(pt_size)
        return mr

    def _cl_geo(self, scene, name: str, cap: int, mat):
        """取（必要时 add）名为 ``name`` 的占位点云，返回其 ``t.PointCloud``。"""
        e = self._cl_objs.get(name)
        if e is None:
            o3d = self.o3d
            pcd = o3d.t.geometry.PointCloud()
            pcd.point.positions = o3d.core.Tensor(np.zeros((cap, 3), np.float32))
            pcd.point.colors = o3d.core.Tensor(np.zeros((cap, 3), np.float32))
            scene.add_geometry(name, pcd, mat)
            self._cl_objs[name] = {"pcd": pcd, "cap": cap}
            self._cl_added[name] = True
            e = self._cl_objs[name]
        return e["pcd"]

    def _cl_show(self, scene, name: str, vis: bool) -> None:
        """切换点云可见性（只在变化时 show_geometry，无新实体、零残留）。"""
        if not self._cl_added.get(name):
            return                                  # 从未 add（空窗期没东西可藏）
        v = bool(vis)
        if self._cl_vis.get(name) != v:
            scene.show_geometry(name, v)
            self._cl_vis[name] = v

    def _cl_set(self, scene, name: str, cap: int, mat, pts, cols, vis: bool) -> None:
        """原地更新点云 ``name`` 到 ``pts/cols``（m≤cap）并设可见性。纯 update，零 remove。"""
        o3d = self.o3d
        pcd = self._cl_geo(scene, name, cap, mat)
        pcd.point.positions = o3d.core.Tensor(np.ascontiguousarray(pts, np.float32))
        pcd.point.colors = o3d.core.Tensor(np.ascontiguousarray(cols, np.float32))
        low = scene.scene
        low.update_geometry(name, pcd,
                            low.UPDATE_POINTS_FLAG | low.UPDATE_COLORS_FLAG)
        self._cl_show(scene, name, vis)

    def _cl_person_nmesh(self, p: int):
        """第 p 槽的 CPU 法线网格（只算法线不进场景；首建拓扑，跨帧只改顶点）。"""
        while len(self._cl_nmesh) <= p:
            self._cl_nmesh.append(None)
        nm = self._cl_nmesh[p]
        if nm is None and self._faces_i32 is not None:
            o3d = self.o3d
            nm = o3d.geometry.TriangleMesh()
            nm.triangles = o3d.utility.Vector3iVector(self._faces_i32)
            self._cl_nmesh[p] = nm
        return nm

    def _cl_body_sample(self, n_v: int):
        """人体表面加密采样表：原顶点（权重 (1,0,0)）+ 每三角形质心。首建缓存。"""
        if self._cl_sample is not None and self._cl_n_v == n_v:
            return self._cl_sample
        F = np.asarray(self._faces_i32, np.int64)   # (F,3)
        vidx = np.repeat(np.arange(n_v, dtype=np.int64)[:, None], 3, axis=1)
        vw = np.zeros((n_v, 3)); vw[:, 0] = 1.0
        cidx = F
        cw = np.full((len(F), 3), 1.0 / 3.0)
        idx = np.concatenate([vidx, cidx], axis=0)
        w = np.concatenate([vw, cw], axis=0)
        self._cl_sample = (idx, w)
        self._cl_n_v = n_v
        return self._cl_sample

    @staticmethod
    def _cl_interp(attr: np.ndarray, idx: np.ndarray, w: np.ndarray) -> np.ndarray:
        """按固定采样表 (idx,w) 从逐顶点属性 attr(V,3) 插出采样点 (N,3)。"""
        return (attr[idx[:, 0]] * w[:, 0:1] + attr[idx[:, 1]] * w[:, 1:2]
                + attr[idx[:, 2]] * w[:, 2:3])

    @staticmethod
    def _bone_dots(joints, edges=None) -> np.ndarray:
        """把关节骨架画成沿线等分点：每根骨线段采 ``_BONE_DOTS`` 个点。"""
        if joints is None or len(joints) < 2:
            return np.empty((0, 3), np.float64)
        j = np.asarray(joints, np.float64)
        E = np.asarray(_SMPL_EDGES if edges is None else edges, np.int64)
        if len(E) == 0:
            return np.empty((0, 3), np.float64)
        ok = (E[:, 0] < len(j)) & (E[:, 1] < len(j))
        E = E[ok]
        if len(E) == 0:
            return np.empty((0, 3), np.float64)
        ts = np.linspace(0.0, 1.0, _BONE_DOTS)[None, :, None]
        a = j[E[:, 0]][:, None, :]           # (Ne,1,3)
        b = j[E[:, 1]][:, None, :]
        seg = a * (1.0 - ts) + b * ts         # (Ne,D,3)
        return seg.reshape(-1, 3)

    def _apply_people_cloud(self, scene, t: int, full: bool = True) -> bool:
        """点云路径：人体 = 表面点 + 地板影点 + 骨骼点，全部原地更新。

        所有几何都 add 一次后只 update_geometry / show_geometry —— 播放与暂停全程
        0 次 remove+add，撞不到段错误上限（详见模块 docstring「点云回放模式」）。
        ``full`` 仅保留签名兼容（点云路径每帧全量更新，无需降频）。
        """
        path = self.tl.person_path_at(t)
        if path is not None and path == self._applied_path and self._last_n_people > 0:
            return True            # 与上一帧同一份结果（hold_gaps/stride 复用）→ 已就位
        people = self.tl.load_people(t)
        self._applied_path = path
        self._last_n_people = len(people)
        shown = 0
        for p, person in enumerate(people):
            if self._cloud_person(scene, p, person):
                shown = p + 1
        # 隐藏超出本帧人数 / 缺内容的槽
        for p in range(shown, max(shown, self._cl_people_shown + 1)):
            for name in (f"body_{p}", f"shadow_{p}", f"bones_{p}"):
                self._cl_show(scene, name, False)
        self._cl_people_shown = max(shown, 0) if shown else -1
        self._last_t = t
        return shown > 0

    def _cloud_person(self, scene, p: int, person: dict) -> bool:
        """把第 p 个人的点云层更新到当前姿势。返回 True 表示有内容显示。"""
        verts = person.get("vertices")
        joints = person.get("joints")
        any_show = False
        if verts is not None and len(verts):
            verts = np.asarray(verts, np.float64)
            skin = _PERSON_COLORS[p % len(_PERSON_COLORS)]
            n_v = len(verts)
            # —— 人体表面点（bake 明暗 → 顶点色）——
            nm = self._cl_person_nmesh(p)
            if nm is not None:
                nm.vertices = self.o3d.utility.Vector3dVector(verts)
                nm.compute_vertex_normals()
                vcols = bake_body_shading(np.asarray(nm.vertex_normals), skin=skin)
                idx, w = self._cl_body_sample(n_v)
                body_pts = self._cl_interp(verts, idx, w)
                body_cols = self._cl_interp(vcols, idx, w)
                self._cl_set(scene, f"body_{p}", len(idx),
                             self._cl_mat(_CLOUD_PT_BODY), body_pts, body_cols, True)
                any_show = True
            else:
                self._cl_show(scene, f"body_{p}", False)
            # —— 地板影：人体顶点沿光水平投影到地板（不透明深色点）——
            if self.cast_shadow:
                proj = project_floor_shadow(verts[::_SHADOW_STRIDE], self.floor_z)
                m = len(proj)
                if m:
                    sp = np.column_stack([proj[:, 0], proj[:, 1],
                                          np.full(m, self.floor_z + _SHADOW_Z_OFF)])
                    sc = np.tile(np.asarray(_SHADOW_PT_COLOR, np.float64), (m, 1))
                    self._cl_set(scene, f"shadow_{p}", n_v, self._cl_mat(_CLOUD_PT_SHADOW),
                                 sp, sc, True)
                else:
                    self._cl_show(scene, f"shadow_{p}", False)
            else:
                self._cl_show(scene, f"shadow_{p}", False)
        else:
            self._cl_show(scene, f"body_{p}", False)
            self._cl_show(scene, f"shadow_{p}", False)
        # —— 骨骼点 ——
        dots = self._bone_dots(joints)
        if len(dots):
            m = len(dots)
            cols = np.tile(np.asarray(_BONE_COLOR, np.float64), (m, 1))
            self._cl_set(scene, f"bones_{p}", max(2, len(_SMPL_EDGES)) * _BONE_DOTS,
                         self._cl_mat(_CLOUD_PT_BONE), dots, cols, True)
            any_show = True
        else:
            self._cl_show(scene, f"bones_{p}", False)
        return any_show

    # 球层：当前位置红球 + 到 t 为止的轨迹（分段连线）
    # ------------------------------------------------------------------
    def _apply_ball_cloud(self, scene, t: int, full: bool = True) -> bool:
        """点云路径的球层：红球单点 + 轨迹逐采样点，原地更新（mesh 路径见 _apply_ball_mesh）。"""
        o3d = self.o3d
        if not self.tl.ball_ok:
            self._cl_show(scene, "ball", False)
            self._cl_show(scene, "trail", False)
            return False
        X = self.tl.ball_pos_at(t)
        if X is None:
            self._cl_show(scene, "ball", False)
            self._cl_show(scene, "trail", False)
            return False
        self._cl_set(scene, "ball", 1, self._cl_mat(_CLOUD_PT_BALL),
                     np.asarray(X, np.float64).reshape(1, 3),
                     np.asarray(_BALL_COLOR, np.float64).reshape(1, 3), True)
        # 轨迹是「到 t 为止」的累积层：逐帧 O(t) 重算 + 点数增长，播放中每帧都多一块
        # 工作。播放中（full=False）冻结轨迹、只让红球逐帧走；暂停/步进/拖条（full=True）
        # 才刷新轨迹（与 mesh 路径「次层降频」同一思路，点云路径把它变成 full 门控）。
        if full:
            if self.ball_trail:
                pts = self.tl.ball_trail_upto(t)
                if pts:
                    allp = np.concatenate([np.asarray(s, np.float64) for s in pts], axis=0)
                    if len(allp) > _TRAIL_CAP:
                        allp = allp[-_TRAIL_CAP:]
                    m = len(allp)
                    cols = np.tile(np.asarray(_BALL_TRAIL_COLOR, np.float64), (m, 1))
                    self._cl_set(scene, "trail", _TRAIL_CAP, self._cl_mat(_CLOUD_PT_TRAIL),
                                 allp, cols, True)
                else:
                    self._cl_show(scene, "trail", False)
            else:
                self._cl_show(scene, "trail", False)
        return True

    # ------------------------------------------------------------------
    # 球层（mesh 路径分派见上 apply_ball）
    # ------------------------------------------------------------------
    def apply_ball(self, scene, t: int, full: bool = True) -> bool:
        """把 t 处的球 + 轨迹喂给场景。返回 True 表示本帧有球显示。

        ``mode="points"`` → :meth:`_apply_ball_cloud`（点云原地更新）；``mode="mesh"``
        → :meth:`_apply_ball_mesh`。球主层每帧必换（小球快速移动，逐帧才不跳）；
        轨迹线属次层，mesh 路径里只在 ``full`` / aux 降频档重挂一次。
        """
        if self.mode == "points":
            return self._apply_ball_cloud(scene, t, full)
        return self._apply_ball_mesh(scene, t, full)

    def _apply_ball_mesh(self, scene, t: int, full: bool = True) -> bool:
        """平滑网格红球 + 轨迹线（render_still 离屏 PNG；新场景少次调用，remove+add 安全）。"""
        o3d = self.o3d
        if not self.tl.ball_ok:
            self._hide(scene, "ball")
            self._hide(scene, "ball_trail")
            return False
        X = self.tl.ball_pos_at(t)
        if X is None:
            self._hide(scene, "ball")
            self._hide(scene, "ball_trail")
            return False
        if self._ball_mesh is None:
            sph = o3d.geometry.TriangleMesh.create_sphere(radius=_BALL_RADIUS)
            self._ball_base = np.asarray(sph.vertices, np.float64).copy()
            self._ball_mesh = sph                    # 之后每帧只平移，不重造
        self._ball_mesh.vertices = o3d.utility.Vector3dVector(
            self._ball_base + np.asarray(X, np.float64))
        self._stage(scene, "ball", self._ball_mesh, self._mat_ball)
        do_aux = full or (self._seq % _AUX_CADENCE == 0)
        if self.ball_trail and do_aux:
            segs = self.tl.ball_trail_upto(t)
            if segs:
                # 多段拼进同一个 LineSet：点拼接、段内相邻连线（段间不连），
                # remove+add 重挂即整段更新。
                pts = [np.asarray(s, np.float64) for s in segs]
                lines = [
                    np.column_stack([base + np.arange(len(s) - 1),
                                     base + np.arange(1, len(s))]).astype(np.int32)
                    for base, s in zip(np.cumsum([0] + [len(s) for s in pts[:-1]]), pts)
                ]
                if self._trail_ls is None:
                    self._trail_ls = o3d.geometry.LineSet()
                    self._trail_mat = o3d.visualization.rendering.MaterialRecord()
                    self._trail_mat.shader = "unlitLine"   # 轨迹逐线着色，材质只带线宽
                    self._trail_mat.line_width = 2.0
                self._trail_ls.points = o3d.utility.Vector3dVector(np.concatenate(pts, axis=0))
                self._trail_ls.lines = o3d.utility.Vector2iVector(np.concatenate(lines, axis=0))
                n = len(self._trail_ls.lines)
                if n:
                    self._trail_ls.colors = o3d.utility.Vector3dVector(
                        np.tile(np.asarray(_BALL_TRAIL_COLOR, np.float64), (n, 1)))
                self._stage(scene, "ball_trail", self._trail_ls, self._trail_mat)
            else:
                self._hide(scene, "ball_trail")
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
                 out_png: str = "", root: Optional[str] = None,
                 cast_shadow: bool = True, ball_trail: bool = True) -> "np.ndarray":
    """渲染主时钟 t 处一帧到 RGB ndarray（可选写 PNG），供回放/验证。"""
    import open3d as o3d
    o3d.visualization.rendering  # noqa: F401
    from tabletennis.reconstruction.triangulate import load_camera_rig

    faces = load_faces(tl.out_dir)
    scene_b = ReconScene(tl, faces, camera_rig=load_camera_rig(root),
                         cast_shadow=cast_shadow, ball_trail=ball_trail)

    r = o3d.visualization.rendering.OffscreenRenderer(width, height)
    sc = r.scene
    sc.set_background(np.array([0.09, 0.10, 0.13, 1.0], np.float32))
    scene_b.add_static(sc)
    scene_b.apply_people(sc, t)
    scene_b.apply_ball(sc, t)
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
# 2D 检测叠加：把重建时的球框 + 姿态关键点叠回四路视频（供回放对比排查）
# ----------------------------------------------------------------------
class Recon2DOverlay:
    """把离线重建时的 2D 观测（球检测框 + 姿态关键点）叠回四路视频画面。

    从 ``pose2d.json`` / ``ball2d.json``（``reconstruct_video.py`` 检测阶段存盘）读
    2D 检测结果，用 ``VideoSource`` 读主时钟帧对应的四路视频，逐相机画球框 + 关键点，
    拼成 2×2 平铺图（RGB uint8），供回放窗口里的 ``gui.ImageWidget`` 显示——方便对比
    排查「人物动作 / 球检测」问题。视频在首次取图时惰性打开，``close()`` 释放。
    """

    def __init__(self, out_dir: str, session_dir: str,
                 per_cam_size: tuple = (480, 360)):
        from ..reconstruction.obs2d import load_ball2d, load_pose2d, load_pred_boxes
        self.pose2d = load_pose2d(out_dir)
        self.ball2d = load_ball2d(out_dir)
        self.pred_boxes = load_pred_boxes(out_dir)   # 纯显示：预测框（灰色虚线）
        self.session_dir = session_dir
        self.per_cam_size = tuple(per_cam_size)
        self._src = None
        self._last_t: Optional[int] = None    # 上次取帧的主时钟帧号（判断是否要 seek）

    @property
    def available(self) -> bool:
        return bool(self.pose2d or self.ball2d or self.pred_boxes)

    def _source(self):
        if self._src is None:
            from ..reconstruction.video_source import VideoSource
            try:
                self._src = VideoSource(self.session_dir)
            except Exception:  # noqa: BLE001 —— session 无视频则整个叠加不可用
                self._src = None
        return self._src

    def tile(self, t: int) -> "Optional[np.ndarray]":
        """主时钟 t 帧的四路画面 2×2 平铺（RGB uint8）；无画面返回 None。

        **每台相机固定占一格**（按 ``src.cids`` 排序），该脉冲号没有帧的相机画
        「camN 无帧」占位块——否则缺一路就会让后面几格整体前移，看起来像「相机接错了」。
        """
        import cv2

        from ..reconstruction.obs2d import dict_to_ball, dict_to_pose
        from .overlay2d import (draw_ball, draw_pose, draw_pred_box, gray_to_bgr,
                                tile_images)

        src = self._source()
        if src is None:
            return None
        # 大步跳（首次打开 / 拖进度条）先 seek，别从 0 顺序解几千帧；顺序步进走快路径
        if self._last_t is None or abs(int(t) - self._last_t) > _2D_SEEK_GAP:
            src.seek(int(t))
        self._last_t = int(t)
        frames = src.frames_for_ref(int(t))
        if not frames:
            return None
        tw, th = self.per_cam_size
        images = []
        for cid in src.cids:
            frame = frames.get(cid)
            if frame is None:
                # 该相机的这一拍被丢帧/没对齐上（见 video_source 脉冲号对齐）
                ph = np.full((th, tw, 3), 24, dtype=np.uint8)
                cv2.putText(ph, f"cam{cid}", (8, 26), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (120, 120, 120), 2, cv2.LINE_AA)
                cv2.putText(ph, "no frame (drop/misalign)", (8, th // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (70, 70, 200), 2, cv2.LINE_AA)
                images.append(ph)
                continue
            bgr = gray_to_bgr(frame.image)
            # 预测框先画（灰色虚线，压在实测框下面）：该相机这一帧没检出人时，
            # 卡尔曼预测位置的框——**只是显示**，重建里这一帧该相机不贡献观测。
            for pb in self.pred_boxes.get(str(int(t)), {}).get(str(cid), []):
                draw_pred_box(bgr, pb[:4], label=f"pred p{int(pb[4])}" if len(pb) > 4 else "pred")
            for pd in self.pose2d.get(str(int(t)), {}).get(str(cid), []):
                draw_pose(bgr, dict_to_pose(pd, camera_id=cid), draw_bbox=True)
            for bd in self.ball2d.get(str(int(t)), {}).get(str(cid), []):
                draw_ball(bgr, dict_to_ball(bd, camera_id=cid))
            if (bgr.shape[1], bgr.shape[0]) != (tw, th):
                bgr = cv2.resize(bgr, (tw, th), interpolation=cv2.INTER_AREA)
            cv2.putText(bgr, f"cam{cid}", (8, 26), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2, cv2.LINE_AA)
            images.append(bgr)
        tile = tile_images(images, cols=2)
        return cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        if self._src is not None:
            self._src.close()
            self._src = None


# ----------------------------------------------------------------------
# 交互回放窗口（gui.Application + SceneWidget，须主线程）
# ----------------------------------------------------------------------
class _PlayerApp:
    def __init__(self, tl: ReconTimeline, faces: Optional[np.ndarray],
                 scene_b: ReconScene, width: int, height: int, fps: float,
                 watch: bool = False, overlay: "Optional[Recon2DOverlay]" = None):
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
        self.overlay = overlay              # 2D 检测叠加（可为 None）
        self._2d_win = None
        self._2d_img_widget = None
        self._last_title = 0.0              # 标题栏更新节流（见 _show）
        self._last_slider_sync = 0.0        # 进度条回显节流（见 _show）
        self._last_2d_wall = 0.0            # 2D 叠加解码节流（见 _refresh_2d）
        self._perf_t0 = 0.0                 # 播放帧率诊断（见 run）
        self._perf_n = 0

        self.app = gui.Application.instance
        self.app.initialize()
        self.win = self.app.create_window(
            f"EasyMocap 重建回放 — {os.path.basename(tl.out_dir)}", width, height)
        self.widget = gui.SceneWidget()
        # 可拖动时间进度条：SceneWidget + Slider 作为窗口**直接子控件**，用自定义
        # on_layout 摆放（SceneWidget 占满、Slider 贴底）。⚠️ 别把 SceneWidget 塞进
        # Vert/Horiz 布局——Open3D 0.19 里那样会让 SceneWidget 坍缩成 0 高度：画面
        # 不刷新、按键失灵、转视角后整屏空白（正是踩过的坑）。官方
        # examples/visualization/vis_gui.py 即用「直接子控件 + set_on_layout」。
        self.slider = gui.Slider(gui.Slider.Type.INT)
        self._slider_sync = False           # 程序设值（回显）时挡掉 seek 回调
        self.slider.set_limits(0, max(1, max(0, tl.n_ref - 1)))
        self.slider.set_on_value_changed(self._on_slider)
        self.win.add_child(self.widget)
        self.win.add_child(self.slider)
        self.win.set_on_layout(self._on_layout)

    def run(self) -> None:
        self._setup()
        self._warmup()
        gc.collect()
        gc.disable()               # 回放期间关自动 GC（几何持久复用后临时对象已很少；
        was_playing = self.playing  # 真需要清积压就暂停时收一次）
        prev_wall = time.perf_counter()   # 上一循环墙钟（含 tick/渲染耗时），驱动实时推进
        # 稳定节拍：播放按 60Hz 目标出帧（屏刷新率），暂停降到 30Hz 省 CPU。旧实现用固定
        # sleep(2ms)，循环节奏随 run_one_tick 返回时长随意抖动、渲染次数远超 60Hz 屏能
        # 显示的量（内容 100fps 时 ~100 次/秒），屏上被迫不均掉帧 → 明显卡顿/抖动。改成
        # 固定节拍后每次 tick 均匀出帧，被跳过的内容帧也是等间隔，肉眼才顺。
        frame_dt = 1.0 / 60.0
        idle_dt = 1.0 / 30.0
        next_wall = prev_wall
        try:
            while self.app.run_one_tick():
                now = time.perf_counter()
                dt = min(max(0.0, now - prev_wall), 0.25)  # 单步最多追 0.25s 内容，防卡顿后瞬移
                prev_wall = now
                if self.watch:
                    self._watch_poll(now)
                n = max(0, self.tl.n_ref - 1)
                if self.playing:
                    if self.t >= float(n):           # 已到尾：自动停（Space 再按则从头）
                        self.playing = False
                    else:
                        # 实时推进：内容时钟按墙钟跑（t += dt×fps×speed）。渲染跟不上时
                        # 每次 tick 的 dt 变大 → t 跳过中间内容帧，仍 1.0× 实时，不会像
                        # 旧实现那样「每帧 t+=1 追不上 → 0.71× 慢动作」。展示只看 int(t)，
                        # 被跳过的帧本来也不会在 60Hz 屏上分得一个刷新。
                        self.t = min(float(n), self.t + dt * self.fps * self.speed)
                self._show()
                if self.playing:
                    # 帧率诊断：每 ~2s 打印一次实测渲染帧率（排查播放卡顿/跳帧）
                    self._perf_n += 1
                    if self._perf_n == 1:
                        self._perf_t0 = time.perf_counter()
                    elif time.perf_counter() - self._perf_t0 >= 2.0:
                        dt = time.perf_counter() - self._perf_t0
                        print(f"[回放] 实测 ≈{self._perf_n / dt:.0f} 帧/秒"
                              f"（{dt * 1000 / self._perf_n:.0f}ms/帧）· t={int(self.t)}")
                        self._perf_n = 0
                if not self.playing and was_playing:
                    gc.collect()      # 刚暂停/到尾时收一次积压
                was_playing = self.playing
                # 睡到下一整拍；渲染/事件超时则重同步，不越掉越多
                next_wall += frame_dt if self.playing else idle_dt
                sleep_t = next_wall - time.perf_counter()
                if sleep_t > 0:
                    time.sleep(sleep_t)
                else:
                    next_wall = time.perf_counter()
        finally:
            gc.enable()
            if self._2d_win is not None:
                self._2d_win.close()
            if self.overlay is not None:
                self.overlay.close()   # 释放四路 VideoCapture，退出不残留句柄/不卡
            self.win.close()

    # -- watch：重建进行中，周期性重扫输出目录，追新帧 ----------------------
    def _watch_poll(self, now: float) -> None:
        if now - self._watch_last < 0.3:
            return
        self._watch_last = now
        prev_n, prev_ok = self._watch_prev_n, self._watch_prev_ok
        self.tl.reload()
        self._update_slider_range()
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
        w.set_view_controls(self.gui.SceneWidget.Controls.ROTATE_CAMERA)  # 左拖旋转/右拖平移/滚轮缩放
        w.set_on_key(self._on_key)
        print(f"[回放] 目标 ≈{self.fps * self.speed:.0f} 帧/秒（录制 {self.fps:.0f}fps = 实时）。"
              f"内容时钟按真实时间跑：渲染跟不上就在 60Hz 屏上自然跳过中间帧，绝不停顿慢放。"
              f"- / + 半速/倍速可调。")
        print("[回放] 鼠标：左拖=旋转/右拖=平移/滚轮=缩放；底部进度条可拖动定位。"
              "Space=播放/暂停（到尾再按=从头），←/→=步进，Home/End=首/尾，"
              "R=复位视角，S=暂停，Esc=退出")
        if self.overlay is not None and self.overlay.available:
            print("[回放] V=开关 2D 检测叠加窗口（四路视频 + 球框 + 关键点；"
                  "播放中冻结，暂停/步进/拖进度条时刷新）")

    def _warmup(self) -> None:
        """首帧预热：渲染几帧真人，提前编译透明阴影等 shader + 摊平 GPU 缓冲分配。

        实测首帧 ``run_one_tick`` 会卡 ~190ms（Filament 首次编译 defaultUnlit/
        defaultLitTransparency 着色器 + GPU 缓冲分配）；在进入回放循环前渲染几帧
        真人把这些一次性开销摊到开窗前，避免播放开头「卡一下」。
        """
        warmed = 0
        for t in self.tl.files:
            if not self.tl.load_people(t):
                continue
            self.scene_b.apply_people(self.widget.scene, t)
            self.scene_b.apply_ball(self.widget.scene, t)
            self.widget.force_redraw()
            self.app.run_one_tick()
            warmed += 1
            if warmed >= 3:
                return

    def _show(self) -> None:
        t0 = int(self.t)
        if t0 == self.last_render_t and not self._need_show:
            return
        force2d = (not self.playing) or self._need_show   # 暂停/步进时 2D 必同步，播放则冻结
        self._need_show = False
        self.last_render_t = t0
        # 播放中次层几何（影/骨骼/轨迹）降频（apply_* 的 full=False）；暂停/单帧/预热全量
        paused = not self.playing
        person = self.scene_b.apply_people(self.widget.scene, t0, full=paused)
        has_ball = self.scene_b.apply_ball(self.widget.scene, t0, full=paused)
        self.widget.force_redraw()     # 场景变了立即重绘，别等事件流捎带（否则卡/跳帧）
        now = time.perf_counter()
        # 进度条回显：播放中降到 ~10Hz——每帧设 Slider 值会反复触发重排/重绘，是进度条
        # 跟着卡顿的来源之一；暂停/步进/拖条（paused）则每帧同步。程序设值用 _slider_sync
        # 挡住 _on_slider 的 seek 回调。
        if paused or now - self._last_slider_sync >= 0.1:
            self._last_slider_sync = now
            self._slider_sync = True
            try:
                self.slider.int_value = t0
            finally:
                self._slider_sync = False
        # 标题栏节流：每帧改 X11 标题会带来卡顿/闪烁，降到 ~4Hz
        if now - self._last_title >= 0.25:
            self._last_title = now
            rate = self.speed * self.fps
            self.win.title = (f"EasyMocap 重建回放 — {os.path.basename(self.tl.out_dir)}"
                              f"  |  t {t0}/{max(0, self.tl.n_ref - 1)}"
                              f"  |  {'● 人物' if person else '— 无人'}"
                              f"  |  {'● 球' if has_ball else ''}"
                              f"  |  x{self.speed:.1f} ≈{rate:.0f}帧/秒")
        self._refresh_2d(force=force2d)

    # -- 2D 检测叠加窗口（V 开关）--------------------------------------
    def _toggle_2d(self) -> None:
        if self.overlay is None or not self.overlay.available:
            print("[回放] 无 2D 检测数据（缺 pose2d.json / ball2d.json），先重跑重建")
            return
        if self._2d_win is None:
            self._open_2d()
        else:
            self._close_2d()

    def _open_2d(self) -> None:
        w = self.app.create_window("2D 检测叠加（V 开关）", 960, 720)
        iw = self.gui.ImageWidget()
        w.add_child(iw)
        w.set_on_close(self._on_2d_close)
        self._2d_win = w
        self._2d_img_widget = iw
        self._refresh_2d(force=True)
        print("[回放] 2D 检测叠加 开")

    def _close_2d(self) -> None:
        if self._2d_win is not None:
            self._2d_win.close()
        self._2d_win = None
        self._2d_img_widget = None
        print("[回放] 2D 检测叠加 关")

    def _on_2d_close(self) -> None:
        self._2d_win = None
        self._2d_img_widget = None

    def _refresh_2d(self, force: bool = False) -> None:
        """把当前 t 的四路画面推给 2D 叠加窗（只在暂停/步进/开窗时解码）。

        4 路 100fps 视频若播放中追帧，等于每秒顺序解 ~400 帧 h264（远超实时），
        每次刷新都要解码上一刷新以来跳过的几十帧，必卡死主循环/叠加窗——这是
        「按 V 后 UI 卡顿甚至强制退出」的根因。故播放中冻结、暂停/步进/拖进度条
        时（force）才解码当前帧：诊断叠加即点即得、绝不拖垮 3D 回放。
        """
        if self._2d_img_widget is None or self.overlay is None:
            return
        if not force:
            return
        # 拖进度条 / 连点会密集触发 force，每次都要解 4 路 h264（~15-40ms）→ 拖条卡。
        # 统一降到 ~8Hz 解码：暂停/步进/拖条够看，又不拖垮主循环。
        now = time.perf_counter()
        if now - self._last_2d_wall < 1.0 / 8.0:
            return
        self._last_2d_wall = now
        try:
            tile = self.overlay.tile(int(self.t))
        except Exception as exc:  # noqa: BLE001 —— 视频解不出来不该拖垮主窗口
            print(f"[回放] ⚠ 2D 叠加取帧失败：{exc}")
            return
        if tile is None:
            return
        try:
            img = _o3d().geometry.Image(np.ascontiguousarray(tile))
            self._2d_img_widget.update_image(img)
            if self._2d_win is not None:
                self._2d_win.post_redraw()   # 独立窗口不随主窗口自动重绘，须显式请求
        except Exception as exc:  # noqa: BLE001
            print(f"[回放] ⚠ 2D 叠加显示失败：{exc}")

    # -- 时间进度条（拖动 seek）--------------------------------------
    def _on_layout(self, layout_context) -> None:
        """自定义窗口布局：SceneWidget 占满、进度条 Slider 贴底。

        SceneWidget 必须是窗口直接子控件才能正常渲染/收键盘（见 __init__ 注释），
        所以不用 Vert/Horiz 布局，改用 on_layout 手动摆两个子控件的位置与大小。
        """
        r = self.win.content_rect
        slider_h = 24                          # 进度条贴底高度（px）
        self.widget.frame = self.gui.Rect(r.x, r.y, r.width, max(0, r.height - slider_h))
        self.slider.frame = self.gui.Rect(r.x, r.get_bottom() - slider_h, r.width, slider_h)

    def _on_slider(self, value: float) -> None:
        """用户拖动进度条 → seek 到该帧并暂停（精确看单帧）。"""
        if self._slider_sync:
            return                        # 程序回显触发的回调，忽略
        t = int(value)
        if t == int(self.t):
            return                        # 数值没变（回显）也忽略
        self.t = float(t)
        self.playing = False
        self._need_show = True

    def _update_slider_range(self) -> None:
        """主时钟帧数变化（watch 追帧）时同步进度条上限。"""
        hi = max(1, max(0, self.tl.n_ref - 1))
        if int(self.slider.get_maximum_value) != hi:
            self._slider_sync = True
            try:
                self.slider.set_limits(0.0, float(hi))
            finally:
                self._slider_sync = False

    def _on_key(self, ev) -> bool:
        k = ev.key
        if ev.type != self.gui.KeyEvent.DOWN:
            return False
        KeyName = self.gui.KeyName
        handled = True
        if k == KeyName.SPACE:
            n = max(0, self.tl.n_ref - 1)
            if self.playing:
                self.playing = False
                self._need_show = True
            elif self.t >= float(n):
                # 停在末尾时按 Space = 从头再放（旧实现是死键：t 到 n 后 _step 直接
                # return，再也播不起来）。残留已在每次播放里被 降频+限速 压到很小，
                # 直接重头放即可。
                self.t = 0.0
                self.playing = True
                self._need_show = True
            else:
                self.playing = True
                self._need_show = True
        elif k == KeyName.LEFT:
            self.playing = False               # 步进即暂停，精确看单帧
            self.t = max(0.0, int(self.t) - 1)
            self._need_show = True
        elif k == KeyName.RIGHT:
            self.playing = False
            self.t = min(max(0, self.tl.n_ref - 1), int(self.t) + 1)
            self._need_show = True
        elif k == KeyName.HOME:
            self.t = 0.0
            self.playing = False
            self._need_show = True
        elif k == KeyName.END:
            self.t = max(0, self.tl.n_ref - 1)
            self.playing = False
            self._need_show = True
        elif k == ord("R"):
            self.widget.setup_camera(50.0, self.scene_b.bounds(), self.scene_b.center())
        elif k == ord("s") or k == ord("S"):
            self.playing = False          # S = 暂停（不再是「从头播放」）
            self._need_show = True
        elif k == ord("V") or k == ord("v"):
            self._toggle_2d()
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
             root: Optional[str] = None, hold_gaps: int = 0,
             cast_shadow: bool = True, ball_trail: bool = True,
             show_2d: bool = True) -> None:
    """在主线程弹出 Open3D 回放窗口并阻塞到关闭。``fps``<=0 用真实出帧率。

    ``hold_gaps``：连续 no_person ≤ 该帧数时播放保持上一姿态（见
    :class:`ReconTimeline`），超过才清空——100Hz 拟合抖动掉的单帧不闪没人体。
    """
    _o3d()  # 尽早暴露缺依赖
    from tabletennis.reconstruction.triangulate import load_camera_rig

    tl = ReconTimeline(out_dir, hold_gaps=hold_gaps)
    # 只重建球（无姿态帧）时不需要 SMPL 网格拓扑，跳过 easymocap 补写（免噪音告警）
    faces = load_faces(out_dir, easymocap_root if tl.files else "")
    # GUI 回放走点云路径（mode="points"）：动态层全部 Scene.update_geometry 原地更新、
    # 无 remove+add —— 不会像网格那样累积 ~1 万次即段错误（render_still 离屏才用 mesh）。
    scene_b = ReconScene(tl, faces, camera_rig=load_camera_rig(root),
                         cast_shadow=cast_shadow, ball_trail=ball_trail,
                         mode="points")
    if fps <= 0:
        fps = tl.ref_rate_hz
    if not watch:
        tl.preload()   # 预热：把全部成功帧读进内存，免回放时逐帧解压磁盘 I/O
    if tl.n_ref > 1 or not watch:
        print(f"[回放] {os.path.basename(out_dir)}：主时钟 {tl.n_ref} 帧，成功 {tl.n_ok()}，"
              f"faces={'有' if faces is not None else '缺'}"
              + ("，watch 追帧中" if watch else ""))
    if tl.n_ref <= 1 and not watch:
        print("[回放] 没有任何已重建帧，无事可播。")
        return
    session_dir = (tl.meta or {}).get("session_dir") or os.path.dirname(os.path.abspath(out_dir))
    overlay = Recon2DOverlay(out_dir, session_dir) if show_2d else None
    _PlayerApp(tl, faces, scene_b, width, height, fps, watch=watch,
               overlay=overlay).run()
