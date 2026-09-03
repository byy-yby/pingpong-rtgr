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
_BALL_RADIUS = 0.02                                          # 乒乓球半径 20mm
_BALL_TRAIL_GAP = 5                                          # 断连阈值（帧）：缺测 >5 帧即断开


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
        self._load_ball()

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
        读失败返回空列表。
        """
        path = self.person_path_at(t)
        if path is None:
            return []
        try:
            with np.load(path) as z:
                n_people = int(z["n_people"]) if "n_people" in z else 1
                out = []
                for p in range(n_people):
                    suf = "" if p == 0 else f"_{p}"
                    verts = np.asarray(z[f"vertices{suf}"], np.float32).reshape(-1, 3)
                    joints = None
                    if f"joints{suf}" in z:
                        joints = np.asarray(z[f"joints{suf}"], np.float32).reshape(-1, 3)
                    out.append({"vertices": verts, "joints": joints})
                return out
        except Exception as exc:  # noqa: BLE001 —— 重建正在写/文件半截
            print(f"[recon_player] ⚠ t={t} 读帧失败：{exc}")
            return []

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
    """二维凸包（Andrew monotone chain，纯 numpy/Python，CCW 有序）。

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
                 ball_trail: bool = True):
        o3d = _o3d()
        self.o3d = o3d
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

    def _add_cast_shadow(self, scene, verts: np.ndarray, name: str = "cast") -> None:
        """在人物脚下画整身投影软影：顶点沿光水平方向投到地面 → 凸包填充盘。

        两片小接触椭圆（~0.3m）在真场景里小到几乎看不见；投影整身轮廓才能给出
        一个「带身形、随姿态伸长」的影子，把 SMPL 人明显锚定在地面上。
        """
        o3d = self.o3d
        v = np.asarray(verts, np.float64)
        if len(v) < 8:
            return
        # 影平面固定在大地板上（不是脚平面）：重投影的人体可能因重建浮空几厘米，
        # 影盘若贴脚平面就会悬在地板上方；画在 floor_z 才是真「影子在地上」。
        z = self.floor_z + 0.004
        proj = project_floor_shadow(v[::_CAST_SUBSAMPLE], self.floor_z)
        hull = convex_hull2d(proj)               # 2D CCW 凸包（同平面，勿用 3D qhull）
        if hull is None or len(hull) < 3:
            return
        # 质心 apex 三角扇填满凸多边形（凸包 CCW → 三角全 CCW → 法线 +Z 朝上）
        c = hull.mean(axis=0)
        ring = np.column_stack([hull, np.full(len(hull), z)])
        verts3 = np.vstack([np.append(c, z), ring])
        m0, m = 1, len(ring)
        tris = np.array([[0, i, i + 1] for i in range(m0, m)] + [[0, m, m0]],
                        np.int64)
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts3)
        mesh.triangles = o3d.utility.Vector3iVector(tris)
        mesh.compute_vertex_normals()
        mr = o3d.visualization.rendering.MaterialRecord()
        mr.shader = "defaultLitTransparency"
        mr.base_color = [0.0, 0.0, 0.0, _CAST_ALPHA]
        scene.add_geometry(name, mesh, mr)

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

    def apply_people(self, scene, t: int) -> bool:
        """把 t 处的（可能多个）人体喂给场景。返回 True 表示本帧有人体被显示。"""
        people = self.tl.load_people(t)
        o3d = self.o3d
        # 总是先移除上一帧的人体几何，再按当前状态重建（简单且无残留）
        for p in range(max(1, self._last_n_people)):
            for name in (f"person_{p}", f"bones_{p}", f"cast_{p}",
                         f"shadow0_{p}", f"shadow1_{p}"):
                try:
                    scene.remove_geometry(name)
                except Exception:  # noqa: BLE001
                    pass
        self._last_n_people = len(people)
        if not people:
            self._last_t = None
            return False

        for p, person in enumerate(people):
            verts = person["vertices"]
            skin = _PERSON_COLORS[p % len(_PERSON_COLORS)]
            # 整身投影软影（把人体锚在地面）→ 再叠两片脚下接触盘（核心深影）
            if self.cast_shadow:
                self._add_cast_shadow(scene, verts, name=f"cast_{p}")
            for d in contact_shadow_planes(verts, self.floor_z):
                self._add_shadow_disc(scene, f"{d['name']}_{p}", d["center"], d["e"],
                                      d["rx"], d["ry"], d["alpha"])
            # 网格（按身份着色：凸凹明暗烘焙进顶点色，defaultUnlit 显示）
            if self.faces is not None and len(verts):
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(verts.astype(np.float64))
                mesh.triangles = o3d.utility.Vector3iVector(self.faces.astype(np.int32))
                mesh.compute_vertex_normals()
                mesh.vertex_colors = o3d.utility.Vector3dVector(
                    bake_body_shading(np.asarray(mesh.vertex_normals), skin=skin))
                scene.add_geometry(f"person_{p}", mesh, self._mat_smpl)
            # 骨骼（若 npz 里有关节 24×3）
            joints = person.get("joints")
            if joints is not None and len(joints):
                ls = o3d.geometry.LineSet()
                ls.points = o3d.utility.Vector3dVector(joints[:24].astype(np.float64))
                ls.lines = o3d.utility.Vector2iVector(np.asarray(_SMPL_EDGES, np.int32))
                self._add_line_geo(scene, f"bones_{p}", ls, _BONE_COLOR * 0.85, 2.0)
        self._last_t = t
        return True

    # ------------------------------------------------------------------
    # 球层：当前位置红球 + 到 t 为止的轨迹（分段连线）
    # ------------------------------------------------------------------
    def apply_ball(self, scene, t: int) -> bool:
        """把 t 处的球 + 轨迹喂给场景。返回 True 表示本帧有球显示。"""
        o3d = self.o3d
        for name in ("ball", "ball_trail"):
            try:
                scene.remove_geometry(name)
            except Exception:  # noqa: BLE001
                pass
        if not self.tl.ball_ok:
            return False
        X = self.tl.ball_pos_at(t)
        if X is None:
            return False
        sph = o3d.geometry.TriangleMesh.create_sphere(radius=_BALL_RADIUS)
        sph.translate(np.asarray(X, np.float64))
        scene.add_geometry("ball", sph, self._mat_ball)
        if self.ball_trail:
            segs = self.tl.ball_trail_upto(t)
            if segs:
                # 多段拼进同一个 LineSet：点拼接、段内相邻连线（段间不连），
                # 单个几何名 "ball_trail" 保证下一帧 remove_geometry 能清干净。
                pts = [np.asarray(s, np.float64) for s in segs]
                lines = [
                    np.column_stack([base + np.arange(len(s) - 1),
                                     base + np.arange(1, len(s))]).astype(np.int32)
                    for base, s in zip(np.cumsum([0] + [len(s) for s in pts[:-1]]), pts)
                ]
                ls = o3d.geometry.LineSet()
                ls.points = o3d.utility.Vector3dVector(np.concatenate(pts, axis=0))
                ls.lines = o3d.utility.Vector2iVector(np.concatenate(lines, axis=0))
                self._add_line_geo(scene, "ball_trail", ls,
                                   np.asarray(_BALL_TRAIL_COLOR, np.float64), 2.0)
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
        from ..reconstruction.obs2d import load_ball2d, load_pose2d
        self.pose2d = load_pose2d(out_dir)
        self.ball2d = load_ball2d(out_dir)
        self.session_dir = session_dir
        self.per_cam_size = tuple(per_cam_size)
        self._src = None

    @property
    def available(self) -> bool:
        return bool(self.pose2d or self.ball2d)

    def _source(self):
        if self._src is None:
            from ..reconstruction.video_source import VideoSource
            try:
                self._src = VideoSource(self.session_dir)
            except Exception:  # noqa: BLE001 —— session 无视频则整个叠加不可用
                self._src = None
        return self._src

    def tile(self, t: int) -> "Optional[np.ndarray]":
        """主时钟 t 帧的四路画面 2×2 平铺（RGB uint8）；无画面返回 None。"""
        import cv2

        from ..reconstruction.obs2d import dict_to_ball, dict_to_pose
        from .overlay2d import draw_ball, draw_pose, gray_to_bgr, tile_images

        src = self._source()
        if src is None:
            return None
        frames = src.frames_for_ref(int(t))
        if not frames:
            return None
        tw, th = self.per_cam_size
        images = []
        for cid in sorted(frames):
            bgr = gray_to_bgr(frames[cid].image)
            for pd in self.pose2d.get(str(int(t)), {}).get(str(cid), []):
                draw_pose(bgr, dict_to_pose(pd, camera_id=cid), draw_bbox=True)
            for bd in self.ball2d.get(str(int(t)), {}).get(str(cid), []):
                draw_ball(bgr, dict_to_ball(bd, camera_id=cid))
            if (bgr.shape[1], bgr.shape[0]) != (tw, th):
                bgr = cv2.resize(bgr, (tw, th), interpolation=cv2.INTER_AREA)
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
            if self.watch:
                time.sleep(0.005)
            elif self.playing:
                # 按内容帧边界 pacing：t 走到下一个整数帧所需时间就是理想的休眠
                # （渲染跟得上时 dt 均匀、丢帧由「整数帧变化才画」决定而非抖动）；
                # 渲染慢则 wait<=0 不睡 → 自然追赶，保持实时语义。
                rate = max(1.0, self.speed * self.fps)
                wait = (int(self.t) + 1 - self.t) / rate
                time.sleep(min(wait, 0.02) if wait > 0 else 0.0)
            else:
                time.sleep(0.02)
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
        print(f"[回放] 目标 ≈{self.fps * self.speed:.0f} 帧/秒（录制 {self.fps:.0f}fps 实时）。"
              f"显示若跟不上会自动掉帧；- / + 半速/倍速可调。")
        print("[回放] 鼠标拖=旋转/平移/滚轮缩放；Space=暂停/继续，←/→=步进，"
              "Home/End=首/尾，R=复位视角，Esc=退出")
        if self.overlay is not None and self.overlay.available:
            print("[回放] V=开关 2D 检测叠加窗口（四路视频 + 球框 + 关键点）")

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
        person = self.scene_b.apply_people(self.widget.scene, t0)
        has_ball = self.scene_b.apply_ball(self.widget.scene, t0)
        self.widget.force_redraw()     # 场景变了立即重绘，别等事件流捎带（否则卡/跳帧）
        rate = self.speed * self.fps
        self.win.title = (f"EasyMocap 重建回放 — {os.path.basename(self.tl.out_dir)}"
                          f"  |  t {t0}/{max(0, self.tl.n_ref - 1)}"
                          f"  |  {'● 人物' if person else '— 无人'}"
                          f"  |  {'● 球' if has_ball else ''}"
                          f"  |  x{self.speed:.1f} ≈{rate:.0f}帧/秒")
        self._refresh_2d()

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
        self._refresh_2d()
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

    def _refresh_2d(self) -> None:
        if self._2d_img_widget is None or self.overlay is None:
            return
        tile = self.overlay.tile(int(self.t))
        if tile is None:
            return
        img = _o3d().geometry.Image(np.ascontiguousarray(tile))
        self._2d_img_widget.update_image(img)

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
    faces = load_faces(out_dir, easymocap_root)
    scene_b = ReconScene(tl, faces, camera_rig=load_camera_rig(root),
                         cast_shadow=cast_shadow, ball_trail=ball_trail)
    if fps <= 0:
        fps = tl.ref_rate_hz
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
