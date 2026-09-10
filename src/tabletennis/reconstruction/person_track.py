"""多人跟踪 / 跨相机身份识别 / 鲁棒三角化（阶段 A–D，宽视野一相机多人场景）。

换广角镜头后**一台相机会同时拍到近处和远处两个人**，旧的
``associate.match_people_fixed``（固定相机分组 [[0,2],[1,3]]，每组只看一人）
前提被打破，身份会逐帧跳。本模块用四阶段替换它：

- **阶段 A 单相机局部跟踪**（:class:`LocalCameraTracker`）：每相机独立，按 IoU
  （退化时用质心最近邻）把本帧检测框接上一帧的 ``local_track_id``；未匹配的检测
  开新 id，未匹配的局部轨迹 ``missed_frames += 1``，超过 ``local_track_max_missed``
  删除。局部 id 只在单相机内有意义，是阶段 C 的**候选来源**与**时序先验**。
- **阶段 B 世界系 3D 轨迹卡尔曼**（:class:`PersonKalman`）：状态
  ``[px,py,pz,vx,vy,vz]``，匀速模型 + DWNA 过程噪声，观测是每帧三角化出的
  3D 根关节，观测噪声 R 由阶段 D 的三角化协方差动态填。
- **阶段 C 跨相机身份关联**（:meth:`MultiPersonTracker._associate`）：把各相机
  轨迹的卡尔曼预测重投影到 4 台相机，**小规模枚举 + 全局重投影误差最小化**决定
  每个检测属于哪个人（无 ReID / 无匈牙利）。再与上一帧的身份映射比较，只有
  显著更优（``switch_margin``）才切换身份，抑制抖动。
- **阶段 D 置信度加权鲁棒三角化**（:meth:`MultiViewTriangulator.triangulate_pose_robust`，
  见 ``triangulate.py``）：逐关键点丢低置信观测 → 2 视角加权 DLT / ≥3 视角
  RANSAC → 输出 3D 点 + 协方差。

**ROI 引导重检测**（阶段 A 的核心补丁）：某人的活跃轨迹在某相机本帧没匹配到检测
时，用其卡尔曼预测 3D 位置重投影到该相机，以预测点为中心开一个边长 = 2×最近 bbox
对角线（缺省 300px）的小窗，用**更低阈值**（``roi_redetect_conf``）再跑一次人检测；
命中则坐标映射回全图、标 ``source="roi_redetect"`` 正常参与后续；仍找不到则记
``source="predicted_only"``（不参与三角化，只进漏检统计）。重检测只在漏检时触发，
不是每帧每相机都跑。

依赖注入：本模块**不 import 任何检测器**，检测/姿态都通过回调传入
（``detect_fn`` / ``pose_fn``），因此可以纯 numpy 单测（见 ``tests/test_person_track.py``）。
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.types import Frame, Pose2D, Skeleton3D
from .associate import anchor_2d
from .triangulate import (
    DEFAULT_ROBUST,
    MultiViewTriangulator,
    RobustPose3D,
    RobustTriangulationConfig,
)

__all__ = [
    "TrackConfig", "BoxTrack", "LocalCameraTracker", "PersonKalman",
    "PersonTrack", "TrackFrameResult", "MultiPersonTracker",
    "iou_xyxy", "select_fit_views", "lower_body_unreliable", "mask_lower_body",
    "LOWER_BODY_HALPE26", "HIP_HALPE26",
]

_EMPTY_BOXES = np.zeros((0, 4), dtype=np.float32)

# bbox 贴到图像边界的判定余量（像素）。
_CLIP_MARGIN_PX = 3.0

# 下半身关键点在 halpe26 里的下标：膝 13/14、踝 15/16、脚尖 20-23、脚跟 24/25。
# **髋部（11/12/19）不在内**——阶段 B/C 的根关节靠它，且髋几乎不被球桌挡住。
LOWER_BODY_HALPE26: Tuple[int, ...] = (13, 14, 15, 16, 20, 21, 22, 23, 24, 25)

# 髋部关键点在 halpe26 里的下标：左髋 11、右髋 12、骨盆中点 19。默认**不**掩码
# （见 mask_lower_body 的说明）；只在 ``mask_hips_when_unreliable`` 打开时才一起丢。
HIP_HALPE26: Tuple[int, ...] = (11, 12, 19)


def lower_body_unreliable(pose: Pose2D, *, ratio: float = 0.5, abs_min: float = 0.5,
                          min_joints: int = 2) -> bool:
    """该视角的姿态是否「下半身不可信」（远端相机隔着球桌看人时的典型症状）。

    判据：下半身关节（膝/踝/脚尖/脚跟）的**中位置信度**既明显低于**同一姿态上半身**
    的中位（``< ratio`` 倍），又低于绝对门限 ``abs_min``。两个条件都满足才判不可信——
    只看相对值会误伤「整具骨架置信度都低的远处小目标」，只看绝对值会误伤挥拍时
    手腕短暂低置信的帧。下半身有效关节少于 ``min_joints`` 时不判（没数据 = 没污染）。

    实测（session 20260908_161147，2 人 4 相机）：远端相机的下半身中位置信
    0.24~0.40 / 同视角上半身 0.76~0.91（比值 0.44~0.46），投回该相机的中位残差
    32~63px；而正常视角下半身 0.81~0.89（比值 0.89~0.94）、残差 10~17px。
    两组比值被 0.5 门限干净分开。
    """
    kp = np.asarray(pose.keypoints, dtype=np.float64)
    if kp.ndim != 2 or kp.shape[1] < 3 or len(kp) <= max(LOWER_BODY_HALPE26):
        return False
    conf = kp[:, 2]
    low_idx = list(LOWER_BODY_HALPE26)
    up_mask = np.ones(len(kp), dtype=bool)
    up_mask[low_idx] = False
    up = conf[up_mask]
    up = up[up > 0]
    low = conf[low_idx]
    low = low[low > 0]
    if len(low) < min_joints or len(up) == 0:
        return False
    m_up, m_low = float(np.median(up)), float(np.median(low))
    return bool(m_low < ratio * m_up and m_low < abs_min)


def mask_lower_body(pose: Pose2D, *, include_hips: bool = False) -> Pose2D:
    """返回副本：下半身关节（膝/踝/脚尖/脚跟）置信度置 0，坐标保留。

    置 0 后该视角的这些关节在下游（阶段 D 鲁棒三角化 / SMPL 拟合）自动被丢弃，
    3D 下半身只由**看得见腿的相机**决定；上半身与髋部完全不受影响。

    ``include_hips=True`` 时连**髋部（halpe26 11/12/19）一起置 0**——即「该视角只用
    上半身」。**默认关闭**：实测远端视角的髋残差 3.5cm（近端 5.5cm）、置信 0.84，
    比近端还准，丢掉它会削弱骨盆（拟合根）的约束；真正烂掉的是膝以下
    （踝 27cm、脚 42~49cm），已由 ``LOWER_BODY_HALPE26`` 覆盖。开关暴露出来只为
    现场 A/B（``TrackConfig.mask_hips_when_unreliable`` / ``--mask-hips-when-unreliable``）。
    """
    kp = np.array(pose.keypoints, dtype=np.float64, copy=True)
    idx = LOWER_BODY_HALPE26 + (HIP_HALPE26 if include_hips else ())
    for i in idx:
        if i < len(kp):
            kp[i, 2] = 0.0
    return replace(pose, keypoints=kp)


def iou_xyxy(a, b) -> float:
    """两个 xyxy 框的 IoU。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    x0 = max(a[0], b[0]); y0 = max(a[1], b[1])
    x1 = min(a[2], b[2]); y1 = min(a[3], b[3])
    iw = max(0.0, x1 - x0); ih = max(0.0, y1 - y0)
    inter = iw * ih
    ua = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return float(inter / ua) if ua > 1e-12 else 0.0


def select_fit_views(
    obs_per_frame: Sequence[Dict[int, Pose2D]],
    intrinsics: Dict[int, object],
    *,
    clip_ratio: float = 0.5,
    min_views: int = 2,
) -> List[int]:
    """为 SMPL 拟合挑视角：**丢掉「人贴在图像边界」的相机**。

    动机（实测 20260908_161147）：某人只在某台相机的画面边缘露出半截时，人检测框
    会被边界裁掉一半，姿态模型对看不见的另一半**外推**出一具完整骨架——中位关键点
    置信度 0.2~0.35、bbox 高度只有 128~168px（正常视角 190~405px），把关节投回该
    视角差 95~290px。把它喂进多视角拟合会明显拉偏 SMPL。

    判定：某相机上该人 bbox 贴边（任一边距图像边界 ≤ 3px）的帧数占出场帧数
    ≥ ``clip_ratio`` 即视为「裁边视角」。若这样筛完不足 ``min_views`` 台，则回退
    到出场最多的 ``min_views`` 台（宁可拟合质量差也不能单视角/无视角）。

    Args:
        obs_per_frame: 逐帧的 ``{cid: Pose2D}``（缺失相机的帧可以是空 dict）。
        intrinsics: ``{cid: CameraIntrinsics}``（需有 ``width`` / ``height``）。

    Returns:
        排序后的相机号列表。
    """
    seen: Dict[int, int] = {}
    clipped: Dict[int, int] = {}
    for obs in obs_per_frame:
        for cid, pose in obs.items():
            seen[cid] = seen.get(cid, 0) + 1
            K = intrinsics.get(cid)
            bb = getattr(pose, "bbox", None)
            if K is None or bb is None:
                continue
            bb = np.asarray(bb, dtype=np.float64)
            if bb.size < 4 or not np.isfinite(bb[:4]).all():
                continue
            W = float(getattr(K, "width", 0) or 0)
            H = float(getattr(K, "height", 0) or 0)
            if W <= 0 or H <= 0:
                continue
            if (bb[0] <= _CLIP_MARGIN_PX or bb[1] <= _CLIP_MARGIN_PX
                    or bb[2] >= W - _CLIP_MARGIN_PX or bb[3] >= H - _CLIP_MARGIN_PX):
                clipped[cid] = clipped.get(cid, 0) + 1
    views = [c for c in sorted(seen) if clipped.get(c, 0) < clip_ratio * seen[c]]
    if len(views) < min_views:
        views = sorted(seen, key=lambda c: (-seen[c], c))[:min_views]
    return sorted(views)


# ----------------------------------------------------------------------
# 配置
# ----------------------------------------------------------------------
@dataclass
class TrackConfig:
    """四阶段跟踪参数（默认值即用户方案里的常量）。"""
    # -- 阶段 A：单相机局部跟踪 --
    iou_thresh: float = 0.3            # IOU_THRESH
    center_gate_ratio: float = 1.2     # IoU 全不达标时的质心最近邻门限（× bbox 对角线）
    local_track_max_missed: int = 10   # LOCAL_TRACK_MAX_MISSED（~0.1s @100fps）
    roi_redetect_conf: float = 0.15    # ROI_REDETECT_CONF（低于全图默认 ~0.4）
    roi_scale: float = 2.0             # 小窗边长 = roi_scale × 最近 bbox 对角线
    roi_fallback_px: float = 300.0     # 无历史 bbox 时的小窗边长
    roi_gate_px: float = 120.0         # ROI 重检测接受门限：检测根像素到预测根的距离上限
    roi_gate_ratio: float = 0.0        # 额外相对门限（× 小窗边长），0=只用 roi_gate_px
    full_detect_interval: int = 30     # FULL_DETECT_INTERVAL：每 N 帧强制全图检测
    # -- 阶段 B：卡尔曼 --
    dt: float = 0.01                   # 帧间隔（秒）；调用方可按录制帧率覆盖
    process_pos_std: float = 0.04      # 过程噪声：位置 std（米）
    process_acc_std: float = 1.5       # 过程噪声：加速度 std（m/s²）
    obs_base_std: float = 0.05         # 观测噪声基线 std（米）
    obs_max_std: float = 1.0           # 观测噪声上限（米）
    max_coast: int = 30                # 连续无观测多少帧后删除轨迹（~0.3s @100fps）
    # -- 阶段 C：跨相机关联 --
    gate_px: float = 120.0             # 检测-预测重投影配对门限（像素）
    switch_margin: float = 0.2         # SWITCH_MARGIN：切换身份需更优 20%
    new_track_min_views: int = 2       # 新建轨迹至少要被几台相机同时看到
    max_assignment_candidates: int = 512   # 枚举上限，超出则按局部代价裁剪
    unmatched_penalty_px: float = 150.0    # 未被任何轨迹认领的检测的代价（× 该检测权重）
    # -- 新建轨迹的防误检门限（宽视野下椅子/墙上照片/腿部碎片会被 YOLO 低阈值检出）--
    spawn_min_score: float = 0.3       # 参与新建轨迹的检测框最低置信度
    spawn_z_range: Tuple[float, float] = (-1.2, 0.8)  # 世界系（桌面系）骨盆高度合理区间
    spawn_max_dist: float = 8.0        # 距世界原点最大水平距离（米）
    confirm_hits: int = 3              # 连续 N 帧有观测才算「确认」，未确认不发观测
    min_pose_joints: int = 5           # 观测姿态至少要有这么多关节 conf ≥ anchor_min_conf
    # -- 阶段 D：鲁棒三角化 --
    robust: RobustTriangulationConfig = field(default_factory=lambda: DEFAULT_ROBUST)
    # 下半身可信度门（见 lower_body_unreliable）：远端相机隔球桌看人时膝/踝是姿态
    # 模型外推的假点，置信度明显低于同视角上半身 → 该视角下半身置 0 不参与重建。
    lower_body_gate: bool = True
    lower_body_conf_ratio: float = 0.5     # 下半身中位置信 < 该比例 × 上半身中位
    lower_body_conf_abs: float = 0.5       # 且下半身中位置信 < 该绝对值
    mask_hips_when_unreliable: bool = False  # 被判不可信的视角**连髋也丢**（只用上身）
                                             # 默认关：远端髋实测 3.5cm、比近端还准
    # -- 通用 --
    max_people: int = 4
    anchor_min_conf: float = 0.3       # anchor_2d 的关节置信度门限


# ----------------------------------------------------------------------
# 阶段 A：单相机局部跟踪
# ----------------------------------------------------------------------
class BoxTrack:
    """单相机内的局部轨迹（阶段 A）。"""

    def __init__(self, track_id: int, box: np.ndarray, score: float,
                 frame_idx: int) -> None:
        self.track_id = int(track_id)
        self.box = np.asarray(box, dtype=np.float32)
        self.score = float(score)
        self.first_frame = int(frame_idx)
        self.last_frame = int(frame_idx)
        self.missed_frames = 0
        self.hits = 1
        self.person_id: Optional[int] = None   # 上一帧跨相机关联到的全局身份

    @property
    def center(self) -> np.ndarray:
        b = self.box
        return np.asarray(((b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5), dtype=np.float64)

    @property
    def diagonal(self) -> float:
        b = self.box
        return float(np.hypot(b[2] - b[0], b[3] - b[1]))


@dataclass
class LocalMatch:
    """本帧某相机的一次检测（已接上局部轨迹 id）。"""
    local_id: int
    box: np.ndarray
    score: float
    is_new: bool
    source: str = "detect"     # detect / predicted / roi_redetect


class LocalCameraTracker:
    """单相机局部跟踪（阶段 A）：IoU 匹配 + 质心最近邻兜底，无卡尔曼。"""

    def __init__(self, camera_id: int, cfg: Optional[TrackConfig] = None) -> None:
        self.camera_id = int(camera_id)
        self.cfg = cfg or TrackConfig()
        self.tracks: Dict[int, BoxTrack] = {}
        self._next_id = 0

    def update(self, boxes, scores=None, frame_idx: int = 0,
               source: str = "detect") -> List[LocalMatch]:
        """本帧检测框 → 接上局部轨迹 id 的 :class:`LocalMatch` 列表。"""
        boxes = _EMPTY_BOXES if boxes is None else np.asarray(boxes, dtype=np.float32)
        if boxes.ndim != 2 or len(boxes) == 0:
            for tr in self.tracks.values():
                tr.missed_frames += 1
            self._prune()
            return []
        if scores is None:
            scores = np.ones(len(boxes), dtype=np.float32)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)

        live = [t for t in self.tracks.values()]
        matches: Dict[int, int] = {}          # det idx -> track_id
        if live:
            pairs: List[Tuple[float, int, int]] = []
            for di, det in enumerate(boxes):
                for ti, tr in enumerate(live):
                    iou = iou_xyxy(det, tr.box)
                    if iou >= self.cfg.iou_thresh:
                        pairs.append((iou, di, ti))
            pairs.sort(key=lambda p: -p[0])
            used_det: set = set()
            used_tr: set = set()
            for iou, di, ti in pairs:
                if di in used_det or ti in used_tr:
                    continue
                matches[di] = live[ti].track_id
                used_det.add(di)
                used_tr.add(ti)
            # 质心最近邻兜底：IoU 全不达标的检测，接给门限内最近的未匹配轨迹
            for di in range(len(boxes)):
                if di in used_det:
                    continue
                det_c = (boxes[di, :2] + boxes[di, 2:]) * 0.5
                best = None
                for ti, tr in enumerate(live):
                    if ti in used_tr:
                        continue
                    dist = float(np.linalg.norm(det_c - tr.center))
                    gate = self.cfg.center_gate_ratio * max(tr.diagonal, 1.0)
                    if dist <= gate and (best is None or dist < best[0]):
                        best = (dist, ti)
                if best is not None:
                    matches[di] = live[best[1]].track_id
                    used_det.add(di)
                    used_tr.add(best[1])

        out: List[LocalMatch] = []
        for di in range(len(boxes)):
            tid = matches.get(di)
            if tid is None:
                tid = self._next_id
                self._next_id += 1
                self.tracks[tid] = BoxTrack(tid, boxes[di], float(scores[di]), frame_idx)
                out.append(LocalMatch(tid, boxes[di], float(scores[di]), True, source))
            else:
                tr = self.tracks[tid]
                tr.box = boxes[di].astype(np.float32)
                tr.score = float(scores[di])
                tr.missed_frames = 0
                tr.hits += 1
                tr.last_frame = int(frame_idx)
                out.append(LocalMatch(tid, tr.box, tr.score, False, source))

        matched_ids = {m.local_id for m in out}
        for tr in self.tracks.values():
            if tr.track_id not in matched_ids:
                tr.missed_frames += 1
        self._prune()
        return out

    def _prune(self) -> None:
        dead = [tid for tid, tr in self.tracks.items()
                if tr.missed_frames > self.cfg.local_track_max_missed]
        for tid in dead:
            del self.tracks[tid]

    def set_person_ids(self, mapping: Dict[int, int]) -> None:
        """把本帧的 ``{local_id: person_id}`` 记到局部轨迹上（下帧时序先验）。"""
        for lid, pid in mapping.items():
            tr = self.tracks.get(lid)
            if tr is not None:
                tr.person_id = int(pid)


# ----------------------------------------------------------------------
# 阶段 B：世界系 3D 卡尔曼
# ----------------------------------------------------------------------
class PersonKalman:
    """6 状态匀速卡尔曼（位置 + 速度），观测 = 三角化出的 3D 根关节。

    过程噪声用 DWNA（离散白噪声加速度）模型；观测噪声 R 由阶段 D 的三角化协方差
    + 基线 ``obs_base_std²`` 动态给出——低置信度 / 遮挡帧 R 大，滤波器自然更信预测。
    """

    def __init__(self, pos: np.ndarray, cfg: TrackConfig,
                 time_s: Optional[float] = None) -> None:
        self.cfg = cfg
        self.x = np.zeros(6, dtype=np.float64)
        self.x[:3] = np.asarray(pos, dtype=np.float64).reshape(3)
        self.P = np.diag([cfg.process_pos_std ** 2] * 3
                         + [(cfg.process_acc_std * cfg.dt) ** 2] * 3)
        self.time_s = float(time_s) if time_s is not None else 0.0

    def _F(self, dt: float) -> np.ndarray:
        F = np.eye(6)
        F[:3, 3:] = np.eye(3) * dt
        return F

    def _Q(self, dt: float) -> np.ndarray:
        a = self.cfg.process_acc_std ** 2
        p = self.cfg.process_pos_std ** 2
        q = np.zeros((6, 6))
        q[:3, :3] = np.eye(3) * (a * dt ** 4 / 4.0 + p)
        q[3:, 3:] = np.eye(3) * (a * dt ** 2)
        q[:3, 3:] = q[3:, :3] = np.eye(3) * (a * dt ** 3 / 2.0)
        return q

    def predict(self, dt: Optional[float] = None,
                time_s: Optional[float] = None) -> np.ndarray:
        """预测到下一时刻；``dt`` 缺省用配置帧间隔。"""
        if time_s is not None:
            dt = max(0.0, float(time_s) - self.time_s)
            self.time_s = float(time_s)
        dt = self.cfg.dt if dt is None else float(dt)
        F = self._F(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self._Q(dt)
        return self.x[:3].copy()

    def update(self, z: Optional[np.ndarray], R: Optional[np.ndarray] = None) -> None:
        """用观测 ``z``（3D 根关节）更新；``z`` 为 None 时只做协方差膨胀（coast）。"""
        if z is None or not np.isfinite(z).all():
            self.P = self.P + self._Q(self.cfg.dt)
            return
        H = np.zeros((3, 6))
        H[:, :3] = np.eye(3)
        if R is None:
            R = np.eye(3) * self.cfg.obs_base_std ** 2
        R = np.asarray(R, dtype=np.float64)
        if not np.isfinite(R).all():
            R = np.eye(3) * self.cfg.obs_base_std ** 2
        S = H @ self.P @ H.T + R
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return
        self.x = self.x + K @ (np.asarray(z, dtype=np.float64) - H @ self.x)
        I_KH = np.eye(6) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ R @ K.T   # Joseph 形式，保正定

    @property
    def position(self) -> np.ndarray:
        return self.x[:3].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.x[3:].copy()


# ----------------------------------------------------------------------
# 人物轨迹（阶段 B 的载体）
# ----------------------------------------------------------------------
class PersonTrack:
    """一个全局身份（person_id）的跨相机轨迹。"""

    def __init__(self, person_id: int, kalman: PersonKalman,
                 cfg: TrackConfig) -> None:
        self.person_id = int(person_id)
        self.kalman = kalman
        self.cfg = cfg
        self.age = 0
        self.hits = 0
        self.misses = 0
        self.miss_total = 0
        self.hit_streak = 0                     # 连续有观测的帧数（用于 tentative → confirmed）
        self.confirmed = cfg.confirm_hits <= 1  # 未确认的轨迹不发观测（防误检开身份）
        self.last_bbox: Dict[int, np.ndarray] = {}
        self.last_pose: Dict[int, Pose2D] = {}
        self.box_offset: Dict[int, np.ndarray] = {}   # EMA：anchor_2d − bbox 中心
        self.err_ema: Dict[int, float] = {}           # 每相机重投影误差 EMA
        self.last_obs: Dict[int, Pose2D] = {}
        self.last_robust: Optional[RobustPose3D] = None
        self.last_sources: Dict[int, str] = {}

    # -- 相机权重 / 衰减 --
    def cam_weight(self, cid: int) -> float:
        """``w_cam = bbox 置信度 × 关键点平均置信度``（阶段 C 的代价权重）。"""
        pose = self.last_pose.get(cid)
        if pose is None:
            return 0.0
        kp = np.asarray(pose.keypoints, dtype=np.float64)
        if kp.ndim != 2 or len(kp) == 0:
            return 0.0
        kpt_conf = float(np.mean(kp[:, 2])) if kp.shape[1] > 2 else 0.0
        return float(max(pose.score, 1e-3) * max(kpt_conf, 1e-3))

    def decay(self, cid: int) -> float:
        """per-camera 误差 EMA 导出的衰减因子（1=好，下限 0.2）。"""
        err = self.err_ema.get(cid)
        if err is None:
            return 1.0
        return float(np.clip(1.0 / (1.0 + err / 8.0), 0.2, 1.0))

    # -- 几何 --
    def root_center(self, cid: int) -> Optional[np.ndarray]:
        """该相机本帧检测框对应的**根关节像素**估计 = bbox 中心 + 历史偏移 EMA。

        ⚠️ 返回的是**原始（带畸变）图像坐标**——bbox 来自检测器。与
        :meth:`MultiViewTriangulator.project` 的**无畸变**坐标口径不同，做几何
        运算前必须经 :meth:`root_center_undist`。
        """
        box = self.last_bbox.get(cid)
        if box is None:
            return None
        center = (np.asarray(box, dtype=np.float64)[:2]
                  + np.asarray(box, dtype=np.float64)[2:]) * 0.5
        return center + self.box_offset.get(cid, np.zeros(2))

    def root_center_undist(self, cid: int, tri: MultiViewTriangulator
                           ) -> Optional[np.ndarray]:
        """:meth:`root_center` 去畸变后的版本（与 ``project``/``_dlt`` 同口径）。"""
        uv = self.root_center(cid)
        if uv is None:
            return None
        return tri.undistort_points(cid, uv)

    def predicted_box(self, cid: int, tri: MultiViewTriangulator
                      ) -> Optional[np.ndarray]:
        """卡尔曼预测位置重投影到该相机 → 预测框（用最近 bbox 尺寸，无畸变坐标）。"""
        if cid not in tri.P:
            return None
        box = self.last_bbox.get(cid)
        if box is None:
            return None
        uv = tri.project(cid, self.kalman.position)
        center = uv - self.box_offset.get(cid, np.zeros(2))
        half = (np.asarray(box, dtype=np.float64)[2:] - np.asarray(box, dtype=np.float64)[:2]) * 0.5
        half = np.maximum(half, 8.0)
        return np.asarray([center[0] - half[0], center[1] - half[1],
                           center[0] + half[0], center[1] + half[1]], dtype=np.float32)

    def note_observation(self, cid: int, pose: Pose2D, source: str) -> None:
        """记录本帧观测：bbox、anchor 偏移 EMA、每相机误差 EMA。"""
        self.last_pose[cid] = pose
        self.last_sources[cid] = source
        if pose.bbox is not None:
            self.last_bbox[cid] = np.asarray(pose.bbox, dtype=np.float32)[:4]
        a = anchor_2d(pose, self.cfg.anchor_min_conf)
        if a is not None and pose.bbox is not None:
            b = np.asarray(pose.bbox, dtype=np.float64)[:4]
            center = (b[:2] + b[2:]) * 0.5
            off = np.asarray(a[:2], dtype=np.float64) - center
            prev = self.box_offset.get(cid)
            self.box_offset[cid] = off if prev is None else 0.7 * prev + 0.3 * off

    def update_error_ema(self, cid: int, err: float, alpha: float = 0.2) -> None:
        if err is None or not np.isfinite(err):
            return
        prev = self.err_ema.get(cid)
        self.err_ema[cid] = float(err) if prev is None else (1 - alpha) * prev + alpha * float(err)

    @property
    def alive(self) -> bool:
        return self.misses <= self.cfg.max_coast


@dataclass
class TrackFrameResult:
    """一帧里某个人的跟踪结果（喂给下游拟合 / 可视化）。"""
    person_id: int
    obs: Dict[int, Pose2D]                    # 参与三角化的 2D 观测（下半身门已生效）
    source: Dict[int, str]                    # detect / predicted / roi_redetect / predicted_only
    boxes: Dict[int, np.ndarray]
    cam_weights: Dict[int, float]
    robust: Optional[RobustPose3D]
    root: Optional[np.ndarray]                # 卡尔曼更新后的 3D 根
    root_cov: np.ndarray
    track: PersonTrack
    raw_obs: Dict[int, Pose2D] = field(default_factory=dict)
    """未做下半身掩码的原始观测（**仅供 2D 叠加显示/诊断**，勿拿去三角化/拟合）。"""
    lower_body_masked: Dict[int, bool] = field(default_factory=dict)
    """每个相机本帧是否被下半身门判为不可信（下半身关节已在 :attr:`obs` 里置 0）。"""


# ----------------------------------------------------------------------
# 阶段 C + 总编排
# ----------------------------------------------------------------------
class MultiPersonTracker:
    """阶段 A–D 总编排：逐帧把多相机检测变成「身份稳定的 3D 人」。

    Args:
        triangulator: 标定好的多视角三角化器。
        cfg: 跟踪参数。
        detect_fn: ``(cid, frame, conf_thresh=None, roi=None) -> (boxes (M,4), scores (M,))``
            全图人检测（roi/conf 只在 ROI 重检测时非 None）。为 None 时改用
            ``step(..., detections=...)`` 由外部喂检测结果（单测用）。
        pose_fn: ``(cid, frame, boxes) -> List[Pose2D]`` 对指定框跑姿态。
    """

    def __init__(
        self,
        triangulator: MultiViewTriangulator,
        cfg: Optional[TrackConfig] = None,
        detect_fn: Optional[Callable] = None,
        pose_fn: Optional[Callable] = None,
    ) -> None:
        self.tri = triangulator
        self.cfg = cfg or TrackConfig()
        self.detect_fn = detect_fn
        self.pose_fn = pose_fn
        self.local: Dict[int, LocalCameraTracker] = {}
        self.tracks: Dict[int, PersonTrack] = {}
        self._next_person = 0
        self._retired: Dict[int, Tuple[np.ndarray, int]] = {}
        self._frame_idx = -1
        self._prev_assignment: Dict[int, Dict[int, int]] = {}   # person -> cam -> local_id
        self._last_time: Optional[float] = None
        self._dt: float = float(self.cfg.dt)

    # -- 对外主入口 --
    def step(
        self,
        frame_idx: int,
        frames: Dict[int, Frame],
        detections: Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]] = None,
        time_s: Optional[float] = None,
    ) -> List[TrackFrameResult]:
        """处理一帧，返回每个（活跃）人的 :class:`TrackFrameResult`。

        Args:
            frame_idx: 帧号（用于 FULL_DETECT_INTERVAL 判定与统计）。
            frames: ``{cid: Frame}`` 本帧各相机图像。
            detections: 可选 ``{cid: (boxes, scores)}`` 外部检测结果；None 时调
                ``self.detect_fn``。给定时**不做** FULL_DETECT_INTERVAL 跳检测。
            time_s: 本帧时间（秒）；给定时卡尔曼用它算真实 dt。
        """
        cams = sorted(frames)
        for cid in cams:
            self.local.setdefault(cid, LocalCameraTracker(cid, self.cfg))
        self._frame_idx = int(frame_idx)
        self._dt = self._frame_dt(time_s)

        # -- 0. 检测（全图 / 预测框替代）--
        need_full = (detections is not None
                     or self.detect_fn is None
                     or not self.tracks
                     or self._should_full_detect(frame_idx))
        dets: Dict[int, List[LocalMatch]] = {}
        for cid in cams:
            if detections is not None:
                boxes, scores = detections.get(cid, (_EMPTY_BOXES, None))
                dets[cid] = self.local[cid].update(boxes, scores, frame_idx, "detect")
            elif need_full:
                boxes, scores = self.detect_fn(cid, frames[cid])
                dets[cid] = self.local[cid].update(boxes, scores, frame_idx, "detect")
            else:
                # 非全图帧**不再**用「卡尔曼预测框当伪检测」：伪框会让 ViTPose 在预测
                # 位置凭空产出姿态（图像里未必有那个人），再被三角化「确认」预测，形成
                # 3D→框→姿态→3D 自激回路——实测 20260908_161147 根关节漂到 1.5m 外。
                # 本帧的观测全部来自下方阶段 A 的 ROI 重检测（YOLO 在真实图像上验证）。
                dets[cid] = []

        # -- 1. 阶段 C：跨相机关联 --
        assign, leftovers = self._associate(dets, cams)

        # -- 2. 阶段 A：ROI 引导重检测（只对漏检的活跃轨迹）--
        # 每台相机本帧的小窗结果**合并成一次**局部跟踪更新，避免逐轨迹调用
        # ``LocalCameraTracker.update`` 把别人的局部轨迹当成漏检反复计数。
        redet: Dict[int, Dict[int, LocalMatch]] = {}
        cam_boxes: Dict[int, List[np.ndarray]] = {}
        cam_scores: Dict[int, List[float]] = {}
        cam_owner: Dict[int, List[int]] = {}
        for pid, tr in self.tracks.items():
            if not tr.alive:
                continue
            for cid in cams:
                if cid in assign.get(pid, {}):
                    continue
                got = self._roi_probe(tr, cid, frames)
                if got is None:
                    tr.last_sources[cid] = "predicted_only"
                    continue
                box, sc = got
                cam_boxes.setdefault(cid, []).append(box)
                cam_scores.setdefault(cid, []).append(sc)
                cam_owner.setdefault(cid, []).append(pid)
        for cid, bl in cam_boxes.items():
            lms = self.local[cid].update(np.asarray(bl, dtype=np.float32),
                                         np.asarray(cam_scores[cid], dtype=np.float32),
                                         frame_idx, "roi_redetect")
            for lm, pid in zip(lms, cam_owner[cid]):
                redet.setdefault(pid, {})[cid] = lm

        # -- 3. 组装每人的本帧观测 + 姿态 --
        results: List[TrackFrameResult] = []
        seen_pid: set = set()
        for pid in set(assign) | set(redet):
            tr = self.tracks.get(pid)
            if tr is None:
                continue
            seen_pid.add(pid)
            per_cam = dict(assign.get(pid, {}))
            per_cam.update(redet.get(pid, {}))
            res = self._build_result(tr, per_cam, frames, time_s)
            if tr.confirmed:
                results.append(res)
            # 未确认的轨迹：状态照常更新（等 confirm_hits 帧），只是本帧不发观测

        # -- 4. 新建轨迹（未被认领的真实检测跨相机聚类）--
        for pid in self._spawn_tracks(leftovers, frames, time_s):
            tr = self.tracks[pid]
            seen_pid.add(pid)
            results.append(TrackFrameResult(
                person_id=pid, obs={}, source={}, boxes=dict(tr.last_bbox),
                cam_weights={}, robust=None, root=tr.kalman.position.copy(),
                root_cov=np.eye(3) * self.cfg.obs_base_std ** 2, track=tr))

        # -- 5. 本帧没观测到的活跃轨迹：漏检计数 + 协方差膨胀 --
        for pid, tr in self.tracks.items():
            if pid in seen_pid:
                continue
            tr.age += 1
            tr.misses += 1
            tr.miss_total += 1
            tr.kalman.update(None)

        # -- 6. 收尾：清理死轨迹 / 记录局部轨迹身份 --
        self._retire()
        for pid, per_cam in assign.items():
            self.local_set_ids(pid, per_cam)
        for pid, per_cam in redet.items():
            self.local_set_ids(pid, per_cam)
        return results

    # -- 内部：检测 --
    def _should_full_detect(self, frame_idx: int) -> bool:
        if frame_idx % max(1, self.cfg.full_detect_interval) == 0:
            return True
        # 有轨迹连续漏检过多 → 提前补一次全图检测，避免长时间盲区
        return any(tr.misses >= 3 for tr in self.tracks.values() if tr.alive)

    def _frame_dt(self, time_s: Optional[float]) -> float:
        if time_s is None:
            return self.cfg.dt
        if self._last_time is None:
            self._last_time = float(time_s)
            return self.cfg.dt
        dt = max(1e-4, float(time_s) - self._last_time)
        self._last_time = float(time_s)
        return dt

    # -- 内部：ROI 重检测（阶段 A，纯探测，无局部跟踪副作用）--
    def _roi_probe(self, tr: PersonTrack, cid: int,
                   frames: Dict[int, Frame]) -> Optional[Tuple[np.ndarray, float]]:
        """在 ``tr`` 的预测位置给 ``cid`` 开小窗跑 YOLO，返回通过几何门限的 ``(box, score)``。

        不做局部跟踪（``LocalCameraTracker.update`` 由 :meth:`step` 每相机统一调一次）；
        找不到就返回 None，调用方记 ``predicted_only``（该相机本帧无观测、不参与三角化）。
        """
        if self.detect_fn is None:
            return None
        frame = frames.get(cid)
        if frame is None:
            return None
        # ROI 要落在**真实图像**上 → 用带畸变的投影（predicted_box 是无畸变口径）
        uv_pred = self.tri.project_distorted(cid, tr.kalman.position)
        if uv_pred is None:
            return None
        cx, cy = float(uv_pred[0]), float(uv_pred[1])
        side = self.cfg.roi_fallback_px
        if cid in tr.last_bbox:
            b = np.asarray(tr.last_bbox[cid], dtype=np.float64)
            diag = float(np.hypot(b[2] - b[0], b[3] - b[1]))
            if np.isfinite(diag) and diag > 1.0:
                side = self.cfg.roi_scale * diag
        half = side * 0.5
        roi = (int(cx - half), int(cy - half), int(cx + half), int(cy + half))
        boxes, scores = self.detect_fn(cid, frame,
                                       conf_thresh=self.cfg.roi_redetect_conf,
                                       roi=roi)
        boxes = _EMPTY_BOXES if boxes is None else np.asarray(boxes, dtype=np.float32)
        # 低阈值小窗检测容易在窗边 hallucinate 出误检 → 只接受中心落在窗内的框，
        # 且尺寸与最近 bbox 同量级（0.25~2.5× 窗边长）
        #
        # ⚠️ 距离门限默认用 **cfg.roi_gate_px**（与阶段 C 的 gate_px 同一口径），不能只按
        # ROI 边长放行：小窗边长 = roi_scale × bbox 对角线（本会话 cam3 里人框对角线
        # ~440px → 窗边长 ~880px），旧写法 ``0.75*side``≈660px 会把窗内的**椅子/显示器/
        # 墙上照片**误检直接收给该轨迹（实测 cam3 第 20 帧：p1 真值在 (649,245)，误检框在
        # (361,278) 差 290px 仍被收下，ViTPose 给出错误姿态 → 拖歪四视角拟合与重投影误差）。
        # 根像素估计 = 框中心 + 该相机的「根-框中心偏移 EMA」，与 ``_det_center_undist`` 一致。
        off = tr.box_offset.get(cid, np.zeros(2))
        gate = max(self.cfg.roi_gate_px, self.cfg.roi_gate_ratio * side)
        cand = []
        for i, b in enumerate(boxes):
            c = (b[:2] + b[2:]) * 0.5
            if not (roi[0] <= c[0] <= roi[2] and roi[1] <= c[1] <= roi[3]):
                continue
            diag = float(np.hypot(b[2] - b[0], b[3] - b[1]))
            if not (0.25 * side <= diag <= 2.5 * side):
                continue
            root = np.asarray(c, dtype=np.float64) + off
            dist = float(np.hypot(root[0] - cx, root[1] - cy))
            if dist > gate:
                continue
            cand.append((dist, i))
        if not cand:
            return None
        i = min(cand)[1]
        sc = float(scores[i]) if scores is not None and len(scores) > i else 0.4
        return boxes[i].copy(), sc

    # -- 内部：阶段 C 跨相机关联 --
    def _associate(self, dets: Dict[int, List[LocalMatch]], cams: Sequence[int]):
        """枚举 + 全局重投影误差最小化，返回 ``({pid: {cid: LocalMatch}}, leftovers)``。"""
        active = [tr for tr in self.tracks.values() if tr.alive]
        for tr in active:
            tr.kalman.predict(self._dt)
        if not active:
            leftovers = {cid: list(d) for cid, d in dets.items()}
            return {}, leftovers

        pids = [tr.person_id for tr in active]
        # 每相机候选：det → 哪个 track（或 None），并做门限裁剪
        per_cam_opts: Dict[int, List[Tuple[Optional[int], ...]]] = {}
        local_of: Dict[int, List[LocalMatch]] = {}
        for cid in cams:
            dl = dets.get(cid, [])
            local_of[cid] = dl
            if not dl:
                per_cam_opts[cid] = [()]
                continue
            # local_cost[i][j] = det i 与 track j 的重投影距离²（None=无预测）
            local_cost: List[List[Optional[float]]] = []
            for m in dl:
                row: List[Optional[float]] = []
                for tr in active:
                    row.append(self._det_track_dist2(cid, m, tr))
                local_cost.append(row)
            opts: List[Tuple[Optional[int], ...]] = []
            self._enum_cam_opts(0, dl, active, local_cost, [], opts)
            # 候选过多时按局部代价保留前 N 个
            if len(opts) > 64:
                def local_sum(o):
                    s = 0.0
                    for i, j in enumerate(o):
                        if j is None:
                            s += self.cfg.unmatched_penalty_px ** 2
                        else:
                            d = local_cost[i][j]
                            s += (self.cfg.gate_px ** 2 if d is None else d)
                    return s
                opts.sort(key=local_sum)
                opts = opts[:64]
            per_cam_opts[cid] = opts

        # 全局枚举（相机间笛卡尔积，上限 max_assignment_candidates）
        combos: List[Dict[int, Tuple[Optional[int], ...]]] = []
        self._enum_global(0, list(cams), per_cam_opts, {}, combos)

        best = None
        for combo in combos:
            cost = self._assignment_cost(combo, local_of, active)
            if best is None or cost < best[0]:
                best = (cost, combo)

        # 时序一致性：把「上一帧的身份映射」也当候选，只有显著更优才切换
        prev_combo = self._prev_combo(local_of, active)
        chosen = best[1] if best is not None else {}
        if prev_combo is not None and best is not None:
            prev_cost = self._assignment_cost(prev_combo, local_of, active)
            if not (best[0] < prev_cost * (1.0 - self.cfg.switch_margin)):
                chosen = prev_combo

        assign: Dict[int, Dict[int, LocalMatch]] = {}
        used: Dict[int, set] = {}
        for cid, opt in chosen.items():
            for i, j in enumerate(opt):
                if j is None:
                    continue
                tr = active[j]
                assign.setdefault(tr.person_id, {})[cid] = local_of[cid][i]
                used.setdefault(cid, set()).add(i)
        # 未认领的**真实检测**才可能开新轨迹；预测伪检测直接丢弃
        leftovers = {cid: [m for i, m in enumerate(local_of[cid])
                           if i not in used.get(cid, set())
                           and m.source in ("detect", "roi_redetect")]
                     for cid in cams}
        self._prev_assignment = {
            pid: {cid: m.local_id for cid, m in per_cam.items()}
            for pid, per_cam in assign.items()
        }
        return assign, leftovers

    def _enum_cam_opts(self, i: int, dl: List[LocalMatch], active: List[PersonTrack],
                       local_cost: List[List[Optional[float]]],
                       cur: List[Optional[int]], out: List) -> None:
        if i == len(dl):
            out.append(tuple(cur))
            return
        used = {j for j in cur if j is not None}
        for j in range(len(active)):
            if j in used:
                continue
            d = local_cost[i][j]
            if d is not None and d > self.cfg.gate_px ** 2:
                continue
            cur.append(j)
            self._enum_cam_opts(i + 1, dl, active, local_cost, cur, out)
            cur.pop()
        cur.append(None)     # 该检测不认领任何已有轨迹
        self._enum_cam_opts(i + 1, dl, active, local_cost, cur, out)
        cur.pop()

    def _enum_global(self, k: int, cams: List[int],
                     per_cam_opts: Dict[int, List[Tuple[Optional[int], ...]]],
                     cur: Dict[int, Tuple[Optional[int], ...]], out: List) -> None:
        if len(out) >= self.cfg.max_assignment_candidates:
            return
        if k == len(cams):
            out.append(dict(cur))
            return
        cid = cams[k]
        for opt in per_cam_opts[cid]:
            cur[cid] = opt
            self._enum_global(k + 1, cams, per_cam_opts, cur, out)
            if len(out) >= self.cfg.max_assignment_candidates:
                return
        cur.pop(cid, None)

    def _det_center_undist(self, cid: int, m: LocalMatch,
                           tr: PersonTrack) -> Optional[np.ndarray]:
        """**该检测框**的根关节像素估计（无畸变），口径同 ``project``/``_dlt``。

        检测框中心 + 该轨迹在该相机的「根相对框中心偏移 EMA」（``box_offset``）。
        ⚠️ 必须用**检测框**而不是轨迹记忆的 ``root_center_undist``：后者与检测无关，
        会让阶段 C 的代价对所有检测都相同 → 身份只能按枚举序（置信度序）乱配。
        """
        box = np.asarray(getattr(m, "box", None), dtype=np.float64)
        if box is None or box.size < 4 or not np.isfinite(box[:4]).all():
            return None
        center = (box[:2] + box[2:]) * 0.5 + tr.box_offset.get(cid, np.zeros(2))
        return self.tri.undistort_points(cid, center)

    def _det_track_dist2(self, cid: int, m: LocalMatch,
                         tr: PersonTrack) -> Optional[float]:
        """检测框根像素 vs 轨迹预测根像素的距离²（无预测返回 None=不设门限）。

        ⚠️ 两侧都必须是**无畸变**坐标（``project`` 不建模畸变）。
        """
        if tr.age == 0 and tr.hits <= 1:
            return None
        center = self._det_center_undist(cid, m, tr)
        if center is None:
            return None
        pred = self.tri.project(cid, tr.kalman.position)
        if pred is None:
            return None
        d = pred - center
        return float(d[0] ** 2 + d[1] ** 2)

    def _assignment_cost(self, combo: Dict[int, Tuple[Optional[int], ...]],
                         local_of: Dict[int, List[LocalMatch]],
                         active: List[PersonTrack]) -> float:
        """``E = Σ_track Σ_已分配相机 w·‖重投影根 − 检测根‖² + 未认领检测罚项``。

        未认领罚项也乘该检测的权重（``score``）：低置信度的误检（椅子/墙上照片）
        本来就该被丢掉，用固定的高罚项会逼着优化器把它硬塞给某个轨迹。
        """
        by_track: Dict[int, Dict[int, LocalMatch]] = {}
        assigned: Dict[int, set] = {}
        for cid, opt in combo.items():
            for i, j in enumerate(opt):
                if j is None:
                    continue
                by_track.setdefault(j, {})[cid] = local_of[cid][i]
                assigned.setdefault(cid, set()).add(i)
        cost = 0.0
        for j, per_cam in by_track.items():
            tr = active[j]
            pts, confs, w = {}, {}, {}
            for cid, m in per_cam.items():
                # 用**检测框**根像素（含该轨迹的根-框中心偏移 EMA）；退化时才回退到
                # 轨迹记忆值。用记忆值会让代价与检测无关 → 身份按枚举序乱配。
                center = self._det_center_undist(cid, m, tr)
                if center is None:
                    center = tr.root_center_undist(cid, self.tri)
                if center is None:
                    continue
                pts[cid] = (float(center[0]), float(center[1]))
                confs[cid] = max(float(m.score), 1e-3)
                w[cid] = max(tr.cam_weight(cid), 1e-3) * max(float(m.score), 1e-3)
            if len(pts) >= 2:
                X = self.tri._dlt(sorted(pts), pts, confs)
                if X is None:
                    cost += self.cfg.gate_px ** 2 * len(pts)
                    continue
                for cid in pts:
                    e = self.tri.reproj(cid, X, pts[cid])
                    cost += w[cid] * e * e
            elif len(pts) == 1:
                cid, uv = next(iter(pts.items()))
                d = self.tri.project(cid, tr.kalman.position) - np.asarray(uv)
                cost += w[cid] * float(d[0] ** 2 + d[1] ** 2)
        for cid, dl in local_of.items():
            used = assigned.get(cid, set())
            for i, m in enumerate(dl):
                if i in used:
                    continue
                cost += max(float(m.score), 1e-3) * self.cfg.unmatched_penalty_px ** 2
        return float(cost)

    def _prev_combo(self, local_of: Dict[int, List[LocalMatch]],
                    active: List[PersonTrack]):
        """上一帧身份映射对应的本帧候选组合（按局部轨迹 id 承接）。"""
        pid_to_j = {tr.person_id: j for j, tr in enumerate(active)}
        combo: Dict[int, Tuple[Optional[int], ...]] = {}
        for cid, dl in local_of.items():
            opt: List[Optional[int]] = []
            seen: set = set()
            ok = True
            for m in dl:
                tr = self.local[cid].tracks.get(m.local_id)
                pid = tr.person_id if tr is not None else None
                j = pid_to_j.get(pid) if pid is not None else None
                if j is not None and j in seen:      # 同一相机同一人被认领两次 → 无效
                    ok = False
                    break
                seen.add(j)
                opt.append(j)
            combo[cid] = tuple(opt) if ok else tuple([None] * len(dl))
        return combo if any(any(o is not None for o in v) for v in combo.values()) else None

    # -- 内部：新建轨迹 --
    def _spawn_tracks(self, leftovers: Dict[int, List[LocalMatch]],
                      frames: Dict[int, Frame],
                      time_s: Optional[float]) -> List[int]:
        """未认领检测 → 跨相机聚类成新人（用**躯干锚点**做多视角一致检验）。

        ⚠️ 不能用 bbox 中心三角化：2D 框中心不是某个固定 3D 点的投影（随视角变化），
        跨视角根本不共线，分组会串人（实测把两人拼成 3 个「人」）。这里对未认领框
        跑一次姿态取骨盆锚点（:func:`anchor_2d`），退而求其次用框底中点（≈双脚间地面
        点，跨视角近似一致）。

        **防误检**：宽视野下 YOLO 低阈值会检出椅子、墙上照片、腿部碎片。只有
        ① 框置信度 ≥ ``spawn_min_score`` 且 ② 三角化出的 3D 根落在世界系合理区间
        （``spawn_z_range`` / ``spawn_max_dist``）的组才允许开新身份；新身份还要
        连续 ``confirm_hits`` 帧有观测才「确认」（未确认不发观测）。
        """
        new_pids: List[int] = []
        pool = {cid: list(v) for cid, v in leftovers.items()}
        # anchors: {cid: [(LocalMatch, anchor_xy (2,), conf)]}，按对象身份索引，pop 无需重排
        anchors: Dict[int, List[Tuple[LocalMatch, np.ndarray, float]]] = {}
        for cid, lst in pool.items():
            if not lst:
                continue
            boxes = np.asarray([m.box for m in lst], dtype=np.float32)
            poses = self.pose_fn(cid, frames[cid], boxes) if self.pose_fn else []
            for i, m in enumerate(lst):
                if float(m.score) < self.cfg.spawn_min_score:
                    continue                      # 低置信度误检不参与开新身份
                a = anchor_2d(poses[i], self.cfg.anchor_min_conf) if i < len(poses) else None
                if a is None:
                    b = np.asarray(m.box, dtype=np.float64)
                    a = ((b[0] + b[2]) * 0.5, b[3], float(m.score))
                uv = self.tri.undistort_points(cid, a[:2])
                if uv is None:
                    continue
                anchors.setdefault(cid, []).append(
                    (m, np.asarray(uv, dtype=np.float64), float(a[2])))

        while len(self.tracks) < self.cfg.max_people:
            best = self._best_leftover_group(anchors)
            if best is None:
                break
            views, X = best
            if not self._plausible_root(X):
                # 几何上不可能的人（椅子/照片拼出来的）→ 丢弃这一组，不再试
                for cid, (m, _a, _c) in views.items():
                    anchors[cid] = [e for e in anchors.get(cid, []) if e[0] is not m]
                continue
            pid = self._new_person_id(X, time_s)
            tr = PersonTrack(pid, PersonKalman(X, self.cfg, time_s), self.cfg)
            tr.age = 1
            tr.hits = 1
            tr.hit_streak = 1
            for cid, (m, _a, _c) in views.items():
                if m in pool.get(cid, []):
                    pool[cid].remove(m)
                anchors[cid] = [e for e in anchors.get(cid, []) if e[0] is not m]
                tr.last_bbox[cid] = np.asarray(m.box, dtype=np.float32)
                ltr = self.local[cid].tracks.get(m.local_id)
                if ltr is not None:
                    ltr.person_id = pid
            self.tracks[pid] = tr
            new_pids.append(pid)
        return new_pids

    def _pose_ok(self, pose: Pose2D) -> bool:
        """姿态质量门：至少 ``min_pose_joints`` 个关节置信度 ≥ ``anchor_min_conf``。

        误检框（椅子/墙上照片）跑姿态只会得到一堆低置信噪声点；把它们喂进阶段 D
        会污染三角化。真人即使被遮挡/穿黑衣服，也总有几个高置信关节（肩/髋/踝）。
        """
        kp = np.asarray(pose.keypoints, dtype=np.float64)
        if kp.ndim != 2 or kp.shape[1] < 3:
            return False
        return int((kp[:, 2] >= self.cfg.anchor_min_conf).sum()) >= self.cfg.min_pose_joints

    def _plausible_root(self, X: np.ndarray) -> bool:
        """3D 根关节是否落在「人可能站的位置」：桌面系高度 + 离原点水平距离。"""
        X = np.asarray(X, dtype=np.float64).reshape(3)
        if not np.isfinite(X).all():
            return False
        z_lo, z_hi = self.cfg.spawn_z_range
        if not (z_lo <= X[2] <= z_hi):
            return False
        return float(np.hypot(X[0], X[1])) <= self.cfg.spawn_max_dist

    def _best_leftover_group(self, anchors):
        """在未认领检测里找「跨相机锚点一致」的一组（≥ new_track_min_views）。

        返回 ``(views, X)``，``views = {cid: (LocalMatch, anchor, conf)}``。
        锚点坐标必须已是**无畸变**像素（与 ``_dlt``/``reproj`` 同口径）。
        """
        cams = sorted([c for c in anchors if anchors.get(c)])
        best = None
        for i, a in enumerate(cams):
            for b in cams[i + 1:]:
                for ma, aa, ca in anchors[a]:
                    for mb, ab, cb in anchors[b]:
                        X = self.tri._dlt([a, b], {a: tuple(aa), b: tuple(ab)},
                                          {a: max(ca, 1e-3), b: max(cb, 1e-3)})
                        if X is None:
                            continue
                        views = {a: (ma, aa, ca), b: (mb, ab, cb)}
                        errs = [self.tri.reproj(a, X, aa), self.tri.reproj(b, X, ab)]
                        for c in cams:
                            if c in (a, b):
                                continue
                            best_c = None
                            for mc, ac, cc in anchors[c]:
                                e = self.tri.reproj(c, X, ac)
                                if e <= self.cfg.gate_px and (best_c is None or e < best_c[0]):
                                    best_c = (e, mc, ac, cc)
                            if best_c is not None:
                                views[c] = (best_c[1], best_c[2], best_c[3])
                                errs.append(best_c[0])
                        if len(views) < self.cfg.new_track_min_views:
                            continue
                        score = (len(views), -float(np.mean(errs)))
                        if best is None or score > best[0]:
                            best = (score, views, X)
        if best is None:
            return None
        return best[1], best[2]

    def _new_person_id(self, X: np.ndarray, time_s: Optional[float]) -> int:
        """优先复用最近刚消失的身份（位置接近且时间接近），否则开新号。"""
        t = float(time_s) if time_s is not None else float(self._frame_idx)
        best = None
        for pid, (pos, ts) in list(self._retired.items()):
            dt = abs(t - ts)
            d = float(np.linalg.norm(np.asarray(X) - pos))
            if dt <= 1.0 and d <= 1.0 and (best is None or d < best[0]):
                best = (d, pid)
        if best is not None:
            self._retired.pop(best[1], None)
            return best[1]
        pid = self._next_person
        self._next_person += 1
        return pid

    def _retire(self) -> None:
        t = float(self._frame_idx)
        for pid, tr in list(self.tracks.items()):
            if tr.alive:
                continue
            self._retired[pid] = (tr.kalman.position, t)
            del self.tracks[pid]

    def local_set_ids(self, pid: int, per_cam: Dict[int, LocalMatch]) -> None:
        for cid, m in per_cam.items():
            ltr = self.local[cid].tracks.get(m.local_id)
            if ltr is not None:
                ltr.person_id = pid

    # -- 内部：组装结果（姿态 + 阶段 D + 阶段 B）--
    def _build_result(self, tr: PersonTrack, per_cam: Dict[int, LocalMatch],
                      frames: Dict[int, Frame],
                      time_s: Optional[float]) -> TrackFrameResult:
        obs: Dict[int, Pose2D] = {}
        sources: Dict[int, str] = {}
        boxes: Dict[int, np.ndarray] = {}
        cam_w: Dict[int, float] = {}
        for cid, m in per_cam.items():
            frame = frames.get(cid)
            if frame is None or self.pose_fn is None:
                continue
            poses = self.pose_fn(cid, frame, np.asarray([m.box], dtype=np.float32))
            if not poses:
                continue
            pose = poses[0]
            if not self._pose_ok(pose):
                # 框落在误检上（椅子/照片）或人太小 → 姿态几乎全是低置信噪声，丢弃
                continue
            obs[cid] = pose
            sources[cid] = m.source
            boxes[cid] = np.asarray(m.box, dtype=np.float32)
            tr.note_observation(cid, pose, m.source)
            kp = np.asarray(pose.keypoints, dtype=np.float64)
            kpt_conf = float(np.mean(kp[:, 2])) if kp.ndim == 2 and kp.shape[1] > 2 else 0.0
            cam_w[cid] = float(max(pose.score, 1e-3) * max(kpt_conf, 1e-3))

        # 下半身可信度门：远端相机隔球桌看人时膝/踝是姿态模型外推的假点（置信度远低于
        # 同视角上半身，实测残差 32~63px），把它们置 0，只让看得见腿的相机决定下半身。
        masked: Dict[int, bool] = {}
        tri_obs = obs
        if self.cfg.lower_body_gate and obs:
            tri_obs = {}
            for cid, pose in obs.items():
                bad = lower_body_unreliable(
                    pose, ratio=self.cfg.lower_body_conf_ratio,
                    abs_min=self.cfg.lower_body_conf_abs)
                masked[cid] = bool(bad)
                tri_obs[cid] = mask_lower_body(
                    pose, include_hips=self.cfg.mask_hips_when_unreliable) if bad else pose

        robust = None
        if len(tri_obs) >= 2:
            robust = self.tri.triangulate_pose_robust(
                tri_obs, cam_weights=cam_w,
                decay={cid: tr.decay(cid) for cid in tri_obs},
                cfg=self.cfg.robust)
            # 每相机误差 EMA（重投影到该相机与观测根关节的距离）
            ri = robust.root_index
            if ri >= 0 and np.isfinite(robust.skeleton.keypoints[ri]).all():
                for cid, pose in obs.items():          # 用原始观测测根关节误差
                    a = anchor_2d(pose, self.cfg.anchor_min_conf)
                    if a is None:
                        continue
                    err = self.tri.reproj(cid, robust.skeleton.keypoints[ri], a[:2])
                    tr.update_error_ema(cid, err)

        # 阶段 B：用三角化根关节更新卡尔曼
        z = None
        R = None
        if robust is not None and robust.root_index >= 0:
            z = robust.skeleton.keypoints[robust.root_index]
            if not np.isfinite(z).all():
                z = None
        if z is not None:
            base = self.cfg.obs_base_std ** 2
            R = np.asarray(robust.root_cov, dtype=np.float64)
            if not np.isfinite(R).all():
                R = np.eye(3) * base
            R = R + np.eye(3) * base
            eig = np.linalg.eigvalsh((R + R.T) * 0.5)
            if eig.min() < base:
                R = R + np.eye(3) * (base - eig.min())
            cap = self.cfg.obs_max_std ** 2
            eig = np.linalg.eigvalsh((R + R.T) * 0.5)
            if eig.max() > cap:
                R = R * (cap / eig.max())
        tr.kalman.update(z, R)
        tr.age += 1
        if z is not None:
            tr.hits += 1
            tr.misses = 0
            tr.hit_streak += 1
        else:
            tr.misses += 1
            tr.miss_total += 1
            tr.hit_streak = 0
        if not tr.confirmed and tr.hit_streak >= self.cfg.confirm_hits:
            tr.confirmed = True
        tr.last_robust = robust
        return TrackFrameResult(
            person_id=tr.person_id, obs=tri_obs, source=sources, boxes=boxes,
            cam_weights=cam_w, robust=robust, root=tr.kalman.position.copy(),
            root_cov=(R if R is not None else np.eye(3) * self.cfg.obs_base_std ** 2),
            track=tr, raw_obs=obs, lower_body_masked=masked,
        )
