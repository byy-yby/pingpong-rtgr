"""跨视角球员匹配：把各相机检测到的人对应到同一身份。

RTMPose 是 top-down 检测，单视角内已经完成「关键点 -> 实例」分组，所以跨视角
只需做**实例级**匹配（哪几台相机里的哪个框是同一个人）。

运动场景球员穿相同队服，外观特征不可靠，这里只用几何约束：

1. 每个实例算一个鲁棒的「躯干锚点」（首选骨盆、退化为躯干关节置信度加权质心）。
2. 两两相机对的锚点做三角化，用**重投影误差**（即极线一致性）门限判断是否同一人，
   每对相机内按误差升序做一对一贪心匹配。
3. 并查集把各相机对的匹配合并成全局身份（同一人被 3~4 台相机看到时自然串成一组）。
4. 一致性精化：>2 视角的身份，先三角化共识锚点，再剔除与共识不一致的错配成员。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core.types import Pose2D
from .triangulate import (
    DEFAULT_MAX_REPROJ_PX,
    DEFAULT_MIN_CONF,
    MultiViewTriangulator,
)

# 用于跨视角匹配的鲁棒锚点关节（halpe26 索引）：骨盆 / 颈 / 双肩 / 双髋。
# 这些关节靠近躯干中心、通常可见且置信度高，比单用脚踝稳健。
_ANCHOR_JOINTS = (19, 18, 5, 6, 11, 12)

# 半固定机位下，球桌两侧各一组相机、各看各的人：cam0/cam2 看一边、cam1/cam3 看另一边。
# 用于 :func:`match_people_fixed` 的默认分组（换机位/换边时改这里或用参数覆盖）。
DEFAULT_PERSON_GROUPS: List[List[int]] = [[0, 2], [1, 3]]


@dataclass
class AssociationConfig:
    """跨视角匹配参数。

    Attributes:
        anchor_min_conf: 锚点关节最低置信度。
        anchor_max_reproj_px: 锚点三角化重投影误差门限（像素），超过即判定不是同一人。
            对 2 视角三角化，该误差 ≈ 极线(Sampson)距离，反映「是不是同一个 3D 点」。
        min_views: 至少多少个视角才能构成一个人（通常 2）。
    """

    anchor_min_conf: float = DEFAULT_MIN_CONF
    anchor_max_reproj_px: float = 30.0
    min_views: int = 2

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "AssociationConfig":
        d = d or {}
        return cls(
            anchor_min_conf=float(d.get("anchor_min_conf", DEFAULT_MIN_CONF)),
            anchor_max_reproj_px=float(d.get("anchor_max_reproj_px", 30.0)),
            min_views=int(d.get("min_views", 2)),
        )


def anchor_2d(
    pose: Pose2D, min_conf: float = DEFAULT_MIN_CONF
) -> Optional[Tuple[float, float, float]]:
    """算一个实例的鲁棒躯干锚点 ``(x, y, conf)``，用于跨视角匹配。

    首选骨盆（halpe26 索引 19）——它是单一 3D 点，极线约束最严格；骨盆不可用时
    退化为锚点关节的置信度加权质心；再退化到全部关键点的置信度加权质心。
    无任何可信关键点返回 None。
    """
    kp = pose.keypoint(19)
    if float(kp[2]) >= min_conf and np.isfinite(kp[0]):
        return (float(kp[0]), float(kp[1]), float(kp[2]))

    pts: List[Tuple[float, float, float]] = []
    for j in _ANCHOR_JOINTS:
        k = pose.keypoint(j)
        if float(k[2]) >= min_conf and np.isfinite(k[0]):
            pts.append((float(k[0]), float(k[1]), float(k[2])))
    if pts:
        w = np.array([p[2] for p in pts], dtype=np.float64)
        x = float(np.average([p[0] for p in pts], weights=w))
        y = float(np.average([p[1] for p in pts], weights=w))
        return (x, y, float(np.mean(w)))

    k = pose.keypoints
    if k.ndim != 2 or k.shape[1] < 3:
        return None
    m = (k[:, 2] >= min_conf) & np.isfinite(k[:, 0])
    if not m.any():
        return None
    w = k[m, 2].astype(np.float64)
    return (
        float(np.average(k[m, 0], weights=w)),
        float(np.average(k[m, 1], weights=w)),
        float(np.mean(w)),
    )


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def match_people(
    poses_per_cam: Dict[int, List[Pose2D]],
    triangulator: MultiViewTriangulator,
    cfg: Optional[AssociationConfig] = None,
) -> List[Dict[int, Pose2D]]:
    """把多相机检测结果匹配成若干「人」。

    Args:
        poses_per_cam: ``{cam_id: [Pose2D, ...]}``，各相机本帧检测到的人。
        triangulator: 用于锚点三角化与重投影误差评估。
        cfg: 匹配参数。

    Returns:
        每个人一个 ``{cam_id: Pose2D}``（至少 ``cfg.min_views`` 个视角）。无匹配
        返回空列表。
    """
    cfg = cfg or AssociationConfig()
    cams = sorted(c for c in poses_per_cam if c in triangulator.P)

    # 展平节点：全局索引 g -> (cam_id, 该相机内第 i 个检测)
    det_cam: List[int] = []
    det_idx: List[int] = []
    anchors: List[Tuple[float, float, float]] = []
    node: Dict[Tuple[int, int], int] = {}
    for cid in cams:
        for i, pose in enumerate(poses_per_cam.get(cid, [])):
            a = anchor_2d(pose, cfg.anchor_min_conf)
            if a is None:
                continue
            g = len(det_cam)
            node[(cid, i)] = g
            det_cam.append(cid)
            det_idx.append(i)
            anchors.append(a)

    if len(det_cam) < 2:
        return []

    uf = _UnionFind(len(det_cam))

    # 两两相机对：候选匹配按锚点三角化重投影误差升序，贪心一对一
    for ai in range(len(cams)):
        for bi in range(ai + 1, len(cams)):
            ca, cb = cams[ai], cams[bi]
            cands: List[Tuple[float, int, int]] = []
            for ia in range(len(poses_per_cam.get(ca, []))):
                ga = node.get((ca, ia))
                if ga is None:
                    continue
                xa, ya, ca_conf = anchors[ga]
                for ib in range(len(poses_per_cam.get(cb, []))):
                    gb = node.get((cb, ib))
                    if gb is None:
                        continue
                    xb, yb, cb_conf = anchors[gb]
                    res = triangulator.triangulate_point(
                        {ca: (xa, ya), cb: (xb, yb)},
                        {ca: ca_conf, cb: cb_conf},
                        min_conf=cfg.anchor_min_conf,
                        max_reproj_px=cfg.anchor_max_reproj_px,
                    )
                    if res is None:
                        continue
                    err = res[2]
                    if err <= cfg.anchor_max_reproj_px:
                        cands.append((err, ga, gb))

            cands.sort(key=lambda t: t[0])
            used_a: set = set()
            used_b: set = set()
            for err, ga, gb in cands:
                if ga in used_a or gb in used_b:
                    continue
                used_a.add(ga)
                used_b.add(gb)
                uf.union(ga, gb)

    # 按根分组 -> 初始身份
    groups: Dict[int, List[int]] = {}
    for g in range(len(det_cam)):
        groups.setdefault(uf.find(g), []).append(g)

    people: List[Dict[int, Pose2D]] = []
    for root, gs in groups.items():
        if len(gs) < cfg.min_views:
            continue
        if len(gs) == 2:
            people.append(_obs(gs, det_cam, det_idx, poses_per_cam))
            continue
        # >2 视角：一致性精化，剔除与共识锚点不一致的错配成员
        pts = {det_cam[g]: anchors[g][:2] for g in gs}
        cs = {det_cam[g]: anchors[g][2] for g in gs}
        res = triangulator.triangulate_point(
            pts, cs, min_conf=cfg.anchor_min_conf,
            max_reproj_px=cfg.anchor_max_reproj_px,
        )
        if res is None:
            continue
        X = res[0]
        keep = [
            g for g in gs
            if triangulator.reproj(det_cam[g], X, anchors[g][:2]) <= cfg.anchor_max_reproj_px
        ]
        if len(keep) >= cfg.min_views:
            people.append(_obs(keep, det_cam, det_idx, poses_per_cam))

    return people


def _obs(
    gs: List[int],
    det_cam: List[int],
    det_idx: List[int],
    poses_per_cam: Dict[int, List[Pose2D]],
) -> Dict[int, Pose2D]:
    return {det_cam[g]: poses_per_cam[det_cam[g]][det_idx[g]] for g in gs}


def match_people_fixed(
    poses_per_cam: Dict[int, List[Pose2D]],
    triangulator: MultiViewTriangulator,
    groups: Optional[List[List[int]]] = None,
    min_conf: float = DEFAULT_MIN_CONF,
) -> List[Dict[int, Pose2D]]:
    """固定相机分组匹配：每组相机各自拍到一个人，直接按组返回观测。

    适合半固定机位下「球桌两侧各一组相机、各看各的人」的场景（如 cam0/cam2 看
    A、cam1/cam3 看 B）：人只被自己那组相机看到，跨组做几何配对反而会错配（把
    一边的人配到另一边的相机上）。这里不做全局匹配，每组相机内部各取一个检测，
    直接作为一个人；组内某相机看到 >1 个检测（偶发串扰/瞥到对面的人）时，用组内
    另一台相机做局部一致性挑选。

    Args:
        poses_per_cam: ``{cam_id: [Pose2D, ...]}``。
        triangulator: 用于组内串扰挑选时锚点三角化。
        groups: 相机分组，如 ``[[0, 2], [1, 3]]``；默认 :data:`DEFAULT_PERSON_GROUPS`。
        min_conf: 锚点置信度下限。

    Returns:
        每个人一个 ``{cam_id: Pose2D}``（至少 2 个视角），列表顺序 = groups 顺序
        （即身份，无需再按 3D 位置重排）。
    """
    groups = groups or DEFAULT_PERSON_GROUPS
    people: List[Dict[int, Pose2D]] = []
    for group in groups:
        obs: Dict[int, Pose2D] = {}
        for cid in group:
            dets = poses_per_cam.get(cid, [])
            if not dets:
                continue
            if len(dets) == 1:
                obs[cid] = dets[0]
            else:
                obs[cid] = _pick_in_group(
                    dets, cid, group, poses_per_cam, triangulator, min_conf
                )
        if len(obs) >= 2:
            people.append(obs)
    return people


def _pick_in_group(
    dets: List[Pose2D],
    cid: int,
    group: List[int],
    poses_per_cam: Dict[int, List[Pose2D]],
    triangulator: MultiViewTriangulator,
    min_conf: float,
) -> Pose2D:
    """组内串扰挑选：该相机看到 >1 个检测时，选与组内另一台相机锚点最一致的那个。

    用锚点 2 视角三角化的重投影误差当「一致性」：正确的人误差小，瞥到的对面的人
    误差大。组内无其它可用相机（都漏检）时回退到第一个检测（通常 NMS 后最高分在前）。
    """
    refs = [c for c in group if c != cid and poses_per_cam.get(c)]
    if not refs:
        return dets[0]
    rcid = refs[0]
    ra = anchor_2d(poses_per_cam[rcid][0], min_conf)
    if ra is None:
        return dets[0]
    best, best_e = dets[0], float("inf")
    for d in dets:
        a = anchor_2d(d, min_conf)
        if a is None:
            continue
        r = triangulator.triangulate_point(
            {cid: a[:2], rcid: ra[:2]},
            {cid: a[2], rcid: ra[2]},
            min_conf=min_conf,
        )
        if r is not None and r[2] < best_e:
            best, best_e = d, r[2]
    return best


def keep_nearest_person(
    poses_per_cam: Dict[int, List[Pose2D]],
) -> Dict[int, List[Pose2D]]:
    """每相机只保留最近的人（bbox 面积最大），排除远处/对面的人。

    换广角镜头后每相机可能同时框到近/远两人，远处那人的检测不稳定（时有时无），
    会扰乱固定分组匹配与三角化（身份错配/骨架闪跳）。这里按 bbox 面积取最大者，
    把每相机压回 ≤1 人，恢复 :func:`match_people_fixed` 所需的「每组相机各看
    一人」前提。bbox 缺失时回退到关键点包络面积。

    Args:
        poses_per_cam: ``{cam_id: [Pose2D, ...]}`` 各相机本帧检测到的人。

    Returns:
        ``{cam_id: [Pose2D]}``，每相机至多 1 人（bbox 面积最大 = 离相机最近、
        属于该相机自己这一侧的人）。无检测的相机不出现在结果里。
    """
    out: Dict[int, List[Pose2D]] = {}
    for cid, poses in poses_per_cam.items():
        best: Optional[Pose2D] = None
        best_area = -1.0
        for p in poses:
            a = _person_area(p)
            if a > best_area:
                best, best_area = p, a
        if best is not None and best_area > 0.0:
            out[cid] = [best]
    return out


def _person_area(pose: Pose2D) -> float:
    """检测框面积（像素²）；bbox 缺失时回退到关键点有效包络面积。"""
    b = pose.bbox
    if b is not None:
        b = np.asarray(b, dtype=np.float64)
        if b.shape[0] >= 4:
            w = float(b[2]) - float(b[0])
            h = float(b[3]) - float(b[1])
            if w > 0.0 and h > 0.0:
                return w * h
    kp = np.asarray(pose.keypoints, dtype=np.float64)
    if kp.ndim != 2 or kp.shape[1] < 2:
        return 0.0
    m = np.isfinite(kp[:, :2]).all(axis=1)
    if not m.any():
        return 0.0
    x = kp[m, 0]
    y = kp[m, 1]
    return float((x.max() - x.min()) * (y.max() - y.min()))
