"""3D 骨架跨帧身份跟踪：按 3D 质心最近邻关联，稳定多人的身份 ID。

背景：``match_people`` 是逐帧独立的几何匹配，返回的骨架顺序每帧可能翻转，
导致两个球员的身份/颜色闪来闪去。本模块在三角化之后加一层时序跟踪：每个
活跃 track 维护一个稳定 ID 与 3D 质心，每帧把新骨架贪心关联到最近的 track
（质心距离 < 门限），输出按 ID 稳定排序的骨架列表，供可视化按 ID 配色。
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ..core.types import Skeleton3D

__all__ = ["PoseTracker"]


def _centroid(skel: Skeleton3D) -> Optional[np.ndarray]:
    """骨架 3D 质心：有效关节的均值（世界系，米）。无有效关节返回 None。"""
    kp = np.asarray(skel.keypoints, dtype=np.float64)
    valid = np.isfinite(kp).all(axis=1)
    if valid.sum() == 0:
        return None
    return kp[valid].mean(axis=0)


class PoseTracker:
    """3D 骨架多目标跟踪器（贪心最近邻 + 按 ID 稳定排序）。"""

    def __init__(self, max_dist: float = 0.8, max_miss: int = 15) -> None:
        """
        Args:
            max_dist: 关联门限（米）。新骨架与某 track 质心的距离小于该值才算同一个人。
                帧间人移动通常 <0.1m，取 0.8m 足够稳健又不会把两个人串起来。
            max_miss: 连续丢失帧数上限，超过则删除该 track。
        """
        self.max_dist = float(max_dist)
        self.max_miss = int(max_miss)
        self._tracks: List[dict] = []  # {id, centroid, miss}
        self._next_id = 0

    def update(self, skeletons: List[Skeleton3D]) -> List[Skeleton3D]:
        """喂入一帧骨架，返回**按稳定 ID 排序**的骨架列表。

        排序保证：同一个人的骨架在跨帧列表中位置稳定（除非其 ID 顺序被新出现的人
        插入），从而可视化按顺序配色的身份不闪变。
        """
        n = len(skeletons)
        centroids = [_centroid(s) for s in skeletons]
        used = [False] * n
        assign = [-1] * n

        # 1) 每个 track 贪心抢最近的未分配骨架
        for track in self._tracks:
            best_j, best_d = -1, self.max_dist
            for j in range(n):
                if used[j] or centroids[j] is None:
                    continue
                d = float(np.linalg.norm(centroids[j] - track["centroid"]))
                if d < best_d:
                    best_j, best_d = j, d
            if best_j >= 0:
                assign[best_j] = track["id"]
                track["centroid"] = centroids[best_j]
                track["miss"] = 0
                used[best_j] = True
            else:
                track["miss"] += 1

        # 2) 未关联的骨架 → 新建 track
        for j in range(n):
            if not used[j] and centroids[j] is not None:
                assign[j] = self._next_id
                self._tracks.append({"id": self._next_id, "centroid": centroids[j], "miss": 0})
                self._next_id += 1
                used[j] = True

        # 3) 删除丢失过久的 track
        self._tracks = [t for t in self._tracks if t["miss"] <= self.max_miss]

        # 4) 按 ID 稳定排序（无有效质心的骨架排最后，保持原顺序）
        ordered = sorted(
            (assign[j], j) for j in range(n)
        )
        ordered.sort(key=lambda t: (t[0] < 0, t[0]))
        return [skeletons[j] for _, j in ordered]

    def reset(self) -> None:
        self._tracks = []
        self._next_id = 0
