"""3D 骨架跨帧身份分配：把多个球员稳定地对应到固定 ID。

背景：``match_people`` 是逐帧独立的几何匹配，返回的骨架顺序每帧可能翻转，
导致两个球员的身份/颜色闪来闪去。本模块提供两种稳定策略：

- **分区模式（推荐，硬编码）**：按 3D 质心在球桌世界系某轴（如长边 Y）的位置，
  把球桌两侧的人固定成 ID=0 / ID=1。绝对稳定，不受时序影响——乒乓球两人分居
  球桌长边两侧，天然适合。
- **时序跟踪模式**：跨帧按 3D 质心最近邻关联稳定 ID（适合没有固定分区语义的场景）。

输出都按 ID 稳定排序，供可视化按 ID 配色。
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from ..core.types import Skeleton3D

__all__ = ["PoseTracker", "centroid"]


def centroid(skel: Skeleton3D) -> Optional[np.ndarray]:
    """骨架 3D 质心：有效关节的均值（世界系，米）。无有效关节返回 None。"""
    kp = np.asarray(skel.keypoints, dtype=np.float64)
    valid = np.isfinite(kp).all(axis=1)
    if valid.sum() == 0:
        return None
    return kp[valid].mean(axis=0)


class PoseTracker:
    """3D 骨架身份分配器（分区硬编码 或 时序最近邻）。"""

    def __init__(
        self,
        max_dist: float = 0.8,
        max_miss: int = 15,
        partition_axis: Optional[int] = None,
        partition_threshold: Optional[float] = None,
    ) -> None:
        """
        Args:
            max_dist: 时序跟踪的关联门限（米）。
            max_miss: 时序跟踪的连续丢失帧数上限。
            partition_axis: 分区模式用的世界系坐标轴（0=X 短边, 1=Y 长边, 2=Z）。
                设为 None 则走时序跟踪。
            partition_threshold: 分区阈值（米）。质心该轴坐标 < 阈值 → ID=0，否则 ID=1。
        """
        self.max_dist = float(max_dist)
        self.max_miss = int(max_miss)
        self.partition_axis = partition_axis
        self.partition_threshold = partition_threshold
        self._tracks: List[dict] = []
        self._next_id = 0

    # ------------------------------------------------------------------
    def update(self, skeletons: List[Skeleton3D]) -> List[Skeleton3D]:
        """喂入一帧骨架，返回**按稳定 ID 排序**的骨架列表。"""
        if self.partition_axis is not None:
            return self._partition_update(skeletons)
        return self._track_update(skeletons)

    # ------------------------------------------------------------------
    def _partition_update(self, skeletons: List[Skeleton3D]) -> List[Skeleton3D]:
        """分区硬编码：按质心在该轴坐标，< 阈值 → ID=0，≥ 阈值 → ID=1。"""
        axis = self.partition_axis
        thr = self.partition_threshold
        key = []
        for s in skeletons:
            c = centroid(s)
            if c is None:
                key.append((1, 1))  # 无质心排最后
            else:
                key.append((0, 0 if float(c[axis]) < thr else 1))
        order = sorted(range(len(skeletons)), key=lambda j: key[j])
        return [skeletons[j] for j in order]

    # ------------------------------------------------------------------
    def _track_update(self, skeletons: List[Skeleton3D]) -> List[Skeleton3D]:
        """时序最近邻：贪心关联到最近的 track，保持 ID 跨帧稳定。"""
        n = len(skeletons)
        centroids = [centroid(s) for s in skeletons]
        used = [False] * n
        assign = [-1] * n

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

        for j in range(n):
            if not used[j] and centroids[j] is not None:
                assign[j] = self._next_id
                self._tracks.append({"id": self._next_id, "centroid": centroids[j], "miss": 0})
                self._next_id += 1
                used[j] = True

        self._tracks = [t for t in self._tracks if t["miss"] <= self.max_miss]

        order = sorted(range(n), key=lambda j: (assign[j] < 0, assign[j]))
        return [skeletons[j] for j in order]

    def reset(self) -> None:
        self._tracks = []
        self._next_id = 0
