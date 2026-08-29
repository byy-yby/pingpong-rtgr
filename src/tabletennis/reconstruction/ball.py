"""球 3D 重建：把多视角的 2D 球检测（Ball2D）三角化成 3D 球心。

球是**唯一的**，每台相机最多检到 0/1 个球，无需跨视角匹配（不像人体姿态）。
这里直接收 ``{cam_id: Ball2D}``，逐视角去畸变后调现成的
:meth:`MultiViewTriangulator.triangulate_point`（置信度加权 DLT + 外点剔除 + 交会角）。
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from ..core.types import Ball2D
from .triangulate import (
    DEFAULT_MIN_CONF,
    MultiViewTriangulator,
    undistort_keypoints,
)

__all__ = ["triangulate_ball"]


def undistort_ball_center(
    ball: Ball2D, K: np.ndarray, dist: np.ndarray
) -> Optional[Tuple[float, float]]:
    """把单个 Ball2D 的球心（畸变后像素坐标）去畸变，返回无畸变 ``(x, y)``。

    复用 :func:`~tabletennis.reconstruction.triangulate.undistort_keypoints`，
    与姿态三角化内部处理一致。坐标非法返回 None。
    """
    kp = np.array(
        [[float(ball.center[0]), float(ball.center[1]), float(ball.confidence)]],
        dtype=np.float64,
    )
    und = undistort_keypoints(kp, np.asarray(K, dtype=np.float64),
                              np.asarray(dist, dtype=np.float64))
    ux, uy = float(und[0, 0]), float(und[0, 1])
    if not (np.isfinite(ux) and np.isfinite(uy)):
        return None
    return (ux, uy)


def triangulate_ball(
    balls: Dict[int, Ball2D],
    triangulator: MultiViewTriangulator,
    min_conf: float = DEFAULT_MIN_CONF,
    **kwargs,
) -> Optional[Tuple[np.ndarray, float, float, int, float]]:
    """把多视角球检测三角化成 3D 球心。

    Args:
        balls: ``{cam_id: Ball2D}``，各视角检测到的球（同一触发时刻）。
        triangulator: 已用内外参初始化的三角化器。
        min_conf: 球检测置信度下限，低于此的视角不参与。
        **kwargs: 透传 :meth:`triangulate_point`（外点 / 交会角阈值）。

    Returns:
        ``(X, conf, reproj_err, n_views, angle_deg)``，与 ``triangulate_point``
        一致；视角 < 2 或全部失败返回 None。
    """
    points: Dict[int, Tuple[float, float]] = {}
    confs: Dict[int, float] = {}
    for cid, ball in balls.items():
        if cid not in triangulator.P:
            continue
        c = float(ball.confidence)
        if c < min_conf:
            continue
        uv = undistort_ball_center(ball, triangulator.K[cid], triangulator.dist[cid])
        if uv is None:
            continue
        points[cid] = uv
        confs[cid] = c

    if len(points) < 2:
        return None
    return triangulator.triangulate_point(points, confs, min_conf=min_conf, **kwargs)
