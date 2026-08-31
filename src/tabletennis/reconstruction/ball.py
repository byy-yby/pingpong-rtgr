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

    与姿态三角化共用 :meth:`MultiViewTriangulator.triangulate_batch`（同一个批量
    DLT 核心）：把各视角球心去畸变后打包成 ``(1, n_cams, 2)`` 单点批量，一次求解。

    Args:
        balls: ``{cam_id: Ball2D}``，各视角检测到的球（同一触发时刻）。
        triangulator: 已用内外参初始化的三角化器。
        min_conf: 球检测置信度下限，低于此的视角不参与。
        **kwargs: 透传 :meth:`MultiViewTriangulator.triangulate_batch`（外点 / 交会角阈值）。

    Returns:
        ``(X, conf, reproj_err, n_views, angle_deg)``，与 ``triangulate_point``
        一致；视角 < 2 或全部失败返回 None。
    """
    cam_ids = triangulator.cameras
    n_cams = len(cam_ids)
    uv = np.full((1, n_cams, 2), np.nan, dtype=np.float64)
    conf = np.zeros((1, n_cams), dtype=np.float64)
    for c, cid in enumerate(cam_ids):
        ball = balls.get(cid)
        if ball is None:
            continue
        cf = float(ball.confidence)
        if cf < min_conf:
            continue
        p = undistort_ball_center(ball, triangulator.K[cid], triangulator.dist[cid])
        if p is None:
            continue
        uv[0, c, 0] = p[0]
        uv[0, c, 1] = p[1]
        conf[0, c] = cf

    X, conf3, err, nviews, angle = triangulator.triangulate_batch(
        uv, conf, min_conf=min_conf, **kwargs
    )
    if nviews[0] < 2:
        return None
    return (X[0], float(conf3[0]), float(err[0]), int(nviews[0]), float(angle[0]))
