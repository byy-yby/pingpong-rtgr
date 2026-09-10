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

__all__ = ["triangulate_ball", "ball_reproj_errors", "undistort_ball_center"]


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


def ball_reproj_errors(
    balls: Dict[int, Ball2D],
    X: np.ndarray,
    triangulator: MultiViewTriangulator,
    min_conf: float = DEFAULT_MIN_CONF,
) -> Tuple[float, float, float, int]:
    """球心的重投影误差（像素）：``(err_all, err_best, err_worst, n_views)``。

    「误差」= 把重建出的 3D 球心用 ``P = K·[R|t]`` 投回各相机，与该相机检出的 2D 球心
    的欧氏距离（都在**无畸变**像素系，与三角化内部一致）。球没有 SMPL 那样的拟合步骤，
    3D 直接由 2D 最小二乘三角化得到，所以这个误差就是**三角化自身的残差**（跨视角一致性），
    **不是「和真值比」**——真值要靠 `scripts/error_budget/aruco_gt.py` 那类刚性靶标。

    两个口径（与姿态的 ``reconstruct_video.py::_proj_err_multi`` 对齐）：
    - ``err_all``：所有参与视角**等权**取均值 —— 就是三角化目标函数的口径。
    - ``err_best``：**只取置信度最高的那台相机**的残差。球检测置信度低时球心会飘
      （反光/遮挡/运动模糊/半个球出画），拿它当基准会污染指标；高置信度那台更接近真值。
    - ``err_worst``：参与视角里最差的一个，长尾用。

    参与视角的判定与 :func:`triangulate_ball` 完全一致（``conf >= min_conf`` 且去畸变成功），
    保证两个口径的**分母与三角化实际用到的视角一致**。

    Returns:
        ``(err_all, err_best, err_worst, n_views)``；3D 非法或无参与视角时误差为 NaN、
        ``n_views=0``。
    """
    X = np.asarray(X, dtype=np.float64).reshape(3)
    if not np.isfinite(X).all():
        return float("nan"), float("nan"), float("nan"), 0

    dists: list = []
    confs: list = []
    for cid in triangulator.cameras:
        ball = balls.get(cid)
        if ball is None:
            continue
        cf = float(ball.confidence)
        if cf < min_conf:
            continue
        p = undistort_ball_center(ball, triangulator.K[cid], triangulator.dist[cid])
        if p is None:
            continue
        ph = triangulator.P[cid] @ np.append(X, 1.0)
        if abs(float(ph[2])) < 1e-12:
            continue
        uv = ph[:2] / ph[2]
        dists.append(float(np.hypot(uv[0] - p[0], uv[1] - p[1])))
        confs.append(cf)
    if not dists:
        return float("nan"), float("nan"), float("nan"), 0

    d = np.asarray(dists, dtype=np.float64)
    c = np.asarray(confs, dtype=np.float64)
    return (float(np.mean(d)),                     # all：视角等权
            float(d[int(np.argmax(c))]),           # best：置信度最高那台
            float(np.max(d)),                      # worst
            int(d.size))
