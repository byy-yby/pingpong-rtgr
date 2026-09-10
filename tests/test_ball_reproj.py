"""球心重投影误差两口径（``ball_reproj_errors``）的纯 numpy 单测。

为什么单独测它：球的 3D 位置**没有拟合步骤**，直接由各视角 2D 检测加权 DLT 得到，
所以「球的 重投影误差」= 三角化自身的残差。口径必须与姿态那边
（``reconstruct_video.py::_proj_err_multi``）一致，否则两个指标不可比：
``all`` = 参与三角化的视角**等权**；``best`` = **只取置信度最高的那台相机**的残差
（低置信度检测的球心会飘，拿它当基准等于用坏点当真值）。
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tabletennis.core.types import Ball2D                      # noqa: E402
from tabletennis.reconstruction.ball import ball_reproj_errors  # noqa: E402


class _FakeTriangulator:
    """只提供 ``ball_reproj_errors`` 用到的四个属性（鸭子类型，免去标定文件）。"""

    def __init__(self, Ps, Ks, dists):
        self.cameras = sorted(Ps)
        self.P = Ps
        self.K = Ks
        self.dist = dists


def _cam(center, look, f=1000.0, cx=720.0, cy=540.0):
    """相机在 ``center`` 看向 ``look`` → ``(P, K)``，其中 ``P = K[R|t]``。

    ⚠️ ``K`` 必须单独返回：``P[:, :3]`` 是 ``K·R`` **不是** K，拿它当内参去畸变
    会算出离谱的坐标（``ball_reproj_errors`` 内部会调 ``undistort_ball_center``）。
    """
    center = np.asarray(center, np.float64)
    look = np.asarray(look, np.float64)
    z = look - center
    assert np.linalg.norm(z) > 1e-6, "相机不能摆在被看点上"
    z /= np.linalg.norm(z)
    up = np.asarray([0.0, 0.0, 1.0])
    if abs(float(z @ up)) > 0.99:
        up = np.asarray([0.0, 1.0, 0.0])
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    t = -R @ center
    K = np.asarray([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    return K @ np.hstack([R, t.reshape(3, 1)]), K


def _zig():
    """3 台相机（无畸变：dist 全 0），P/K/dist 三件套。"""
    cams = {0: _cam((0.0, -3.0, 1.5), (0.0, 0.0, 1.2)),
            1: _cam((3.0, 0.0, 1.5), (0.0, 0.0, 1.2)),
            2: _cam((-2.0, 2.0, 2.5), (0.0, 0.0, 1.2))}
    Ps = {c: pk[0] for c, pk in cams.items()}
    Ks = {c: pk[1] for c, pk in cams.items()}
    return _FakeTriangulator(Ps, Ks, {c: np.zeros(5) for c in Ps})


def _project(P, X):
    h = P @ np.append(np.asarray(X, np.float64), 1.0)
    return h[:2] / h[2]


def _balls(P, X, confs, shifts=None):
    """按各相机把 3D 球心 ``X`` 投到 2D 当检测（可加 ``shifts`` 像素偏移）。"""
    out = {}
    for c, Pc in P.items():
        uv = _project(Pc, X)
        if shifts and shifts.get(c):
            uv = uv + np.asarray(shifts[c], np.float64)
        out[c] = Ball2D(camera_id=c, center=uv, radius=8.0, confidence=confs[c])
    return out


X_TRUE = np.asarray([0.15, 0.40, 1.05])       # 桌面系上方一个球


def test_perfect_detections_give_zero():
    tri = _zig()
    balls = _balls(tri.P, X_TRUE, {0: 0.9, 1: 0.9, 2: 0.9})
    ea, eb, ew, nv = ball_reproj_errors(balls, X_TRUE, tri)
    assert nv == 3
    assert ea == pytest.approx(0.0, abs=1e-3)
    assert eb == pytest.approx(0.0, abs=1e-3)
    assert ew == pytest.approx(0.0, abs=1e-3)


def test_all_is_equal_weight_best_follows_max_conf():
    """把**最低置信度**那台相机的球心挪偏 21px：
    ``all`` 被拉高（坏视角等权进来了），``best`` 只跟最高置信度那台、不受影响。
    """
    tri = _zig()
    shifts = {0: (21.0, 0.0)}
    balls = _balls(tri.P, X_TRUE, {0: 0.3, 1: 0.6, 2: 0.9}, shifts)
    ea, eb, ew, nv = ball_reproj_errors(balls, X_TRUE, tri)
    assert nv == 3
    assert ea == pytest.approx(21.0 / 3.0, abs=1e-3)      # 视角等权
    assert eb == pytest.approx(0.0, abs=1e-3)             # cam2 置信度最高且没偏
    assert ew == pytest.approx(21.0, abs=1e-3)


def test_best_is_the_highest_conf_view_not_the_closest():
    """``best`` 取的是**置信度最高**那台，哪怕它偏得比别的视角多。"""
    tri = _zig()
    shifts = {0: (0.0, 0.0), 1: (30.0, 0.0), 2: (0.0, 0.0)}
    balls = _balls(tri.P, X_TRUE, {0: 0.2, 1: 0.99, 2: 0.2}, shifts)
    _ea, eb, ew, _nv = ball_reproj_errors(balls, X_TRUE, tri)
    assert eb == pytest.approx(30.0, abs=1e-3)             # cam1 最可信 → 就认它，哪怕偏 30px
    assert ew == pytest.approx(30.0, abs=1e-3)


def test_views_below_min_conf_are_excluded():
    """低于 ``min_conf`` 的视角不参与 —— 分母要跟三角化实际用到的视角一致。"""
    tri = _zig()
    shifts = {0: (100.0, 0.0)}                            # 挪飞但置信度太低
    balls = _balls(tri.P, X_TRUE, {0: 0.1, 1: 0.8, 2: 0.8}, shifts)
    ea, eb, ew, nv = ball_reproj_errors(balls, X_TRUE, tri, min_conf=0.5)
    assert nv == 2
    assert ea == pytest.approx(0.0, abs=1e-3)
    assert eb == pytest.approx(0.0, abs=1e-3)
    assert ew == pytest.approx(0.0, abs=1e-3)


def test_missing_camera_is_skipped():
    tri = _zig()
    balls = _balls(tri.P, X_TRUE, {0: 0.9, 1: 0.9, 2: 0.9})
    del balls[1]
    _ea, _eb, _ew, nv = ball_reproj_errors(balls, X_TRUE, tri)
    assert nv == 2


def test_nan_3d_and_no_views():
    tri = _zig()
    balls = _balls(tri.P, X_TRUE, {0: 0.9, 1: 0.9, 2: 0.9})
    ea, eb, ew, nv = ball_reproj_errors(balls, np.full(3, np.nan), tri)
    assert nv == 0 and all(np.isnan(v) for v in (ea, eb, ew))

    ea, eb, ew, nv = ball_reproj_errors({}, X_TRUE, tri)
    assert nv == 0 and all(np.isnan(v) for v in (ea, eb, ew))
