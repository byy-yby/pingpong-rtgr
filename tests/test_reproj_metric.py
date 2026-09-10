"""重投影误差两种口径（``_proj_err_multi``）的纯 numpy 单测。

为什么单测它：这个函数是「重投影误差」这个指标的全部实现，而指标本身是判断
「改这个参数有没有让重建变好」的依据——算错了会得出反向结论。两个口径的语义：
``mean_all`` 所有观测视角等权；``mean_best`` 每个关节只取**置信度最高**的那台相机
（低置信度视角的 2D 是外推的，拿它当基准会污染指标）。
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "reconstruct_video.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("_recon_video_for_test", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


rv = _load_module()

N_J = 25


def _cam(center, look=(0.0, 0.0, 1.5), f=1000.0, cx=720.0, cy=540.0):
    """相机在 world 的 ``center``，看向 ``look``，返回 3x4 投影矩阵 P = K[R|t]。"""
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
    R = np.stack([x, y, z])                       # world → camera
    t = -R @ center
    K = np.asarray([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    return K @ np.hstack([R, t.reshape(3, 1)])


def _project(P, X):
    h = np.hstack([X, np.ones((len(X), 1))]) @ P.T
    return h[:, :2] / h[:, 2:3]


@pytest.fixture()
def rig():
    """3 台相机 + 一具合成骨架的 3D 关节与「理想」2D 观测。"""
    rng = np.random.default_rng(0)
    X = rng.normal(0.0, 0.3, size=(N_J, 3))
    X[:, 2] += 1.5
    Ps = [_cam((0.0, -3.0, 1.5)), _cam((3.0, 0.0, 1.5)), _cam((-2.0, 2.0, 2.5))]
    kp2d = np.zeros((3, N_J, 3), np.float32)
    for i, P in enumerate(Ps):
        kp2d[i, :, :2] = _project(P, X)
        kp2d[i, :, 2] = 0.9
    return np.stack(Ps), kp2d, X


def test_perfect_observation_gives_zero(rig):
    Pall, kp2d, X = rig
    mean_all, mean_best, worst = rv._proj_err_multi(Pall, kp2d, X)
    assert mean_all == pytest.approx(0.0, abs=1e-3)
    assert mean_best == pytest.approx(0.0, abs=1e-3)
    assert worst == pytest.approx(0.0, abs=1e-3)


def test_mean_all_averages_views_mean_best_picks_max_conf(rig):
    """故意把**低置信度**那台相机的 2D 挪偏 20px：
    ``mean_all`` 被拉高（坏视角等权进来了），``mean_best`` 不受影响（它只信高置信度那台）。
    """
    Pall, kp2d, X = rig
    kp2d = kp2d.copy()
    kp2d[0, :, 0] += 20.0            # cam0 整体右移 20px
    kp2d[0, :, 2] = 0.3             # 且它是置信度最低的
    kp2d[1, :, 2] = 0.6
    kp2d[2, :, 2] = 0.9             # cam2 置信度最高，且没被动过

    mean_all, mean_best, worst = rv._proj_err_multi(Pall, kp2d, X)
    # 3 视角等权：均值 = (20 + 0 + 0)/3
    assert mean_all == pytest.approx(20.0 / 3.0, abs=1e-3)
    # 最高置信视角口径：每关节都取 cam2（误差 0），完全不被坏视角污染
    assert mean_best == pytest.approx(0.0, abs=1e-3)
    assert worst == pytest.approx(20.0, abs=1e-3)


def test_conf_zero_joints_excluded(rig):
    """conf==0 的观测（下半身门掩掉的）不进任何口径。"""
    Pall, kp2d, X = rig
    kp2d = kp2d.copy()
    kp2d[:, :5, 0] = 500.0          # 前 5 个关节全视角都挪飞
    kp2d[:, :5, 2] = 0.0            # 但置信度为 0 → 应被完全忽略
    mean_all, mean_best, worst = rv._proj_err_multi(Pall, kp2d, X)
    assert mean_all == pytest.approx(0.0, abs=1e-3)
    assert mean_best == pytest.approx(0.0, abs=1e-3)
    assert worst == pytest.approx(0.0, abs=1e-3)


def test_best_view_chosen_per_joint_not_per_person(rig):
    """「最高置信视角」是**逐关节**选的，不是整人一个视角。"""
    Pall, kp2d, X = rig
    kp2d = kp2d.copy()
    # 关节 0 在 cam0 最可信且偏 10px；关节 1 在 cam1 最可信且偏 30px
    kp2d[0, :, 2] = 0.5
    kp2d[1, :, 2] = 0.5
    kp2d[2, :, 2] = 0.5
    kp2d[0, 0, 2] = 0.99
    kp2d[0, 0, 0] += 10.0
    kp2d[1, 1, 2] = 0.99
    kp2d[1, 1, 0] += 30.0

    _mean_all, mean_best, _worst = rv._proj_err_multi(Pall, kp2d, X)
    # 其余 23 个关节误差 0；关节 0 取 10px、关节 1 取 30px → 均值 (10+30)/25
    assert mean_best == pytest.approx(40.0 / 25.0, abs=1e-3)


def test_joint_with_no_valid_view_is_skipped(rig):
    """某关节所有视角都 conf==0/无效 → 整个关节从 best 口径里去掉，而不是当 0 混进去。"""
    Pall, kp2d, X = rig
    kp2d = kp2d.copy()
    kp2d[:, 3, 2] = 0.0
    kp2d[0, 7, 0] += 12.0           # 只有 cam0 看得到关节 7，且偏 12px、conf 最高
    kp2d[1:, 7, 2] = 0.0
    _mean_all, mean_best, _worst = rv._proj_err_multi(Pall, kp2d, X)
    assert mean_best == pytest.approx(12.0 / 24.0, abs=1e-3)   # 分母 24，不是 25


def test_all_conf_zero_returns_nan(rig):
    Pall, kp2d, X = rig
    kp2d = np.zeros_like(kp2d)
    mean_all, mean_best, worst = rv._proj_err_multi(Pall, kp2d, X)
    assert np.isnan(mean_all) and np.isnan(mean_best) and np.isnan(worst)


def test_err_summary_reports_mean_above_median_on_tail():
    """长尾时 平均数 > 中位数：这正是「两个数都要报」的理由。"""
    series = [10.0] * 95 + [40.0] * 5          # 5% 的坏帧
    s = rv._err_summary(series)
    assert s["median"] == pytest.approx(10.0)
    assert s["mean"] == pytest.approx(11.5)
    assert s["mean"] > s["median"]
    assert s["p90"] == pytest.approx(10.0)


def test_err_summary_skips_nan_and_empty():
    s = rv._err_summary([10.0, float("nan"), 20.0, None])
    assert s["median"] == pytest.approx(15.0)
    empty = rv._err_summary([])
    assert all(np.isnan(v) for v in empty.values())
