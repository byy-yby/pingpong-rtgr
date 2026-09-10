"""obs_filter（多视角共识降权）与 obs_cache（Pass-1 观测缓存）单元测试。"""
import numpy as np
import pytest

from tabletennis.core.types import Pose2D
from tabletennis.reconstruction.easymocap import HALPE26_TO_BODY25
from tabletennis.reconstruction.obs_cache import load_pass1, save_pass1
from tabletennis.reconstruction.obs_filter import (
    ConsensusConfig,
    filter_obs_by_consensus,
    format_stats,
)
from test_reconstruction import make_rig, project


def _body25_points():
    """一具合成 body25 骨架（站立，米）。"""
    X = np.zeros((25, 3))
    X[:, 2] = 1.0
    for j in range(25):
        X[j] = [0.15 * np.cos(j), 0.15 * np.sin(j), 1.0 + 0.02 * j]
    return X


def _halpe26_from_body25(X, intrinsics, extrinsics, cid, offset=None, conf=0.9):
    """把 body25 3D 投到某相机，按 HALPE26_TO_BODY25 反填 halpe26 关键点。"""
    kp = np.zeros((26, 3), dtype=np.float64)
    for h, b in HALPE26_TO_BODY25:
        uv = project(X[b], intrinsics, extrinsics, cid)
        kp[h, :2] = uv
        kp[h, 2] = conf
    if offset is not None:
        kp[:, :2] += np.asarray(offset, dtype=np.float64)
    return kp


def _rig():
    return make_rig([(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 2.0, 0.0)])


def test_consensus_filter_keeps_good_drops_outlier():
    intr, extr = _rig()
    X = _body25_points()
    kp3d = np.concatenate([X, np.ones((25, 1))], axis=1)      # conf=1
    obs = {
        0: Pose2D(camera_id=0, keypoints=_halpe26_from_body25(X, intr, extr, 0)),
        1: Pose2D(camera_id=1, keypoints=_halpe26_from_body25(
            X, intr, extr, 1, offset=(40.0, -25.0))),          # 离群视角
        2: Pose2D(camera_id=2, keypoints=_halpe26_from_body25(X, intr, extr, 2)),
    }
    cfg = ConsensusConfig(sigma_px=10.0, max_px=30.0)
    out, st = filter_obs_by_consensus([obs], np.asarray([kp3d]), intr, extr, cfg)

    good = out[0][0].keypoints[:, 2]
    bad = out[0][1].keypoints[:, 2]
    assert np.allclose(good[good > 0], 0.9), "一致视角的权重应≈1（conf 不变）"
    assert np.all(bad == 0.0), "偏差 47px > max_px=30 → 该视角该关节权重 0"
    assert st["n_zero"] > 0
    assert "共识降权" in format_stats(st)


def test_consensus_filter_partial_weight():
    """偏差介于 σ 与 max 之间 → 权重在 (0,1)。"""
    intr, extr = _rig()
    X = _body25_points()
    kp3d = np.concatenate([X, np.ones((25, 1))], axis=1)
    obs = {
        0: Pose2D(camera_id=0, keypoints=_halpe26_from_body25(X, intr, extr, 0)),
        1: Pose2D(camera_id=1, keypoints=_halpe26_from_body25(
            X, intr, extr, 1, offset=(12.0, 0.0))),
        2: Pose2D(camera_id=2, keypoints=_halpe26_from_body25(X, intr, extr, 2)),
    }
    cfg = ConsensusConfig(sigma_px=10.0, max_px=30.0)
    out, _ = filter_obs_by_consensus([obs], np.asarray([kp3d]), intr, extr, cfg)
    w = out[0][1].keypoints[:, 2]
    w = w[w > 0] / 0.9
    assert np.all(w > 0.0) and np.all(w < 1.0)
    # Cauchy: r=12, σ=10 → w = 1/(1+1.44) = 0.410
    assert np.allclose(w, 1.0 / (1.0 + (12.0 / 10.0) ** 2), atol=1e-6)


def test_consensus_filter_disabled_is_identity():
    intr, extr = _rig()
    X = _body25_points()
    kp3d = np.concatenate([X, np.ones((25, 1))], axis=1)
    obs = {0: Pose2D(camera_id=0,
                     keypoints=_halpe26_from_body25(X, intr, extr, 0))}
    cfg = ConsensusConfig(sigma_px=0.0)
    out, st = filter_obs_by_consensus([obs], np.asarray([kp3d]), intr, extr, cfg)
    assert out[0][0] is obs[0], "关闭时原样返回（不复制）"
    assert st["n_obs"] == 0


def test_consensus_filter_no_3d_reference_is_identity():
    intr, extr = _rig()
    X = _body25_points()
    obs = {0: Pose2D(camera_id=0,
                     keypoints=_halpe26_from_body25(X, intr, extr, 0))}
    cfg = ConsensusConfig(sigma_px=10.0)
    out, st = filter_obs_by_consensus([obs], None, intr, extr, cfg)
    assert out[0][0] is obs[0]
    assert st["n_obs"] == 0


def test_consensus_filter_does_not_mutate_input():
    intr, extr = _rig()
    X = _body25_points()
    kp3d = np.concatenate([X, np.ones((25, 1))], axis=1)
    kp = _halpe26_from_body25(X, intr, extr, 1, offset=(40.0, 0.0))
    obs = {1: Pose2D(camera_id=1, keypoints=kp)}
    before = kp.copy()
    filter_obs_by_consensus([obs], np.asarray([kp3d]), intr, extr,
                            ConsensusConfig())
    assert np.array_equal(obs[1].keypoints, before)


def test_obs_cache_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    cids = [0, 1, 3]
    frames = []
    for _ in range(3):
        f = {}
        for cid in cids:
            kp = rng.random((26, 3)) * 100.0
            kp[:, 2] = rng.random(26)
            f[cid] = Pose2D(camera_id=cid, keypoints=kp, score=0.7,
                            bbox=np.array([10.0, 20.0, 110.0, 220.0]),
                            skeleton="halpe26")
        frames.append(f)
    frames[1] = {cids[0]: frames[1][cids[0]]}      # 中间帧只有一台相机
    p = str(tmp_path / "p1.npz")
    save_pass1(p, {0: frames, 1: frames}, {}, [5, 6, 7], cids)
    d = load_pass1(p)
    assert d["cids"] == cids
    assert d["indices"] == [5, 6, 7]
    assert d["n_people"] == 2
    back = d["frames_obs_by_pid"][0]
    assert len(back) == 3
    assert sorted(back[0].keys()) == cids
    assert sorted(back[1].keys()) == [cids[0]], "缺帧的相机不应被还原成观测"
    for cid in cids:
        assert np.allclose(back[0][cid].keypoints, frames[0][cid].keypoints,
                           atol=1e-5)
        assert np.allclose(back[0][cid].bbox, frames[0][cid].bbox)
    assert d["kp3ds_by_pid"][0].shape == (3, 25, 4)
    assert not d["kp3ds_by_pid"][0].any(), "无阶段 D 结果时 3D 覆盖应为全 0"


def test_obs_cache_rejects_garbage_bbox(tmp_path):
    """bbox 含 NaN 时应存 0，读回为 None（不参与裁边判定）。"""
    kp = np.zeros((26, 3))
    kp[:, 2] = 0.5
    frames = [{0: Pose2D(camera_id=0, keypoints=kp,
                         bbox=np.array([np.nan, 0.0, 1.0, 2.0]))}]
    p = str(tmp_path / "p1.npz")
    save_pass1(p, {0: frames}, {}, [0], [0])
    back = load_pass1(p)["frames_obs_by_pid"][0]
    assert back[0][0].bbox is None
