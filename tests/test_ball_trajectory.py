"""ReconTimeline 球轨迹加载 / 查询 / 分段轨迹的纯 numpy 单测（不碰 open3d）。

``scripts/reconstruct_video.py`` 写 ``ball_trajectory.npz``（``ref_frame`` + ``X``，
失败帧 ``X=NaN``）；``recon_player.ReconTimeline`` 把它读成 ``ball_pos_at`` /
``ball_trail_upto`` 供渲染——球位置按 ``hold_gaps`` coast 防闪没，轨迹按「缺测
>5 帧」断开，不把不同回合连成一条大线。
"""
import json

import numpy as np

from tabletennis.visualization.recon_player import ReconTimeline


def _write_meta(out, n_ref=12, ref_cam="0", period_s=0.01):
    (out / "recon_meta.json").write_text(json.dumps({
        "n_ref_frames": n_ref, "ref_cam": ref_cam,
        "source": {"ref_cam": ref_cam, "cams": {ref_cam: {"period_s": period_s}}},
    }))


def _write_ball(out, refs, X):
    """refs (M,) 主时钟帧号；X (M,3)，NaN 行 = 无球。"""
    np.savez(out / "ball_trajectory.npz",
             ref_frame=np.asarray(refs, np.int64),
             X=np.asarray(X, np.float64))


def test_ball_timeline_load_and_pos_coast(tmp_path):
    out = tmp_path / "recon"
    out.mkdir()
    _write_meta(out, n_ref=10)
    nan = np.nan
    refs = [1, 2, 3, 4, 5, 6]
    X = np.array([
        [0.1, 0.1, 0.2],   # frame 1
        [0.2, 0.2, 0.3],   # frame 2
        [nan, nan, nan],   # frame 3 无球
        [nan, nan, nan],   # frame 4 无球
        [0.3, 0.3, 0.4],   # frame 5
        [0.4, 0.4, 0.5],   # frame 6
    ])
    _write_ball(out, refs, X)
    tl = ReconTimeline(str(out), hold_gaps=2)
    assert tl.ball_ok
    # 直接命中
    np.testing.assert_allclose(tl.ball_pos_at(1), [0.1, 0.1, 0.2])
    np.testing.assert_allclose(tl.ball_pos_at(5), [0.3, 0.3, 0.4])
    # 无球帧 coast 回看（hold_gaps=2 → 最多回看 2 帧）
    np.testing.assert_allclose(tl.ball_pos_at(3), [0.2, 0.2, 0.3])   # 回看 1 帧
    np.testing.assert_allclose(tl.ball_pos_at(4), [0.2, 0.2, 0.3])   # 回看 2 帧
    np.testing.assert_allclose(tl.ball_pos_at(7), [0.4, 0.4, 0.5])   # 回看 1 帧
    np.testing.assert_allclose(tl.ball_pos_at(8), [0.4, 0.4, 0.5])   # 回看 2 帧
    assert tl.ball_pos_at(9) is None                                   # 回看 2 帧都无球
    assert tl.ball_pos_at(-1) is None


def test_ball_timeline_absent(tmp_path):
    out = tmp_path / "recon"
    out.mkdir()
    _write_meta(out, n_ref=5)
    tl = ReconTimeline(str(out))
    assert not tl.ball_ok
    assert tl.ball_pos_at(0) is None
    assert tl.ball_trail_upto(0) == []


def test_ball_trail_segmentation_breaks_on_gap(tmp_path):
    out = tmp_path / "recon"
    out.mkdir()
    _write_meta(out, n_ref=100)
    # 两段连续球：帧 0..9、帧 50..59，中间大缺口
    refs = list(range(0, 10)) + list(range(50, 60))
    X = np.array([[float(i), 0.0, 0.1] for i in refs])
    _write_ball(out, refs, X)
    tl = ReconTimeline(str(out))
    segs = tl.ball_trail_upto(59)
    assert len(segs) == 2
    assert segs[0].shape == (10, 3)   # 帧 0..9
    assert segs[1].shape == (10, 3)   # 帧 50..59
    # 只看到一半：只回第一段
    segs_half = tl.ball_trail_upto(5)
    assert len(segs_half) == 1 and segs_half[0].shape == (6, 3)


def test_ball_trail_short_segments_dropped(tmp_path):
    out = tmp_path / "recon"
    out.mkdir()
    _write_meta(out, n_ref=10)
    nan = np.nan
    # 单点孤立（前后都缺）→ 段长 1 < 2 被丢弃（成不了线）
    refs = [0, 1, 2]
    X = np.array([[nan, nan, nan], [0.5, 0.5, 0.5], [nan, nan, nan]])
    _write_ball(out, refs, X)
    tl = ReconTimeline(str(out))
    assert tl.ball_pos_at(1) is not None      # 位置可查
    assert tl.ball_trail_upto(2) == []        # 但成不了线（<2 点）
