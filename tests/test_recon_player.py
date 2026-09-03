"""recon_player 时间线语义 + 程序化接触阴影的纯 numpy 单测（不碰 open3d）。

ReconTimeline 只依赖 numpy；contact_shadow_planes 只依赖 numpy。open3d
（ReconScene / render_still / play_gui）在方法内才惰性 import。
"""
import json

import numpy as np
import pytest

from tabletennis.visualization.recon_player import (
    ReconTimeline,
    contact_shadow_planes,
)


def _write_meta(out, n_ref=10, ref_cam="0", period_s=0.01):
    (out / "recon_meta.json").write_text(json.dumps({
        "n_ref_frames": n_ref,
        "ref_cam": ref_cam,
        "source": {"ref_cam": ref_cam, "cams": {ref_cam: {"period_s": period_s}}},
        "ok": 7, "gap": 0, "no_person": 1, "fail": 0, "error": 0,
    }))


def _write_index(out, refs, status):
    """refs/status 长度相同；数组对齐 recon_index.npz 真实键名。"""
    np.savez(out / "recon_index.npz",
             ref_frame=np.asarray(refs, np.int64),
             status=np.asarray(status, np.int64),
             wall_ms=np.zeros(len(refs), np.float64),
             err_mean_px=np.zeros(len(refs), np.float64),
             err_worst_px=np.zeros(len(refs), np.float64))


def _write_frame(out, t, with_joints=True):
    d = {"vertices": np.random.rand(6890, 3).astype(np.float32)}
    if with_joints:
        d["joints"] = np.random.rand(24, 3).astype(np.float32)
    np.savez(out / f"frame_{t:06d}.npz", **d)


def _make_timeline(tmp_path):
    """典型重建：10 主时钟帧；t0..3 ok、t4 no_person、t5..6 ok；t7..9 被 stride 跳过。
    文件只写 ok 帧。
    """
    out = tmp_path / "recon"
    out.mkdir()
    _write_meta(out, n_ref=10)
    _write_index(out, refs=[0, 1, 2, 3, 4, 5, 6],
                 status=[0, 0, 0, 0, 2, 0, 0])
    for t in (0, 1, 2, 3, 5, 6):
        _write_frame(out, t)
    return ReconTimeline(str(out)), out


def test_timeline_person_hold_empty_clear(tmp_path):
    tl, out = _make_timeline(tmp_path)
    assert tl.index_ready
    assert tl.n_ref == 10
    assert tl.n_ok() == 6

    # ok 帧区间内 → 上一成功帧一直显示
    assert tl.person_path_at(0) == str(out / "frame_000000.npz")
    assert tl.person_path_at(1) == str(out / "frame_000001.npz")
    # t4 = no_person → 清空
    assert tl.person_path_at(4) is None
    # 被 stride 跳过的 t7..9 → 保持最近成功帧（t6）
    last = str(out / "frame_000006.npz")
    for t in (7, 8, 9):
        assert tl.person_path_at(t) == last
    assert tl.person_path_at(5) == str(out / "frame_000005.npz")
    assert tl.person_path_at(-1) is None
    assert tl.person_path_at(999) is None


def test_timeline_missing_ok_file_treated_empty(tmp_path):
    out = tmp_path / "recon"
    out.mkdir()
    _write_meta(out, n_ref=5)
    # index 说 t1 是 ok，但 frame_000001.npz 缺失（写入半截/被删）→ 应当空窗
    _write_index(out, refs=[0, 1, 2, 3, 4], status=[0, 0, 0, 0, 0])
    for t in (0, 2, 3, 4):
        _write_frame(out, t)
    tl = ReconTimeline(str(out))
    assert tl.person_path_at(1) is None
    assert tl.person_path_at(0) == str(out / "frame_000000.npz")
    assert tl.person_path_at(2) == str(out / "frame_000002.npz")


def test_timeline_no_index_watch_reload(tmp_path):
    """重建进行中：没有 recon_index.npz，靠逐帧 npz 推断；reload 追新帧。"""
    out = tmp_path / "recon"
    out.mkdir()
    _write_meta(out, n_ref=10)
    _write_frame(out, 0)
    _write_frame(out, 1)
    tl = ReconTimeline(str(out))
    assert not tl.index_ready
    assert tl.n_ok() == 2
    # t2 还没写好 → 暂时保持 t1；越界也不报错
    assert tl.person_path_at(2) == str(out / "frame_000001.npz")
    # 新帧出现 → reload 追上
    _write_frame(out, 2)
    tl.reload()
    assert tl.n_ok() == 3
    assert tl.person_path_at(2) == str(out / "frame_000002.npz")
    # index 就绪后 reload → 精确语义生效
    _write_index(out, refs=[0, 1, 2], status=[0, 0, 0])
    tl.reload()
    assert tl.index_ready


def test_load_person_ok_and_missing_joints(tmp_path):
    out = tmp_path / "recon"
    out.mkdir()
    _write_meta(out, n_ref=2)
    _write_index(out, refs=[0, 1], status=[0, 0])
    _write_frame(out, 0, with_joints=True)
    _write_frame(out, 1, with_joints=False)
    tl = ReconTimeline(str(out))
    p0 = tl.load_person(0)
    assert p0["vertices"].shape == (6890, 3)
    assert p0["joints"].shape == (24, 3)
    p1 = tl.load_person(1)
    assert p1["joints"] is None            # 无 joints 键不崩


# ----------------------------------------------------------------------
# contact_shadow_planes（程序化接触阴影）
# ----------------------------------------------------------------------
def test_shadow_planes_under_feet_and_follow_sun():
    # 一个“站立”在 z=0 地面上的人（脚底 0，头顶 1.75）
    verts = np.zeros((8, 3))
    verts[:, 2] = [0.0, 0.0, 0.0, 0.0, 1.0, 1.4, 1.75, 1.75]
    verts[:, 0] = np.arange(8) * 0.05          # 沿 +x 散开一点
    sun = np.array([1.0, 0.0, 0.5])
    sun = sun / np.linalg.norm(sun)

    layers = contact_shadow_planes(verts, floor_z=-0.76, sun_dir=sun)
    assert len(layers) == 2
    for d in layers:
        assert d["rx"] > d["ry"] > 0
        assert abs(d["center"][2] - 0.004) < 1e-6   # 略高于脚底平面防 z-fight
        assert d["center"][1] == pytest.approx(0.0, abs=1e-9)  # y 居中
        # 影子沿 +x（太阳水平方向）偏移 → 盘心在立足点 x 正侧
        assert d["center"][0] > 0.0
        # e 与太阳水平同向
        assert np.dot(d["e"], np.array([1.0, 0.0])) > 0.99

    # 太阳从另一侧 → 影子翻到 -x
    sun2 = np.array([-1.0, 0.0, 0.5])
    layers2 = contact_shadow_planes(verts, floor_z=-0.76, sun_dir=sun2)
    for d, d2 in zip(layers, layers2):
        assert d2["center"][0] < 0.0
        assert abs(d["center"][1] - d2["center"][1]) < 1e-9


def test_shadow_planes_empty_verts():
    assert contact_shadow_planes(np.empty((0, 3)), -0.76) == []
    assert contact_shadow_planes(None, -0.76) == []


def test_shadow_clamps_below_floor():
    # 脚底陷到地面以下（SMPL 常轻微穿透）→ 阴影盘贴回地板上方而非埋在地里
    verts = np.zeros((6, 3))
    verts[:, 2] = -0.79                       # 低于 floor_z=-0.76
    layers = contact_shadow_planes(verts, floor_z=-0.76)
    assert len(layers) == 2
    for d in layers:
        assert abs(d["center"][2] - (-0.756)) < 1e-6   # max(-0.79,-0.76)+0.004
