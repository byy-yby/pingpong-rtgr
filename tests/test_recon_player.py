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
    # 一个“站立”在地面上的人：脚底 = 地板 = 0，头顶 1.75
    verts = np.zeros((8, 3))
    verts[:, 2] = [0.0, 0.0, 0.0, 0.0, 1.0, 1.4, 1.75, 1.75]
    verts[:, 0] = np.arange(8) * 0.05          # 沿 +x 散开一点
    sun = np.array([1.0, 0.0, 0.5])
    sun = sun / np.linalg.norm(sun)

    layers = contact_shadow_planes(verts, floor_z=0.0, sun_dir=sun)
    assert len(layers) == 2
    for d in layers:
        assert d["rx"] > d["ry"] > 0
        assert abs(d["center"][2] - 0.004) < 1e-6   # 略高于地板平面防 z-fight
        assert d["center"][1] == pytest.approx(0.0, abs=1e-9)  # y 居中
        # 影子沿 +x（太阳水平方向）偏移 → 盘心在立足点 x 正侧
        assert d["center"][0] > 0.0
        # e 与太阳水平同向
        assert np.dot(d["e"], np.array([1.0, 0.0])) > 0.99

    # 太阳从另一侧 → 影子翻到 -x
    sun2 = np.array([-1.0, 0.0, 0.5])
    layers2 = contact_shadow_planes(verts, floor_z=0.0, sun_dir=sun2)
    for d, d2 in zip(layers, layers2):
        assert d2["center"][0] < 0.0
        assert abs(d["center"][1] - d2["center"][1]) < 1e-9


def test_shadow_planes_empty_verts():
    assert contact_shadow_planes(np.empty((0, 3)), -0.76) == []
    assert contact_shadow_planes(None, -0.76) == []


def test_shadow_disc_pinned_to_floor_plane():
    # 盘心永远钉在地板平面上（floor_z+0.004），不随脚底抬起：
    # SMPL 脚底悬空地板上方几 cm 是常态（实测 median ~-0.71 vs floor -0.76），
    # 若盘子贴脚平面就会悬在地板上方、低视角看是脱开的深斑。
    for foot_z in (-0.68, -0.79):            # 悬浮 / 穿透 两种都钉地板
        verts = np.zeros((6, 3))
        verts[:, 2] = foot_z
        layers = contact_shadow_planes(verts, floor_z=-0.76)
        assert len(layers) == 2
        for d in layers:
            assert abs(d["center"][2] - (-0.756)) < 1e-6   # floor_z + 0.004


# ----------------------------------------------------------------------
# 整身投影软影（project_floor_shadow + ReconTimeline.hold_gaps）
# ----------------------------------------------------------------------
def test_project_floor_shadow_elongates_with_height():
    from tabletennis.visualization.recon_player import project_floor_shadow
    # 站在 z=0 地面：脚底 (0,0)、头顶 1m 处一点
    verts = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.2, 0.1, 0.5]])
    sun = np.array([1.0, 0.0, 0.5]); sun = sun / np.linalg.norm(sun)
    proj = project_floor_shadow(verts, floor_z=0.0, sun_dir=sun, k=0.5)
    assert proj.shape == (3, 2)
    # 脚底不动；1m 高处沿 +x 伸 0.5m
    assert proj[0, 0] == pytest.approx(0.0, abs=1e-12)
    assert proj[0, 1] == pytest.approx(0.0, abs=1e-12)
    assert proj[1, 0] == pytest.approx(0.5, abs=1e-12)   # 1.0m * 0.5
    assert proj[1, 1] == pytest.approx(0.0, abs=1e-12)   # y 分量 0
    # 中间高度按比例：0.5m → 0.25m
    assert proj[2, 0] == pytest.approx(0.2 + 0.25, abs=1e-12)
    # 太阳反向 → 影朝 -x
    sun2 = np.array([-1.0, 0.0, 0.5])
    proj2 = project_floor_shadow(verts, 0.0, sun_dir=sun2, k=0.5)
    assert proj2[1, 0] == pytest.approx(-0.5, abs=1e-12)


def test_project_floor_shadow_clips_below_floor_and_empty():
    from tabletennis.visualization.recon_player import project_floor_shadow
    # 陷到地面以下的顶点不再往回缩（clip 到 0）
    verts = np.array([[1.0, 2.0, -0.79], [0.0, 0.0, -0.8]])
    sun = np.array([1.0, 0.0, 1.0])
    proj = project_floor_shadow(verts, floor_z=-0.76, sun_dir=sun, k=0.5)
    assert proj.shape == (2, 2)
    assert proj[0, 0] == pytest.approx(1.0, abs=1e-12)    # (-0.79 < floor) → 原位
    assert project_floor_shadow(np.empty((0, 3)), 0.0).shape == (0, 2)
    assert project_floor_shadow(None, 0.0).shape == (0, 2)


def test_convex_hull2d_ccw_square_and_area():
    from tabletennis.visualization.recon_player import convex_hull2d
    # 含内部点的正方形 → 只留 4 角，CCW 有序
    pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0.5, 0.5], [0.2, 0.7]])
    hull = convex_hull2d(pts)
    assert hull is not None and len(hull) == 4
    # CCW + 面积为 1（标准 shoelace，>0 即 CCW）
    a = 0.5 * (hull[:, 0] * np.roll(hull[:, 1], -1)
               - hull[:, 1] * np.roll(hull[:, 0], -1)).sum()
    assert a == pytest.approx(1.0, abs=1e-9)   # 正值 = CCW → 三角扇法线 +Z


def test_convex_hull2d_degenerate_returns_none():
    from tabletennis.visualization.recon_player import convex_hull2d
    assert convex_hull2d(np.empty((0, 2))) is None
    assert convex_hull2d(np.array([[0, 0], [1, 1]])) is None       # <3 点
    collinear = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [0.5, 0.5]])
    assert convex_hull2d(collinear) is None                        # 全共线
    # 共面但投影在 xy 非共线 → 正常返回（cast 的实际使用形态）
    ring = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float64)
    assert convex_hull2d(ring) is not None


def test_hold_gaps_keeps_through_short_gaps_only(tmp_path):
    out = tmp_path / "recon"
    out.mkdir()
    # t0..2 ok；t3 no_person；t4 ok；t5..7 连续 no_person；t8..9 ok；t10..11 stride 跳过
    _write_meta(out, n_ref=12)
    _write_index(out, refs=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                 status=[0, 0, 0, 1, 0, 1, 1, 1, 0, 0])
    for t in (0, 1, 2, 4, 8, 9):
        _write_frame(out, t)
    f4 = str(out / "frame_000004.npz")

    # hold=0：语义精确——遇 no_person 立即清空
    tl0 = ReconTimeline(str(out), hold_gaps=0)
    assert tl0.person_path_at(3) is None
    assert tl0.person_path_at(5) is None and tl0.person_path_at(7) is None

    # hold=2：单帧空窗(t3)保持；连续 3 帧空窗(t5..7)第 3 帧开始清空
    tl2 = ReconTimeline(str(out), hold_gaps=2)
    assert tl2.person_path_at(3) == str(out / "frame_000002.npz")
    assert tl2.person_path_at(5) == f4          # 第 1 个 empty 保持
    assert tl2.person_path_at(6) == f4          # 第 2 个 empty 保持
    assert tl2.person_path_at(7) is None        # 第 3 个 > hold → 清空
    assert tl2.person_path_at(8) == str(out / "frame_000008.npz")

    # hold=3：t5..7 全保持（3 ≤ hold）；stride 跳过帧照常保持最近姿态
    tl3 = ReconTimeline(str(out), hold_gaps=3)
    assert tl3.person_path_at(7) == f4
    assert tl3.person_path_at(11) == str(out / "frame_000009.npz")


# ----------------------------------------------------------------------
# 人体凸凹明暗烘焙（bake_body_shading）：纯 numpy，凸/凹按法线·光源方向明暗
# ----------------------------------------------------------------------
def test_bake_body_shading_shape_and_bounds():
    from tabletennis.visualization.recon_player import bake_body_shading
    n = np.random.RandomState(0).normal(size=(500, 3))
    c = bake_body_shading(n, skin=np.array([0.8, 0.7, 0.6]))
    assert c.shape == (500, 3)
    assert c.dtype == np.float64
    assert np.isfinite(c).all()
    assert c.min() >= 0.0 and c.max() <= 1.0


def test_bake_body_shading_facing_light_brighter_than_away():
    from tabletennis.visualization.recon_player import bake_body_shading
    lf = np.array([0.0, 0.0, 1.0])          # 光从正上方来
    skin = np.array([1.0, 1.0, 1.0])
    amb, key = 0.4, 0.8
    up = bake_body_shading(np.array([[0, 0, 1.0]]), skin=skin, ambient=amb,
                           key=key, light_from=lf)[0]
    down = bake_body_shading(np.array([[0, 0, -1.0]]), skin=skin, ambient=amb,
                             key=key, light_from=lf)[0]
    # 顶面吃满主光（0.4+0.8=1.2 → clip 1.0）；底面只吃环境光（=0.4）
    assert up[0] == 1.0
    assert down[0] == 0.4
    assert (up > down).all()


def test_bake_body_shading_tilt_faces_shade_continuously():
    from tabletennis.visualization.recon_player import bake_body_shading
    lf = np.array([0.0, 0.0, 1.0])
    # 法线从上仰 0°→90°，亮度应单调下降（Lambert，凸凹可读的来源）
    lums = []
    for deg in (0, 30, 60, 90):
        th = np.deg2rad(deg)
        c = bake_body_shading(np.array([[0, np.sin(th), np.cos(th)]]),
                              skin=np.ones(3), ambient=0.0, key=1.0, light_from=lf)[0]
        lums.append(float(c.mean()))
    assert lums == sorted(lums, reverse=True)
    assert lums[0] > lums[-1]


def test_bake_body_shading_nan_normal_degrades_to_ambient():
    from tabletennis.visualization.recon_player import bake_body_shading
    nrm = np.array([[0, 0, 1.0], [np.nan, 0, 0], [0, np.nan, np.nan]], np.float64)
    c = bake_body_shading(nrm, skin=np.ones(3), ambient=0.5, key=0.8,
                          light_from=np.array([0.0, 0.0, 1.0]))
    assert np.isfinite(c).all()
    np.testing.assert_allclose(c[1], c[2])          # NaN 行退化为环境光底 0.5
    assert c[1, 0] == 0.5 and c[0, 0] == 1.0        # 正常顶面仍吃满主光


def test_bake_body_shading_skin_tints_and_xyz_convex():
    from tabletennis.visualization.recon_player import bake_body_shading
    lf = np.array([-0.45, -0.25, 0.86]); lf = lf / np.linalg.norm(lf)
    skin = np.array([0.82, 0.71, 0.60])
    c = bake_body_shading(np.array([[0.0, 0.0, 1.0]]), skin=skin,
                          ambient=0.42, key=0.80, light_from=lf)[0]
    # 朝光面的三通道都显著大于环境光底
    assert (c > skin * 0.42).all()
    assert np.argmax(c) == 0                       # R 通道最多（肤色暖调保持）
    assert c[1] > c[2]
