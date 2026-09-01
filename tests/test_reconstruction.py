"""reconstruction 模块单元测试：三角化精度 + 跨视角匹配正确性（合成数据，无需相机）。"""
import numpy as np

from tabletennis.core.types import (
    CameraExtrinsics,
    CameraIntrinsics,
    Pose2D,
)
from tabletennis.reconstruction import (
    AssociationConfig,
    MultiViewTriangulator,
    match_people,
    match_people_fixed,
    undistort_keypoints,
)


def make_rig(cam_centers, look_at=(0.0, 0.0, 2.0), K=None):
    """构造一组「看着同一点」的相机内外参（无畸变）。

    Args:
        cam_centers: 各相机光心世界坐标列表 [(x,y,z), ...]。
        look_at: 各相机都看向的世界点。
        K: 3x3 内参，默认 1000 焦距、主点在 640x480 中心。

    Returns:
        ``(intrinsics, extrinsics)``，均为 ``{cam_id: ...}``。
    """
    if K is None:
        K = np.array([[1000.0, 0.0, 320.0],
                      [0.0, 1000.0, 240.0],
                      [0.0, 0.0, 1.0]])
    intrinsics, extrinsics = {}, {}
    for cid, C in enumerate(cam_centers):
        C = np.asarray(C, dtype=np.float64)
        # 相机 Z 轴指向 look_at，Y 轴尽量朝上，X = Y x Z（右手系）
        z = look_at - C
        z = z / np.linalg.norm(z)
        y = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        x = np.cross(y, z)
        if np.linalg.norm(x) < 1e-6:
            y = np.array([0.0, 1.0, 0.0])
            x = np.cross(y, z)
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)
        R = np.column_stack([x, y, z])  # 世界 -> 相机
        t = -R @ C
        intrinsics[cid] = CameraIntrinsics(
            width=640, height=480, K=K.copy(), dist=np.zeros(5)
        )
        extrinsics[cid] = CameraExtrinsics(R=R, t=t.reshape(3, 1))
    return intrinsics, extrinsics


def project(X, intrinsics, extrinsics, cid):
    """把世界点 X 投到相机 cid 的无畸变像素坐标。"""
    e = extrinsics[cid]
    p = e.project(intrinsics[cid].K)
    x = p @ np.append(np.asarray(X, dtype=np.float64), 1.0)
    return x[:2] / x[2]


def make_parallel_rig(cam_centers):
    """构造一组「平行光轴」（R=I，都看向 +Z）的相机，光心在 cam_centers。

    平行相机的极线是水平线：两点只要 y 不同，2 视角就能严格区分，避免
    汇聚相机带来的极线退化（人眼/射线共面导致的歧义），是匹配测试的干净基准。
    """
    K = np.array([[1000.0, 0.0, 320.0], [0.0, 1000.0, 240.0], [0.0, 0.0, 1.0]])
    intrinsics, extrinsics = {}, {}
    for cid, C in enumerate(cam_centers):
        C = np.asarray(C, dtype=np.float64)
        intrinsics[cid] = CameraIntrinsics(
            width=640, height=480, K=K.copy(), dist=np.zeros(5)
        )
        extrinsics[cid] = CameraExtrinsics(R=np.eye(3), t=(-C).reshape(3, 1))
    return intrinsics, extrinsics


def test_triangulate_point_accuracy():
    """两个正交视角三角化一个已知点，误差应很小（亚毫米级）。"""
    intrinsics, extrinsics = make_rig(
        cam_centers=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], look_at=(0.5, 0.5, 2.0)
    )
    tri = MultiViewTriangulator(intrinsics, extrinsics)

    X_gt = np.array([0.5, 0.4, 2.0])
    pts = {cid: tuple(project(X_gt, intrinsics, extrinsics, cid)) for cid in (0, 1)}
    confs = {0: 0.9, 1: 0.9}

    res = tri.triangulate_point(pts, confs)
    assert res is not None
    X, conf, err, nv, angle = res
    assert nv == 2
    assert np.linalg.norm(X - X_gt) < 1e-3  # 无噪声时几乎精确
    assert err < 1e-3
    assert angle > 20.0  # 1m 基线在 2m 深度处交会角约 28°


def test_triangulate_point_rejects_degenerate_angle():
    """两相机近共线（交会角≈0）时应返回 None。"""
    intrinsics, extrinsics = make_rig(
        cam_centers=[[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]], look_at=(0.5, 0.5, 2.0)
    )
    tri = MultiViewTriangulator(intrinsics, extrinsics)
    X_gt = np.array([0.5, 0.4, 2.0])
    pts = {cid: tuple(project(X_gt, intrinsics, extrinsics, cid)) for cid in (0, 1)}
    res = tri.triangulate_point(pts, {0: 0.9, 1: 0.9})
    assert res is None


def test_triangulate_pose_with_noise_and_nan():
    """两个视角 + 加噪，三角化整条姿态，未观测关节为 NaN。"""
    intrinsics, extrinsics = make_rig(
        cam_centers=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], look_at=(0.5, 0.5, 2.0)
    )
    tri = MultiViewTriangulator(intrinsics, extrinsics)
    rng = np.random.default_rng(0)

    # 3 个关节的真值
    gt = np.array([[0.5, 0.4, 2.0], [0.5, 0.4, 1.7], [0.5, 0.4, 0.5]])
    obs = {}
    for cid in (0, 1):
        kpts = np.zeros((3, 3), dtype=np.float32)
        for j in range(3):
            u, v = project(gt[j], intrinsics, extrinsics, cid)
            kpts[j, :2] = [u + rng.normal(0, 1.0), v + rng.normal(0, 1.0)]
            kpts[j, 2] = 0.9
        # 关节 2 在相机 1 里置信度极低（模拟遮挡）→ 只剩 1 视角 → NaN
        if cid == 1:
            kpts[2, 2] = 0.05
        obs[cid] = Pose2D(camera_id=cid, keypoints=kpts, skeleton="coco17")

    skel = tri.triangulate_pose(obs, min_conf=0.3)
    assert skel.keypoints.shape == (3, 3)
    assert np.isfinite(skel.keypoints[0]).all()
    assert np.isfinite(skel.keypoints[1]).all()
    assert np.isnan(skel.keypoints[2]).all()  # 关节 2 只剩 1 视角
    # 关节 0 误差应 < 2cm（1px 噪声）
    assert np.linalg.norm(skel.keypoints[0] - gt[0]) < 0.02


def make_pose3(cid, X, intrinsics, extrinsics):
    """构造一个 3 关节的 Pose2D：关节 0 精确落在世界点 X，关节 1/2 在 X 上方。"""
    kpts = np.zeros((3, 3), dtype=np.float32)
    for j, dz in enumerate((0.0, 0.4, 0.6)):
        u, v = project(np.asarray(X, dtype=np.float64) + [0, 0, dz],
                       intrinsics, extrinsics, cid)
        kpts[j, :2] = [u, v]
        kpts[j, 2] = 0.9
    return Pose2D(camera_id=cid, keypoints=kpts, skeleton="coco17")


def test_match_people_two_people_two_views():
    """两人都被同一对相机看到，匹配应正确分出 2 人且不错配。"""
    intrinsics, extrinsics = make_parallel_rig(
        cam_centers=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    )
    tri = MultiViewTriangulator(intrinsics, extrinsics)

    # 两人 x/y 都不同，平行相机极线为水平线，y 不同即可严格区分
    gt = [np.array([0.2, 0.1, 2.0]), np.array([-0.2, 0.4, 2.0])]

    poses_per_cam = {
        0: [make_pose3(0, gt[0], intrinsics, extrinsics),
            make_pose3(0, gt[1], intrinsics, extrinsics)],
        1: [make_pose3(1, gt[0], intrinsics, extrinsics),
            make_pose3(1, gt[1], intrinsics, extrinsics)],
    }

    people = match_people(poses_per_cam, tri)
    assert len(people) == 2
    for person in people:
        assert set(person.keys()) == {0, 1}
        pts = {cid: (float(p.keypoints[0][0]), float(p.keypoints[0][1]))
               for cid, p in person.items()}
        cs = {cid: float(p.keypoints[0][2]) for cid, p in person.items()}
        res = tri.triangulate_point(pts, cs)
        assert res is not None
        d = min(np.linalg.norm(res[0] - g) for g in gt)
        assert d < 0.1


def test_match_people_no_cross_merge():
    """一人被 cam0/1 看到、另一人被 cam2/3 看到，不应跨相机合并。"""
    # 4 台平行相机沿 X 一字排开，极线水平
    intrinsics, extrinsics = make_parallel_rig(
        cam_centers=[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0],
                     [1.0, 0.0, 0.0], [1.5, 0.0, 0.0]]
    )
    tri = MultiViewTriangulator(intrinsics, extrinsics)
    # 两人 y 不同（0.1 / 0.4），跨视角（如 cam0 与 cam2）极线不一致 → 不该匹配
    gt0 = np.array([0.3, 0.1, 2.0])
    gt1 = np.array([0.3, 0.4, 2.0])

    poses_per_cam = {
        0: [make_pose3(0, gt0, intrinsics, extrinsics)],
        1: [make_pose3(1, gt0, intrinsics, extrinsics)],
        2: [make_pose3(2, gt1, intrinsics, extrinsics)],
        3: [make_pose3(3, gt1, intrinsics, extrinsics)],
    }
    people = match_people(poses_per_cam, tri)
    assert len(people) == 2
    cam_sets = sorted([tuple(sorted(p.keys())) for p in people])
    assert cam_sets == [(0, 1), (2, 3)]


def test_match_people_fixed_groups():
    """固定分组匹配：每组相机各自看到一个人，直接按组返回，不跨组错配。"""
    intrinsics, extrinsics = make_rig(
        cam_centers=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        look_at=(0.5, 0.5, 2.0),
    )
    tri = MultiViewTriangulator(intrinsics, extrinsics)

    # A 只在 cam0/cam2 可见，B 只在 cam1/cam3 可见
    A = np.array([0.3, 0.3, 2.0])
    B = np.array([0.7, 0.7, 2.0])
    poses_per_cam = {
        0: [make_pose3(0, A, intrinsics, extrinsics)],
        1: [make_pose3(1, B, intrinsics, extrinsics)],
        2: [make_pose3(2, A, intrinsics, extrinsics)],
        3: [make_pose3(3, B, intrinsics, extrinsics)],
    }
    people = match_people_fixed(poses_per_cam, tri, groups=[[0, 2], [1, 3]])
    assert len(people) == 2
    assert set(people[0].keys()) == {0, 2}
    assert set(people[1].keys()) == {1, 3}


def test_match_people_fixed_glimpse_picks_consistent():
    """固定分组下 cam0 瞥到对面的人（看到 2 人）时，应挑与组内 cam2 一致的那一个。"""
    intrinsics, extrinsics = make_rig(
        cam_centers=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                     [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        look_at=(0.5, 0.5, 2.0),
    )
    tri = MultiViewTriangulator(intrinsics, extrinsics)

    A = np.array([0.3, 0.3, 2.0])
    B = np.array([0.7, 0.7, 2.0])
    poses_per_cam = {
        0: [make_pose3(0, A, intrinsics, extrinsics),
            make_pose3(0, B, intrinsics, extrinsics)],  # cam0 串扰看到两人
        1: [make_pose3(1, B, intrinsics, extrinsics)],
        2: [make_pose3(2, A, intrinsics, extrinsics)],
        3: [make_pose3(3, B, intrinsics, extrinsics)],
    }
    people = match_people_fixed(poses_per_cam, tri, groups=[[0, 2], [1, 3]])
    assert len(people) == 2
    # cam0/cam2 组：三角化后 X/Y 应更接近 A（挑对了 A，而不是瞥到的 B）
    skel = tri.triangulate_pose(people[0])
    valid = np.isfinite(skel.keypoints).all(axis=1)
    centroid = skel.keypoints[valid].mean(axis=0)
    assert np.linalg.norm(centroid[:2] - A[:2]) < 0.3


def _halpe26_skeleton(x=0.5, y=0.4, h=1.7) -> np.ndarray:
    """一个站立的 halpe26 骨架（真实比例：脚跟几乎在踝正下方、脚趾在前）。"""
    k = np.zeros((26, 3))
    k[19] = [x, y, 0.53 * h]; k[18] = [x, y, 0.82 * h]; k[17] = [x, y, 0.90 * h]
    k[0] = [x, y + 0.03, 0.93 * h]
    k[1] = [x - 0.03, y + 0.05, 0.91 * h]; k[2] = [x + 0.03, y + 0.05, 0.91 * h]
    k[3] = [x - 0.06, y + 0.03, 0.91 * h]; k[4] = [x + 0.06, y + 0.03, 0.91 * h]
    k[5] = [x - 0.18, y, 0.81 * h]; k[6] = [x + 0.18, y, 0.81 * h]
    k[7] = [x - 0.22, y, 0.62 * h]; k[8] = [x + 0.22, y, 0.62 * h]
    k[9] = [x - 0.24, y, 0.44 * h]; k[10] = [x + 0.24, y, 0.44 * h]
    k[11] = [x - 0.09, y, 0.52 * h]; k[12] = [x + 0.09, y, 0.52 * h]
    k[13] = [x - 0.10, y, 0.26 * h]; k[14] = [x + 0.10, y, 0.26 * h]
    k[15] = [x - 0.10, y, 0.06 * h]; k[16] = [x + 0.10, y, 0.06 * h]
    k[20] = [x - 0.10, y + 0.15, 0.02]; k[21] = [x + 0.10, y + 0.15, 0.02]
    k[22] = [x - 0.14, y + 0.06, 0.02]; k[23] = [x + 0.14, y + 0.06, 0.02]
    k[24] = [x - 0.10, y - 0.03, 0.02]; k[25] = [x + 0.10, y - 0.03, 0.02]
    return k


def test_fill_missing_joints_head_feet():
    """单相机遮挡头/脚时，fill_missing_joints 用骨长+射线补出 3D（相机俯视）。"""
    from tabletennis.reconstruction import fill_missing_joints
    # 相机放上方俯视（与真实 2.1m 机位一致）
    intrinsics, extrinsics = make_rig(
        cam_centers=[[0.0, 0.0, 2.5], [1.0, 0.0, 2.5]],
        look_at=(0.5, 0.4, 0.8),
    )
    tri = MultiViewTriangulator(intrinsics, extrinsics)
    gt = _halpe26_skeleton(0.5, 0.4, 1.7)
    head = {0, 1, 2, 3, 4, 17}
    feet = {20, 21, 22, 23, 24, 25}

    obs = {}
    for cid in (0, 1):
        kps = np.zeros((26, 3), dtype=np.float32)
        for j in range(26):
            u, v = project(gt[j], intrinsics, extrinsics, cid)
            kps[j] = [u, v, 0.9]
        obs[cid] = Pose2D(camera_id=cid, keypoints=kps, skeleton="halpe26")
    for j in head | feet:
        obs[1].keypoints[j, 2] = 0.0  # cam1 遮挡头/脚

    skel = tri.triangulate_pose(obs)
    for j in head | feet:
        assert np.isnan(skel.keypoints[j]).all(), f"j{j} 应先为 NaN"

    filled = fill_missing_joints(skel, obs, tri)
    for j in head | feet:
        assert np.isfinite(filled.keypoints[j]).all(), f"j{j} 应被补上"

    # 核心关节（头顶/鼻/脚）补点应较准；眼耳为头侧向关节，容差放宽
    core = {17, 0} | feet
    for j in core:
        assert np.linalg.norm(filled.keypoints[j] - gt[j]) < 0.10, f"j{j} 误差过大"
    for j in {1, 2, 3, 4}:
        assert np.linalg.norm(filled.keypoints[j] - gt[j]) < 0.25, f"j{j} 误差过大"


def test_undistort_keypoints_passthrough():
    """无畸变（dist=0）时去畸变应近似恒等（仅主点附近精确）。"""
    K = np.array([[1000.0, 0.0, 320.0], [0.0, 1000.0, 240.0], [0.0, 0.0, 1.0]])
    kpts = np.array([[320.0, 240.0, 0.9], [400.0, 300.0, 0.8]], dtype=np.float32)
    out = undistort_keypoints(kpts, K, np.zeros(5))
    assert np.allclose(out[:, :2], kpts[:, :2], atol=1e-3)
    assert np.allclose(out[:, 2], kpts[:, 2])


def test_association_config_from_dict():
    cfg = AssociationConfig.from_dict({"anchor_max_reproj_px": 20.0, "min_views": 3})
    assert cfg.anchor_max_reproj_px == 20.0
    assert cfg.min_views == 3
