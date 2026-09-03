"""球检测 / 三角化 / 轨迹滤波 单元测试（合成数据，无需相机）。"""
import cv2
import numpy as np

from tabletennis.core.types import (
    Ball2D,
    CameraExtrinsics,
    CameraIntrinsics,
    Frame,
)
from tabletennis.reconstruction import (
    BallTracker,
    MultiViewTriangulator,
    triangulate_ball,
)
from tabletennis.vision.ball import ClassicalBallDetector, refine_ball_center


# ----------------------------------------------------------------------
# 合成相机 rig（与 test_reconstruction.py 同款模式）
# ----------------------------------------------------------------------
def make_rig(cam_centers, look_at, K=None):
    """构造一组「看着同一点」的相机内外参（无畸变，1440x1080，fx=1765）。"""
    if K is None:
        K = np.array([[1765.0, 0.0, 720.0],
                      [0.0, 1765.0, 540.0],
                      [0.0, 0.0, 1.0]])
    intrinsics, extrinsics = {}, {}
    for cid, C in enumerate(cam_centers):
        C = np.asarray(C, dtype=np.float64)
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
            width=1440, height=1080, K=K.copy(), dist=np.zeros(5)
        )
        extrinsics[cid] = CameraExtrinsics(R=R, t=t.reshape(3, 1))
    return intrinsics, extrinsics


def project(X, intrinsics, extrinsics, cid):
    e = extrinsics[cid]
    p = e.project(intrinsics[cid].K)
    x = p @ np.append(np.asarray(X, dtype=np.float64), 1.0)
    return x[:2] / x[2]


# ----------------------------------------------------------------------
# refine
# ----------------------------------------------------------------------
def test_refine_ball_center_subpixel():
    """合成高斯 blob，亚像素球心应恢复到 <0.2px。"""
    rng = np.random.default_rng(0)
    H = W = 120
    bg, cx, cy, sigma, amp = 30.0, 61.3, 58.7, 3.0, 180.0
    yy, xx = np.mgrid[0:H, 0:W]
    img = bg + amp * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    img = np.clip(img + rng.normal(0.0, 1.0, size=(H, W)), 0, 255).astype(np.uint8)

    x, y, r, conf = refine_ball_center(img, 61.0, 59.0, radius_hint=8.0)
    assert conf > 0.5
    assert abs(x - cx) < 0.2
    assert abs(y - cy) < 0.2
    assert 2.0 <= r <= 12.0


def test_refine_ball_center_no_ball():
    """纯背景（无对比）应返回 conf=0、原中心。"""
    img = np.full((100, 100), 40, dtype=np.uint8)
    x, y, r, conf = refine_ball_center(img, 50.0, 50.0, radius_hint=8.0)
    assert conf == 0.0
    assert (x, y) == (50.0, 50.0)


# ----------------------------------------------------------------------
# triangulate_ball
# ----------------------------------------------------------------------
def test_triangulate_ball_accuracy():
    """4 视角 + 0.3px 噪声，3D 球心误差应 <2mm。"""
    look_at = np.array([0.76, 1.37, 0.3])
    centers = [[-2.0, -2.0, 3.0], [4.0, -2.0, 3.0],
               [-2.0, 5.0, 3.0], [4.0, 5.0, 3.0]]
    intrinsics, extrinsics = make_rig(centers, look_at)
    tri = MultiViewTriangulator(intrinsics, extrinsics)

    rng = np.random.default_rng(0)
    X_gt = np.array([0.76, 1.37, 0.3])
    balls = {}
    for cid in tri.cameras:
        u, v = project(X_gt, intrinsics, extrinsics, cid)
        u += rng.normal(0.0, 0.3)
        v += rng.normal(0.0, 0.3)
        balls[cid] = Ball2D(
            camera_id=cid, center=np.array([u, v], dtype=np.float32),
            radius=8.0, confidence=0.9,
        )

    res = triangulate_ball(balls, tri)
    assert res is not None
    X, conf, err, nv, angle = res
    assert nv == 4
    assert np.linalg.norm(X - X_gt) < 0.002


def test_triangulate_ball_insufficient_views():
    """少于 2 视角应返回 None。"""
    look_at = np.array([0.76, 1.37, 0.3])
    intrinsics, extrinsics = make_rig([[-2.0, -2.0, 3.0], [4.0, -2.0, 3.0]], look_at)
    tri = MultiViewTriangulator(intrinsics, extrinsics)
    one = {0: Ball2D(camera_id=0, center=np.array([720.0, 540.0], np.float32),
                     radius=8.0, confidence=0.9)}
    assert triangulate_ball(one, tri) is None


# ----------------------------------------------------------------------
# tracker
# ----------------------------------------------------------------------
def test_tracker_smooths_noise():
    """卡尔曼对加噪直线轨迹应降低误差（跳过前 10 帧收敛瞬态）。"""
    tr = BallTracker(dt=0.01, process_noise=10.0, meas_noise_m=0.01)
    gt = np.array([[0.2 * t, 0.1 * t, 1.5] for t in range(60)], dtype=np.float64)
    rng = np.random.default_rng(1)
    noisy = gt + rng.normal(0.0, 0.005, size=gt.shape)

    raw, filt = [], []
    for i in range(60):
        out = tr.update(noisy[i], conf=0.9)
        raw.append(float(np.linalg.norm(noisy[i] - gt[i])))
        filt.append(float(np.linalg.norm(out - gt[i])))

    assert np.mean(filt[10:]) < np.mean(raw[10:])


def test_tracker_uninitialized_returns_none():
    tr = BallTracker()
    assert tr.update(None, conf=0.0) is None


def _tracker_with_track(max_coast):
    """初始化一条已跟踪的轨迹（沿 X 匀速 +Z 恒定），返回 tracker。"""
    tr = BallTracker(dt=0.01, process_noise=10.0, meas_noise_m=0.002,
                     gate_m=0.5, max_coast=max_coast)
    for i in range(5):
        tr.update(np.array([0.2 * i, 0.0, 1.5]), conf=0.9)
    assert tr.initialized
    return tr


def test_tracker_coasts_through_short_occlusion():
    """短遮挡（缺测 ≤ max_coast）应外推预测、不失联，仍返回位置。"""
    tr = _tracker_with_track(max_coast=10)
    for _ in range(5):
        out = tr.update(None, conf=0.0)
        assert out is not None          # 补帧：仍给位置
    assert tr.initialized
    assert tr.coast == 5


def test_tracker_loses_track_after_long_occlusion():
    """长遮挡（缺测 > max_coast）应失联：返回 None 且 reset。"""
    tr = _tracker_with_track(max_coast=5)
    results = [tr.update(None, conf=0.0) for _ in range(6)]
    assert all(r is not None for r in results[:5])   # 前 5 帧还在 coast
    assert results[5] is None                        # 第 6 帧失联
    assert not tr.initialized
    assert tr.coast == 0


def test_tracker_reinitializes_after_loss():
    """失联后再出现测量应从新位置重新初始化（不跨断点连轨迹）。"""
    tr = _tracker_with_track(max_coast=3)
    for _ in range(4):
        tr.update(None, conf=0.0)
    assert not tr.initialized
    new = np.array([2.0, 2.0, 0.5])
    out = tr.update(new, conf=0.9)
    assert tr.initialized
    assert np.allclose(out, new)


def test_tracker_outlier_counts_as_coast():
    """门限外点只预测不更新，且计入 coast（连续外点最终失联）。"""
    tr = _tracker_with_track(max_coast=3)
    far = np.array([10.0, 10.0, 10.0])
    out = tr.update(far, conf=0.9)
    assert out is not None                    # 预测返回
    assert np.linalg.norm(out - far) > 1.0    # 位置没被外点拉走（仍沿原轨迹预测）
    assert tr.coast == 1                      # 外点计 coast
    for _ in range(3):
        tr.update(far, conf=0.9)
    assert not tr.initialized                 # 连续外点 → 失联


def test_tracker_max_coast_none_never_loses():
    """max_coast=None（旧行为）：初始化后永不因缺测失联。"""
    tr = _tracker_with_track(max_coast=None)
    for _ in range(50):
        out = tr.update(None, conf=0.0)
        assert out is not None
    assert tr.initialized


def test_tracker_tracks_fast_ball_with_correct_dt():
    """回归老 bug：用真实 dt 时快速球应被跟随，不被门限误判冻结。"""
    tr = BallTracker(dt=0.05, process_noise=1000.0, meas_noise_m=0.002,
                     gate_m=0.5, max_coast=5)
    tr.update(np.array([0.0, 0.0, 1.5]), conf=0.9)
    speed = 8.0                 # 8 m/s；dt=0.05 → 每帧 0.4m < gate 0.5m
    last = None
    for i in range(1, 20):
        x = np.array([speed * 0.05 * i, 0.0, 1.5])
        last = tr.update(x, conf=0.9)
    assert tr.initialized
    assert tr.coast == 0                        # 从未被判离群
    assert abs(last[0] - speed * 0.05 * 19) < 0.2   # 位置跟上了球


def test_tracker_gravity_coast_follows_ballistic_arc():
    """重力模型下，coast 按抛物线外推而非直线（回归「球击高后一直往上飞」）。"""
    g = 9.81
    dt = 0.02
    tr = BallTracker(dt=dt, process_noise=50.0, meas_noise_m=0.001,
                     gravity=(0.0, 0.0, -g), max_coast=50)

    def true_pos(i):
        t = i * dt
        return np.array([3.0 * t, 0.0, 1.0 + 5.0 * t - 0.5 * g * t * t])

    for i in range(15):                 # 前 15 帧喂测量，建立位置+速度
        tr.update(true_pos(i), conf=0.9)
    errs = []
    for i in range(15, 35):             # 20 帧 coast（无测量）
        out = tr.update(None, conf=0.0)
        errs.append(float(np.linalg.norm(out - true_pos(i))))
    assert tr.initialized               # coast 20 帧 ≤ max_coast 50，未失联
    assert np.mean(errs) < 0.2          # 抛物线贴合真实（直线外推到 0.4s 会差 ~0.8m）


def test_tracker_gravity_decelerates_on_coast():
    """coast 时重力应使竖直速度持续减小（不会笔直往上飞）。"""
    g = 9.81
    tr = BallTracker(dt=0.02, process_noise=1.0, meas_noise_m=0.001,
                     gravity=(0.0, 0.0, -g), max_coast=100)
    for i in range(5):
        tr.update(np.array([0.0, 0.0, 1.0 + 5.0 * 0.02 * i]), conf=0.9)
    vz0 = float(tr.velocity[2])
    assert vz0 > 0.5                    # 已建立向上的速度
    for _ in range(5):
        tr.update(None, conf=0.0)       # coast
    assert float(tr.velocity[2]) < vz0 - 0.5   # 重力减速（≈ g·dt·5 ≈ 0.98）


def test_tracker_mahalanobis_rejects_clear_outlier():
    """马氏门限：明显的离群点（远超不确定度）仍应被拒，且计 coast。"""
    tr = BallTracker(dt=0.01, meas_noise_m=0.002, gate_sigma=3.0, max_coast=5)
    tr.update(np.array([0.0, 0.0, 1.5]), conf=0.9)
    out = tr.update(np.array([100.0, 0.0, 1.5]), conf=0.9)
    assert out is not None              # 预测返回（不更新）
    assert tr.coast == 1                # 判为外点


def test_tracker_floor_bounce_prevents_piercing():
    """球下落 coast 穿过桌面时应反弹，输出 z 永不 < floor_z（防穿模）。"""
    g = 9.81
    tr = BallTracker(dt=0.02, process_noise=1.0, meas_noise_m=0.001,
                     gravity=(0.0, 0.0, -g), floor_z=0.02, restitution=0.9,
                     max_coast=200)
    z, vz = 0.5, 0.0
    for _ in range(100):                # 喂自由落体测量，建立向下速度，直到接近桌面
        vz -= g * 0.02
        z += vz * 0.02
        if z < 0.02:
            break
        tr.update(np.array([0.0, 0.0, z]), conf=0.9)
    assert tr.velocity[2] < 0.0         # 下落速度已建立（负）
    min_z = float("inf")
    for _ in range(40):                 # coast：预测向下穿 → 应在桌面反弹
        out = tr.update(None, conf=0.0)
        assert out is not None
        min_z = min(min_z, float(out[2]))
    assert min_z >= 0.02 - 1e-6         # 永不击穿桌面
    assert tr.velocity[2] > 0.0         # 已反弹，速度朝上


def test_tracker_floor_bounce_follows_bounce_measurements():
    """真实弹跳测量序列（下落→反弹）应被滤波跟住，且不穿桌。"""
    g = 9.81
    dt = 0.01
    tr = BallTracker(dt=dt, process_noise=500.0, meas_noise_m=0.001,
                     gravity=(0.0, 0.0, -g), floor_z=0.02, restitution=0.9,
                     gate_sigma=3.0, max_coast=10)
    z, vz = 1.0, 0.0
    for _ in range(100):                # 解析弹道：自由落体→桌面反弹（e=0.9）
        vz -= g * dt
        z += vz * dt
        if z < 0.02:
            z = 0.02 + (0.02 - z)       # 反射位置
            vz = abs(vz) * 0.9
        out = tr.update(np.array([0.0, 0.0, z]), conf=0.9)
        assert out[2] >= 0.02 - 1e-6    # 全程不穿桌（多次反弹仍被跟住）
    assert tr.initialized               # 从未失联


# ----------------------------------------------------------------------
# classical detector
# ----------------------------------------------------------------------
def _frame(img, i, cid=0):
    return Frame(
        camera_id=cid, serial="s", frame_num=i, device_timestamp=0,
        host_timestamp=0, image=img, pixel_format=17301505, width=img.shape[1], height=img.shape[0],
    )


def test_classical_detector_finds_moving_ball():
    """静态背景上移动的小亮球应被检出，且球心接近真值。"""
    det = ClassicalBallDetector(radius_px=(5.0, 15.0))
    bg = np.full((200, 200), 40, dtype=np.uint8)
    found = None
    for i in range(5):
        img = bg.copy()
        cx, cy = 100, 60 + i * 10
        cv2.circle(img, (cx, cy), 6, 200, -1)
        out = det.detect(_frame(img, i))
        if out:
            found = out[0]
    assert found is not None
    assert abs(found.center[0] - 100.0) < 4.0
    assert abs(found.center[1] - 100.0) < 4.0
    assert found.confidence > 0.5


def test_classical_detector_multicamera_isolated():
    """单实例跨两相机复用：静止相机不应「看到」另一相机移动球的残影。"""
    det = ClassicalBallDetector(radius_px=(5.0, 15.0))
    bg = np.full((200, 200), 40, dtype=np.uint8)
    found_cam0 = None
    for i in range(6):
        # 相机 0：有移动的亮球
        img0 = bg.copy()
        cv2.circle(img0, (100, 60 + i * 10), 6, 200, -1)
        out0 = det.detect(_frame(img0, i, cid=0))
        # 相机 1：完全静止、无球（背景与相机 0 相同，但须按相机隔离背景）
        img1 = bg.copy()
        out1 = det.detect(_frame(img1, i, cid=1))
        assert out1 == [], "静止相机不应检出球（背景必须按 camera_id 隔离）"
        if out0:
            found_cam0 = out0[0]
    assert found_cam0 is not None
