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


# ----------------------------------------------------------------------
# classical detector
# ----------------------------------------------------------------------
def _frame(img, i):
    return Frame(
        camera_id=0, serial="s", frame_num=i, device_timestamp=0,
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
