"""阶段 A–D 多人跟踪 / 身份识别 / 鲁棒三角化的合成测试（无相机、无检测器）。

构造 4 台环绕相机 + 2 个合成人（一相机同时看到两人 = 宽视野场景），用**注入的**
``detect_fn`` / ``pose_fn`` 喂真值投影出来的框与 halpe26 姿态，检验：

1. 身份稳定：两人全程各拿一个 person_id，不互换；
2. ROI 引导重检测：某相机「看不到」某人时，用预测位置开小窗 + 低阈值重新检测回来，
   标 ``source="roi_redetect"`` 且身份不变；
3. 阶段 D RANSAC：单相机整具骨架被注入 150px 外点时，3D 根仍贴近真值；
4. ``select_shape_frames``（β 只用 top-K 帧）的纯函数行为。
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tabletennis.core.types import (  # noqa: E402
    CameraExtrinsics,
    CameraIntrinsics,
    Frame,
    Pose2D,
)
from tabletennis.reconstruction.person_track import (  # noqa: E402
    LOWER_BODY_HALPE26,
    MultiPersonTracker,
    TrackConfig,
    lower_body_unreliable,
    mask_lower_body,
)
from tabletennis.reconstruction.triangulate import MultiViewTriangulator  # noqa: E402

_W, _H = 1280, 960
_FX = 800.0


# ----------------------------------------------------------------------
# 合成相机 / 人
# ----------------------------------------------------------------------
def make_ring_rig(radius=4.0, height=1.16, n=4):
    """n 台相机均匀分布在半径 radius、高 height 的圆上，都看向 (0,0,0.06)。

    世界系用**桌面系**约定（与真实标定一致）：z=0 是桌面高度，地面在 z=-0.76，
    站立人的骨盆（halpe26 idx19）在 z≈+0.06。
    """
    K = np.array([[_FX, 0.0, _W / 2], [0.0, _FX, _H / 2], [0.0, 0.0, 1.0]])
    target = np.array([0.0, 0.0, 0.06])
    intr, ext = {}, {}
    for cid in range(n):
        ang = 2 * np.pi * cid / n
        C = np.array([radius * np.cos(ang), radius * np.sin(ang), height])
        fwd = target - C
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        R = np.stack([right, up, fwd])          # X_cam = R X_world + t
        intr[cid] = CameraIntrinsics(width=_W, height=_H, K=K.copy(), dist=np.zeros(5))
        ext[cid] = CameraExtrinsics(R=R, t=(-R @ C).reshape(3, 1))
    return intr, ext


def project(X, intr, ext, cid):
    p = ext[cid].project(intr[cid].K)
    x = p @ np.append(np.asarray(X, dtype=np.float64), 1.0)
    return x[:2] / x[2]


# 26 个关节相对骨盆的世界系偏移（够用来做几何检验，不追求解剖学精度）
_JOINT_OFFSETS = {
    19: (0.0, 0.0, 0.0),        # pelvis = 根
    18: (0.0, 0.0, 0.6),        # neck
    0: (0.0, 0.0, 0.75),        # nose
    17: (0.0, 0.0, 0.85),       # head
    1: (-0.05, 0.0, 0.78), 2: (0.05, 0.0, 0.78),
    3: (-0.1, 0.0, 0.76), 4: (0.1, 0.0, 0.76),
    5: (-0.2, 0.0, 0.5), 6: (0.2, 0.0, 0.5),
    7: (-0.35, 0.0, 0.25), 8: (0.35, 0.0, 0.25),
    9: (-0.45, 0.0, 0.0), 10: (0.45, 0.0, 0.0),
    11: (-0.12, 0.0, 0.0), 12: (0.12, 0.0, 0.0),
    13: (-0.12, 0.0, -0.45), 14: (0.12, 0.0, -0.45),
    15: (-0.12, 0.0, -0.85), 16: (0.12, 0.0, -0.85),
    20: (-0.14, 0.05, -0.88), 21: (-0.1, -0.06, -0.88),
    22: (0.14, 0.05, -0.88), 23: (0.1, -0.06, -0.88),
    24: (-0.14, 0.0, -0.9), 25: (0.14, 0.0, -0.9),
}


def person_joints(root):
    """由骨盆世界坐标生成 (26,3) 关节（静止站姿）。"""
    root = np.asarray(root, dtype=np.float64)
    j = np.zeros((26, 3))
    for idx, off in _JOINT_OFFSETS.items():
        j[idx] = root + np.asarray(off)
    return j


def make_pose(cid, joints, intr, ext, shift_px=(0.0, 0.0)):
    """把 (26,3) 关节投到相机 cid，生成 halpe26 Pose2D（可选整体像素偏移）。"""
    kpts = np.zeros((26, 3), dtype=np.float32)
    for idx in range(26):
        u, v = project(joints[idx], intr, ext, cid)
        kpts[idx] = (u + shift_px[0], v + shift_px[1], 0.9)
    return Pose2D(camera_id=cid, keypoints=kpts, score=0.9, skeleton="halpe26")


def box_of(cid, joints, intr, ext):
    """由关节投影出的紧致人框（xyxy）。"""
    pts = np.asarray([project(j, intr, ext, cid) for j in joints], dtype=np.float64)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    pad = 0.08 * max(x1 - x0, y1 - y0)
    return np.asarray([x0 - pad, y0 - pad, x1 + pad, y1 + pad], dtype=np.float32)


def make_frame(cid, idx=0):
    return Frame(camera_id=cid, serial="t", frame_num=idx, device_timestamp=idx,
                 host_timestamp=idx, image=np.zeros((_H, _W), np.uint8),
                 pixel_format=17301505, width=_W, height=_H)


class World:
    """合成世界：两个人在 4 台相机下的真值投影，供 detect_fn / pose_fn 查询。"""

    def __init__(self, n_frames=30, cams=(0, 1, 2, 3)):
        self.intr, self.ext = make_ring_rig()
        self.cams = list(cams)
        self.gt = []            # [frame][pid] -> (26,3)
        for t in range(n_frames):
            a = np.array([-0.8 + 0.01 * t, -0.5 + 0.005 * t, 0.06])
            b = np.array([0.8 - 0.008 * t, 0.5 - 0.004 * t, 0.06])
            self.gt.append([person_joints(a), person_joints(b)])
        self.hidden = {}        # {(frame, cid, pid)} -> True 表示该相机此帧「看不见」
        self.pose_shift = {}    # {(frame, cid, pid)} -> (dx, dy) 姿态外点

    def boxes(self, cid, t, only_visible=True):
        out, scores = [], []
        for pid, joints in enumerate(self.gt[t]):
            if only_visible and self.hidden.get((t, cid, pid)):
                continue
            out.append(box_of(cid, joints, self.intr, self.ext))
            scores.append(0.9)
        if not out:
            return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)
        return np.asarray(out, np.float32), np.asarray(scores, np.float32)

    def poses_for(self, cid, t, boxes):
        """对给定框，按框中心最近的 GT 人生成姿态。"""
        out = []
        for box in np.asarray(boxes, np.float32):
            c = (box[:2] + box[2:]) * 0.5
            best, best_d = None, None
            for pid, joints in enumerate(self.gt[t]):
                gb = box_of(cid, joints, self.intr, self.ext)
                gc = (gb[:2] + gb[2:]) * 0.5
                d = float(np.hypot(*(c - gc)))
                if best_d is None or d < best_d:
                    best, best_d = pid, d
            shift = self.pose_shift.get((t, cid, best), (0.0, 0.0))
            out.append(make_pose(cid, self.gt[t][best], self.intr, self.ext, shift))
        return out

    def pose_truth(self, cid, t, pid):
        return make_pose(cid, self.gt[t][pid], self.intr, self.ext)


def T_anchor(world, cid, t, pid):
    """GT 第 pid 人在相机 cid 第 t 帧的骨盆像素坐标。"""
    return project(world.gt[t][pid][19], world.intr, world.ext, cid)


def run_tracker(world, cfg=None, n_frames=None, skip_frames=None, gt_pid=None):
    """跑一遍 tracker，返回逐帧的 ``List[TrackFrameResult]``。"""
    cfg = cfg or TrackConfig(full_detect_interval=1, dt=0.01)
    tri = MultiViewTriangulator(world.intr, world.ext)
    t_now = {"t": 0}

    def detect_fn(cid, frame, conf_thresh=None, roi=None):
        t = t_now["t"]
        # 全图检测看不到 hidden 的人；ROI 小窗 + 低阈值能找回来（设计的核心假设）
        boxes, scores = world.boxes(cid, t, only_visible=(roi is None))
        if roi is not None:
            # ROI 小窗只返回落在窗内的框（模拟低阈值重检测）
            keep = []
            for i, b in enumerate(boxes):
                c = (b[:2] + b[2:]) * 0.5
                if roi[0] <= c[0] <= roi[2] and roi[1] <= c[1] <= roi[3]:
                    keep.append(i)
            boxes = boxes[keep]
            scores = scores[keep]
        return boxes, scores

    def pose_fn(cid, frame, boxes):
        return world.poses_for(cid, t_now["t"], boxes)

    trk = MultiPersonTracker(tri, cfg, detect_fn=detect_fn, pose_fn=pose_fn)
    frames = {cid: make_frame(cid) for cid in world.cams}
    out = []
    n = n_frames if n_frames is not None else len(world.gt)
    for t in range(n):
        t_now["t"] = t
        out.append(trk.step(t, frames, time_s=t * cfg.dt))
    return out, trk


# ----------------------------------------------------------------------
# 测试
# ----------------------------------------------------------------------
def test_two_people_stable_identity_wide_fov():
    """一相机同时看到两人：身份全程稳定、每人 4 视角、3D 根贴近真值。"""
    world = World(n_frames=30)
    results, trk = run_tracker(world)

    assert len(trk.tracks) == 2, f"应跟踪到 2 人，实际 {len(trk.tracks)}"
    last = results[-1]
    assert len(last) == 2
    pids = {r.person_id for r in last}
    assert len(pids) == 2, "最后两帧的人身份应互不相同"

    for r in last:
        assert len(r.obs) == 4, f"身份 {r.person_id} 应被 4 台相机看到，实际 {sorted(r.obs)}"
        assert r.robust is not None
        ri = r.robust.root_index
        assert ri >= 0
        X = r.robust.skeleton.keypoints[ri]
        # 找出与真值最接近的那一侧（身份号本身是任意的，只要稳定）
        errs = [np.linalg.norm(X - world.gt[-1][p][19]) for p in (0, 1)]
        assert min(errs) < 0.05, f"3D 根误差 {min(errs)*1000:.1f}mm 过大"

    # 身份不互换：把每帧结果按 person_id 归位，检查轨迹连续（根位置帧间跳变 < 0.15m）
    by_pid = {}
    for frame_res in results[1:]:
        for r in frame_res:
            if r.robust is None:
                continue
            ri = r.robust.root_index
            by_pid.setdefault(r.person_id, []).append(
                r.robust.skeleton.keypoints[ri])
    assert len(by_pid) == 2
    for pid, seq in by_pid.items():
        jumps = np.linalg.norm(np.diff(np.asarray(seq), axis=0), axis=1)
        assert jumps.max() < 0.15, f"身份 {pid} 帧间跳变 {jumps.max():.3f}m（疑似身份互换）"


def test_roi_redetect_recovers_missed_person():
    """某相机连续几帧「看不见」某人 → ROI 重检测找回，身份不变、标 roi_redetect。"""
    world = World(n_frames=30)
    for t in range(10, 15):
        world.hidden[(t, 1, 1)] = True      # cam1 在 10..14 帧看不到 1 号人
    results, trk = run_tracker(world)
    assert len(trk.tracks) == 2

    # 找出「被 cam1 隐藏的那个人」的结果（身份号是任意的，按 anchor 距 GT 最近认）
    gt_anchor = np.asarray(T_anchor(world, 1, 12, 1))
    hit, best_d = None, None
    for r in results[12]:
        if 1 not in r.obs:
            continue
        a = np.asarray(r.obs[1].keypoints[19, :2])
        d = float(np.hypot(*(a - gt_anchor)))
        if best_d is None or d < best_d:
            hit, best_d = r, d
    assert hit is not None, "12 帧应有人的观测包含 cam1"
    assert best_d < 20.0, f"cam1 观测的 anchor 距 GT 差 {best_d:.1f}px，选错了人"
    assert hit.source[1] == "roi_redetect", f"cam1 应为 ROI 重检测，实际 {hit.source[1]}"
    assert len(hit.obs) == 4
    # 该身份在隐藏期前后保持同一个 person_id
    ids_before = {r.person_id for r in results[8]}
    assert hit.person_id in ids_before, "ROI 重检测后身份不应换号"


def test_stage_d_ransac_rejects_single_view_outlier():
    """单相机整具骨架注入 150px 外点 → RANSAC 剔除，3D 根仍准。"""
    world = World(n_frames=12)
    world.pose_shift[(8, 2, 0)] = (150.0, 120.0)
    results, trk = run_tracker(world)
    frame8 = results[8]
    assert len(frame8) == 2
    # 找到被注入外点的那个人（cam2 的观测与其余视角差很多）
    bad = None
    for r in frame8:
        if r.robust is None or r.robust.root_index < 0:
            continue
        X = r.robust.skeleton.keypoints[r.robust.root_index]
        err = min(np.linalg.norm(X - world.gt[8][p][19]) for p in (0, 1))
        if err > 0.3:
            bad = r
    assert bad is None, "RANSAC 未能剔除单视角外点（3D 根偏差 > 0.3m）"

    # 内点率应反映「4 视角里剔掉 1 个」
    r = [x for x in frame8 if x.robust is not None][0]
    ri = r.robust.root_index
    assert r.robust.inlier_ratio[ri] <= 0.8, (
        f"根关节内点率 {r.robust.inlier_ratio[ri]:.2f} 应 < 1（有一个视角是外点）")


def test_full_detect_interval_skips_yolo_but_keeps_tracking():
    """full_detect_interval>1 时跳全图检测（用预测框），轨迹仍连续、不新建身份。"""
    world = World(n_frames=12)
    calls = {"n": 0}
    cfg = TrackConfig(full_detect_interval=4, dt=0.01)
    tri = MultiViewTriangulator(world.intr, world.ext)
    t_now = {"t": 0}

    def detect_fn(cid, frame, conf_thresh=None, roi=None):
        if roi is None:
            calls["n"] += 1
        return world.boxes(cid, t_now["t"])

    def pose_fn(cid, frame, boxes):
        return world.poses_for(cid, t_now["t"], boxes)

    trk = MultiPersonTracker(tri, cfg, detect_fn=detect_fn, pose_fn=pose_fn)
    frames = {cid: make_frame(cid) for cid in world.cams}
    for t in range(12):
        t_now["t"] = t
        trk.step(t, frames, time_s=t * cfg.dt)

    # 12 帧 × 4 相机，每 4 帧一次全图检测 → 远少于 48 次
    assert calls["n"] <= 12 * 4 // 2, f"全图检测调用 {calls['n']} 次，未按间隔跳过"
    assert len(trk.tracks) == 2, "跳检测期间不应多出/丢掉身份"


def test_select_shape_frames_picks_high_confidence():
    """β 帧选择：选出的应是置信度最高、关节最全的帧。"""
    from tabletennis.reconstruction.easymocap import select_shape_frames

    kp = np.zeros((10, 25, 4))
    kp[:, :, 3] = 0.5
    kp[:, :, :3] = 1.0
    # 帧 3/5 关节全、置信度高；帧 7 只有 3 个关节但都是 0.99
    kp[3, :, 3] = 0.9
    kp[5, :, 3] = 0.85
    kp[7, :, 3] = 0.0
    kp[7, :3, 3] = 0.99
    idx = select_shape_frames(kp, top_k=2)
    assert list(idx) == [3, 5], f"选出的帧 {idx.tolist()} 应避开只有 3 个关节的帧 7"

    # 全零输入不崩，且至少给 2 帧（optimizeShape 需要成对骨长）
    idx0 = select_shape_frames(np.zeros((6, 25, 4)), top_k=5)
    assert len(idx0) >= 2


def test_select_shape_frames_uses_top_k_only():
    """top_k 决定选出几帧；帧数不足时按可用帧数返回。"""
    from tabletennis.reconstruction.easymocap import select_shape_frames
    kp = np.zeros((20, 25, 4))
    kp[:, :, 3] = 0.8
    assert len(select_shape_frames(kp, top_k=5)) == 5
    assert len(select_shape_frames(kp, top_k=3)) == 3
    assert len(select_shape_frames(kp, top_k=100)) == 20


def test_select_fit_views_drops_clipped_camera():
    """贴边相机（人只露半截、姿态是外推的）应被剔出 SMPL 拟合视角。"""
    from tabletennis.reconstruction.person_track import select_fit_views

    intr, _ = make_ring_rig(n=4)
    kpts = np.zeros((26, 3), dtype=np.float32)
    kpts[:, 2] = 0.8

    def pose(cid, bbox):
        return Pose2D(camera_id=cid, keypoints=kpts.copy(), score=0.9,
                      bbox=np.asarray(bbox, dtype=np.float32), skeleton="halpe26")

    inside = [100.0, 100.0, 300.0, 700.0]                 # 完全在画面内
    clipped = [900.0, 100.0, float(_W), 700.0]            # 右边缘贴边
    obs = [{0: pose(0, inside), 1: pose(1, clipped), 2: pose(2, inside)}
           for _ in range(10)]

    views = select_fit_views(obs, intr)
    assert views == [0, 2], f"应剔掉裁边的 c1、保留 c0/c2，实际 {views}"

    # 只有一台不裁边 → 仍剩 1 视角 < min_views，回退到出场最多的 2 台
    obs_two = [{0: pose(0, inside), 1: pose(1, clipped)} for _ in range(10)]
    assert len(select_fit_views(obs_two, intr)) == 2

    # 全部贴边 → 不能返回空，回退到出场最多的 2 台
    obs_all = [{0: pose(0, clipped), 1: pose(1, clipped), 2: pose(2, clipped)}
               for _ in range(10)]
    assert len(select_fit_views(obs_all, intr)) == 2

    # 只有少数帧贴边（<50%）→ 该相机保留
    obs_mixed = [{0: pose(0, inside), 1: pose(1, clipped) if i < 3 else pose(1, inside),
                  2: pose(2, inside)} for i in range(10)]
    assert select_fit_views(obs_mixed, intr) == [0, 1, 2]

    # 空输入不崩
    assert select_fit_views([], intr) == []


def _pose_with_conf(upper=0.9, lower=0.9):
    kpts = np.zeros((26, 3), dtype=np.float32)
    kpts[:, :2] = 100.0
    kpts[:, 2] = upper
    for i in LOWER_BODY_HALPE26:
        kpts[i, 2] = lower
    return Pose2D(camera_id=0, keypoints=kpts, score=0.9, skeleton="halpe26")


def test_lower_body_gate_predicate():
    """下半身不可信的判据：相对 + 绝对两个条件都要满足。"""
    # 远端相机典型值：上半身 0.9、下半身 0.3（实测 0.76~0.91 / 0.24~0.40）
    assert lower_body_unreliable(_pose_with_conf(0.9, 0.3))
    # 正常视角：下半身 0.85（实测 0.81~0.89）→ 保留
    assert not lower_body_unreliable(_pose_with_conf(0.9, 0.85))
    # 整具骨架都低（远处小目标）→ 相对比值接近 1，不该被判成「下半身污染」
    assert not lower_body_unreliable(_pose_with_conf(0.3, 0.28))
    # 绝对够高但相对偏低（0.9 vs 0.5）→ 不算不可信
    assert not lower_body_unreliable(_pose_with_conf(0.9, 0.55))
    # 下半身关节全缺（conf=0）→ 没有数据也就没有污染，不判
    assert not lower_body_unreliable(_pose_with_conf(0.9, 0.0))
    # 有效下半身关节太少（< min_joints=2）→ 不判
    p = _pose_with_conf(0.9, 0.3)
    kp = p.keypoints.copy()
    kp[list(LOWER_BODY_HALPE26)[1:], 2] = 0.0
    assert not lower_body_unreliable(Pose2D(camera_id=0, keypoints=kp, score=0.9,
                                            skeleton="halpe26"))


def test_mask_lower_body_keeps_hips_and_coords():
    """掩码只把下半身置信度置 0，坐标与髋部（根关节要用）保持不变。"""
    p = _pose_with_conf(0.9, 0.3)
    m = mask_lower_body(p)
    assert np.allclose(m.keypoints[:, :2], p.keypoints[:, :2]), "坐标不该被改"
    for i in LOWER_BODY_HALPE26:
        assert m.keypoints[i, 2] == 0.0
    for i in (11, 12, 19):                       # 双髋 + 骨盆（阶段 B/C 的根）
        assert m.keypoints[i, 2] == p.keypoints[i, 2] > 0
    assert p.keypoints[13, 2] > 0, "原对象不该被就地修改"
    assert not lower_body_unreliable(m), "掩码后不该再判为不可信"


def test_tracker_masks_lower_body_of_polluted_view():
    """tracker 集成：某视角下半身低置信 → 只该视角的膝/踝被置 0，其余视角不受影响。"""
    world = World(n_frames=8, cams=(0, 1, 2, 3))
    bad_cid = 1
    orig = world.poses_for

    def poses_for(cid, t, boxes):
        out = orig(cid, t, boxes)
        if cid != bad_cid:
            return out
        for p in out:                            # 模拟「隔球桌看人」：膝/踝低置信
            kp = np.array(p.keypoints, dtype=np.float64, copy=True)
            for i in LOWER_BODY_HALPE26:
                kp[i, 1] += 45.0                 # 同时几何上是错的（外推）
                kp[i, 2] = 0.25
            p.keypoints = kp.astype(np.float32)
        return out

    world.poses_for = poses_for
    results, _ = run_tracker(world, n_frames=8)
    last = results[-1]
    assert last, "应有跟踪结果"
    for r in last:
        assert r.lower_body_masked.get(bad_cid) is True, "c1 的下半身应被判不可信"
        assert r.obs[bad_cid].keypoints[13, 2] == 0.0, "掩码后该视角膝不该参与重建"
        assert r.raw_obs[bad_cid].keypoints[13, 2] > 0.0, "原始观测（显示用）应保留"
        for cid in r.obs:
            if cid != bad_cid:
                assert not r.lower_body_masked.get(cid)
                assert r.obs[cid].keypoints[13, 2] > 0.0, "好视角不该被掩码"


def test_tracker_lower_body_gate_can_be_disabled():
    """--no-lower-body-gate 等价开关：关掉后不掩码、raw_obs 与 obs 一致。"""
    world = World(n_frames=6)
    cfg = TrackConfig(full_detect_interval=1, dt=0.01, lower_body_gate=False)
    results, _ = run_tracker(world, cfg=cfg, n_frames=6)
    for r in results[-1]:
        assert r.lower_body_masked == {}
        for cid in r.obs:
            assert np.allclose(r.obs[cid].keypoints, r.raw_obs[cid].keypoints)
