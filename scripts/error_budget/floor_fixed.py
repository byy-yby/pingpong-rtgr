"""用修正后的外参重算**姿态**的跨视角三角化地板：验证 ArUco BA 的修正是否真的有用。

若地板从 6.2px 掉到 ~4px → 那 0.2~0.4° 的外参误差确实是姿态误差的一部分；
若几乎不变 → 说明标记上的修正只对桌面平面有效（平面过拟合），别去动外参。

用法：python floor_fixed.py <session_dir> <fixed_extrinsics.npz>
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, 'src')
from tabletennis.core.types import Pose2D  # noqa: E402
from tabletennis.reconstruction.easymocap import _pose_to_body25_view  # noqa: E402
from tabletennis.reconstruction.person_track import lower_body_unreliable  # noqa: E402
from tabletennis.reconstruction.triangulate import (  # noqa: E402
    MultiViewTriangulator, load_camera_rig,
)

S = sys.argv[1].rstrip('/')
FIX = np.load(sys.argv[2])
RUN = os.path.join(S, 'recon')
CAMS = [0, 1, 2, 3]
MIN_CONF = 0.15
LOWER25 = [10, 11, 13, 14, 19, 20, 21, 22, 23, 24]

intr, extr0 = load_camera_rig(None)


class E:
    def __init__(self, R, t):
        self.R, self.t = R, t.reshape(3, 1)


extr1 = {c: E(FIX[f'R{c}'], FIX[f't{c}']) for c in CAMS}
tri0 = MultiViewTriangulator(intr, extr0)
tri1 = MultiViewTriangulator(intr, extr1)
P = {c: intr[c].K @ np.hstack([extr0[c].R, extr0[c].t.reshape(3, 1)]) for c in CAMS}

poses = json.load(open(os.path.join(RUN, 'pose2d.json'), encoding='utf-8'))
frame_files = sorted(glob.glob(os.path.join(RUN, 'frame_*.npz')))


def proj(cid, X):
    c = np.hstack([X, np.ones((len(X), 1))]) @ P[cid].T
    return c[:, :2] / c[:, 2:3]


def med(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if a.size else float('nan')


rec = {}
for fp in frame_files:
    d = np.load(fp, allow_pickle=True)
    t = int(os.path.basename(fp).split('_')[1].split('.')[0])
    pd = poses.get(str(t))
    if not pd:
        continue
    j3 = {}
    for pid, suf in ((0, ''), (1, '_1')):
        key = 'joints_body25' + suf
        if key in d.files:
            X = np.asarray(d[key], dtype=np.float64)
            if np.isfinite(X).all():
                j3[pid] = X
    if not j3:
        continue
    for cid_s, plist in pd.items():
        cid = int(cid_s)
        if cid not in intr:
            continue
        for p in plist:
            kp26 = np.asarray(p['kpts'], dtype=np.float64)
            v = _pose_to_body25_view(kp26, intr[cid].K, intr[cid].dist, 0.0)
            if v is None:
                continue
            best, best_e = None, 1e18
            for pid, X in j3.items():
                e = np.linalg.norm(proj(cid, X) - v[:, :2], axis=1)
                e = np.where(v[:, 2] > 0, e, np.nan)
                m = np.nanmean(e)
                if np.isfinite(m) and m < best_e:
                    best_e, best = m, pid
            if best is None or best_e > 60:
                continue
            gated = lower_body_unreliable(
                Pose2D(camera_id=cid, keypoints=kp26, score=float(p.get('score', 1.0))))
            rec.setdefault((best, t), {})[cid] = (v, bool(gated))

print(f'===== {os.path.basename(S)}  姿态地板：标定外参 vs ArUco BA 修正外参 =====')


def floor(tri, use_gate=True):
    out = []
    bynv = {}
    for (pid, t), views in rec.items():
        uv = np.full((25, len(CAMS), 2), np.nan)
        cf = np.zeros((25, len(CAMS)))
        for i, cid in enumerate(CAMS):
            if cid not in views:
                continue
            v, gated = views[cid]
            conf = v[:, 2].astype(float).copy()
            if gated and use_gate:
                conf[LOWER25] = 0.0
            uv[:, i, :] = v[:, :2]
            cf[:, i] = conf
        uv[cf < MIN_CONF] = np.nan
        X, cfs, errs, nvs, _ = tri.triangulate_batch(uv, cf, min_conf=MIN_CONF)
        for j in range(25):
            if nvs[j] >= 2 and np.isfinite(errs[j]):
                out.append(errs[j])
                bynv.setdefault(int(nvs[j]), []).append(errs[j])
    return out, bynv


o0, n0 = floor(tri0)
o1, n1 = floor(tri1)
print(f'  全部关节：原 {med(o0):.2f}px  → 修正后 {med(o1):.2f}px   '
      f'(n={len(o0)})')
for nv in sorted(n0):
    print(f'    {nv} 视角：原 {med(n0[nv]):5.2f}px → {med(n1.get(nv, [])):5.2f}px  '
          f'n={len(n0[nv])}')
o0u, _ = floor(tri0, use_gate=False)
o1u, _ = floor(tri1, use_gate=False)
print(f'  不开下半身门：原 {med(o0u):.2f}px → {med(o1u):.2f}px')
