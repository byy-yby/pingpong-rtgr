"""桌面 ArUco 标记上的**交替式 bundle adjustment**：三角度量级的外参修正能拿回多少。

标记静止刚体 → 每相机每角点 2D 按帧平均（角点帧间噪声 0.1~0.5px）。
交替：① 用当前外参把 16 角点三角化 → ② 每台相机对 3D 做 PnP（内参固定）
收敛后报告：逐相机位姿修正量、修正前后「跨视角重投影残差」、标记几何是否变好。

用法：python aruco_ba.py <session_dir> [step] [iters]
"""
import os
import sys
from collections import defaultdict

import cv2
import numpy as np

sys.path.insert(0, 'src')
from tabletennis.reconstruction.triangulate import (  # noqa: E402
    MultiViewTriangulator, load_camera_rig,
)
from tabletennis.reconstruction.video_source import VideoSource  # noqa: E402

S = sys.argv[1].rstrip('/')
STEP = int(sys.argv[2]) if len(sys.argv) > 2 else 10
ITERS = int(sys.argv[3]) if len(sys.argv) > 3 else 5
intr, extr = load_camera_rig(None)
tri = MultiViewTriangulator(intr, extr)
CAMS = sorted(intr)
det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50))
dp = det.getDetectorParameters()
dp.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
dp.minMarkerPerimeterRate = 0.001
dp.maxMarkerPerimeterRate = 4.0
det.setDetectorParameters(dp)

obs = defaultdict(list)
src = VideoSource(S)
for k in range(0, src.n_ref, STEP):
    for cid, f in src.frames_for_ref(k).items():
        corners, ids, _ = det.detectMarkers(getattr(f, 'image', f))
        if ids is None:
            continue
        for q, mid in enumerate(ids.ravel().tolist()):
            if mid not in (0, 1, 2, 3):
                continue
            raw = corners[q].reshape(4, 2).astype(np.float64)
            for j in range(4):
                und = tri.undistort_points(cid, raw[j])
                if und is not None:
                    obs[(mid, j, cid)].append(und)
src.close()

avg2d = {k: np.mean(v, axis=0) for k, v in obs.items() if len(v) >= 5}
sd2d = {k: float(np.linalg.norm(np.std(v, axis=0))) for k, v in obs.items() if len(v) >= 5}
KEYS = sorted({(m, j) for (m, j, c) in avg2d})


def med(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if a.size else float('nan')


def triangulate(ext_use):
    """用给定外参三角化所有角点 → {key: X}。"""
    tri_u = MultiViewTriangulator(intr, ext_use)
    out = {}
    for key in KEYS:
        pts = {c: tuple(avg2d[(key[0], key[1], c)]) for c in CAMS
               if (key[0], key[1], c) in avg2d}
        if len(pts) < 2:
            continue
        X = tri_u._dlt(list(pts), pts, {c: 1.0 for c in pts})
        if X is not None:
            out[key] = X
    return out


def resid(ext_use, X3):
    """各相机对各角点的重投影残差（px）。"""
    out = defaultdict(list)
    for c in CAMS:
        for key, X in X3.items():
            if (key[0], key[1], c) not in avg2d:
                continue
            u = tri.project(c, X) if ext_use is extr else \
                cv2.projectPoints(X.reshape(1, 1, 3), cv2.Rodrigues(ext_use[c].R)[0],
                                  ext_use[c].t.reshape(3, 1), intr[c].K,
                                  np.zeros(5))[0].reshape(2)
            out[c].append(float(np.linalg.norm(np.asarray(u) - avg2d[(key[0], key[1], c)])))
    return out


class E:
    def __init__(self, R, t):
        self.R, self.t = R, t.reshape(3, 1)


print(f'===== {os.path.basename(S)}  ArUco 交替 BA (step={STEP}) =====')
print(f'  可用角点 {len(KEYS)}/16  ' +
      ' '.join(f'ID{m}={sum(1 for k in KEYS if k[0]==m)*4}' for m in (0, 1, 2, 3)))

X3 = triangulate(extr)
r0 = resid(extr, X3)
print('  修正前逐相机残差（px）：' +
      '  '.join(f'c{c}={med(r0[c]):.2f}(n={len(r0[c])})' for c in CAMS))

cur = {c: E(extr[c].R.copy(), extr[c].t.copy()) for c in CAMS}
for it in range(ITERS):
    X3 = triangulate(cur)
    for c in CAMS:
        ks = [k for k in X3 if (k[0], k[1], c) in avg2d]
        if len(ks) < 4:
            continue
        objp = np.array([X3[k] for k in ks], np.float64).reshape(-1, 1, 3)
        imgp = np.array([avg2d[(k[0], k[1], c)] for k in ks], np.float64).reshape(-1, 1, 2)
        rvec0, _ = cv2.Rodrigues(cur[c].R)
        _, rvec, tvec = cv2.solvePnP(objp, imgp, intr[c].K, np.zeros(5), rvec0,
                                     cur[c].t.reshape(3, 1), True,
                                     cv2.SOLVEPNP_ITERATIVE)
        R, _ = cv2.Rodrigues(rvec)
        cur[c] = E(R, tvec)

X3f = triangulate(cur)
r1 = resid(cur, X3f)
if len(sys.argv) > 4:
    out = {f'R{c}': cur[c].R for c in CAMS}
    out.update({f't{c}': cur[c].t.ravel() for c in CAMS})
    np.savez(sys.argv[4], **out)
    print(f'  已存修正外参 -> {sys.argv[4]}')
print('  修正后逐相机残差（px）：' +
      '  '.join(f'c{c}={med(r1[c]):.2f}' for c in CAMS))
print('\n  逐相机位姿修正量（相对标定值）：')
for c in CAMS:
    dR = cur[c].R @ extr[c].R.T
    ang = float(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))
    # 分解成世界系轴角
    rv, _ = cv2.Rodrigues(dR)
    ax = rv.ravel()
    n = np.linalg.norm(ax)
    ax = ax / n if n > 1e-12 else ax
    dt = (cur[c].t.ravel() - extr[c].t.ravel()) * 100
    print(f'    c{c}: {ang:5.2f}°  轴 ({ax[0]:+.2f},{ax[1]:+.2f},{ax[2]:+.2f})  '
          f'平移 ({dt[0]:+5.2f},{dt[1]:+5.2f},{dt[2]:+5.2f})cm  |Δt| {np.linalg.norm(dt):5.2f}cm  '
          f'角点噪声 {med([sd2d[(k[0],k[1],c)] for k in KEYS if (k[0],k[1],c) in sd2d]):.2f}px')

# 标记几何（修正后）
sides = []
for mid in (0, 1, 2, 3):
    if all((mid, j) in X3f for j in range(4)):
        P4 = np.array([X3f[(mid, j)] for j in range(4)])
        sides.extend(float(np.linalg.norm(P4[i] - P4[(i + 1) % 4])) for i in range(4))
cen = {m: np.mean([X3f[(m, j)] for j in range(4)], axis=0) for m in (0, 1, 2, 3)
       if all((m, j) in X3f for j in range(4))}
print(f'\n  修正后标记几何：边长中位 {med(sides):.4f}m (真值 0.1751)')
if len(cen) == 4:
    for a, b, nom in ((0, 1, 1.3099), (0, 3, 2.5249), (0, 2, 2.8444)):
        d = float(np.linalg.norm(cen[a] - cen[b]))
        print(f'    ID{a}→ID{b} {d:.4f}m  真值 {nom:.4f}m  比 {d/nom:.4f}')
