"""ArUco 桌面标记 v2：把标定误差拆成「哪台相机 / 哪个方向 / 内参还是外参」。

v1 的结论：角点跨视角残差 3.4px（姿态地板 6.2px），c1/c3 各 ~4.8px、c0/c2 ~1.7px。
v2 加三件事：
  1. 留一视角（LOO）：用其余相机三角化再投回被留的那台 → 逐相机**与共识的一致性**
     （不受「2 视角残差天然小」污染，只用 ≥3 视角的角点）
  2. 残差随**像面半径**：随半径涨 → 内参（焦距/畸变）；基本平 → 外参（旋转/平移）
  3. 已知尺寸：用**实测**标记边长当基准算标记间距真值（打印机缩放的坑）

用法：python aruco_gt2.py <session_dir> [step]
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
STEP = int(sys.argv[2]) if len(sys.argv) > 2 else 5
BORDER = 0.02          # 白边宽度（config/extrinsics.yaml）
TAB_W, TAB_L = 1.525, 2.74

intr, extr = load_camera_rig(None)
tri = MultiViewTriangulator(intr, extr)
CAMS = sorted(intr)
det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50))
dp = det.getDetectorParameters()
dp.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
dp.minMarkerPerimeterRate = 0.001
dp.maxMarkerPerimeterRate = 4.0
dp.cornerRefinementWinSize = 5
det.setDetectorParameters(dp)


def med(a):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if a.size else float('nan')


def p90(a):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.percentile(a, 90)) if a.size else float('nan')


rec = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
src = VideoSource(S)
n_ref = src.n_ref
for k in range(0, n_ref, STEP):
    for cid, f in src.frames_for_ref(k).items():
        img = getattr(f, 'image', f)
        corners, ids, _ = det.detectMarkers(img)
        if ids is None:
            continue
        for q, mid in enumerate(ids.ravel().tolist()):
            if mid not in (0, 1, 2, 3):
                continue
            raw = corners[q].reshape(4, 2).astype(np.float64)
            for j in range(4):
                und = tri.undistort_points(cid, raw[j])
                if und is not None:
                    rec[mid][j][k][cid] = (und, raw[j])
src.close()
print(f'===== {os.path.basename(S)}  ArUco 标定真值 v2  (step={STEP}, {n_ref} 帧) =====')


def dlt_subset(views, pts):
    cf = {c: 1.0 for c in views}
    return tri._dlt(list(views), pts, cf)


# ------------------------------------------------ A 边长 → 局部尺度
side = {}
for mid in (0, 1, 2, 3):
    ss = []
    for k in rec[mid][0]:
        P4 = []
        for j in range(4):
            v = rec[mid][j].get(k)
            if not v or len(v) < 2:
                break
            X = dlt_subset(v.keys(), {c: tuple(v[c][0]) for c in v})
            if X is None:
                break
            P4.append(X)
        if len(P4) == 4:
            P4 = np.asarray(P4)
            ss.extend(float(np.linalg.norm(P4[i] - P4[(i + 1) % 4])) for i in range(4))
    side[mid] = med(ss)
print('\n-- A 标记黑方块边长（配置真值 0.180m；打印机缩放会整体偏移）--')
for mid in (0, 1, 2, 3):
    print(f'  ID{mid}: {side[mid]:.4f}m  比 {side[mid] / 0.18:.4f}')
s_meas = med(list(side.values()))
print(f'  四标记一致边长中位 {s_meas:.4f}m（后文以它为基准算标记间距真值）')

# ------------------------------------------------ B 间距
cen = {}
for mid in (0, 1, 2, 3):
    acc = []
    for k in rec[mid][0]:
        P4 = []
        for j in range(4):
            v = rec[mid][j].get(k)
            if not v or len(v) < 2:
                break
            X = dlt_subset(v.keys(), {c: tuple(v[c][0]) for c in v})
            if X is None:
                break
            P4.append(X)
        if len(P4) == 4:
            acc.append(np.mean(P4, axis=0))
    cen[mid] = np.median(np.asarray(acc), axis=0) if acc else None
    if cen[mid] is not None:
        print(f'  ID{mid} 世界坐标中位 ({cen[mid][0]:7.3f},{cen[mid][1]:7.3f},{cen[mid][2]:7.3f})'
              f'  n={len(acc)}')
off = s_meas / 2 + BORDER                      # 标记中心到桌角（白边外角）
print(f'\n-- B 标记间距（真值按实测边长算：中心间距 = 桌边 − 2×{off:.4f}m）--')
for (a, b), nom0 in {(0, 1): TAB_W - 2 * off, (0, 3): TAB_L - 2 * off,
                     (0, 2): np.hypot(TAB_W - 2 * off, TAB_L - 2 * off)}.items():
    if cen.get(a) is None or cen.get(b) is None:
        continue
    d = float(np.linalg.norm(cen[a] - cen[b]))
    print(f'  ID{a}→ID{b}: 实测 {d:.4f}m  真值 {nom0:.4f}m  比 {d / nom0:.4f}'
          f'  差 {(d - nom0) * 100:+.1f}cm')

# ------------------------------------------------ C 跨视角残差（按视角数）
print('\n-- C 跨视角三角化残差地板（角点；按视角数分）--')
by_nv = defaultdict(list)
for mid in (0, 1, 2, 3):
    for j in range(4):
        for k, v in rec[mid][j].items():
            if len(v) < 2:
                continue
            pts = {c: tuple(v[c][0]) for c in v}
            X = dlt_subset(v.keys(), pts)
            if X is None:
                continue
            r = [float(np.linalg.norm(tri.project(c, X) - np.asarray(pts[c]))) for c in v]
            by_nv[len(v)].extend(r)
for nv in sorted(by_nv):
    v = by_nv[nv]
    print(f'  {nv} 视角: 中位 {med(v):5.2f}px  p90 {p90(v):6.2f}px  n={len(v)}')

# ------------------------------------------------ D 留一视角逐相机一致性
print('\n-- D 留一视角（≥3 视角才做）：该相机与其余相机共识的偏差 --')
loo = defaultdict(list)
for mid in (0, 1, 2, 3):
    for j in range(4):
        for k, v in rec[mid][j].items():
            if len(v) < 3:
                continue
            pts = {c: tuple(v[c][0]) for c in v}
            for c in v:
                rest = [x for x in v if x != c]
                X = dlt_subset(rest, pts)
                if X is None:
                    continue
                loo[c].append(float(np.linalg.norm(tri.project(c, X) - np.asarray(pts[c]))))
for c in CAMS:
    v = loo.get(c, [])
    if v:
        print(f'  c{c}: LOO 中位 {med(v):5.2f}px  p90 {p90(v):6.2f}px  n={len(v)}')

# ------------------------------------------------ E 残差 vs 像面半径（逐相机）
print('\n-- E 残差 vs 像面半径（LOO，逐相机；随半径涨=内参，平=外参）--')
EDGES = [0, 200, 400, 600, 900]
for c in CAMS:
    bins = defaultdict(list)
    for mid in (0, 1, 2, 3):
        for j in range(4):
            for k, v in rec[mid][j].items():
                if len(v) < 3 or c not in v:
                    continue
                pts = {x: tuple(v[x][0]) for x in v}
                rest = [x for x in v if x != c]
                X = dlt_subset(rest, pts)
                if X is None:
                    continue
                e = float(np.linalg.norm(tri.project(c, X) - np.asarray(pts[c])))
                px = np.asarray(pts[c])
                K = intr[c].K
                r = float(np.hypot(px[0] - K[0, 2], px[1] - K[1, 2]))
                for i in range(len(EDGES) - 1):
                    if EDGES[i] <= r < EDGES[i + 1]:
                        bins[i].append(e)
    s = ' '.join(f'{EDGES[i]}-{EDGES[i+1]}px: {med(bins[i]):5.2f}(n={len(bins[i])})'
                 for i in range(len(EDGES) - 1) if bins[i])
    print(f'  c{c}: {s}')

# ------------------------------------------------ F 静态散布
print('\n-- F 静态目标跨帧 3D 散布（真值 0 = 2D 噪声经三角化放大）--')
for mid in (0, 1, 2, 3):
    xs = []
    for k in rec[mid][0]:
        P4 = []
        for j in range(4):
            v = rec[mid][j].get(k)
            if not v or len(v) < 2:
                break
            X = dlt_subset(v.keys(), {c: tuple(v[c][0]) for c in v})
            if X is None:
                break
            P4.append(X)
        if len(P4) == 4:
            xs.append(np.mean(P4, axis=0))
    if len(xs) < 10:
        continue
    xs = np.asarray(xs)
    sd = xs.std(axis=0)
    print(f'  ID{mid}: std X/Y/Z = {sd[0]*100:5.2f}/{sd[1]*100:5.2f}/{sd[2]*100:5.2f} cm'
          f'  合 {np.linalg.norm(sd)*100:.2f}cm  n={len(xs)}')

# ------------------------------------------------ G 共面性
pts = []
for mid in (0, 1, 2, 3):
    for j in range(4):
        for k, v in rec[mid][j].items():
            if len(v) < 2:
                continue
            X = dlt_subset(v.keys(), {c: tuple(v[c][0]) for c in v})
            if X is not None:
                pts.append(X)
if pts:
    pts = np.asarray(pts)
    c0 = pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts - c0)
    d = (pts - c0) @ Vt[2]
    print(f'\n-- G 共面性：法向 {Vt[2]}  点面距 中位 {med(np.abs(d))*100:.2f}cm'
          f'  p90 {p90(np.abs(d))*100:.2f}cm  n={len(pts)}')
