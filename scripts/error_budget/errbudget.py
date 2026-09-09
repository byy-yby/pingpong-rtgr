"""误差来源 / 分布分析（三段视频）。

层次（每层独立可测，同时给 px 与该视角深度下的 cm）：
  L0  2D 关键点识别噪声 σ：逐相机逐关节时间二阶差分（仅低运动帧）
  L1  三角化地板 r_in：同批 2D 点加权 DLT 三角化后回投各视角的残差
       └ 若只有 2D 白噪声，应有 r_pred = σ·sqrt((2N-3)/(2N))
       └ 超出部分 r_sys = sqrt(max(0, r_in²-r_pred²)) = 标定/系统性
  L2  SMPL 拟合残差 e_fit；e_fit² ≈ r_pred² + r_sys² + e_model²
  系统性 vs 随机：逐相机逐关节的**带符号**残差 DC(均值) / AC(标准差)
  标定诊断：极线（Sampson）误差、残差随像面半径的分布

用法：python errbudget.py <session_dir> [out.npz]
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, 'src')
from tabletennis.core.types import Pose2D  # noqa: E402
from tabletennis.reconstruction.easymocap import _pose_to_body25_view  # noqa: E402
from tabletennis.reconstruction.person_track import lower_body_unreliable  # noqa: E402
from tabletennis.reconstruction.triangulate import (  # noqa: E402
    MultiViewTriangulator, load_camera_rig,
)

SESSION = sys.argv[1].rstrip('/')
OUT = sys.argv[2] if len(sys.argv) > 2 else None
RUN = os.path.join(SESSION, 'recon')
CAMS = [0, 1, 2, 3]
MIN_CONF = 0.15
LOWER25 = [10, 11, 13, 14, 19, 20, 21, 22, 23, 24]
HIP25 = [8, 9, 12]
BONES = [(5, 12), (6, 11), (11, 13), (12, 14), (13, 15), (14, 16), (5, 6), (11, 12)]
NAME25 = {0: '鼻', 1: '颈', 2: 'R肩', 3: 'R肘', 4: 'R腕', 5: 'L肩', 6: 'L肘', 7: 'L腕',
          8: '骨盆中', 9: 'R髋', 10: 'R膝', 11: 'R踝', 12: 'L髋', 13: 'L膝', 14: 'L踝',
          15: 'R眼', 16: 'L眼', 17: 'R耳', 18: 'L耳', 19: 'L大趾', 20: 'L小趾',
          21: 'L跟', 22: 'R大趾', 23: 'R小趾', 24: 'R跟'}
PART = np.array(['上身'] * 25, dtype=object)
for j in HIP25:
    PART[j] = '髋'
for j in LOWER25:
    PART[j] = '下身'

intr, extr = load_camera_rig(None)
tri = MultiViewTriangulator(intr, extr)
P = {c: intr[c].K @ np.hstack([extr[c].R, extr[c].t.reshape(3, 1)]) for c in intr}
meta = json.load(open(os.path.join(RUN, 'recon_meta.json'), encoding='utf-8'))
FIT_VIEWS = {i: sorted(g) for i, g in
             enumerate(meta.get('person_groups', [[0, 1, 2], [0, 1, 3]]))}
poses = json.load(open(os.path.join(RUN, 'pose2d.json'), encoding='utf-8'))
frame_files = sorted(glob.glob(os.path.join(RUN, 'frame_*.npz')))
NC = len(CAMS)
K_ALL = np.stack([intr[c].K for c in CAMS])


def proj(cid, X):
    c = np.hstack([X, np.ones((len(X), 1))]) @ P[cid].T
    return c[:, :2] / c[:, 2:3]


def med(a):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if a.size else float('nan')


# ---------------------------------------------------------------- 逐帧解析
rec = defaultdict(lambda: defaultdict(dict))
bbox = defaultdict(lambda: defaultdict(dict))     # pid -> t -> cid -> (x1,y1,x2,y2)
smpl = defaultdict(dict)
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
                uv = proj(cid, X)
                e = np.linalg.norm(uv - v[:, :2], axis=1)
                e = np.where(v[:, 2] > 0, e, np.nan)
                m = np.nanmean(e)
                if np.isfinite(m) and m < best_e:
                    best_e, best = m, pid
            if best is None or best_e > 60:
                continue
            gated = lower_body_unreliable(
                Pose2D(camera_id=cid, keypoints=kp26, score=float(p.get('score', 1.0))))
            rec[best][t][cid] = (v, kp26, bool(gated))
            if p.get('bbox') is not None:
                bbox[best][t][cid] = np.asarray(p['bbox'], dtype=float)
    for pid, X in j3.items():
        smpl[pid][t] = X

# ---------------------------------------------------------------- 根速度（运动门）
speed = defaultdict(dict)        # pid -> t -> m/s
for pid, by_t in smpl.items():
    ts = sorted(by_t)
    for i, t in enumerate(ts):
        if i == 0 or i == len(ts) - 1 or ts[i + 1] - ts[i - 1] > 4:
            continue
        dt = (ts[i + 1] - ts[i - 1]) * 0.01
        speed[pid][t] = float(np.linalg.norm(by_t[ts[i + 1]][8] - by_t[ts[i - 1]][8])) / dt

# ---------------------------------------------------------------- L0：2D 噪声 σ
sig_vals = defaultdict(list)
for pid, by_t in rec.items():
    ts = sorted(by_t)
    for cid in CAMS:
        ser = [(t, by_t[t][cid][0]) for t in ts if cid in by_t[t]]
        if len(ser) < 3:
            continue
        T = np.array([s[0] for s in ser])
        K = np.stack([s[1] for s in ser])
        d2 = K[2:, :, :2] - 0.5 * (K[:-2, :, :2] + K[1:-1, :, :2])
        ok = (K[2:, :, 2] > 0.5) & (K[1:-1, :, 2] > 0.5) & (K[:-2, :, 2] > 0.5)
        ok &= ((T[2:] - T[:-2]) <= 4)[:, None]
        mid = T[1:-1]
        sp = np.array([speed.get(pid, {}).get(t, 99.0) for t in mid])
        ok &= (sp < 0.30)[:, None]                     # 只用低运动帧，避免真实加速度混入
        mag = np.linalg.norm(d2, axis=2)
        for j in range(25):
            m = mag[ok[:, j], j]
            if m.size:
                sig_vals[(pid, cid, j)].append(m)
SIGMA = {k: float(np.median(np.concatenate(v))) / 1.177 / (1.5 ** 0.5)
         for k, v in sig_vals.items()}

# ---------------------------------------------------------------- 逐帧误差层
K_KEYS = ('pid', 't', 'cid', 'j', 'gated', 'in_fit', 'conf', 'nv', 'r_in', 'r_fit',
          'r_loo', 'mpp', 'sigma', 'd3', 'dx', 'dy', 'fdx', 'fdy', 'rad')
ACC = {k: [] for k in K_KEYS}
FRAME = defaultdict(list)
for pid in sorted(rec):
    for t in sorted(rec[pid]):
        if t not in smpl.get(pid, {}):
            continue
        views = rec[pid][t]
        Xs = smpl[pid][t]
        uv = np.full((25, NC, 2), np.nan)
        cf = np.zeros((25, NC))
        gate = np.zeros(NC, dtype=bool)
        for i, cid in enumerate(CAMS):
            if cid not in views:
                continue
            v, _kp26, gated = views[cid]
            conf = v[:, 2].astype(np.float64).copy()
            gate[i] = gated
            if gated:
                conf[LOWER25] = 0.0
            uv[:, i, :] = v[:, :2]
            cf[:, i] = conf
        uv[cf < MIN_CONF] = np.nan
        Xt, _c, _e, nv, _a = tri.triangulate_batch(uv, cf, min_conf=MIN_CONF)
        valid3 = np.isfinite(Xt).all(axis=1)
        rp = np.stack([proj(cid, np.where(valid3[:, None], Xt, 0.0)) for cid in CAMS], 1)
        dxy = rp - uv
        res = np.linalg.norm(dxy, axis=2)
        res[~np.isfinite(uv).all(axis=2)] = np.nan
        res[~valid3] = np.nan
        mpp = np.full(NC, np.nan)
        for i in range(NC):
            vals = []
            for a, b in BONES:
                if not (valid3[a] and valid3[b]):
                    continue
                if not (np.isfinite(uv[a, i]).all() and np.isfinite(uv[b, i]).all()):
                    continue
                L3 = float(np.linalg.norm(Xt[a] - Xt[b]))
                L2 = float(np.linalg.norm(uv[a, i] - uv[b, i]))
                if L3 > 0.2 and L2 > 5:
                    vals.append(L3 / L2)
            if vals:
                mpp[i] = float(np.median(vals))
        loo = np.full((25, NC), np.nan)
        d3_list = [[] for _ in range(25)]
        for i in range(NC):
            if not np.isfinite(uv[:, i]).any():
                continue
            cf2 = cf.copy()
            cf2[:, i] = 0.0
            Xl, _c2, _e2, _n2, _a2 = tri.triangulate_batch(uv, cf2, min_conf=MIN_CONF)
            good = np.isfinite(Xl).all(axis=1) & np.isfinite(uv[:, i]).all(axis=1)
            if good.any():
                loo[good, i] = np.linalg.norm(proj(CAMS[i], Xl[good]) - uv[good, i], axis=1)
            # 只在「该视角确实参与了」时算 LOO→3D 抖动
            contrib = np.isfinite(uv[:, i]).all(axis=1)
            for j in np.where(contrib & np.isfinite(Xl).all(axis=1) & valid3)[0]:
                d3_list[j].append(float(np.linalg.norm(Xl[j] - Xt[j])))
        d3 = np.array([float(np.median(v)) if v else np.nan for v in d3_list])
        fdxy = np.stack([proj(cid, Xs) - uv[:, i] for i, cid in enumerate(CAMS)], 1)
        fit = np.linalg.norm(fdxy, axis=2)
        fit[~np.isfinite(uv).all(axis=2)] = np.nan
        # 帧级报告口径
        pp = []
        for i, cid in enumerate(CAMS):
            if cid not in views or cid not in FIT_VIEWS.get(pid, []):
                continue
            m = np.isfinite(fit[:, i])
            if m.any():
                pp.extend(fit[m, i].tolist())
        if pp:
            FRAME[t].append(float(np.mean(pp)))
        for i, cid in enumerate(CAMS):
            if cid not in views:
                continue
            for j in range(25):
                if not np.isfinite(uv[j, i]).all():
                    continue
                ACC['pid'].append(pid); ACC['t'].append(t); ACC['cid'].append(cid)
                ACC['j'].append(j); ACC['gated'].append(gate[i])
                ACC['in_fit'].append(cid in FIT_VIEWS.get(pid, []))
                ACC['conf'].append(cf[j, i]); ACC['nv'].append(nv[j])
                ACC['r_in'].append(res[j, i]); ACC['r_fit'].append(fit[j, i])
                ACC['r_loo'].append(loo[j, i]); ACC['mpp'].append(mpp[i])
                ACC['sigma'].append(SIGMA.get((pid, cid, j), np.nan))
                ACC['d3'].append(d3[j])
                ACC['dx'].append(dxy[j, i, 0]); ACC['dy'].append(dxy[j, i, 1])
                ACC['fdx'].append(fdxy[j, i, 0]); ACC['fdy'].append(fdxy[j, i, 1])
                ACC['rad'].append(float(np.linalg.norm(
                    uv[j, i] - np.array([intr[cid].K[0, 2], intr[cid].K[1, 2]]))))
A = {k: np.asarray(v) for k, v in ACC.items()}
for k in ('pid', 't', 'cid', 'j', 'nv'):
    A[k] = A[k].astype(int)
for k in ('gated', 'in_fit'):
    A[k] = A[k].astype(bool)
A['cm'] = A['mpp'] * 100.0
series = {t: float(np.mean(v)) for t, v in FRAME.items() if v}


def sel(mask, **kw):
    m = mask
    for k, v in kw.items():
        m = m & (A[k] == v)
    return m


fitm = A['in_fit']
np.savez_compressed(OUT, series=np.array(sorted(series.items())), **A)   # 先存档，后面打印挂了也不丢
print(f'===== {os.path.basename(SESSION)}  帧 {len(frame_files)}  '
      f'meta {meta.get("reproj_err_mean_px_median", float("nan")):.2f}px =====')
print(f'复现口径 = {med(list(series.values())):.2f}px   '
      f'[meta {meta.get("reproj_err_mean_px_median", float("nan")):.2f}px]')

print('\n-- 逐人（拟合视角）--')
for pid in sorted(set(A['pid'].tolist())):
    m = sel(fitm, pid=pid)
    fr = []
    for t in np.unique(A['t'][m]):
        v = A['r_fit'][m & (A['t'] == t)]
        if v.size:
            fr.append(float(np.mean(v)))
    print(f'  p{pid}: 帧 {len(fr):5d} | 帧均中位 {med(fr):5.2f}px | 关节级 '
          f'{med(A["r_fit"][m]):5.2f}px = {med(A["r_fit"][m]*A["cm"][m]):5.1f}cm | '
          f'视角 {med(A["nv"][m]):.2f} | 远端占比 {A["gated"][m].mean():.2f}')

print('\n-- 误差预算（拟合视角，关节级中位）--')
for part in ('上身', '髋', '下身'):
    m = fitm & (PART[A['j']] == part)
    nv, sig = A['nv'][m], A['sigma'][m]
    rpred = sig * np.sqrt(np.maximum(2 * nv - 3, 0) / np.maximum(2 * nv, 1))
    rin, rfit, rloo, cmf = A['r_in'][m], A['r_fit'][m], A['r_loo'][m], A['cm'][m]
    rsys = np.sqrt(np.maximum(rin ** 2 - rpred ** 2, 0))
    rmod = np.sqrt(np.maximum(rfit ** 2 - rin ** 2, 0))
    print(f'  {part}: n={m.sum():7d}  视角中位 {med(nv):.2f}')
    print(f'    2D 噪声 σ              {med(sig):6.2f}px  {med(sig*cmf):6.1f}cm(单视角)')
    print(f'    三角化地板 r_in        {med(rin):6.2f}px  {med(rin*cmf):6.1f}cm')
    print(f'      ├ 2D 噪声 r_pred     {med(rpred):6.2f}px  {med(rpred*cmf):6.1f}cm')
    print(f'      └ 标定/系统性 r_sys  {med(rsys):6.2f}px  {med(rsys*cmf):6.1f}cm')
    print(f'    SMPL 拟合 e_fit        {med(rfit):6.2f}px  {med(rfit*cmf):6.1f}cm')
    print(f'      └ 拟合/模型 e_model  {med(rmod):6.2f}px  {med(rmod*cmf):6.1f}cm')
    print(f'    LOO 留一视角           {med(rloo):6.2f}px  {med(rloo*cmf):6.1f}cm')
    print(f'    2D→3D 位置抖动         {med(A["d3"][m])*100:6.2f}cm')

print('\n-- 近端 / 远端（下半身门命中）--')
for g in (False, True):
    m = fitm & (A['gated'] == g)
    if not m.any():
        continue
    for part in ('上身', '髋', '下身'):
        mm = m & (PART[A['j']] == part)
        if mm.any():
            print(f'  {"远端" if g else "近端"} {part}: n={mm.sum():7d} '
                  f'{med(A["r_fit"][mm]):5.2f}px = {med(A["r_fit"][mm]*A["cm"][mm]):5.1f}cm  '
                  f'σ={med(A["sigma"][mm]):.2f}px 视角 {med(A["nv"][mm]):.2f} '
                  f'LOO {med(A["r_loo"][mm]*A["cm"][mm]):5.1f}cm')

print('\n-- 逐相机（拟合视角）--')
for cid in CAMS:
    m = sel(fitm, cid=cid)
    if not m.any():
        continue
    mppc = med(A['mpp'][m])
    dx, dy = A['dx'][m] * mppc * 100, A['dy'][m] * mppc * 100
    print(f'  c{cid}: n={m.sum():7d} 拟合 {med(A["r_fit"][m]*A["cm"][m]):5.1f}cm  '
          f'地板 {med(A["r_in"][m]*A["cm"][m]):5.1f}cm  LOO '
          f'{med(A["r_loo"][m]*A["cm"][m]):5.1f}cm  σ={med(A["sigma"][m]):.2f}px  '
          f'mpp={med(A["mpp"][m])*1000:.2f}mm/px  视角 {med(A["nv"][m]):.2f}')
    print(f'        带符号残差 DC=({np.nanmean(dx):+5.2f},{np.nanmean(dy):+5.2f})cm  '
          f'AC=({np.nanstd(dx):4.2f},{np.nanstd(dy):4.2f})cm  '
          f'|DC|/|AC|={np.hypot(np.nanmean(dx), np.nanmean(dy))/max(np.hypot(np.nanstd(dx), np.nanstd(dy)), 1e-9):5.2f}')

print('\n-- 系统性 vs 随机（拟合残差，逐相机逐关节带符号 DC/AC，cm）--')
for cid in CAMS:
    m = sel(fitm, cid=cid)
    if not m.any():
        continue
    dc, ac = [], []
    mppc = med(A['mpp'][m])
    for j in range(25):
        mm = m & (A['j'] == j)
        if mm.sum() < 200:
            continue
        fx = A['fdx'][mm] * mppc * 100
        fy = A['fdy'][mm] * mppc * 100
        dc.append(float(np.hypot(np.nanmean(fx), np.nanmean(fy))))
        ac.append(float(np.hypot(np.nanstd(fx), np.nanstd(fy))))
    print(f'  c{cid}: |DC| 中位 {med(dc):4.2f}cm  |AC| 中位 {med(ac):4.2f}cm  '
          f'（DC 占 √(DC²+AC²) 的 {med([d/np.hypot(d,a) for d,a in zip(dc,ac)])*100:.0f}%）')

print('\n-- 残差 vs 像面半径（拟合视角，地板残差，cm）--')
m = fitm
edges = [0, 200, 400, 600, 800, 1000, 1400]
for k in range(len(edges) - 1):
    mm = m & (A['rad'] >= edges[k]) & (A['rad'] < edges[k + 1])
    if mm.sum() > 500:
        print(f'  半径 {edges[k]:4d}-{edges[k+1]:4d}px: 地板 '
              f'{med(A["r_in"][mm]*A["cm"][mm]):5.2f}cm  拟合 '
              f'{med(A["r_fit"][mm]*A["cm"][mm]):5.2f}cm  n={mm.sum()}')

print('\n-- 残差 vs 运动速度 / 2D 置信度（拟合视角，地板残差）--')
spd = np.array([max(speed.get(pid, {}).get(t, np.nan), 0.0)
                for pid, t in zip(A['pid'], A['t'])])
for lo, hi in ((0, 0.1), (0.1, 0.3), (0.3, 0.8), (0.8, 1.6), (1.6, 99)):
    m = fitm & (spd >= lo) & (spd < hi)
    if m.sum() > 500:
        print(f'  根速度 {lo:4.1f}-{hi:4.1f} m/s: 地板 '
              f'{med(A["r_in"][m]*A["cm"][m]):5.2f}cm  拟合 '
              f'{med(A["r_fit"][m]*A["cm"][m]):5.2f}cm  n={m.sum()}')
for lo, hi in ((0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)):
    m = fitm & (A['conf'] >= lo) & (A['conf'] < hi)
    if m.sum() > 500:
        print(f'  2D conf {lo:.1f}-{hi:.1f}: 地板 '
              f'{med(A["r_in"][m]*A["cm"][m]):5.2f}cm  拟合 '
              f'{med(A["r_fit"][m]*A["cm"][m]):5.2f}cm  n={m.sum()}')

print('\n-- 极线（Sampson）误差：标定几何自洽性，与深度无关 --')


def fund(a, b):
    Ra, ta = np.asarray(extr[a].R), np.asarray(extr[a].t).ravel()
    Rb, tb = np.asarray(extr[b].R), np.asarray(extr[b].t).ravel()
    Rba = Rb @ Ra.T
    tba = tb - Rba @ ta
    tx = np.array([[0, -tba[2], tba[1]], [tba[2], 0, -tba[0]], [-tba[1], tba[0], 0]])
    return np.linalg.inv(intr[b].K).T @ tx @ Rba @ np.linalg.inv(intr[a].K)


PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
epi = defaultdict(list)
FM = {(a, b): fund(a, b) for a, b in PAIRS}
for pid, by_t in rec.items():
    for t, views in by_t.items():
        for a, b in PAIRS:
            if a not in views or b not in views:
                continue
            va, vb = views[a][0], views[b][0]
            F = FM[(a, b)]
            x1 = np.concatenate([va[:, :2], np.ones((25, 1))], axis=1)      # (25,3)
            x2 = np.concatenate([vb[:, :2], np.ones((25, 1))], axis=1)
            Fx1 = x1 @ F.T                       # (25,3)
            Ftx2 = x2 @ F                        # (25,3)
            num = np.einsum('ij,ij->i', x2, Fx1) ** 2
            den = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2
            ok = (va[:, 2] > 0.3) & (vb[:, 2] > 0.3) & (den > 1e-12)
            if ok.any():
                sv = np.sqrt(num[ok] / den[ok])
                epi[(a, b)].extend(sv.tolist())
                for j, s in zip(np.where(ok)[0], sv):
                    epi[('j', int(j))].append(float(s))
for (a, b), v in sorted(epi.items(), key=lambda kv: str(kv[0])):
    if a == 'j':
        continue
    print(f'  c{a}-c{b}: Sampson 中位 {med(v):5.2f}px  p90 {np.percentile(v,90):6.2f}px  '
          f'n={len(v)}')
print('  逐关节 Sampson（中位 px）：刚性高置信部位 = 标定；四肢 = 2D 定位')
for j in range(25):
    v = epi.get(('j', j), [])
    if len(v) > 1000:
        print(f'    #{j:2d} {NAME25[j]:>5}: {med(v):5.2f}px  (n={len(v)})')

print('\n-- 逐关节（拟合视角，cm）--')
MPP_G = med(A['mpp'][fitm])
for j in range(25):
    m = fitm & (A['j'] == j)
    if m.sum() > 200:
        fx = A['fdx'][m] * MPP_G * 100
        fy = A['fdy'][m] * MPP_G * 100
        print(f'  #{j:2d} {NAME25[j]:>5}: 拟合 {med(A["r_fit"][m]*A["cm"][m]):5.2f}  '
              f'地板 {med(A["r_in"][m]*A["cm"][m]):4.2f}  '
              f'DC {np.hypot(np.nanmean(fx), np.nanmean(fy)):4.2f}  '
              f'AC {np.hypot(np.nanstd(fx), np.nanstd(fy)):4.2f} cm')

print('\n-- 时间分布（按帧号 10 等分）--')
ts = sorted(series)
if ts:
    edges = np.linspace(ts[0], ts[-1] + 1, 11)
    for k in range(10):
        sub = [t for t in ts if edges[k] <= t < edges[k + 1]]
        if not sub:
            continue
        v = [series[t] for t in sub]
        mk = fitm & np.isin(A['t'], sub)
        print(f'  帧 {int(edges[k]):5d}-{int(edges[k+1])-1:5d}: 中位 {np.median(v):5.2f}px  '
              f'p90 {np.percentile(v, 90):6.2f}px  视角 {med(A["nv"][mk]):.2f}  '
              f'远端 {A["gated"][mk].mean():.2f}')

print('\n-- 最差 10 帧 --')
for t, v in sorted(series.items(), key=lambda kv: -kv[1])[:10]:
    mk = fitm & (A['t'] == t)
    lo = mk & (PART[A['j']] == '下身')
    print(f'  t={t:5d}: {v:6.2f}px  视角 {med(A["nv"][mk]):.1f}  '
          f'远端 {A["gated"][mk].mean():.2f}  下身 '
          f'{med(A["r_fit"][lo]*A["cm"][lo]):5.1f}cm')

if OUT:
    np.savez_compressed(OUT, series=np.array(sorted(series.items())), **A)
    print(f'\n已存 {OUT}')
