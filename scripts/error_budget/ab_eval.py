"""A/B 评估：用**固定观测集**比较两份重建结果的重投影误差。

为什么要专门写：`reconstruct_video.py` 报的「重投影误差中位」口径是「conf>0 的关节」，
抬高 ``--fit-conf`` / 给低置信度关节降权后，这些关节会掉出分母 → 指标自己变好，
是**假提升**（CLAUDE.md 已记过这个坑）。本脚本把评估集固定在**原始 2D 观测**上，
两份重建用同一批 (帧, 人, 视角, 关节) 比，才是真比较。

口径：
    fit   —— 只算该人**参与拟合**的视角（``recon_meta.person_groups``，对齐现有指标）
    outfit—— 该人有观测但**未参与拟合**的视角（真·留出视角，衡量泛化）
    all   —— 两者合并
每个口径都按两个置信度门槛各算一次：
    conf>0    —— 全部观测（含姿态模型外推的垃圾关节，最保守）
    conf>0.5  —— 可信 2D（实测这一档 2D 自身误差 2~6cm）

用法：
    python ab_eval.py <recon_dir> [--poses <pose2d.json>] [--conf-floor 0.5]
                      [--label 名字] [--json out.json]

``<recon_dir>`` 需含 ``frame_*.npz``；``--poses`` 缺省取 ``<recon_dir>/pose2d.json``
（``--load-pass1`` 重跑出的目录不写 pose2d.json，要显式指向基线目录那份）。
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, 'src')
from tabletennis.reconstruction.easymocap import _pose_to_body25_view  # noqa: E402
from tabletennis.reconstruction.triangulate import load_camera_rig  # noqa: E402

MAX_ASSIGN_PX = 60.0        # 2D 归人到谁：重投影均值超过它就不计入（明显是别人）


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('recon_dir')
    ap.add_argument('--poses', default=None)
    ap.add_argument('--meta', default=None)
    ap.add_argument('--conf-floor', type=float, default=0.5)
    ap.add_argument('--label', default=None)
    ap.add_argument('--json', default=None)
    a = ap.parse_args()

    R = a.recon_dir.rstrip('/')
    label = a.label or os.path.basename(R)
    poses = json.load(open(a.poses or os.path.join(R, 'pose2d.json'), encoding='utf-8'))
    meta_p = a.meta or os.path.join(R, 'recon_meta.json')
    fit_views = {}
    if os.path.exists(meta_p):
        m = json.load(open(meta_p, encoding='utf-8'))
        for i, g in enumerate(m.get('person_groups') or []):
            fit_views[i] = sorted(g)
    intr, ext = load_camera_rig(None)
    P = {c: intr[c].K @ np.hstack([ext[c].R, ext[c].t.reshape(3, 1)])
         for c in ext if c in intr}

    def proj(cid, X):
        h = np.hstack([X, np.ones((len(X), 1))]) @ P[cid].T
        return h[:, :2] / h[:, 2:3]

    # (mode, pid, floor) -> [逐帧每人均值]
    acc = defaultdict(list)
    n_obs = defaultdict(int)
    for fp in sorted(glob.glob(os.path.join(R, 'frame_*.npz'))):
        t = int(os.path.basename(fp).split('_')[1].split('.')[0])
        pd = poses.get(str(t))
        if not pd:
            continue
        d = np.load(fp, allow_pickle=True)
        j3 = {}
        for pid, suf in ((0, ''), (1, '_1')):
            k = 'joints_body25' + suf
            if k in d.files:
                X = np.asarray(d[k], np.float64)
                if np.isfinite(X).all():
                    j3[pid] = X
        if not j3:
            continue
        for cid_s, plist in pd.items():
            cid = int(cid_s)
            if cid not in intr:
                continue
            for p in plist:
                kp26 = np.asarray(p['kpts'], np.float64)
                v = _pose_to_body25_view(kp26, intr[cid].K, intr[cid].dist, 0.0)
                if v is None:
                    continue
                best, best_e = None, 1e18
                for pid, X in j3.items():
                    e = np.linalg.norm(proj(cid, X) - v[:, :2], axis=1)
                    e = np.where(v[:, 2] > 0, e, np.nan)
                    mm = np.nanmean(e)
                    if np.isfinite(mm) and mm < best_e:
                        best_e, best = mm, pid
                if best is None or best_e > MAX_ASSIGN_PX:
                    continue
                mode = 'fit' if cid in fit_views.get(best, []) else 'outfit'
                err = np.linalg.norm(proj(cid, j3[best]) - v[:, :2], axis=1)
                for fl in (0.0, a.conf_floor):
                    m = (v[:, 2] > fl) & np.isfinite(err)
                    if m.any():
                        acc[(mode, best, fl)].append(float(np.mean(err[m])))
                        n_obs[(mode, best, fl)] += int(m.sum())

    out = {'label': label, 'dir': R, 'conf_floor': a.conf_floor, 'metrics': {}}
    print(f"===== {label}  ({R}) =====")
    for fl in (0.0, a.conf_floor):
        for mode in ('fit', 'outfit'):
            for pid in (0, 1, -1):
                pids = (0, 1) if pid == -1 else (pid,)
                vals = [v for p in pids for v in acc.get((mode, p, fl), [])]
                cnt = sum(n_obs.get((mode, p, fl), 0) for p in pids)
                if not vals:
                    continue
                name = 'p0' if pid == 0 else ('p1' if pid == 1 else '两人合计')
                med = float(np.median(vals))
                key = f'{mode}_conf{fl:g}_{name}'
                out['metrics'][key] = {'median_px': med, 'n_frames': len(vals),
                                       'n_obs': cnt}
                print(f"  {mode:6s} conf>{fl:<4g} {name}: 中位 {med:6.2f}px  "
                      f"p90 {np.percentile(vals, 90):6.2f}px  "
                      f"n帧={len(vals):5d} n观测={cnt}")
    if a.json:
        json.dump(out, open(a.json, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        print(f"  → {a.json}")


if __name__ == '__main__':
    main()
