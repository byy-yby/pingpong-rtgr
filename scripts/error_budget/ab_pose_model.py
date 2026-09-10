"""换 2D 模型后的**公平**比较：两个模型在同一批「公共 2D 观测」上互相打分。

为什么不能直接比各自的「重投影误差中位」：换模型 = 换 2D 观测集，分母都变了，
数字不可比（CLAUDE.md 记的「假提升」坑的另一个面）。本脚本做两件事：

1. **公共观测集**：只保留「A、B 两个模型都给出该关节且 conf ≥ 阈值」的
   (帧, 人, 视角, 关节)。同一批点，分别用 A 的 3D 和 B 的 3D 算重投影误差 → 可比。
2. **2D 互差**：同一批点上两模型 2D 坐标的距离（px）。它衡量「换模型到底改了多少」，
   配合各自的时序抖动（`errbudget.py` 口径）能看出是修好了还是只是挪了位置。

分部位统计（上身/下身/脚）——换更强模型的主要假设收益就在下半身与脚。

用法：
    python ab_pose_model.py --a <reconA> --b <reconB> [--conf-floor 0.5]
                            [--poses-a PATH] [--poses-b PATH] [--json out.json]

``--a/--b`` 需含 ``frame_*.npz``；``pose2d.json`` 缺省取各自目录里的。
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, 'src')
from tabletennis.reconstruction.easymocap import (  # noqa: E402
    LOWER_BODY_BODY25,
    _pose_to_body25_view,
)
from tabletennis.reconstruction.triangulate import load_camera_rig  # noqa: E402

MAX_ASSIGN_PX = 60.0        # 2D 归人：重投影均值超过它就不计入（明显是别人）
FEET_BODY25 = (20, 21, 22, 23, 24)      # LBigToe..RHeel
UPPER_BODY25 = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14)


def _load_poses(path, recon_dir):
    p = path or os.path.join(recon_dir, 'pose2d.json')
    if not os.path.exists(p):
        raise SystemExit(f"缺 pose2d.json：{p}")
    return json.load(open(p, encoding='utf-8'))


def _load_j3(recon_dir, t):
    fp = os.path.join(recon_dir, f'frame_{t:06d}.npz')
    if not os.path.exists(fp):
        return {}
    d = np.load(fp, allow_pickle=True)
    out = {}
    for pid, suf in ((0, ''), (1, '_1')):
        k = 'joints_body25' + suf
        if k in d.files:
            X = np.asarray(d[k], np.float64)
            if np.isfinite(X).all():
                out[pid] = X
    return out


def _assign(cid, v, j3, P):
    """把某相机某帧的一条 2D 观测归到某个人（用该模型自己的 3D 重投影最近者）。"""
    best, best_e = None, 1e18
    for pid, X in j3.items():
        h = np.hstack([X, np.ones((len(X), 1))]) @ P[cid].T
        uv = h[:, :2] / h[:, 2:3]
        e = np.linalg.norm(uv - v[:, :2], axis=1)
        e = np.where(v[:, 2] > 0, e, np.nan)
        mm = np.nanmean(e)
        if np.isfinite(mm) and mm < best_e:
            best_e, best = mm, pid
    return (best, best_e) if best is not None and best_e <= MAX_ASSIGN_PX else (None, None)


def _proj(cid, X, P):
    h = np.hstack([X, np.ones((len(X), 1))]) @ P[cid].T
    return h[:, :2] / h[:, 2:3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True, help='重建目录 A（基线模型）')
    ap.add_argument('--b', required=True, help='重建目录 B（新模型）')
    ap.add_argument('--poses-a', default=None)
    ap.add_argument('--poses-b', default=None)
    ap.add_argument('--conf-floor', type=float, default=0.5)
    ap.add_argument('--json', default=None)
    a = ap.parse_args()

    A, B = a.a.rstrip('/'), a.b.rstrip('/')
    pa, pb = _load_poses(a.poses_a, A), _load_poses(a.poses_b, B)
    intr, ext = load_camera_rig(None)
    P = {c: intr[c].K @ np.hstack([ext[c].R, ext[c].t.reshape(3, 1)])
         for c in ext if c in intr}

    # (部位, 模型) -> [逐观测像素误差]
    err = defaultdict(list)
    diff = defaultdict(list)          # 2D 互差
    n_common = 0

    common_t = sorted(set(pa) & set(pb))
    for t_s in common_t:
        t = int(t_s)
        j3a, j3b = _load_j3(A, t), _load_j3(B, t)
        if not j3a or not j3b:
            continue
        # 逐相机把 A/B 的 2D 各自归人（用各自的 3D），再取「同一个人」的配对
        by = defaultdict(dict)        # (cid, pid) -> {'a': (v, kp), 'b': (...)}
        for tag, poses, j3 in (('a', pa, j3a), ('b', pb, j3b)):
            for cid_s, plist in poses[t_s].items():
                cid = int(cid_s)
                if cid not in intr:
                    continue
                for p in plist:
                    kp = np.asarray(p['kpts'], np.float64)
                    v = _pose_to_body25_view(kp, intr[cid].K, intr[cid].dist, 0.0)
                    if v is None:
                        continue
                    pid, _ = _assign(cid, v, j3, P)
                    if pid is not None:
                        by[(cid, pid)][tag] = (v, kp)
        for (cid, pid), d in by.items():
            if 'a' not in d or 'b' not in d:
                continue
            va, kpa = d['a']
            vb, kpb = d['b']
            if pid not in j3a or pid not in j3b:
                continue
            ea = np.linalg.norm(_proj(cid, j3a[pid], P) - va[:, :2], axis=1)
            eb = np.linalg.norm(_proj(cid, j3b[pid], P) - vb[:, :2], axis=1)
            dd = np.linalg.norm(va[:, :2] - vb[:, :2], axis=1)
            m = (va[:, 2] > a.conf_floor) & (vb[:, 2] > a.conf_floor) \
                & np.isfinite(ea) & np.isfinite(eb)
            for j in np.where(m)[0]:
                n_common += 1
                parts = []
                if j in LOWER_BODY_BODY25:
                    parts.append('下身')
                else:
                    parts.append('上身')
                if j in FEET_BODY25:
                    parts.append('脚')
                parts.append('全部')
                for pn in parts:
                    err[(pn, 'a')].append(float(ea[j]))
                    err[(pn, 'b')].append(float(eb[j]))
                    diff[pn].append(float(dd[j]))

    out = {'a': A, 'b': B, 'conf_floor': a.conf_floor, 'n_common_obs': n_common,
           'metrics': {}}
    print(f"===== A={os.path.basename(A)}  vs  B={os.path.basename(B)} =====")
    print(f"公共观测（两模型都 conf>{a.conf_floor:g} 的关节）：{n_common} 个")
    print(f"{'部位':<6}{'A 中位':>10}{'B 中位':>10}{'变化':>10}{'2D互差中位':>12}{'n':>9}")
    for pn in ('全部', '上身', '下身', '脚'):
        ea, eb, dd = err.get((pn, 'a')), err.get((pn, 'b')), diff.get(pn)
        if not ea:
            continue
        ma, mb = float(np.median(ea)), float(np.median(eb))
        md = float(np.median(dd))
        out['metrics'][pn] = {'a_median_px': ma, 'b_median_px': mb,
                              'delta_px': mb - ma, 'd2d_median_px': md,
                              'n_obs': len(ea)}
        print(f"{pn:<6}{ma:>10.2f}{mb:>10.2f}{mb - ma:>+10.2f}{md:>12.2f}{len(ea):>9}")
    print("（变化为负 = B 更好；A/B 用的是同一批点，可直接比）")
    if a.json:
        json.dump(out, open(a.json, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        print(f"  → {a.json}")


if __name__ == '__main__':
    main()
