"""重投影误差的**分布**与**两种口径**（不是只报一个中位数）。

**姿态**与**球**都测：姿态读 ``recon_index.npz``，球读 ``ball_trajectory.npz``
（没有就跳过，提示用 ``reconstruct_video.py <session> --ball-only --out <本目录>`` 补跑）。
两者的「误差」含义**不一样**，别混：
  - 姿态：SMPL **拟合**出的 3D 关节投回 2D —— 误差里含**模型拟合的偏差**。
  - 球：3D 由各视角 2D **直接加权 DLT** 得到，没有拟合步骤 —— 误差就是**三角化自身
    的残差**（跨视角一致性），**不含「和真值差多少」**。球的真值得靠刚性靶标那类外部参照。
  球的数还要按**参与视角数**拆开看：2 视角时 DLT 能把两条根本不相交的射线"解"出一个点，
  中位数反而比 4 视角更小（161147 实测 2 视角中位 1.67px 但平均 927px）。


数据源 = 重建目录的 ``recon_index.npz``（``err_mean_px`` / ``err_best_px``，逐帧）
+ ``frame_*.npz``（只用来取根关节位置算速度）+ ``recon_meta.json``（真实帧周期）。
**故意不复算 2D**：``pose2d.json`` 存的是**未掩码**的原始姿态，拿它复算会把下半身门
掩掉的外推关节算进来（实测 161147 会从 11.54px 虚高到 15.2px），口径就和脚本对不上了。

两种口径（都只统计 conf>0 的观测，见 ``reconstruct_video.py::_proj_err_multi``）：
    all  —— 所有观测视角**等权**取均值（拟合的目标口径）
    best —— **每个关节只取置信度最高的那台相机**再取均值。置信度高的 2D 更接近真值
            （实测 conf 0.9-1.0 → 2.2cm、0.3-0.5 → 14.9cm，见 error_budget_report §1），
            所以这个口径更接近「和 Ground Truth 比」。

回答三个问题：
    1. 两种口径各自的 中位/平均/p90/p99/max？平均值 > 中位数就说明有长尾。
    2. 误差在**帧之间**怎么分布？开头几秒、结尾几秒是不是明显更差？
    3. 只取「动作稳定」的帧，误差是多少？（用**根关节速度**当客观判据，不靠肉眼挑段）

用法：
    python reproj_stats.py <recon_dir> [--head-s 2] [--tail-s 2] [--json out.json]
"""
import argparse
import glob
import json
import os

import numpy as np

ROOT_IDX = 19       # body25 骨盆


def _pct(x, q):
    return float(np.percentile(x, q)) if len(x) else float('nan')


def _line(name, x, unit='px'):
    if not len(x):
        return f"  {name:<22} （无数据）"
    return (f"  {name:<22} n={len(x):5d}  中位 {np.median(x):6.2f}  "
            f"平均 {np.mean(x):6.2f}  p90 {_pct(x, 90):6.2f}  "
            f"p99 {_pct(x, 99):6.2f}  max {np.max(x):7.2f} {unit}")


def _brief(aa, bb):
    """一行式：all 中位/平均 + best 中位/平均。"""
    bb = bb[np.isfinite(bb)]
    bm = f'{np.median(bb):6.2f}' if len(bb) else '   n/a'
    be = f'{np.mean(bb):6.2f}' if len(bb) else '   n/a'
    return (f"all 中位 {np.median(aa):6.2f} / 平均 {np.mean(aa):6.2f}"
            f"   best 中位 {bm} / 平均 {be} px")


def _deciles(rf, allv, bestv, period):
    """按帧序 10 等分的表格（两种口径的中位/平均）+ 结构化结果。"""
    print(f"  {'段位':<9}{'帧区间':>15}{'秒区间':>15}"
          f"{'all 中位':>10}{'all 平均':>10}{'best 中位':>11}{'best 平均':>11}")
    out = []
    for i, seg in enumerate(np.array_split(np.arange(len(rf)), 10)):
        if not len(seg):
            continue
        aa, bb = allv[seg], bestv[seg][np.isfinite(bestv[seg])]
        f0, f1 = int(rf[seg[0]]), int(rf[seg[-1]])
        s0 = (f0 - int(rf[0])) * period
        s1 = (f1 - int(rf[0])) * period
        bm = float(np.median(bb)) if len(bb) else float('nan')
        be = float(np.mean(bb)) if len(bb) else float('nan')
        out.append({'seg': f'{i * 10}-{(i + 1) * 10}%', 'frames': [f0, f1],
                    'seconds': [round(s0, 2), round(s1, 2)],
                    'all_median': float(np.median(aa)), 'all_mean': float(np.mean(aa)),
                    'best_median': bm, 'best_mean': be})
        print(f"  {f'{i * 10}-{(i + 1) * 10}%':<9}{f'{f0}-{f1}':>15}"
              f"{f'{s0:.1f}-{s1:.1f}s':>15}"
              f"{np.median(aa):>10.2f}{np.mean(aa):>10.2f}{bm:>11.2f}{be:>11.2f}")
    return out


def _head_tail(rf, allv, bestv, period, head_s, tail_s):
    """首尾 vs 中间；帧数太少时跳过。返回结构化结果或 None。"""
    n_head = int(round(head_s / period))
    n_tail = int(round(tail_s / period))
    if not (len(rf) > 3 * max(n_head, n_tail) and n_head > 0):
        return None
    print(f'【首尾 vs 中间】（头 {head_s:g}s ≈ {n_head} 帧、尾 {tail_s:g}s ≈ {n_tail} 帧）')
    res = {'head_frames': n_head, 'tail_frames': n_tail}
    for nm, sl in (('开头', slice(0, n_head)),
                   ('中间', slice(n_head, len(rf) - n_tail)),
                   ('结尾', slice(len(rf) - n_tail, len(rf)))):
        print(f'  {nm:<5} n={len(allv[sl]):5d}   {_brief(allv[sl], bestv[sl])}')
        res[nm] = {'n': int(len(allv[sl])),
                   'all_median': float(np.median(allv[sl])),
                   'all_mean': float(np.mean(allv[sl])),
                   'best_median': float(np.median(bestv[sl][np.isfinite(bestv[sl])]))
                   if np.isfinite(bestv[sl]).any() else float('nan')}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('recon_dir')
    ap.add_argument('--head-s', type=float, default=2.0, help='头部多少秒算「开头」')
    ap.add_argument('--tail-s', type=float, default=2.0, help='尾部多少秒算「结尾」')
    ap.add_argument('--label', default=None)
    ap.add_argument('--json', default=None)
    a = ap.parse_args()

    R = a.recon_dir.rstrip('/')
    label = a.label or os.path.basename(R)
    idx_p = os.path.join(R, 'recon_index.npz')
    if not os.path.exists(idx_p):
        print(f'✗ 找不到 {idx_p}')
        return
    z = np.load(idx_p, allow_pickle=True)
    if 'err_best_px' not in z.files:
        print('✗ recon_index.npz 里没有 err_best_px —— 这是「最高置信视角」口径，'
              '本次代码改动后才开始写。请用新版 reconstruct_video.py 重跑该 session。')
        return
    rf = np.asarray(z['ref_frame'], np.int64)
    ok = np.asarray(z['status'], np.int8) == 0
    allv = np.asarray(z['err_mean_px'], np.float64)
    bestv = np.asarray(z['err_best_px'], np.float64)
    rf, allv, bestv = rf[ok], allv[ok], bestv[ok]     # 只留有人的帧

    meta = {}
    mp = os.path.join(R, 'recon_meta.json')
    if os.path.exists(mp):
        meta = json.load(open(mp, encoding='utf-8'))
    period = float('nan')
    src = meta.get('source') or {}
    cams = src.get('cams') or {}
    if cams:
        period = float(np.median([c['period_s'] for c in cams.values()]))
    if not np.isfinite(period) or period <= 0:
        period = 0.01
        print('  ⚠ recon_meta 无真实帧周期，按 100fps 假定')
    fps = 1.0 / period

    # 根关节位置（逐帧）→ 速度分箱
    root = {}
    for fp in glob.glob(os.path.join(R, 'frame_*.npz')):
        t = int(os.path.basename(fp).split('_')[1].split('.')[0])
        d = np.load(fp, allow_pickle=True)
        ps = []
        for suf in ('', '_1'):
            k = 'joints_body25' + suf
            if k in d.files:
                X = np.asarray(d[k], np.float64)
                if np.isfinite(X).all():
                    ps.append(X[ROOT_IDX])
        if ps:
            root[t] = np.mean(ps, axis=0)
    speed = np.full(len(rf), np.nan)
    for i in range(len(rf)):
        if rf[i] in root and (rf[i] - 1) in root:
            speed[i] = float(np.linalg.norm(root[rf[i]] - root[rf[i] - 1])) * 100.0

    print(f'===== {label}  ({R}) =====')
    print(f'  有人帧 {len(rf)}（主时钟 {int(rf[0])}..{int(rf[-1])}），'
          f'帧周期 {period * 1000:.2f}ms（≈{fps:.1f}fps），'
          f'时长 ≈{(rf[-1] - rf[0] + 1) * period:.1f}s')

    print()
    print('【总体】每帧一个数，再对帧聚合：')
    print(_line('全部视角(all)', allv))
    print(_line('最高置信视角(best)', bestv))
    print(f'  平均/中位 = {np.mean(allv) / np.median(allv):.2f}（all）、'
          f'{np.mean(bestv) / np.median(bestv):.2f}（best）—— >1 即长尾')

    # ---- 沿时间轴 10 等分 ----
    print()
    print('【沿时间轴】按帧序 10 等分：')
    dec = _deciles(rf, allv, bestv, period)

    # ---- 首尾 vs 中间 ----
    t_rel = (rf - rf[0]) * period
    total_s = float(t_rel[-1]) if len(t_rel) else 0.0
    n_head = int(round(a.head_s / period))
    n_tail = int(round(a.tail_s / period))
    print()
    head_tail = _head_tail(rf, allv, bestv, period, a.head_s, a.tail_s)
    if head_tail:
        print()
        print(f'  全程 {total_s:.1f}s 里的中位数：开头 {np.median(allv[:n_head]):.2f} / '
              f'中间 {np.median(allv[n_head:len(rf) - n_tail]):.2f} / '
              f'结尾 {np.median(allv[len(rf) - n_tail:]):.2f} px（all 口径）')

        # 去掉首尾之后，误差最高的若干帧落在哪
        mid = slice(n_head, len(rf) - n_tail)
        order = np.argsort(-allv[mid])
        mid_rf = rf[mid]
        print(f'  去掉首尾后最差的 5 帧（all 口径，主时钟帧号 / 秒 / all / best）：')
        for j in order[:5]:
            k = mid_rf[j]
            print(f'    frame {int(k):5d}  t={(k - rf[0]) * period:6.2f}s   '
                  f'all {allv[mid][j]:6.2f}  best {bestv[mid][j]:6.2f} px')

    # ---- 按「动作稳定度」分箱 ----
    print()
    print('【按动作稳定度】根关节（骨盆）逐帧位移分箱（= 动作有多剧烈）：')
    edges = [0, 0.2, 0.5, 1.0, 2.0, 5.0, np.inf]
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = np.isfinite(speed) & (speed >= lo) & (speed < hi)
        if not m.any():
            continue
        tag = f'{lo:g}-{hi:g}' if np.isfinite(hi) else f'>{lo:g}'
        bb = bestv[m][np.isfinite(bestv[m])]
        bins.append({'speed_cm_per_frame': tag, 'n': int(m.sum()),
                     'all_median': float(np.median(allv[m])),
                     'all_mean': float(np.mean(allv[m])),
                     'best_median': float(np.median(bb)) if len(bb) else float('nan'),
                     'best_mean': float(np.mean(bb)) if len(bb) else float('nan')})
        print(f'  {tag + " cm/帧":<14} n={int(m.sum()):5d}   {_brief(allv[m], bestv[m])}')
    stable = np.isfinite(speed) & (speed < 0.5)
    if stable.any():
        print(f'  → 只取「稳定帧」（根位移 <0.5cm/帧，占 {stable.sum() / len(rf) * 100:.0f}%，'
              f'n={int(stable.sum())}）：{_brief(allv[stable], bestv[stable])}')

    # ---- 球：3D 球心由各视角 2D 检测直接加权 DLT 得到，误差 = 三角化自身残差 ----
    ball_res = None
    bp = os.path.join(R, 'ball_trajectory.npz')
    print()
    if not os.path.exists(bp):
        print('【球】没有 ball_trajectory.npz（本次重建带了 --no-ball？'
              '可只补跑球：`reconstruct_video.py <session> --ball-only --out <本目录>`）')
    else:
        bz = np.load(bp, allow_pickle=True)
        if 'reproj_err_all' not in bz.files:
            print('【球】ball_trajectory.npz 里没有 reproj_err_all/reproj_err_best —— '
                  '这两个键是本次改动才开始写的，请用 --ball-only 重跑该 session。')
        else:
            brf = np.asarray(bz['ref_frame'], np.int64)
            ba = np.asarray(bz['reproj_err_all'], np.float64)
            bb = np.asarray(bz['reproj_err_best'], np.float64)
            tot = int(brf.size)
            m = np.isfinite(ba)
            brf, ba, bb = brf[m], ba[m], bb[m]
            print(f'【球】3D 三角化成功 {len(brf)}/{tot} 帧'
                  f'（成功率 {len(brf) / max(1, tot) * 100:.0f}%）；'
                  f'误差 = 三角化自身残差（跨视角一致性），不是和真值比')
            print(_line('全部视角(all)', ba))
            print(_line('最高置信视角(best)', bb))
            print(f'  平均/中位 = {np.mean(ba) / np.median(ba):.2f}（all）、'
                  f'{np.mean(bb) / np.median(bb):.2f}（best）—— >1 即长尾')

            # 按「参与三角化的视角数」拆开：**读球误差必须先看这张表**
            nvb = np.asarray(bz['n_views'], np.int32)[np.isfinite(
                np.asarray(bz['reproj_err_all'], np.float64))]
            print()
            print('【球·按参与视角数】2 视角是**欠约束**的：DLT 能把两条射线"解"出一个点，'
                  '哪怕它们根本不相交 ⇒ 中位数反而更小，别看错')
            vbins = []
            for v in np.unique(nvb):
                s = nvb == v
                vbins.append({'n_views': int(v), 'n': int(s.sum()),
                              'all_median': float(np.median(ba[s])),
                              'all_mean': float(np.mean(ba[s])),
                              'all_max': float(np.max(ba[s])),
                              'best_median': float(np.median(bb[s])) if np.isfinite(bb[s]).any()
                              else float('nan')})
                print(f'  {int(v)} 视角  n={int(s.sum()):5d} ({s.mean() * 100:4.0f}%)   '
                      f'all 中位 {np.median(ba[s]):7.2f} 平均 {np.mean(ba[s]):10.2f} '
                      f'max {np.max(ba[s]):10.1f}   |  best 中位 {np.median(bb[s]):7.2f}')
            s3 = nvb >= 3
            if s3.any():
                print(f'  → 只取 ≥3 视角（有冗余，n={int(s3.sum())}）：'
                      f'all 中位 {np.median(ba[s3]):.2f} 平均 {np.mean(ba[s3]):.2f} '
                      f'p99 {_pct(ba[s3], 99):.2f} max {np.max(ba[s3]):.1f} px')
            print()
            print('【球的沿时间轴】按帧序 10 等分：')
            bdec = _deciles(brf, ba, bb, period)
            print()
            bht = _head_tail(brf, ba, bb, period, a.head_s, a.tail_s)
            ball_res = {
                'n_ok': int(len(brf)), 'n_frames': tot,
                'overall': {
                    'all': {'median': float(np.median(ba)), 'mean': float(np.mean(ba)),
                            'p90': _pct(ba, 90), 'p99': _pct(ba, 99),
                            'max': float(np.max(ba))},
                    'best': {'median': float(np.median(bb)), 'mean': float(np.mean(bb)),
                             'p90': _pct(bb, 90), 'p99': _pct(bb, 99),
                             'max': float(np.max(bb))}},
                'deciles': bdec, 'head_tail': bht, 'by_n_views': vbins}

    if a.json:
        json.dump({'label': label, 'dir': R, 'n_frames': int(len(rf)),
                   'period_s': period,
                   'overall': {
                       'all': {'median': float(np.median(allv)), 'mean': float(np.mean(allv)),
                               'p90': _pct(allv, 90), 'p99': _pct(allv, 99),
                               'max': float(np.max(allv))},
                       'best': {'median': float(np.median(bestv)),
                                'mean': float(np.mean(bestv)),
                                'p90': _pct(bestv, 90), 'p99': _pct(bestv, 99),
                                'max': float(np.max(bestv))}},
                   'deciles': dec, 'speed_bins': bins,
                   'head_tail': head_tail, 'ball': ball_res},
                  open(a.json, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        print(f'  → {a.json}')


if __name__ == '__main__':
    main()
