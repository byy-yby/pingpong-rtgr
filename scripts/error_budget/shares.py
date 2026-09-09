"""最终误差来源占比（逐样本可加，均值即精确分解）。

每样本：r_fit² = r_pred²(2D白噪声) + r_sys²(地板超出) + r_mod²(SMPL超出地板)
再把 r_sys² 拆成 标定 c(nv)² + 2D关键点系统性偏差。
c(nv) = ArUco 刚性标记在同样 DLT 下的跨视角残差地板（实测）：
  2 视角 0.88px、3 视角 4.52px、4 视角 3.83px
——这是**同口径**的纯标定量（角点亚像素、无姿态模型）。
"""
import sys

import numpy as np

AR_FLOOR = {2: 0.88, 3: 4.52, 4: 3.83}
LOWER25 = [10, 11, 13, 14, 19, 20, 21, 22, 23, 24]
HIP25 = [8, 9, 12]
PART = np.array(['上身'] * 25, dtype=object)
for j in HIP25:
    PART[j] = '髋'
for j in LOWER25:
    PART[j] = '下身'


def med(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if a.size else float('nan')


for path in sys.argv[1:]:
    d = np.load(path, allow_pickle=True)
    A = {k: d[k] for k in d.files if k != 'series'}
    print(f'\n===== {path.split("/")[-1]} =====')
    fit = A['in_fit']
    part = PART[A['j']]
    nv, sig = A['nv'].astype(float), A['sigma']
    rpred = sig * np.sqrt(np.maximum(2 * nv - 3, 0) / np.maximum(2 * nv, 1))
    cm = A['cm']
    cf = np.array([AR_FLOOR.get(int(v), 3.0) for v in A['nv']])
    for p in ('上身', '髋', '下身'):
        m = fit & (part == p) & np.isfinite(A['r_fit']) & np.isfinite(A['r_in']) \
            & np.isfinite(rpred) & np.isfinite(A['cm'])
        if not m.any():
            continue
        rf, rp = A['r_fit'][m], rpred[m]
        ri = A['r_in'][m]
        rs = np.sqrt(np.maximum(ri ** 2 - rp ** 2, 0))
        rm = np.sqrt(np.maximum(rf ** 2 - ri ** 2, 0))
        c = cf[m]
        kp = np.sqrt(np.maximum(rs ** 2 - c ** 2, 0))
        ok = rf > 1.0                   # 丢掉「拟合残差≈0」的退化样本（比值会爆）
        den = rf[ok] ** 2
        sh = np.nanmedian(np.vstack([rp[ok] ** 2 / den, c[ok] ** 2 / den,
                                     kp[ok] ** 2 / den, rm[ok] ** 2 / den]), axis=1)
        print(f'  {p}: n={m.sum():6d}  拟合中位 {med(rf):5.2f}px / {med(rf*cm[m]):4.1f}cm  '
              f'p90 {np.percentile(rf, 90):5.2f}px  视角中位 {med(nv[m]):.0f}')
        print(f'      2D白噪声 {sh[0]*100:5.1f}% | 标定 {sh[1]*100:5.1f}% | '
              f'2D关键点系统性偏差 {sh[2]*100:5.1f}% | SMPL/模型 {sh[3]*100:5.1f}%  '
              f'(和 {sh.sum()*100:.1f}%)')
        print(f'      → 2D 关键点合计 {100*(sh[0]+sh[2]):.1f}%  '
              f'标定 {100*sh[1]:.1f}%  模型 {100*sh[3]:.1f}%')
