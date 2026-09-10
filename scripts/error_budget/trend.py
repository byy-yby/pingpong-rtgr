"""误差随时间增长的原因：是置信度掉了、人走远了、还是标定漂了？

逐 10 段看：拟合残差中位、2D 置信度中位、mpp(深度代理)中位、σ(2D 抖动)。
若 σ 不变而残差涨 → 不是 2D 随机噪声；若 conf 掉/mpp 涨 → 是观测条件变差。
"""
import sys

import numpy as np

LOWER25 = [10, 11, 13, 14, 19, 20, 21, 22, 23, 24]


def med(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if a.size else float('nan')


for path in sys.argv[1:]:
    d = np.load(path, allow_pickle=True)
    A = {k: d[k] for k in d.files if k != 'series'}
    fit = A['in_fit'] & (A['j'] < 25)
    is_low = np.isin(A['j'], LOWER25)
    t = A['t']
    t0, t1 = t.min(), t.max()
    print(f'\n===== {path.split("/")[-1]} =====')
    print('  段    帧区间     n     拟合中位  地板中位  conf中位  mpp中位  σ中位  '
          '下身拟合')
    for q in range(10):
        lo = t0 + (t1 - t0) * q / 10
        hi = t0 + (t1 - t0) * (q + 1) / 10
        m = fit & (t >= lo) & (t < hi)
        ml = m & is_low
        if not m.any():
            continue
        print(f'  {q+1:2d}  {int(lo):5d}-{int(hi):5d} {m.sum():6d}  '
              f'{med(A["r_fit"][m]):7.2f}  {med(A["r_in"][m]):7.2f}  '
              f'{med(A["conf"][m]):7.3f}  {med(A["mpp"][m])*1000:6.2f}  '
              f'{med(A["sigma"][m]):5.2f}  {med(A["r_fit"][ml]):7.2f}')
