"""乒乓球 2D 亚像素精修：在粗定位附近，用强度加权质心 + 二阶矩估计球心与半径。

思路：黑白 Mono8 图像里，球是一个与局部背景有高对比的小 blob（亮球或暗球）。
在粗中心 ``(cx, cy)`` 附近抠 ROI，自动判极性（亮/暗），用阈值/极值定出 blob 像素
集合，再对这些像素做**强度加权质心**（天然亚像素）与**二阶矩**（半径），返回
亚像素球心、半径与置信度。

为什么不用霍夫圆：乒乓球小（本系统 12~24px）且高速运动常带模糊，边缘不成正圆，
霍夫累加不稳；而强度加权质心对对称 blob 是亚像素精确的，numpy 一步完成、快且无依赖。

只依赖 numpy + cv2（项目环境无 scipy）。
"""
from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

__all__ = ["refine_ball_center"]


def _connected_centroid(
    mask: np.ndarray,
    cx_roi: float,
    cy_roi: float,
    min_area: int = 3,
) -> Optional[np.ndarray]:
    """在二值 mask 上选「离中心最近且面积达标」的连通域，返回其 (y, x) 像素索引。

    用于丢弃阈值后散落的噪声点 / 反光点，只保留靠近粗中心的那个 blob。
    ``cx_roi/cy_roi`` 是粗中心在 ROI 坐标系下的坐标。
    """
    n_labels, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    best_idx: Optional[np.ndarray] = None
    best_d = float("inf")
    for lab in range(1, n_labels):
        ys, xs = np.where(labels == lab)
        if len(ys) < min_area:
            continue
        ccx = float(xs.mean())
        ccy = float(ys.mean())
        d = (ccx - cx_roi) ** 2 + (ccy - cy_roi) ** 2
        if d < best_d:
            best_d = d
            best_idx = np.stack([ys, xs], axis=1)
    return best_idx


def refine_ball_center(
    gray: np.ndarray,
    cx: float,
    cy: float,
    radius_hint: float = 8.0,
    contrast_floor: float = 8.0,
) -> Tuple[float, float, float, float]:
    """在粗中心 ``(cx, cy)`` 附近精修球心，返回 ``(x, y, radius, conf)``。

    Args:
        gray: 灰度图 (H, W) uint8。
        cx, cy: 粗定位球心（像素，图像坐标系）。
        radius_hint: 期望球半径（像素），决定 ROI 半宽。
        contrast_floor: 判定「确有球」的最小对比度（灰度级），低于此返回 conf=0。

    Returns:
        ``(x, y, radius, conf)``：亚像素球心、估计半径（像素）、置信度 0..1。
        未找到有效 blob 时返回原粗中心、``radius_hint``、``conf=0``。
    """
    if gray is None or gray.size == 0:
        return (float(cx), float(cy), float(radius_hint), 0.0)

    H, W = gray.shape
    pad = max(3, int(round(radius_hint * 2.5)))
    x0 = max(0, int(round(cx)) - pad)
    x1 = min(W, int(round(cx)) + pad + 1)
    y0 = max(0, int(round(cy)) - pad)
    y1 = min(H, int(round(cy)) + pad + 1)
    if x1 <= x0 or y1 <= y0:
        return (float(cx), float(cy), float(radius_hint), 0.0)

    roi = gray[y0:y1, x0:x1].astype(np.float32)
    cx_roi = float(cx) - x0
    cy_roi = float(cy) - y0

    bg = float(np.median(roi))
    peak = float(roi.max())
    valley = float(roi.min())

    # 极性：亮球 / 暗球
    if (peak - bg) >= (bg - valley):
        w = roi - bg
        contrast = peak - bg
    else:
        w = bg - roi
        contrast = bg - valley

    if contrast < contrast_floor:
        return (float(cx), float(cy), float(radius_hint), 0.0)

    wmax = float(w.max())
    if wmax <= 1e-6:
        return (float(cx), float(cy), float(radius_hint), 0.0)

    mask = w >= 0.5 * wmax
    idx = _connected_centroid(mask, cx_roi, cy_roi)
    if idx is None:
        return (float(cx), float(cy), float(radius_hint), 0.0)

    ys = idx[:, 0]
    xs = idx[:, 1]
    wt = w[ys, xs]
    total = float(wt.sum())
    if total <= 1e-9:
        return (float(cx), float(cy), float(radius_hint), 0.0)

    xc = float((xs * wt).sum() / total)
    yc = float((ys * wt).sum() / total)

    # 半径：强度加权二阶矩 → 等效高斯 sigma → radius ≈ 2.5σ（覆盖绝大部分能量）
    varx = float((wt * (xs - xc) ** 2).sum() / total)
    vary = float((wt * (ys - yc) ** 2).sum() / total)
    sigma = float(np.sqrt(max(varx + vary, 0.0) / 2.0))
    radius = float(max(1.0, 2.5 * sigma))

    # 置信度：对比度 SNR → 0..1（snr≈10 即接近满置信，对应清晰高对比 blob）
    noise = float(np.median(np.abs(roi - bg))) + 1e-6
    snr = contrast / noise
    conf = float(np.clip(snr / 10.0, 0.0, 1.0))

    return (xc + x0, yc + y0, radius, conf)
