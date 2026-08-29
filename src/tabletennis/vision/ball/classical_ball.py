"""经典球检测器（无训练）：背景减除 + 帧差 + 尺寸先验 → 亚像素精修。

适用前提（受控场景，与计划 Phase 0 一致）：
- 相机固定，桌面 / 地面 / 背景静止；
- 球是画面里唯一「小而快」的高对比 blob（球员大、拍 / 手更大）；
- 曝光足够短（≤100µs），球不成拖影（否则 blob 被拉长，尺寸先验会把它滤掉）。

流程（每帧）：
1. 运行均值维护背景（静态场景）+ 相邻帧差（球最快，捕捉短暂近乎静止的球），
   两路运动信号取并集 → 运动 mask。
2. 尺寸先验：连通域面积 ∈ [π·r_min², π·r_max²]（球 12~24px → r 5~15px）。
3. 候选 blob 质心作粗中心，喂 :func:`~tabletennis.vision.ball.refine.refine_ball_center`
   得亚像素球心，产出 :class:`~tabletennis.core.types.Ball2D`。

只依赖 numpy + cv2。相比 YOLO 路线，本实现零标注、CPU 即可，且直接出亚像素质心。
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np

from ...core.types import Ball2D, Frame
from ..detector import BallDetector
from .refine import refine_ball_center

__all__ = ["ClassicalBallDetector"]


class ClassicalBallDetector(BallDetector):
    """经典乒乓球检测器（有状态：按 camera_id 跨帧维护背景与上一帧，可跨相机复用）。"""

    def __init__(
        self,
        radius_px: Tuple[float, float] = (5.0, 15.0),
        diff_thresh: float = 25.0,
        bg_alpha: float = 0.02,
        min_contrast: float = 8.0,
        min_area: Optional[float] = None,
        max_area: Optional[float] = None,
    ) -> None:
        """
        Args:
            radius_px: 球半径像素范围 ``(r_min, r_max)``，用于尺寸先验与 ROI 半宽。
            diff_thresh: 前景 / 帧差的灰度差阈值。
            bg_alpha: 背景运行均值更新系数（越小越稳定，越大越适应光照变化）。
            min_contrast: 传给精修的对比度下限（灰度级），低于此判定无球。
            min_area / max_area: 连通域面积过滤（像素），默认由 radius_px 推导。
        """
        self.radius_min, self.radius_max = radius_px
        self.diff_thresh = float(diff_thresh)
        self.bg_alpha = float(bg_alpha)
        self.min_contrast = float(min_contrast)
        self.min_area = min_area if min_area is not None else np.pi * self.radius_min ** 2
        self.max_area = max_area if max_area is not None else np.pi * self.radius_max ** 2

        self._bg: Dict[int, np.ndarray] = {}        # cam_id -> float32 运行均值背景
        self._prev: Dict[int, np.ndarray] = {}      # cam_id -> uint8 上一帧
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    def detect(self, frame: Frame) -> List[Ball2D]:
        gray = frame.image
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
        if gray.dtype != np.uint8:
            gray = gray.astype(np.uint8)

        cid = frame.camera_id
        # 该相机首帧只做初始化，等待背景建立
        bg = self._bg.get(cid)
        if bg is None or bg.shape != gray.shape:
            self._bg[cid] = gray.astype(np.float32)
            self._prev[cid] = gray.copy()
            return []

        cv2.accumulateWeighted(gray, self._bg[cid], self.bg_alpha)
        bg_img = self._bg[cid].astype(np.uint8)

        # 两路运动信号取并集：背景减除（静态背景下的新物体）+ 帧差（快球）
        fg = cv2.absdiff(gray, bg_img)
        fd = cv2.absdiff(gray, self._prev[cid])
        motion = ((fg > self.diff_thresh) | (fd > self.diff_thresh)).astype(np.uint8) * 255
        self._prev[cid] = gray.copy()

        motion = cv2.morphologyEx(motion, cv2.MORPH_OPEN, self._kernel)

        contours, _ = cv2.findContours(motion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        balls: List[Ball2D] = []
        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < self.min_area or area > self.max_area:
                continue
            m = cv2.moments(cnt)
            if m["m00"] <= 0:
                continue
            cx = float(m["m10"] / m["m00"])
            cy = float(m["m01"] / m["m00"])
            radius_hint = float(np.sqrt(area / np.pi))
            x, y, r, conf = refine_ball_center(
                gray, cx, cy, radius_hint, self.min_contrast
            )
            if conf <= 0.0:
                continue
            balls.append(
                Ball2D(
                    camera_id=frame.camera_id,
                    center=np.array([x, y], dtype=np.float32),
                    radius=float(r),
                    confidence=float(conf),
                )
            )
        return balls

    def reset(self) -> None:
        """清空各相机背景与上一帧，重新预热（换场景 / 相机挪动后调用）。"""
        self._bg.clear()
        self._prev.clear()
