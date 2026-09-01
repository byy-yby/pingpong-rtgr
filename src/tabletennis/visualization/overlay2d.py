"""2D 姿态叠加：把检测出的骨架画到相机图像上（实时预览用）。

输入是灰度图（本机 Mono8），先复制成 BGR 再画，避免污染原始数据。
关键点颜色固定成 17 种，骨骼连线用青绿色，与常见人体姿态可视化一致，
方便和 RTMPose 官方 demo 对齐。
"""
from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np

from ..core.types import Pose2D
from ..vision.skeleton import get_skeleton

# COCO-17 每类关键点的标准颜色（BGR）
_KEYPOINT_COLORS = [
    (203, 192, 255), (0, 0, 255), (0, 255, 255), (0, 255, 0), (255, 0, 0),
    (255, 255, 0), (255, 0, 255), (0, 128, 255), (0, 255, 128), (255, 128, 0),
    (255, 0, 128), (128, 255, 0), (128, 0, 255), (0, 128, 128), (128, 128, 0),
    (0, 0, 128), (128, 0, 0),
]

_EDGE_COLOR = (0, 255, 255)  # BGR 黄色
_TEXT_COLOR = (255, 255, 255)
_CONF_THRESHOLD = 0.3


def gray_to_bgr(image: np.ndarray) -> np.ndarray:
    """灰度图 -> 3 通道 BGR（复制通道）。输入已经是 3 通道则原样返回。"""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def draw_pose(
    image: np.ndarray,
    pose: Pose2D,
    *,
    conf_threshold: float = _CONF_THRESHOLD,
    draw_bbox: bool = False,
    scale: float = 1.0,
    index: Optional[int] = None,
) -> np.ndarray:
    """把单个 :class:`Pose2D` 画到 BGR 图像上（原地返回同一张图）。``scale`` 用于
    在降采样后的图上按比例缩放关键点坐标；``index`` 为该相机内检测序号（画在框
    左上角，排查跨相机匹配错配时用）。"""
    skeleton = get_skeleton(pose.skeleton)
    names = skeleton["names"]
    edges = skeleton["edges"]

    # 骨骼连线
    for a, b in edges:
        ka = pose.keypoint(a)
        kb = pose.keypoint(b)
        if ka[2] < conf_threshold or kb[2] < conf_threshold:
            continue
        cv2.line(
            image,
            (int(ka[0] * scale), int(ka[1] * scale)),
            (int(kb[0] * scale), int(kb[1] * scale)),
            _EDGE_COLOR,
            2,
            cv2.LINE_AA,
        )

    # 关键点
    for i in range(len(pose.keypoints)):
        x, y, c = pose.keypoints[i]
        if c < conf_threshold:
            continue
        color = _KEYPOINT_COLORS[i % len(_KEYPOINT_COLORS)]
        cv2.circle(image, (int(x * scale), int(y * scale)), 4, color, -1, cv2.LINE_AA)

    # 检测框（含该相机内的检测序号，便于跨视角核对是否同一个人）
    if draw_bbox and pose.bbox is not None:
        x1, y1, x2, y2 = [int(v * scale) for v in pose.bbox]
        cv2.rectangle(image, (x1, y1), (x2, y2), _EDGE_COLOR, 1)
        label = f"#{index} {pose.score:.2f}" if index is not None else f"{pose.score:.2f}"
        cv2.putText(image, label, (x1, max(0, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, _TEXT_COLOR, 1, cv2.LINE_AA)

    return image


def annotate_frame(
    frame_image: np.ndarray,
    poses: List[Pose2D],
    *,
    title: Optional[str] = None,
    conf_threshold: float = _CONF_THRESHOLD,
    draw_bbox: bool = False,
) -> np.ndarray:
    """把一帧的所有姿态画到（灰度）图上，返回 BGR 图。

    无姿态时也返回带标题的 BGR 图，便于多路平铺时看清是哪台相机。
    """
    image = gray_to_bgr(frame_image)
    for i, pose in enumerate(poses or []):
        draw_pose(image, pose, conf_threshold=conf_threshold, draw_bbox=draw_bbox, index=i)
    if title:
        cv2.putText(image, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, _TEXT_COLOR, 2, cv2.LINE_AA)
    return image


def tile_images(images: List[np.ndarray], cols: int = 2) -> np.ndarray:
    """把多张（尺寸相同的）BGR 图平铺成一格，缺的地方补黑。

    用于多相机实时预览：4 路 → 2×2。
    """
    if not images:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    images = [gray_to_bgr(im) for im in images]
    h, w = images[0].shape[:2]
    rows = (len(images) + cols - 1) // cols
    canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, im in enumerate(images):
        r, c = divmod(i, cols)
        canvas[r * h:(r + 1) * h, c * w:(c + 1) * w] = im
    return canvas


def draw_ball(
    image: np.ndarray,
    ball,
    *,
    color=(0, 0, 255),
    conf_threshold: float = 0.3,
    scale: float = 1.0,
) -> np.ndarray:
    """把单个球检测结果（:class:`Ball2D`）画到 BGR 图像上（原地返回）。

    ``scale`` 用于在降采样后的图上按比例缩放球心坐标与半径。
    """
    if ball.confidence < conf_threshold:
        return image
    x, y = int(ball.center[0] * scale), int(ball.center[1] * scale)
    r = max(int(ball.radius * scale), 1)
    cv2.circle(image, (x, y), r, color, 2, cv2.LINE_AA)
    return image


def draw_table(
    image: np.ndarray,
    table,
    *,
    color=(0, 255, 0),
    conf_threshold: float = 0.3,
) -> np.ndarray:
    """把单个球桌检测结果（:class:`Table2D`）的 4 个角点画成多边形（原地返回）。"""
    if table.confidence < conf_threshold:
        return image
    pts = table.corners.astype(int).reshape(-1, 1, 2)
    cv2.polylines(image, [pts], True, color, 2, cv2.LINE_AA)
    return image


# ---- 球桌 3D 线框投影（桌面边框 + 4 腿 + 地面框 + 球网） ----
_TABLE_TOP_COLOR = (0, 255, 0)       # 桌面边框：绿色（醒目）
_TABLE_LEG_COLOR = (0, 150, 0)       # 桌腿：深绿
_TABLE_FLOOR_COLOR = (110, 110, 110)  # 地面框：灰
_TABLE_NET_COLOR = (0, 220, 255)     # 球网：青


def draw_table_model(
    image: np.ndarray,
    table,
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    scale: float = 1.0,
) -> np.ndarray:
    """把标准尺寸球桌（:class:`~tabletennis.core.types.Table3D`）的 3D 线框画到 BGR 图上。

    桌面边框（绿）+ 4 条腿（深绿）+ 地面框（灰）+ 球网（青），全部按世界系 ->
    相机系外参 ``(R, t)`` 与内参 ``K/dist`` 投影。任一线段端点落到相机后方
    （z <= 2cm）时跳过该线段，避免畸变投影画错。

    Args:
        image: BGR 图（原地返回同一张）。
        table: :class:`Table3D`。
        R / t: 桌面世界系 -> 相机系外参 ``X_cam = R @ X_world + t``。
        K / dist: 相机内参矩阵 / 畸变系数。
    """
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    rvec, _ = cv2.Rodrigues(R)
    segs = table.segments()

    for group, color, thickness in (
        (segs["top"], _TABLE_TOP_COLOR, 3),
        (segs["legs"], _TABLE_LEG_COLOR, 1),
        (segs["floor"], _TABLE_FLOOR_COLOR, 1),
        (segs["net"], _TABLE_NET_COLOR, 1),
    ):
        seg = np.asarray(group, dtype=np.float64)  # (M, 2, 3)
        a = seg[:, 0, :]
        b = seg[:, 1, :]
        # 相机系 Z，判断端点是否在相机前方
        za = (R @ a.T).T[:, 2] + t[2]
        zb = (R @ b.T).T[:, 2] + t[2]
        keep = (za > 0.02) & (zb > 0.02)
        a, b = a[keep], b[keep]
        if len(a) == 0:
            continue

        proj, _ = cv2.projectPoints(np.vstack([a, b]), rvec, t.reshape(3, 1), K, dist)
        proj = proj.reshape(-1, 2) * scale
        m = len(a)
        for i in range(m):
            cv2.line(
                image,
                tuple(proj[i].astype(int)),
                tuple(proj[m + i].astype(int)),
                color,
                thickness,
                cv2.LINE_AA,
            )
    return image
