"""场景重建：把多视角 2D 检测 + 标定内外参转成 3D。

- :mod:`triangulate`：置信度加权的多视角 DLT 三角化（含去畸变、重投影/交会角质量）。
- :mod:`associate`：跨视角实例级球员匹配（几何锚点 + 并查集 + 一致性精化）。
"""
from .associate import AssociationConfig, anchor_2d, match_people
from .ball import triangulate_ball, undistort_ball_center
from .pose_track import PoseTracker
from .track import BallTracker
from .triangulate import (
    MultiViewTriangulator,
    load_camera_rig,
    undistort_keypoints,
)

__all__ = [
    "MultiViewTriangulator",
    "load_camera_rig",
    "undistort_keypoints",
    "AssociationConfig",
    "anchor_2d",
    "match_people",
    "triangulate_ball",
    "undistort_ball_center",
    "BallTracker",
    "PoseTracker",
]
