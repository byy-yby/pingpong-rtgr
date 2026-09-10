"""场景重建：把多视角 2D 检测 + 标定内外参转成 3D。

- :mod:`triangulate`：置信度加权的多视角 DLT 三角化（含去畸变、重投影/交会角质量）。
- :mod:`associate`：跨视角实例级球员匹配（几何锚点 + 并查集 + 一致性精化）。
"""
from .associate import (
    DEFAULT_PERSON_GROUPS,
    AssociationConfig,
    anchor_2d,
    keep_nearest_person,
    match_people,
    match_people_fixed,
)
from .ball import ball_reproj_errors, triangulate_ball, undistort_ball_center
from .easymocap import HALPE26_TO_BODY25, EasymocapReconstructor
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
    "keep_nearest_person",
    "match_people",
    "match_people_fixed",
    "DEFAULT_PERSON_GROUPS",
    "triangulate_ball",
    "ball_reproj_errors",
    "undistort_ball_center",
    "BallTracker",
    "PoseTracker",
    "EasymocapReconstructor",
    "HALPE26_TO_BODY25",
]
