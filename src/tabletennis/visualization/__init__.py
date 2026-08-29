"""可视化模块。

2D 叠加：``overlay2d``（姿态骨架 / 球 / 球桌边框，多路平铺）。
3D 场景：``viewer3d``（Open3D，相机视锥 + 标准尺寸球桌），按需 import。
"""
from .overlay2d import (
    annotate_frame,
    draw_ball,
    draw_pose,
    draw_table,
    draw_table_model,
    gray_to_bgr,
    tile_images,
)

__all__ = [
    "annotate_frame",
    "draw_ball",
    "draw_pose",
    "draw_table",
    "draw_table_model",
    "gray_to_bgr",
    "tile_images",
]
