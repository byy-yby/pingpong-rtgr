"""相机标定模块：内参标定（棋盘格 / ChArUco）+ 外参标定（ChArUco 板）。

对外提供：

- :mod:`intrinsics`：内参标定（棋盘格 + ChArUco 两种板）/ 精度校验 / 结果读写；
  GUI 入口见 ``scripts/calibrate_intrinsics.py``。
- :mod:`extrinsics`：ChArUco 检测 / 板位姿（外参）求解 / 多帧平均 / 结果读写；
  GUI 入口见 ``scripts/calibrate_extrinsics.py``。
"""
from .intrinsics import (
    CalibrationConfig,
    calibrate_camera,
    calibrate_charuco,
    count_captures,
    detect_corners,
    load_intrinsics,
    save_capture,
    save_intrinsics,
    scan_charuco_images,
    validate_intrinsics,
)
from .extrinsics import (
    CharucoConfig,
    average_extrinsics,
    build_table_frame,
    build_table_frame_from_corners,
    compute_relative_extrinsics,
    create_board,
    detect_charuco,
    detect_charuco_detailed,
    detect_markers,
    estimate_board_pose,
    estimate_marker_pose,
    find_marker_pose,
    fuse_marker_poses,
    generate_board_image,
    generate_marker_image,
    load_extrinsics,
    reprojection_error,
    save_extrinsics,
)

__all__ = [
    # 内参
    "CalibrationConfig",
    "calibrate_camera",
    "calibrate_charuco",
    "count_captures",
    "detect_corners",
    "load_intrinsics",
    "save_capture",
    "save_intrinsics",
    "scan_charuco_images",
    "validate_intrinsics",
    # 外参
    "CharucoConfig",
    "average_extrinsics",
    "build_table_frame",
    "build_table_frame_from_corners",
    "compute_relative_extrinsics",
    "create_board",
    "detect_charuco",
    "detect_charuco_detailed",
    "detect_markers",
    "estimate_board_pose",
    "estimate_marker_pose",
    "find_marker_pose",
    "fuse_marker_poses",
    "generate_board_image",
    "generate_marker_image",
    "load_extrinsics",
    "reprojection_error",
    "save_extrinsics",
]
