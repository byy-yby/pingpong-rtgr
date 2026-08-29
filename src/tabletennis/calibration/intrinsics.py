"""相机内参标定核心：棋盘角点检测 + 张正友标定 + 结果读写。

与相机模块解耦：本模块只吃灰度图（``(H, W)`` uint8，Mono8 相机直接给
``Frame.image``），不关心相机怎么打开。结果以 :class:`~tabletennis.core.types.CameraIntrinsics`
产出，并可通过 :func:`load_intrinsics` 读回，供后续重建 / 三角化直接使用。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import yaml

from ..core.types import CameraIntrinsics

# ChArUco 角点检测复用外参模块里的通用实现（同属 calibration 包，无分层问题）
from .extrinsics import detect_charuco, detect_charuco_detailed

# 角点亚像素细化终止准则
_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-4)


@dataclass
class CalibrationConfig:
    """标定参数（来自 ``config/calibration.yaml``）。

    Attributes:
        cols / rows: 棋盘格内角点列数 / 行数（= 格子数 - 1）。
        square_size_mm: 每格物理边长（毫米）。
        min_images: 建议的最少采集张数（低于此值仅提示）。
        min_images_hard: 硬下限，低于此值该相机直接跳过标定。
        output_dir: 图像与结果输出目录（相对项目根目录）。
    """

    cols: int = 8
    rows: int = 12
    square_size_mm: float = 30.0
    min_images: int = 15
    min_images_hard: int = 5
    output_dir: str = "data/calibration"

    @property
    def pattern(self) -> Tuple[int, int]:
        return (self.cols, self.rows)

    @classmethod
    def from_dict(cls, d: dict) -> "CalibrationConfig":
        d = d or {}
        cb = d.get("chessboard") or {}
        return cls(
            cols=int(cb.get("cols", 8)),
            rows=int(cb.get("rows", 12)),
            square_size_mm=float(cb.get("square_size_mm", 30.0)),
            min_images=int(d.get("min_images", 15)),
            min_images_hard=int(d.get("min_images_hard", 5)),
            output_dir=d.get("output_dir", "data/calibration"),
        )


def detect_corners(gray: np.ndarray, pattern: Tuple[int, int]) -> Tuple[bool, Optional[np.ndarray]]:
    """检测棋盘内角点，返回 ``(是否找到, 角点)``。

    角点为 ``(N, 1, 2)`` float32 亚像素坐标。失败时返回 ``(False, None)``。
    """
    ret, corners = cv2.findChessboardCorners(gray, pattern, None)
    if not ret:
        return False, None
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), _CRITERIA)
    return True, corners


def capture_dir(out_dir: str, cam_id: int) -> str:
    d = os.path.join(out_dir, f"cam_{cam_id}")
    os.makedirs(d, exist_ok=True)
    return d


def count_captures(out_dir: str, cam_id: int) -> int:
    d = capture_dir(out_dir, cam_id)
    return len([f for f in os.listdir(d) if f.endswith(".png")])


def save_capture(out_dir: str, cam_id: int, gray: np.ndarray) -> str:
    """把当前灰度帧存为 ``data/calibration/cam_N/NNN.png``，返回保存路径。"""
    d = capture_dir(out_dir, cam_id)
    path = os.path.join(d, f"{count_captures(out_dir, cam_id):03d}.png")
    cv2.imwrite(path, gray)
    return path


def _object_points(pattern: Tuple[int, int], square_size_mm: float) -> np.ndarray:
    c, r = pattern
    objp = np.zeros((c * r, 3), np.float32)
    objp[:, :2] = np.mgrid[0:c, 0:r].T.reshape(-1, 2) * square_size_mm
    return objp


def _reprojection_rms(
    obj_pts: List[np.ndarray],
    img_pts: List[np.ndarray],
    rvecs: List[np.ndarray],
    tvecs: List[np.ndarray],
    K: np.ndarray,
    dist: np.ndarray,
) -> float:
    """总重投影误差（RMS，像素）。"""
    total = 0.0
    npts = 0
    for i in range(len(obj_pts)):
        proj, _ = cv2.projectPoints(obj_pts[i], rvecs[i], tvecs[i], K, dist)
        proj = np.asarray(proj, dtype=np.float32).reshape(-1, 2)
        img = np.asarray(img_pts[i], dtype=np.float32).reshape(-1, 2)
        err = cv2.norm(img, proj, cv2.NORM_L2)
        total += err * err
        npts += img.shape[0]
    return float(np.sqrt(total / npts))


def calibrate_camera(
    out_dir: str,
    cam_id: int,
    pattern: Tuple[int, int],
    square_size_mm: float,
    min_images_hard: int = 5,
) -> Tuple[Optional[CameraIntrinsics], Optional[float], int]:
    """从 ``out_dir/cam_N/*.png`` 标定单个相机。

    Returns:
        ``(intrinsics, rms, n)``。张数不足或失败时 ``intrinsics`` 与 ``rms`` 为 None，
        ``n`` 为成功用到的图像张数。
    """
    d = capture_dir(out_dir, cam_id)
    paths = sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".png"))

    objp = _object_points(pattern, square_size_mm)
    obj_pts: List[np.ndarray] = []
    img_pts: List[np.ndarray] = []
    img_size: Optional[Tuple[int, int]] = None

    for p in paths:
        gray = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        ok, corners = detect_corners(gray, pattern)
        if not ok:
            continue
        if img_size is None:
            img_size = (gray.shape[1], gray.shape[0])  # (w, h)
        obj_pts.append(objp)
        img_pts.append(corners)

    n = len(obj_pts)
    if n < min_images_hard or img_size is None:
        return None, None, n

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_pts, img_pts, img_size, None, None
    )
    rms = _reprojection_rms(obj_pts, img_pts, rvecs, tvecs, K, dist)
    intrinsics = CameraIntrinsics(
        width=img_size[0], height=img_size[1], K=K, dist=dist.ravel()
    )
    return intrinsics, rms, n


def save_intrinsics(
    out_dir: str,
    cam_id: int,
    intrinsics: CameraIntrinsics,
    rms: float,
    n: int,
    extra: Optional[dict] = None,
) -> str:
    """把标定结果写成 ``out_dir/cam_N.yaml``，返回路径。

    ``extra`` 可追加校验信息（如 ``verdict`` / ``problems`` / ``per_view_rms``）。
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"cam_{cam_id}.yaml")
    data = {
        "camera_id": cam_id,
        "image_size": [intrinsics.width, intrinsics.height],
        "num_images": n,
        "camera_matrix": intrinsics.K.tolist(),
        "distortion": intrinsics.dist.tolist(),
        "rms_error": rms,
    }
    if extra:
        data.update(extra)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    return path


def load_intrinsics(path: str) -> CameraIntrinsics:
    """从 yaml 读回 :class:`CameraIntrinsics`，供重建 / 三角化使用。"""
    with open(path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)
    return CameraIntrinsics(
        width=int(d["image_size"][0]),
        height=int(d["image_size"][1]),
        K=np.asarray(d["camera_matrix"], dtype=np.float64),
        dist=np.asarray(d["distortion"], dtype=np.float64),
    )


def calibrate_charuco(
    out_dir: str,
    cam_id: int,
    detector,
    board,
    min_corners: int = 10,
    min_images_hard: int = 5,
) -> Tuple[Optional[CameraIntrinsics], Optional[float], List[float], int]:
    """用 ChArUco 板做单相机内参标定（张正友法）。

    从 ``out_dir/cam_N/*.png`` 逐张检测 ChArUco 角点（复用
    :func:`~tabletennis.calibration.extrinsics.detect_charuco`），经
    ``board.matchImagePoints`` 得到物点/像点后交给 ``cv2.calibrateCamera``。

    Args:
        out_dir: 照片目录根（其下 ``cam_N/`` 放该相机照片）。
        cam_id: 相机逻辑索引。
        detector: ``cv2.aruco.CharucoDetector(board)``。
        board: ``cv2.aruco.CharucoBoard``（须与打印板一致）。
        min_corners: 判定单张「检测成功」的最少角点数。
        min_images_hard: 硬下限，低于此张数判定失败。

    Returns:
        ``(intrinsics, rms, per_view_rms, n)``；失败时前两项为 None、``n`` 为
        成功用到的图像张数。
    """
    d = capture_dir(out_dir, cam_id)
    paths = sorted(
        os.path.join(d, f) for f in os.listdir(d)
        if f.endswith(".png") and "_annot" not in f
    )

    obj_pts: List[np.ndarray] = []
    img_pts: List[np.ndarray] = []
    img_size: Optional[Tuple[int, int]] = None

    for p in paths:
        gray = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        corners, ids = detect_charuco(gray, detector, min_corners)
        if corners is None:
            continue
        obj, img = board.matchImagePoints(corners, ids)
        obj = np.asarray(obj, dtype=np.float32).reshape(-1, 3)
        img = np.asarray(img, dtype=np.float32).reshape(-1, 2)
        if len(obj) < min_corners:
            continue
        if img_size is None:
            img_size = (gray.shape[1], gray.shape[0])  # (w, h)
        obj_pts.append(obj)
        img_pts.append(img)

    n = len(obj_pts)
    if n < min_images_hard or img_size is None:
        return None, None, [], n

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_pts, img_pts, img_size, None, None
    )
    rms = _reprojection_rms(obj_pts, img_pts, rvecs, tvecs, K, dist)

    per_view: List[float] = []
    for i in range(n):
        proj, _ = cv2.projectPoints(obj_pts[i], rvecs[i], tvecs[i], K, dist)
        proj = np.asarray(proj, dtype=np.float32).reshape(-1, 2)
        img = np.asarray(img_pts[i], dtype=np.float32).reshape(-1, 2)
        err = cv2.norm(img, proj, cv2.NORM_L2)
        per_view.append(float(np.sqrt(err * err / img.shape[0])))

    intrinsics = CameraIntrinsics(
        width=img_size[0], height=img_size[1], K=K, dist=dist.ravel()
    )
    return intrinsics, rms, per_view, n


def scan_charuco_images(out_dir: str, cam_id: int, detector) -> List[dict]:
    """逐张检测某相机已采集照片，返回每张的诊断信息（供「拍完统一识别」）。

    Returns:
        ``[{file, n_markers, n_corners}, ...]``，按文件名排序。``n_corners`` 是
        原始 ChArUco 角点数（未按 min_corners 过滤），调用方据此判断可用性。
    """
    d = capture_dir(out_dir, cam_id)
    paths = sorted(
        os.path.join(d, f) for f in os.listdir(d)
        if f.endswith(".png") and "_annot" not in f
    )
    out: List[dict] = []
    for p in paths:
        gray = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        corners, ids, n_markers = detect_charuco_detailed(gray, detector)
        out.append({
            "file": os.path.basename(p),
            "n_markers": n_markers,
            "n_corners": 0 if corners is None else len(corners),
        })
    return out


def validate_intrinsics(
    intrinsics: CameraIntrinsics,
    rms: float,
    per_view_rms: Optional[List[float]] = None,
) -> dict:
    """对内参结果做「准不准」检查，返回报告 dict。

    主要看三点：

    - **重投影误差 rms**：标准精度指标，<0.3px 优秀、<0.5px 良好、>1px 可疑。
    - **主点偏移**：cx/cy 应接近图像中心（通常偏离 < 图像边长的 5%）。
    - **fx/fy 比值**：方形像素应 ≈1，偏差 >10% 可疑。

    Returns:
        含 ``rms`` / ``per_view_rms`` / ``fx`` / ``fy`` / ``cx`` / ``cy`` /
        ``focal_ratio`` / ``principal_point_offset_px`` / ``problems`` /
        ``verdict``（ok | warn | bad）。
    """
    w, h = intrinsics.width, intrinsics.height
    K = intrinsics.K
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    pp_offset = (cx - w / 2.0, cy - h / 2.0)
    focal_ratio = fx / fy if fy else 0.0

    problems: List[str] = []
    if rms > 1.0:
        problems.append(f"重投影误差 {rms:.3f}px 过大(>1)")
    elif rms > 0.5:
        problems.append(f"重投影误差 {rms:.3f}px 偏高(>0.5)")
    if abs(pp_offset[0]) > 0.05 * w or abs(pp_offset[1]) > 0.05 * h:
        problems.append(f"主点偏移 ({pp_offset[0]:.1f}, {pp_offset[1]:.1f})px 过大")
    if abs(focal_ratio - 1.0) > 0.1:
        problems.append(f"fx/fy 比值 {focal_ratio:.3f} 偏离 1 过多")

    if rms > 1.0 or len(problems) >= 2:
        verdict = "bad"
    elif problems:
        verdict = "warn"
    else:
        verdict = "ok"

    return {
        "rms": float(rms),
        "per_view_rms": [float(x) for x in (per_view_rms or [])],
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "focal_ratio": focal_ratio,
        "principal_point_offset_px": pp_offset,
        "problems": problems,
        "verdict": verdict,
    }
