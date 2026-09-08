"""多相机外参标定核心：ChArUco 检测 + 位姿估计 + 外参求解 + 结果读写。

约定（与 :class:`~tabletennis.core.types.CameraExtrinsics` 一致）：

- 外参 ``(R, t)`` 表示「世界系 -> 相机系」的刚体变换 ``X_cam = R @ X_world + t``。
- 本模块以 **ChArUco 板局部坐标系为世界系**：板所在平面为 z=0，原点在板的第一格
  角点（棋盘角），X/Y 沿格边、Z 垂直板面。``estimate_board_pose`` 用 solvePnP 解出
  「板 -> 相机」的 ``(R, t)``，就是该相机的外参。

球桌定原点：把板**平放在桌面**，板的原点角压在你想定的那个球桌角上、板 X/Y 边对齐
球桌长/短边，则板系 == 世界系，直接得到桌面世界系下的各相机外参。

依赖每台相机的内参（``data/calibration/cam_N.yaml``），用
:func:`tabletennis.calibration.intrinsics.load_intrinsics` 读入。

OpenCV 5.0 已移除旧 aruco API（``detectMarkers`` / ``interpolateCornersCharuco`` /
``estimatePoseCharucoBoard``），本模块全部走 ``CharucoBoard`` + ``CharucoDetector``
新接口（detectBoard → matchImagePoints → solvePnP）。
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

from ..core.types import CameraExtrinsics, CameraIntrinsics

# 可用 ArUco 字典名 -> 枚举值。注意 13x9 的板有 58 个标记，必须用 ≥58 的字典
# （DICT_*_50 只有 50 个，不够；用 DICT_*_100 / _250 / _1000）。
_DICTS = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}

# solvePnP 用迭代法（Charuco 角点通常 10+ 个，不适合 IPPE_SQUARE）
_SOLVEPNP_FLAGS = cv2.SOLVEPNP_ITERATIVE


@dataclass
class CharucoConfig:
    """ChArUco 标定板参数（来自 ``config/extrinsics.yaml`` 的 ``charuco`` 段）。

    Attributes:
        squares_x / squares_y: 格子列数 / 行数（= 板长边 / 短边的格子数）。13x9。
        square_length_m: 每格边长（米）。A4 板 = 0.021，1 米板 = 0.08。
        marker_length_m: 格内 ArUco 标记边长（米），**须按实际打印板测量**。
        dictionary: ArUco 字典名，**须与打印板一致**。
        legacy_pattern: True 表示打印板由 OpenCV <4.6 生成（偶数行布局有差异）。
        min_corners: 判定「检测成功」的最少 ChArUco 角点数。
    """

    squares_x: int = 13
    squares_y: int = 9
    square_length_m: float = 0.021
    marker_length_m: float = 0.015
    dictionary: str = "DICT_4X4_100"
    legacy_pattern: bool = False
    min_corners: int = 10

    @property
    def board_size(self) -> Tuple[int, int]:
        return (self.squares_x, self.squares_y)

    @classmethod
    def from_dict(cls, d: dict) -> "CharucoConfig":
        d = d or {}
        cb = d.get("charuco") or {}
        return cls(
            squares_x=int(cb.get("squares_x", 13)),
            squares_y=int(cb.get("squares_y", 9)),
            square_length_m=float(cb.get("square_length_m", 0.021)),
            marker_length_m=float(cb.get("marker_length_m", 0.015)),
            dictionary=str(cb.get("dictionary", "DICT_4X4_100")),
            legacy_pattern=bool(cb.get("legacy_pattern", False)),
            min_corners=int(cb.get("min_corners", 10)),
        )


def resolve_dictionary(name: str):
    """按名字取 OpenCV 预定义 ArUco 字典。"""
    key = str(name).upper()
    if key not in _DICTS:
        raise ValueError(f"未知 ArUco 字典 {name!r}，可选: {sorted(_DICTS)}")
    return cv2.aruco.getPredefinedDictionary(_DICTS[key])


def create_board(cfg: CharucoConfig):
    """按配置创建 ChArUco 板对象（需与打印板一致才能正确检测）。"""
    dictionary = resolve_dictionary(cfg.dictionary)
    try:
        board = cv2.aruco.CharucoBoard(
            cfg.board_size, cfg.square_length_m, cfg.marker_length_m, dictionary
        )
    except cv2.error as e:
        raise ValueError(
            f"创建 ChArUco 板失败（{cfg.squares_x}x{cfg.squares_y}，字典 {cfg.dictionary}）: {e}"
        ) from e
    if cfg.legacy_pattern:
        board.setLegacyPattern(True)

    # 字典太小会导致部分 marker 永远检测不到，提前报错而不是静默失败
    n_dict = len(dictionary.bytesList)
    n_need = len(board.getIds())
    if n_need > n_dict:
        raise ValueError(
            f"字典 {cfg.dictionary} 只有 {n_dict} 个标记，但 {cfg.squares_x}x{cfg.squares_y} "
            f"板需要 {n_need} 个标记；请换更大的字典（如 DICT_4X4_100 / DICT_6X6_250）。"
        )
    return board


_DETECTOR_CORNER_REFINE = {
    "none": cv2.aruco.CORNER_REFINE_NONE,
    "subpix": cv2.aruco.CORNER_REFINE_SUBPIX,
    "contour": cv2.aruco.CORNER_REFINE_CONTOUR,
}


def make_detector_parameters(detector_cfg: Optional[dict] = None):
    """构建 ArUco 检测参数，默认按短焦/广角镜头放宽。

    OpenCV 默认 ``minMarkerPerimeterRate=0.03``（标记周长须 >= 图像最短边的 3%），
    换短焦镜头后同一块板在画面里变小、标记常被该阈值**静默丢弃**，表现为「这台
    相机识别不到板」而非报错。这里默认放宽到 0.01 并开启亚像素角点精化，使小标记
    也能被检出（外参标定是离线操作，检测只在按 Enter 时跑一次，放宽不会拖慢预览）。

    Args:
        detector_cfg: ``config/extrinsics.yaml`` 里 ``detector`` 段的 dict，可选。

    Returns:
        ``cv2.aruco.DetectorParameters``。
    """
    d = detector_cfg or {}
    p = cv2.aruco.DetectorParameters()
    p.minMarkerPerimeterRate = float(d.get("min_marker_perimeter_rate", 0.01))
    p.maxMarkerPerimeterRate = float(d.get("max_marker_perimeter_rate", 4.0))
    p.adaptiveThreshWinSizeMin = int(d.get("adaptive_thresh_win_min", 3))
    p.adaptiveThreshWinSizeMax = int(d.get("adaptive_thresh_win_max", 23))
    p.minCornerDistanceRate = float(d.get("min_corner_distance_rate", 0.05))
    refine = str(d.get("corner_refinement", "subpix")).lower()
    p.cornerRefinementMethod = _DETECTOR_CORNER_REFINE.get(
        refine, cv2.aruco.CORNER_REFINE_SUBPIX
    )
    p.cornerRefinementWinSize = int(d.get("corner_refinement_win_size", 5))
    return p


def create_charuco_detector(board, detector_cfg: Optional[dict] = None):
    """创建 CharucoDetector 并套用放宽的检测参数（见 :func:`make_detector_parameters`）。"""
    detector = cv2.aruco.CharucoDetector(board)
    detector.setDetectorParameters(make_detector_parameters(detector_cfg))
    return detector


def detect_charuco(
    gray: np.ndarray, detector, min_corners: int = 0
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """检测 ChArUco 板角点。

    Returns:
        ``(corners, ids)``：corners 形状 ``(N, 2)`` float32，ids 形状 ``(N,)`` int32；
        失败或角点数不足 ``min_corners`` 时返回 ``(None, None)``。
    """
    cc, cid, _mc, _mid = detector.detectBoard(gray)
    if cid is None or len(cid) == 0:
        return None, None
    corners = np.asarray(cc, dtype=np.float32).reshape(-1, 2)
    ids = np.asarray(cid, dtype=np.int32).reshape(-1)
    if len(corners) < min_corners:
        return None, None
    return corners, ids


def detect_charuco_detailed(
    gray: np.ndarray, detector
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
    """检测 ChArUco，额外返回原始 ArUco 标记数（供诊断）。

    Returns:
        ``(corners, ids, n_markers)``。``n_markers`` 是检测到的原始标记数，
        用于区分「0 标记（字典/图像问题）」与「有标记但角点少（布局问题）」。
    """
    cc, cid, _mc, mid = detector.detectBoard(gray)
    n_markers = 0 if mid is None else len(mid)
    if cid is None or len(cid) == 0:
        return None, None, n_markers
    corners = np.asarray(cc, dtype=np.float32).reshape(-1, 2)
    ids = np.asarray(cid, dtype=np.int32).reshape(-1)
    return corners, ids, n_markers


def estimate_board_pose(
    corners: np.ndarray,
    ids: np.ndarray,
    board,
    intrinsics: Optional[CameraIntrinsics],
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]]:
    """solvePnP 解「板 -> 相机」外参。

    Returns:
        ``(rvec, tvec, R, t, n_pts)``，其中 ``R`` 3x3、``t`` 3x1 即该相机外参；
        无内参或点数不足时返回 None。
    """
    if intrinsics is None:
        return None
    obj, img = board.matchImagePoints(corners, ids)
    obj = np.asarray(obj, dtype=np.float64).reshape(-1, 3)
    img = np.asarray(img, dtype=np.float64).reshape(-1, 2)
    if len(obj) < 4:
        return None

    ok, rvec, tvec = cv2.solvePnP(
        obj, img, intrinsics.K, intrinsics.dist, flags=_SOLVEPNP_FLAGS
    )
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    return rvec, tvec, R, tvec.reshape(3, 1), len(obj)


def reprojection_error(
    corners: np.ndarray, ids: np.ndarray, board, intrinsics, rvec, tvec
) -> float:
    """ChArUco 角点重投影误差（RMS，像素）。"""
    obj, _ = board.matchImagePoints(corners, ids)
    obj = np.asarray(obj, dtype=np.float64).reshape(-1, 3)
    proj, _ = cv2.projectPoints(obj, rvec, tvec, intrinsics.K, intrinsics.dist)
    proj = np.asarray(proj).reshape(-1, 2)
    return float(np.linalg.norm(np.asarray(corners).reshape(-1, 2) - proj, axis=1).mean())


# ---- 旋转平均（四元数） ----
def _mat_to_quat(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
    m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
    m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
    tr = m00 + m11 + m22
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2.0
        q = np.array([0.25 * S, (m21 - m12) / S, (m02 - m20) / S, (m10 - m01) / S])
    elif m00 > m11 and m00 > m22:
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2.0
        q = np.array([(m21 - m12) / S, 0.25 * S, (m01 + m10) / S, (m02 + m20) / S])
    elif m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2.0
        q = np.array([(m02 - m20) / S, (m01 + m10) / S, 0.25 * S, (m12 + m21) / S])
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2.0
        q = np.array([(m10 - m01) / S, (m02 + m20) / S, (m12 + m21) / S, 0.25 * S])
    return q / np.linalg.norm(q)


def _quat_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def average_rotation(mats: List[np.ndarray]) -> np.ndarray:
    """多组旋转矩阵的四元数均值（保证正交）。"""
    qs = [_mat_to_quat(R) for R in mats]
    for i in range(1, len(qs)):
        if np.dot(qs[i], qs[0]) < 0:  # 统一到同一半球，避免符号翻转抵消
            qs[i] = -qs[i]
    return _quat_to_mat(np.sum(qs, axis=0))


def average_extrinsics(
    poses: List[Tuple[np.ndarray, np.ndarray]]
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """对同一相机多次采集的 ``(R, t)`` 取平均（旋转四元数均值、平移均值）。空则 None。"""
    if not poses:
        return None
    R = average_rotation([p[0] for p in poses])
    t = np.mean([p[1].reshape(3) for p in poses], axis=0).reshape(3, 1)
    return R, t


def compute_relative_extrinsics(
    board_poses: List[Dict[int, Tuple[np.ndarray, np.ndarray]]],
    reference_cam: int = 0,
) -> Tuple[Dict[int, Tuple[np.ndarray, np.ndarray]], Dict[int, int]]:
    """从多组「板 -> 相机」位姿求相机间**相对外参**（世界系 = 参考相机）。

    标定板可以每拍一次就移动一个位置（不必固定），只要每次快门里参考相机和
    目标相机**同时**看到板即可。对每个快照：若参考相机与相机 i 都看到板，则

        R_{ref→i} = R_i @ R_ref^T
        t_{ref→i} = t_i - R_{ref→i} @ t_ref

    再对多个快照取平均（旋转四元数均值、平移均值）。

    Args:
        board_poses: 每快照一个 ``{cam_id: (R, t)}``，R/t 是「板 -> 相机」。
        reference_cam: 世界系原点相机（其外参恒等）。

    Returns:
        ``({cam_id: (R, t)}, {cam_id: n_used})``，``(R, t)`` 表示「参考相机系
        -> cam_id」，参考相机自身为恒等变换；``n_used`` 是参与平均的快照数。
    """
    per_cam: Dict[int, List[Tuple[np.ndarray, np.ndarray]]] = {}
    for snap in board_poses:
        ref = snap.get(reference_cam)
        if ref is None:
            continue
        Rr, tr = ref
        for cid, (Ri, ti) in snap.items():
            R_ri = Ri @ Rr.T
            t_ri = ti - R_ri @ tr
            per_cam.setdefault(cid, []).append((R_ri, t_ri))

    results: Dict[int, Tuple[np.ndarray, np.ndarray]] = {
        reference_cam: (np.eye(3), np.zeros((3, 1)))
    }
    counts: Dict[int, int] = {
        reference_cam: sum(1 for s in board_poses if reference_cam in s)
    }
    for cid, poses in per_cam.items():
        if cid == reference_cam:
            continue
        avg = average_extrinsics(poses)
        if avg is not None:
            results[cid] = avg
            counts[cid] = len(poses)
    return results, counts


def draw_charuco_overlay(
    gray: np.ndarray, corners: np.ndarray, ids: np.ndarray, intrinsics, rvec, tvec
) -> np.ndarray:
    """把检测到的角点 + 坐标轴画到 BGR 图上，便于人工核对。"""
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.aruco.drawDetectedCornersCharuco(
        img, corners.reshape(-1, 1, 2), ids.reshape(-1, 1), (0, 255, 0)
    )
    if intrinsics is not None:
        cv2.drawFrameAxes(img, intrinsics.K, intrinsics.dist, rvec, tvec, 0.02)
    return img


# ---- 结果读写 ----
def save_extrinsics(
    path: str,
    results: Dict[int, Tuple[np.ndarray, np.ndarray]],
    world_frame: str = "charuco_board",
    extra: Optional[dict] = None,
) -> str:
    """把各相机外参 ``{cam_id: (R, t)}`` 写成 yaml（``world_frame`` 记录世界系）。"""
    data = {"world_frame": world_frame, "cameras": {}}
    for cid in sorted(results):
        R, t = results[cid]
        data["cameras"][f"cam_{cid}"] = {
            "camera_id": int(cid),
            "R": np.asarray(R).tolist(),
            "t": np.asarray(t).reshape(3).tolist(),
        }
    if extra:
        data["meta"] = extra
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    return path


def load_extrinsics(path: str) -> Dict[int, CameraExtrinsics]:
    """从 yaml 读回外参，返回 ``{cam_id: CameraExtrinsics}``。"""
    with open(path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)
    out: Dict[int, CameraExtrinsics] = {}
    for key, v in (d.get("cameras") or {}).items():
        cid = int(v["camera_id"])
        out[cid] = CameraExtrinsics(
            R=np.asarray(v["R"], dtype=np.float64),
            t=np.asarray(v["t"], dtype=np.float64).reshape(3, 1),
        )
    return out


# ---- 打印板生成（用与检测完全一致的配置，避免字典/布局不匹配） ----
def generate_board_image(
    cfg: CharucoConfig, px_per_square: int = 60, margin_px: int = 40
) -> np.ndarray:
    """生成与配置一致的 ChArUco 打印板灰度图（含留白）。

    打印时按 1:1 输出、不缩放，即可得到物理边长正确的板。
    """
    board = create_board(cfg)
    w = cfg.squares_x * px_per_square + 2 * margin_px
    h = cfg.squares_y * px_per_square + 2 * margin_px
    return board.generateImage((w, h), None, margin_px, 1)


# =========================================================================
# 桌面坐标系标定：两个大 ArUco 标记（ID0=原点角，ID1=对角）平放桌面定世界系。
# OpenCV 5.0 移除了 estimatePoseSingleMarkers，这里手动 ArucoDetector.detectMarkers
# + solvePnP（单标记 4 个共面点，用 IPPE_SQUARE）。
# =========================================================================

def detect_markers(gray: np.ndarray, detector):
    """检测 ArUco 标记。返回 ``(corners, ids)``：corners 形状 ``(N,1,4,2)``、
    ids 形状 ``(N,1)``；未检测到返回 ``(None, None)``。
    """
    corners, ids, _rej = detector.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return None, None
    return corners, ids


def estimate_marker_pose(corners4, marker_length_m: float, intrinsics, white_border_m: float = 0.0) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """单标记 solvePnP。返回 ``(R, t)`` 或 None。

    ``t`` 默认是标记**黑色部分左上角**在相机系的坐标；``R`` 的列 = 标记 X/Y/Z 轴
    在相机系的表示（X 沿标记上边向右、Y 沿左边向下、Z 垂直板面朝相机，即
    标记平放朝上时 Z 向上）。对象点以黑角为原点；用 IPPE（平面 4 点、自动
    消歧，避免 IPPE_SQUARE 要求的中心原点约定）求解。

    ``white_border_m``：白边（静默区）宽度（米）。>0 时把原点从黑角移到**白边角**
    （白边角在黑角的 -X/-Y 方向各退 white_border_m），便于把印刷件的物理外角对齐桌角。
    """
    L = float(marker_length_m)
    obj = np.array([[0, 0, 0], [L, 0, 0], [L, L, 0], [0, L, 0]], dtype=np.float64)
    img = np.asarray(corners4, dtype=np.float64).reshape(-1, 2)
    if intrinsics is None:
        return None
    ok, rvec, tvec = cv2.solvePnP(
        obj, img, intrinsics.K, intrinsics.dist, flags=cv2.SOLVEPNP_IPPE
    )
    if not ok:
        return None
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3, 1)
    if white_border_m > 0:
        t = t + R @ np.array([-white_border_m, -white_border_m, 0.0], dtype=np.float64).reshape(3, 1)
    return R, t


def find_marker_pose(markers, ids, marker_id: int, marker_length_m: float, intrinsics, white_border_m: float = 0.0):
    """在检测结果里找指定 ``marker_id`` 并解位姿。返回 ``(R, t)`` 或 None。

    ``white_border_m`` 透传给 :func:`estimate_marker_pose`，用于把原点移到白边角。
    """
    if ids is None:
        return None
    for i, mid in enumerate(np.asarray(ids).ravel()):
        if int(mid) == int(marker_id):
            return estimate_marker_pose(np.asarray(markers[i]), marker_length_m, intrinsics, white_border_m)
    return None


def build_table_frame(pose_origin, pose_second, origin_offset=None, up_hint=None) -> Tuple[np.ndarray, np.ndarray]:
    """用两个大标记的位姿构造桌面坐标系（桌面 -> 相机）。

    Args:
        pose_origin: 原点角标记的位姿 ``(R0, t0)``（marker -> 相机）。
        pose_second: 对角标记的位姿 ``(R1, t1)``。
        origin_offset: 原点相对原点标记角点的偏移（米，3 分量），默认 0。
        up_hint: 世界「上」方向的稳健估计（参考相机系里的单位向量，如由天花板
            相机位置估出）。给定则优先用它做桌面 Z 轴——标记法向在「相机俯视 +
            标记平放」时几乎退化、噪声极大，会导致桌面反复摆动。

    桌面系定义：Z = 向上（优先 up_hint，否则标记平均法向取反）；**Y = 两标记平均
    X 轴（沿桌面长边，需两标记同朝向摆放）**；X = Y x Z（右手系，即沿桌面短边）；
    原点 = 原点标记角点 + offset。
    """
    R0, t0 = pose_origin
    R1, t1 = pose_second
    origin_offset = (
        np.zeros((3, 1)) if origin_offset is None
        else np.asarray(origin_offset, dtype=np.float64).reshape(3, 1)
    )

    # Z：优先用 up_hint（稳健）；否则用标记平均法向取反。
    # 注意：estimate_marker_pose 的对象点用「X 向右 / Y 向下」，其 Z = X×Y 指向
    # 标记内部（平放朝上时即朝桌面里、朝下），故桌面「向上」要取反。
    if up_hint is not None:
        Z = np.asarray(up_hint, dtype=np.float64).reshape(3)
        Z = Z / np.linalg.norm(Z)
        # 符号对齐到「-标记平均法向」（≈上）同向
        marker_up = -(R0[:, 2] + R1[:, 2])
        if float(Z @ marker_up) < 0:
            Z = -Z
    else:
        Z1 = R1[:, 2].copy()
        if float(Z1 @ R0[:, 2]) < 0:
            Z1 = -Z1
        Z = -(R0[:, 2] + Z1)
        Z = Z / np.linalg.norm(Z)

    # Y = 平均标记 X 轴（沿桌面长边），符号对齐后投影到垂直 Z 的平面
    Y1 = R1[:, 0].copy()
    if float(Y1 @ R0[:, 0]) < 0:
        Y1 = -Y1
    Y = R0[:, 0] + Y1
    Y = Y - float(Y @ Z) * Z
    Y = Y / np.linalg.norm(Y)

    # X = 短边 = Y x Z（右手系）
    X = np.cross(Y, Z)
    X = X / np.linalg.norm(X)

    R_table = np.column_stack([X, Y, Z])
    t_table = t0 + R_table @ origin_offset
    return R_table, t_table


def build_table_frame_from_corners(positions, up_hint=None) -> Tuple[np.ndarray, np.ndarray]:
    """从 4 个标记角点位置构造桌面坐标系（桌面 -> 相机）。

    4 个标记分居球桌四角，它们的**位置**（三角化后）完全确定桌面平面和
    短/长边方向，不再依赖标记的噪声法向，因而远比 2 标记方案稳健。

    Args:
        positions: dict，key 为 ``'origin' / 'short' / 'diagonal' / 'long'``，
            value 为该角标记原点角在参考相机系里的 3D 位置（已含白边偏移）。
            至少需要 origin / short / long 三个角。
        up_hint: 世界「上」方向（参考相机系，用于 Z 符号校正），可选。

    Returns:
        ``(R, t)``，「桌面 -> 参考相机系」。
    """
    origin = np.asarray(positions['origin'], dtype=np.float64).reshape(3)
    short = np.asarray(positions['short'], dtype=np.float64).reshape(3) - origin
    long = np.asarray(positions['long'], dtype=np.float64).reshape(3) - origin

    X = short / np.linalg.norm(short)          # 短边
    Z = np.cross(X, long)                       # 法向（短×长，可能朝下）
    Z = Z / np.linalg.norm(Z)
    if up_hint is not None:
        up = np.asarray(up_hint, dtype=np.float64).reshape(3)
        if float(Z @ up) < 0:
            Z = -Z
    Y = np.cross(Z, X)                          # 长边（正交化，右手系）
    Y = Y / np.linalg.norm(Y)

    R_table = np.column_stack([X, Y, Z])
    t_table = origin.reshape(3, 1)
    return R_table, t_table


def _triangulate_point(rays: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """从多条射线（origin, direction，参考相机系）三角化出 3D 点（最小二乘）。

    每条射线约束：3D 点 P 应落在射线 ``o + t*d`` 上，即 ``(I - d d^T) P = (I - d d^T) o``。
    """
    A: List[np.ndarray] = []
    b: List[np.ndarray] = []
    for o, d in rays:
        d = np.asarray(d, dtype=np.float64).reshape(3)
        d = d / np.linalg.norm(d)
        P = np.eye(3) - np.outer(d, d)  # 投影到垂直于 d 的平面
        A.append(P)
        b.append(P @ np.asarray(o, dtype=np.float64).reshape(3))
    x, *_ = np.linalg.lstsq(np.vstack(A), np.concatenate(b), rcond=None)
    return x


def fuse_marker_poses(
    marker_obs: Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray]]],
    relative_extrinsics: Dict[int, Tuple[np.ndarray, np.ndarray]],
    reference_cam: int = 0,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """把各相机各自看到的标记位姿融合到参考相机系（配合相对外参）。

    位置用**三角化**（各相机指向标记的射线求交），只依赖标记的**方向**（稳健），
    不依赖单标记的**深度**（弱、且可能有系统性偏差）；方向用四元数平均。单个相机
    看到标记时才回退到该相机的深度。

    这样即使没有任何一台相机同时看到两个标记，只要标记 0 和标记 1 各自被至少
    一台相机看到，就能融合出两个标记在参考相机系里的位姿，进而建立桌面系。

    Args:
        marker_obs: ``{marker_id: {cam_id: (R, t)}}``，R/t 是「标记 -> 相机」
            （:func:`estimate_marker_pose` / :func:`find_marker_pose` 的返回值）。
        relative_extrinsics: ``{cam_id: (R, t)}``，「参考相机系 -> cam_id」
            （参考相机自身恒等，可由 :func:`compute_relative_extrinsics` 得到）。
        reference_cam: 参考相机索引（默认 0，仅作语义标注，实际用 map 里的值）。

    Returns:
        ``{marker_id: (R, t)}``，「标记 -> 参考相机系」；某标记无任何相机看到则不出现。
    """
    out: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for mid, obs in marker_obs.items():
        rays: List[Tuple[np.ndarray, np.ndarray]] = []
        rots: List[np.ndarray] = []
        fallback: Optional[np.ndarray] = None
        for cid, (R_mi, t_mi) in obs.items():
            ext = relative_extrinsics.get(cid)
            if ext is None:
                continue
            R_ri, t_ri = ext  # 参考相机系 -> cid：X_cid = R_ri @ X_ref + t_ri
            t_ri = np.asarray(t_ri, dtype=np.float64).reshape(3)
            t_mi = np.asarray(t_mi, dtype=np.float64).reshape(3)
            # 相机位置（参考系）
            C = (-R_ri.T @ t_ri).reshape(3)
            # 标记方向（参考系，只取方向、忽略弱深度）
            d = (R_ri.T @ t_mi).reshape(3)
            d = d / np.linalg.norm(d)
            rays.append((C, d))
            rots.append(R_ri.T @ R_mi)
            # 单相机回退位置（用该相机的深度）
            fallback = (R_ri.T @ (t_mi - t_ri)).reshape(3, 1)

        if not rays:
            continue
        if len(rays) >= 2:
            t_mr = _triangulate_point(rays).reshape(3, 1)
        else:
            t_mr = fallback
        out[mid] = (average_rotation(rots), t_mr)
    return out


def _as_rt(ext) -> Tuple[np.ndarray, np.ndarray]:
    """把外参统一成 ``(R, t)`` 元组（兼容 :class:`CameraExtrinsics` 与 ``(R, t)``）。"""
    if hasattr(ext, "R") and hasattr(ext, "t"):
        R, t = ext.R, ext.t
    else:
        R, t = ext
    return np.asarray(R, dtype=np.float64), np.asarray(t, dtype=np.float64).reshape(3)


def camera_plane_up_direction(relative_extrinsics) -> Optional[np.ndarray]:
    """从相机位置估计世界「上」方向（相机大致共面于天花板）。

    相机位置 = ``-R_ri^T @ t_ri``（参考相机系）。对位置做 SVD，取最小方差方向
    （= 相机平面法向 ≈ 天花板法向 ≈ 桌面法向）。相机数不足 3 时返回 None，
    调用方应回退到标记法向。
    """
    pts: List[np.ndarray] = []
    for _cid, ext in relative_extrinsics.items():
        R, t = _as_rt(ext)
        pts.append((-R.T @ t).reshape(3))
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 3:
        return None
    centroid = pts.mean(axis=0)
    _, _, vh = np.linalg.svd(pts - centroid)
    up = vh[-1]  # 最小方差方向
    return up / np.linalg.norm(up)


def localize_table_bundle(
    gray_frames: Dict[int, np.ndarray],
    intrinsics: Dict[int, CameraIntrinsics],
    relative_extrinsics: Dict[int, object],
    marker_detector,
    marker_ids: Dict[str, int],
    marker_length_m: float,
    white_border_m: float,
    expected_diag_m: Optional[float] = None,
    diag_tol_m: float = 0.5,
    reference_cam: int = 0,
) -> Tuple[Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]], dict]:
    """跨相机融合定位桌面（与 ``calibrate_extrinsics.py`` 的 T 键一致）。

    对每台相机单独检测四角大标记并解单标记位姿，再用相对外参把各相机看到的
    标记统一到参考相机系做**三角化**，由 origin/short/long 三个融合角点位置
    构造桌面系（diagonal 角仅做对角线校验），最后把桌面系外参转到每台相机。

    这是「桌面定位」的单一实现：``calibrate_extrinsics.py`` 的 T 键与
    ``vision/table`` 检测器 / ``live_control.py`` 都调它，保证结果一致。

    Args:
        gray_frames: ``{cam_id: 灰度图}``（同一时钟周期的一组同步帧）。
        intrinsics: ``{cam_id: CameraIntrinsics}``。
        relative_extrinsics: ``{cam_id: (R, t)}`` 或 :class:`CameraExtrinsics`，
            「参考相机系 -> cam_id」（参考相机自身恒等）。
        marker_detector: ``cv2.aruco.ArucoDetector``（大标记字典）。
        marker_ids: ``{'origin' / 'short' / 'diagonal' / 'long': id}``。
        marker_length_m / white_border_m: 大标记黑色边长 / 白边宽（米）。
        expected_diag_m / diag_tol_m: 对角线校验期望值与容差；``expected_diag_m``
            为 None 时跳过校验（校验失败仅记入 info，不中断）。
        reference_cam: 参考相机索引。

    Returns:
        ``(table_extrinsics, info)``：
        - ``table_extrinsics``: ``{cam_id: (R, t)}`` 桌面系 -> 相机系；失败为 None。
        - ``info``: 诊断字典，含 ``marker_obs``（各相机看到的标记位姿）、``fused``
          （融合后标记位姿）、``diag_m``、``missing``、``per_cam``（各相机原始
          检测结果 ``{'corners','ids','origins','got'}``，供画叠加 / 状态显示）。
    """
    info: dict = {
        "marker_obs": {}, "fused": {}, "per_cam": {},
        "diag_m": None, "missing": [], "diag_warn": False,
    }

    # 归一化相对外参为 (R, t) 元组（兼容 CameraExtrinsics）
    rel: Dict[int, Tuple[np.ndarray, np.ndarray]] = {
        cid: _as_rt(ext) for cid, ext in relative_extrinsics.items()
    }

    # 每台相机各自检测四角大标记并解单标记位姿
    marker_obs: Dict[int, Dict[int, Tuple[np.ndarray, np.ndarray]]] = {}
    for cid, gray in gray_frames.items():
        if gray is None or cid not in intrinsics:
            continue
        corners, ids = detect_markers(gray, marker_detector)
        if ids is None:
            info["per_cam"][cid] = {"corners": None, "ids": None, "origins": {}, "got": []}
            continue
        got: List[int] = []
        origins: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        for mid in set(marker_ids.values()):
            p = find_marker_pose(corners, ids, mid, marker_length_m,
                                 intrinsics[cid], white_border_m)
            if p is not None:
                marker_obs.setdefault(mid, {})[cid] = p
                got.append(mid)
                origins[mid] = p
        info["per_cam"][cid] = {"corners": corners, "ids": ids, "origins": origins, "got": got}
    info["marker_obs"] = marker_obs

    if not marker_obs:
        return None, info

    fused = fuse_marker_poses(marker_obs, rel, reference_cam)
    info["fused"] = fused

    need = {k: marker_ids[k] for k in ("origin", "short", "long")}
    missing = [f"{k}(ID{v})" for k, v in need.items() if v not in fused]
    info["missing"] = missing
    if missing:
        return None, info

    # 对角线校验：ID0↔ID2 应≈桌面对角线（只警告，不中断）
    diag_id = marker_ids.get("diagonal")
    if diag_id is not None and diag_id in fused:
        dist = float(np.linalg.norm(fused[diag_id][1] - fused[marker_ids["origin"]][1]))
        info["diag_m"] = dist
        if expected_diag_m is not None and abs(dist - expected_diag_m) > diag_tol_m:
            info["diag_warn"] = True

    # 由 origin/short/long 三个融合角点位置构造桌面系（Z 用相机平面法向校正符号）
    up = camera_plane_up_direction(rel)
    positions = {k: fused[marker_ids[k]][1] for k in ("origin", "short", "long")}
    R_tbl_ref, t_tbl_ref = build_table_frame_from_corners(positions, up_hint=up)

    # 转到每台相机（含没直接看到标记的相机）
    table_extrinsics: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for cid, (R_ri, t_ri) in rel.items():
        if cid not in intrinsics:
            continue
        R = R_ri @ R_tbl_ref
        t = R_ri @ t_tbl_ref + t_ri.reshape(3, 1)
        table_extrinsics[cid] = (R, t)
    return table_extrinsics, info


def generate_marker_image(dict_name: str, marker_id: int, side_px: int = 1500, margin_px: int = 300) -> np.ndarray:
    """生成带白边 + 朝向标注的单标记打印图（BGR）。

    图上标注了「原点角」（红色圆点，须压在球桌原点角上）和 X 箭头（沿桌面长边）。
    按 1:1 原尺寸打印，标记物理边长 = 打印边长。
    """
    dictionary = resolve_dictionary(dict_name)
    mk = cv2.aruco.generateImageMarker(dictionary, int(marker_id), side_px)
    img = np.full((side_px + 2 * margin_px, side_px + 2 * margin_px), 255, np.uint8)
    img[margin_px:margin_px + side_px, margin_px:margin_px + side_px] = mk
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    x0 = y0 = margin_px
    # 原点角（左上角 = 对象原点），红色圆点
    cv2.circle(bgr, (x0, y0), 24, (0, 0, 255), -1)
    cv2.putText(bgr, "origin", (x0 + 8, y0 + 64), cv2.FONT_HERSHEY_SIMPLEX,
                2.2, (0, 0, 255), 5, cv2.LINE_AA)
    # X 箭头（沿上边向右）
    cv2.arrowedLine(bgr, (x0 + 50, y0 + 32), (x0 + side_px - 50, y0 + 32),
                    (0, 180, 0), 6, tipLength=0.05)
    cv2.putText(bgr, "X", (x0 + side_px - 150, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX,
                2.6, (0, 180, 0), 6, cv2.LINE_AA)
    # 底部信息
    cv2.putText(bgr, f"ID={marker_id}  {dict_name}", (margin_px, side_px + 2 * margin_px - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 2.4, (0, 0, 0), 5, cv2.LINE_AA)
    return bgr
