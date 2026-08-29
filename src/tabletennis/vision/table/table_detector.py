"""球桌检测器：四个大 ArUco 标记（ID0/1/2/3 = 原点/短边/对角/长边）定位桌面世界系。

桌面系定义同 :func:`~tabletennis.calibration.extrinsics.build_table_frame_from_corners`：
原点 = 标记 0 原点角（白边角），X 沿短边、Y 沿长边、Z 向上。

**主路径**是 :meth:`TableDetector.localize_bundle`：对一组同步帧做跨相机三角化
融合（与 ``calibrate_extrinsics.py`` 的 T 键一致），返回 ``{cam_id: (R, t)}``
桌面系 -> 相机系外参。单帧 :meth:`TableDetector.detect` 作为**单相机回退**
（只依赖单相机 solvePnP 深度，精度较弱），把标准尺寸球桌
（:class:`~tabletennis.core.types.Table3D`）边框投影到该相机图像；若某相机没
检测到标记，则回退到已保存的 ``data/extrinsics/table_extrinsics.yaml`` 外参。
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from ...calibration.extrinsics import (
    build_table_frame_from_corners,
    detect_markers,
    find_marker_pose,
    load_extrinsics,
    localize_table_bundle,
    resolve_dictionary,
)
from ...calibration.intrinsics import load_intrinsics
from ...core.config import load_yaml_optional, project_root
from ...core.types import (
    CameraExtrinsics,
    CameraIntrinsics,
    Frame,
    Table3D,
    TableDetection,
)
from ..detector import TableDetector as _BaseTableDetector


class TableDetector(_BaseTableDetector):
    """球桌检测器：四个大 ArUco 标记定位桌面世界系，投影标准尺寸球桌。

    Args:
        intrinsics: ``{cam_id: CameraIntrinsics}``，用于解标记位姿与投影。
        table_cfg: ``config/extrinsics.yaml`` 的 ``table`` 段（标记字典 / 边长 / ID）。
        fallback_extrinsics: ``{cam_id: CameraExtrinsics}``，检测失败时回退（已存桌面系外参）。
        relative_extrinsics: ``{cam_id: CameraExtrinsics}``，相对外参（参考相机系 -> cam_id），
            跨相机融合定位用；缺失则 :meth:`localize_bundle` 不可用。
        table: 标准尺寸球桌模型，默认 :class:`Table3D`。
    """

    def __init__(
        self,
        intrinsics: Optional[Dict[int, CameraIntrinsics]] = None,
        table_cfg: Optional[dict] = None,
        fallback_extrinsics: Optional[Dict[int, CameraExtrinsics]] = None,
        relative_extrinsics: Optional[Dict[int, CameraExtrinsics]] = None,
        table: Optional[Table3D] = None,
    ) -> None:
        self.intrinsics = intrinsics or {}
        self.table_cfg = table_cfg or {}
        self.fallback = fallback_extrinsics or {}
        self.relative = relative_extrinsics or {}
        self.table = table or Table3D()

        self.marker_length_m = float(self.table_cfg.get("marker_length_m", 0.18))
        self.white_border_m = float(self.table_cfg.get("white_border_m", 0.0))
        self.marker_ids = {
            "origin": int(self.table_cfg.get("marker_origin_id", 0)),
            "short": int(self.table_cfg.get("marker_short_id", 1)),
            "diagonal": int(self.table_cfg.get("marker_diagonal_id", 2)),
            "long": int(self.table_cfg.get("marker_long_id", 3)),
        }
        self.origin_offset = self.table_cfg.get("origin_offset_m", [0.0, 0.0, 0.0])

        # 四角大标记 ID0↔ID2 应相距≈桌面对角线；用它做几何校验，拒绝误检
        # （如标定板上的 0/1 号格子——DICT_5X5_50 是 DICT_5X5_250 的子集）。
        self.expected_marker_dist = float(
            self.table_cfg.get("expected_marker_distance_m", self.table.diagonal)
        )
        self.marker_dist_tol = float(self.table_cfg.get("marker_distance_tol_m", 0.5))

        dictionary = resolve_dictionary(self.table_cfg.get("marker_dict", "DICT_5X5_50"))
        self._marker_detector = cv2.aruco.ArucoDetector(dictionary)

    # ------------------------------------------------------------------
    # 接口
    # ------------------------------------------------------------------
    def detect(self, frame: Frame) -> Optional[TableDetection]:
        """识别一帧里的球桌，返回 :class:`TableDetection` 或 None。

        单相机回退路径：优先实时检测四角大标记（``live=True``）；该相机没同时看到
        足够标记时，回退到已保存外参（``live=False``）；既无标记又无外参则返回 None。
        跨相机融合请用 :meth:`localize_bundle`。
        """
        cid = frame.camera_id
        K = self.intrinsics.get(cid)
        if K is None:
            return None

        pose = self._detect_pose(frame.image, K, cid)
        if pose is None:
            ext = self.fallback.get(cid)
            if ext is None:
                return None
            R, t = ext.R, ext.t
            live = False
        else:
            R, t = pose
            live = True

        corners = self.project_corners(R, t, K)
        if corners is None:
            return None
        return TableDetection(camera_id=cid, R=R, t=t, corners=corners, live=live)

    def _detect_pose(
        self, gray: np.ndarray, K: CameraIntrinsics, cid: int = -1
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """单相机检测四角大标记并解「桌面 -> 相机」位姿 ``(R, t)``，失败返回 None。

        这是**单相机回退**：由单帧内 origin/short/long 三个标记的 solvePnP 深度直接
        构造桌面系，精度弱于 :meth:`localize_bundle`（跨相机三角化）。需至少看到
        origin/short/long 三个角；diagonal 角仅用于几何校验（ID0↔ID2 应≈对角线）。
        """
        corners, ids = detect_markers(gray, self._marker_detector)
        if ids is None:
            return None
        poses = {}
        for role, mid in self.marker_ids.items():
            p = find_marker_pose(corners, ids, mid, self.marker_length_m, K, self.white_border_m)
            if p is not None:
                poses[role] = p

        if not {"origin", "short", "long"} <= set(poses):
            return None

        if "diagonal" in poses:
            dist = float(np.linalg.norm(poses["diagonal"][1] - poses["origin"][1]))
            if abs(dist - self.expected_marker_dist) > self.marker_dist_tol:
                return None

        positions = {k: poses[k][1] for k in ("origin", "short", "long")}
        return build_table_frame_from_corners(positions)

    def localize_bundle(
        self, gray_frames: Dict[int, np.ndarray]
    ) -> Optional[Dict[int, Tuple[np.ndarray, np.ndarray]]]:
        """跨相机融合定位桌面（与 ``calibrate_extrinsics.py`` 的 T 键一致）。

        用相对外参把各相机看到的四角大标记三角化到参考相机系，由 origin/short/long
        三个融合角点位置构造桌面系，返回 ``{cam_id: (R, t)}``（桌面系 -> 相机系）。
        无相对外参或融合失败返回 None（调用方应回退到 :attr:`fallback`）。
        """
        if not self.relative:
            return None
        table_extrinsics, _info = localize_table_bundle(
            gray_frames, self.intrinsics, self.relative, self._marker_detector,
            self.marker_ids, self.marker_length_m, self.white_border_m,
            expected_diag_m=self.expected_marker_dist, diag_tol_m=self.marker_dist_tol,
        )
        return table_extrinsics

    def project_corners(
        self, R: np.ndarray, t: np.ndarray, K: CameraIntrinsics
    ) -> np.ndarray:
        """把标准球桌桌面 4 角点投影到图像，返回 ``(4,2)`` 像素坐标。"""
        rvec, _ = cv2.Rodrigues(np.asarray(R, dtype=np.float64))
        proj, _ = cv2.projectPoints(
            self.table.top_corners,
            rvec,
            np.asarray(t, dtype=np.float64).reshape(3, 1),
            K.K,
            K.dist,
        )
        return proj.reshape(-1, 2)

    # ------------------------------------------------------------------
    # 从项目配置加载（供检测器工厂调用）
    # ------------------------------------------------------------------
    @classmethod
    def load_default(cls) -> "TableDetector":
        """从项目根目录加载标记参数、内参与外参，构造一个开箱即用的检测器。"""
        root = project_root()
        cfg = load_yaml_optional(os.path.join(root, "config", "extrinsics.yaml"), {}) or {}
        table_cfg = cfg.get("table", {}) or {}

        intrinsics: Dict[int, CameraIntrinsics] = {}
        cal_dir = os.path.join(root, "data", "calibration")
        if os.path.isdir(cal_dir):
            for name in sorted(os.listdir(cal_dir)):
                if not (name.startswith("cam_") and name.endswith(".yaml")):
                    continue
                try:
                    cid = int(name[len("cam_"):-len(".yaml")])
                except ValueError:
                    continue
                intrinsics[cid] = load_intrinsics(os.path.join(cal_dir, name))

        fallback: Dict[int, CameraExtrinsics] = {}
        ext_path = os.path.join(root, "data", "extrinsics", "table_extrinsics.yaml")
        if os.path.exists(ext_path):
            fallback = load_extrinsics(ext_path)

        # 相对外参（参考相机系 -> 各相机），跨相机融合定位用
        relative: Dict[int, CameraExtrinsics] = {}
        rel_path = os.path.join(root, "data", "extrinsics", "extrinsics.yaml")
        if os.path.exists(rel_path):
            relative = load_extrinsics(rel_path)

        return cls(
            intrinsics=intrinsics,
            table_cfg=table_cfg,
            fallback_extrinsics=fallback,
            relative_extrinsics=relative,
        )
