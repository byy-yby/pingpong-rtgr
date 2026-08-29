"""多视角三角化：把多个相机视角的 2D 关键点转成 3D 世界坐标。

核心是**置信度加权的齐次 DLT**（SVD），并利用相机几何（光心 / 射线交会角）做
质量评估与多视角外点剔除。

约定（与 :class:`~tabletennis.core.types.CameraExtrinsics` 一致）：外参 ``(R, t)``
表示「世界系 -> 相机系」``X_cam = R @ X_world + t``，投影矩阵 ``P = K [R | t]``
把世界点投到（无畸变的）像素坐标。

关键点来自 RTMPose 时是**畸变后**的原始像素坐标，须先用内参去畸变，再用线性
投影矩阵三角化——本模块在 :meth:`MultiViewTriangulator.triangulate_pose` 里内部
完成去畸变；:meth:`triangulate_point` 则假定输入已经是无畸变像素坐标。

权重设计（对应「利用关节置信度 + 相机位置关系」）：

- **置信度**：DLT 每一行按该视角关键点置信度缩放，高分视角主导解；
  综合置信度 = 视角置信度均值 × 几何因子 × 重投影误差衰减。
- **相机位置**：投影矩阵由真实内外参导出；交会角（两视角光心到 3D 点射线的
  夹角）越大，深度越可靠，交会角过小（近共线）判定几何退化直接放弃。
"""
from __future__ import annotations

import itertools
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..core.types import (
    CameraExtrinsics,
    CameraIntrinsics,
    Pose2D,
    Skeleton3D,
)

# 判定「关键点可用」的最低 2D 置信度；低于此值的观测不参与三角化。
DEFAULT_MIN_CONF = 0.3
# 多视角外点剔除：单视角重投影误差超过该值（像素）视为外点剔除（视角 > 2 时）。
DEFAULT_MAX_REPROJ_PX = 12.0
# 交会角低于该值（度）判定几何退化（两相机近共线），深度不可靠，放弃该关键点。
DEFAULT_MIN_ANGLE_DEG = 8.0


def undistort_keypoints(
    keypoints: np.ndarray, K: np.ndarray, dist: np.ndarray
) -> np.ndarray:
    """把 (N, 3) 关键点 ``[x, y, conf]`` 的坐标去畸变，返回同样形状的数组。

    用 ``cv2.undistortPoints(..., P=K)`` 得到与内参 K 一致的**无畸变像素坐标**
    （畸变模型由 ``dist`` 描述），置信度列原样保留。坐标越界 / 数值异常的点
    置为 NaN（后续按不可用处理，不参与三角化）。
    """
    pts = np.asarray(keypoints, dtype=np.float32).reshape(-1, 3)
    xys = pts[:, :2].reshape(-1, 1, 2)
    try:
        undist = cv2.undistortPoints(xys, np.asarray(K, dtype=np.float64),
                                     np.asarray(dist, dtype=np.float64), P=K)
    except cv2.error:  # 畸变系数 / 内参不合法
        return np.full_like(pts, np.nan)
    out = np.asarray(undist, dtype=np.float32).reshape(-1, 2)
    bad = ~np.isfinite(out).all(axis=1)
    if bad.any():
        out[bad] = np.nan
    return np.concatenate([out, pts[:, 2:3]], axis=1)


class MultiViewTriangulator:
    """多视角置信度加权 DLT 三角化器。

    Args:
        intrinsics: ``{cam_id: CameraIntrinsics}``，用于去畸变与构建投影矩阵。
        extrinsics: ``{cam_id: CameraExtrinsics}``，世界系 -> 相机系外参。
            只保留同时具备内参与外参的相机。
    """

    def __init__(
        self,
        intrinsics: Dict[int, CameraIntrinsics],
        extrinsics: Dict[int, CameraExtrinsics],
    ) -> None:
        self.P: Dict[int, np.ndarray] = {}       # cam_id -> (3,4) 投影矩阵
        self.centers: Dict[int, np.ndarray] = {}  # cam_id -> (3,) 相机光心（世界系）
        self.K: Dict[int, np.ndarray] = {}        # cam_id -> (3,3) 内参
        self.dist: Dict[int, np.ndarray] = {}     # cam_id -> 畸变系数

        for cid in sorted(extrinsics):
            Kobj = intrinsics.get(cid)
            if Kobj is None:
                continue
            ext = extrinsics[cid]
            K = np.asarray(Kobj.K, dtype=np.float64)
            R = np.asarray(ext.R, dtype=np.float64)
            t = np.asarray(ext.t, dtype=np.float64).reshape(3)
            self.P[cid] = K @ np.hstack([R, t.reshape(3, 1)])
            self.centers[cid] = (-R.T @ t).reshape(3)
            self.K[cid] = K
            self.dist[cid] = np.asarray(Kobj.dist, dtype=np.float64)

    @property
    def cameras(self) -> List[int]:
        return sorted(self.P)

    # ------------------------------------------------------------------
    # 几何工具
    # ------------------------------------------------------------------
    def project(self, cid: int, X: np.ndarray) -> np.ndarray:
        """世界点 ``X`` (3,) -> 该相机无畸变像素坐标 (2,)。"""
        x = self.P[cid] @ np.append(np.asarray(X, dtype=np.float64), 1.0)
        return x[:2] / x[2]

    def reproj(self, cid: int, X: np.ndarray, uv) -> float:
        """单视角重投影误差（像素）：世界点 X 投影到相机 cid 与观测 uv 的距离。"""
        proj = self.project(cid, X)
        return float(np.linalg.norm(proj - np.asarray(uv, dtype=np.float64)))

    def _dlt(self, views: List[int], points: Dict[int, Tuple[float, float]],
             confs: Dict[int, float]) -> Optional[np.ndarray]:
        """置信度加权齐次 DLT。返回世界坐标 (3,)，退化返回 None。"""
        A = np.zeros((2 * len(views), 4), dtype=np.float64)
        for r, cid in enumerate(views):
            x, y = points[cid]
            w = float(confs.get(cid, 0.0))
            Pi = self.P[cid]
            A[2 * r] = w * (x * Pi[2] - Pi[0])
            A[2 * r + 1] = w * (y * Pi[2] - Pi[1])
        _, s, Vh = np.linalg.svd(A)
        # 数值退化：次小奇异值过小说明近共线，解不可信
        if s[-2] < 1e-9:
            return None
        Xh = Vh[-1]
        if abs(Xh[3]) < 1e-12:
            return None
        return Xh[:3] / Xh[3]

    def _max_angle_deg(self, X: np.ndarray, views: List[int]) -> float:
        """最大交会角（度）：任意两视角「光心 -> X」射线夹角的最大值。"""
        best = 0.0
        for a, b in itertools.combinations(views, 2):
            da = np.asarray(X) - self.centers[a]
            db = np.asarray(X) - self.centers[b]
            na, nb = np.linalg.norm(da), np.linalg.norm(db)
            if na < 1e-9 or nb < 1e-9:
                continue
            cosang = float(np.clip(np.dot(da, db) / (na * nb), -1.0, 1.0))
            best = max(best, float(np.degrees(np.arccos(cosang))))
        return best

    # ------------------------------------------------------------------
    # 三角化
    # ------------------------------------------------------------------
    def triangulate_point(
        self,
        points: Dict[int, Tuple[float, float]],
        confs: Dict[int, float],
        min_conf: float = DEFAULT_MIN_CONF,
        max_reproj_px: float = DEFAULT_MAX_REPROJ_PX,
        min_angle_deg: float = DEFAULT_MIN_ANGLE_DEG,
    ) -> Optional[Tuple[np.ndarray, float, float, int, float]]:
        """三角化单个 3D 点（输入为**无畸变**像素坐标）。

        Args:
            points: ``{cam_id: (x, y)}`` 各视角观测。
            confs: ``{cam_id: conf}`` 各视角置信度。
            min_conf: 观测可用性阈值。
            max_reproj_px: 外点剔除阈值（视角 > 2 时，剔除重投影误差超标的视角）。
            min_angle_deg: 最小交会角，低于即放弃。

        Returns:
            ``(X, conf, reproj_err, n_views, angle_deg)`` 或 None（失败）：
            - X: 3D 世界坐标 (3,)
            - conf: 综合置信度 0..1（视角置信度均值 × 几何因子 × 误差衰减）
            - reproj_err: 加权重投影误差（像素）
            - n_views: 实际用到的视角数
            - angle_deg: 最大交会角（度）
        """
        views = []
        for cid in points:
            if cid not in self.P:
                continue
            c = float(confs.get(cid, 0.0))
            uv = np.asarray(points[cid], dtype=np.float64)
            if c < min_conf or not np.isfinite(uv).all():
                continue
            views.append(cid)
        if len(views) < 2:
            return None

        active = list(views)
        X = None
        # 视角 > 2 时迭代剔除最差视角（重投影误差超阈值且仍多于 2 个视角）
        while True:
            X = self._dlt(active, points, confs)
            if X is None:
                return None
            errs = {cid: self.reproj(cid, X, points[cid]) for cid in active}
            if len(active) <= 2:
                break
            worst = max(active, key=lambda c: errs[c])
            if errs[worst] > max_reproj_px:
                active.remove(worst)
                continue
            break

        err = float(np.mean([errs[c] for c in active]))
        angle = self._max_angle_deg(X, active)
        if angle < min_angle_deg:
            return None

        mean_conf = float(np.mean([float(confs[c]) for c in active]))
        geo = 0.5 + 0.5 * float(np.clip(angle / 90.0, 0.0, 1.0))
        conf = float(np.clip(mean_conf * geo / (1.0 + err / 8.0), 0.0, 1.0))
        return X, conf, err, len(active), angle

    def triangulate_pose(
        self,
        observations: Dict[int, Pose2D],
        min_conf: float = DEFAULT_MIN_CONF,
        **kwargs,
    ) -> Skeleton3D:
        """把同一球员在多相机里的 2D 姿态三角化成 3D 骨架。

        Args:
            observations: ``{cam_id: Pose2D}``，同一球员在各相机的匹配观测，
                须同属一个骨架（halpe26 / coco17）。内部先对关键点去畸变。
            min_conf: 关键点可用性阈值。
            **kwargs: 透传给 :meth:`triangulate_point`（外点 / 交会角阈值）。

        Returns:
            :class:`Skeleton3D`，未三角化的关键点为 NaN、置信度 0、n_views 0。
        """
        if not observations:
            return Skeleton3D(
                keypoints=np.zeros((0, 3)), confidence=np.zeros(0), skeleton="coco17"
            )

        skeleton = next(iter(observations.values())).skeleton
        n_joints = max(len(obs.keypoints) for obs in observations.values())

        # 每个观测先内部去畸变，坐标换到无畸变像素系、置信度不变
        undist: Dict[int, np.ndarray] = {}
        for cid, obs in observations.items():
            if cid not in self.P:
                continue
            undist[cid] = undistort_keypoints(obs.keypoints, self.K[cid], self.dist[cid])

        kp3 = np.full((n_joints, 3), np.nan, dtype=np.float64)
        conf3 = np.zeros(n_joints, dtype=np.float64)
        nviews = np.zeros(n_joints, dtype=np.int32)
        rerr = np.full(n_joints, np.nan, dtype=np.float64)

        for j in range(n_joints):
            pts: Dict[int, Tuple[float, float]] = {}
            cs: Dict[int, float] = {}
            for cid, kps in undist.items():
                if j >= len(kps):
                    continue
                kp = kps[j]
                if not np.isfinite(kp[0]) or float(kp[2]) < min_conf:
                    continue
                pts[cid] = (float(kp[0]), float(kp[1]))
                cs[cid] = float(kp[2])
            res = self.triangulate_point(pts, cs, min_conf=min_conf, **kwargs)
            if res is None:
                continue
            X, c, e, nv, _ang = res
            kp3[j] = X
            conf3[j] = c
            nviews[j] = nv
            rerr[j] = e

        return Skeleton3D(
            keypoints=kp3,
            confidence=conf3,
            skeleton=skeleton,
            n_views=nviews,
            reproj_err=rerr,
        )


def load_camera_rig(root: Optional[str] = None):
    """从项目默认目录读内参 / 外参，返回 ``(intrinsics, extrinsics)``。

    内参读 ``data/calibration/cam_N.yaml``，外参读
    ``data/extrinsics/table_extrinsics.yaml``（世界系 = 桌面系，X 短边 / Y 长边 /
    Z 向上，与 3D 场景一致）。缺某相机内参或外参则相应字典里没有该相机。
    """
    from ..calibration.extrinsics import load_extrinsics
    from ..calibration.intrinsics import load_intrinsics
    from ..core.config import project_root

    root = root or project_root()
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

    ext_path = os.path.join(root, "data", "extrinsics", "table_extrinsics.yaml")
    extrinsics: Dict[int, CameraExtrinsics] = (
        load_extrinsics(ext_path) if os.path.exists(ext_path) else {}
    )
    return intrinsics, extrinsics
