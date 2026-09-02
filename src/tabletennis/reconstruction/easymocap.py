"""EasyMocap 多视角 SMPL 重建（不依赖三角测量）。

背景：现有姿态重建走「RTMPose 2D → 多视角三角化」，身体某部位若不同时被
≥2 台相机看到就无法重建。EasyMocap 的思路是把 SMPL 参数化人体模型**直接拟合到
多视角 2D 关键点**（重投影损失 + 姿态/形状先验），模型先验会补全「只有单视角可见」
的部位，因此不依赖三角测量（三角化仅用于给根部平移一个粗略初值）。

本模块只做基础版：

- 单人、SMPL 身体模型（24 关节，无手/脸细节，符合「面部/手部不用精细」的要求）；
- 用 EasyMocap 自带的 ``easymocap.bodymodel.smpl.SMPLModel``（纯 torch+numpy，无
  smplx/chumpy 依赖）做前向；
- Adam 直接拟合 pose/shape/Rh/Th 到多视角重投影误差，速度与精度后续再优化。

约定（与 :class:`~tabletennis.core.types.CameraExtrinsics` 一致）：外参 ``(R, t)``
表示「世界系（桌面系）→ 相机系」，``X_cam = R @ X_world + t``。拟合出的 SMPL 顶点 /
关节直接就是桌面系坐标（米），与 ``viewer3d`` 的世界系一致。

用法：见 ``scripts/live_control.py`` 的 ``s`` 键。模型文件路径优先级：
环境变量 ``SMPL_MODEL_PATH`` > 构造参数 ``model_path`` > 项目 ``data/bodymodels/``
下的常见文件名（``SMPL_NEUTRAL.npz`` / ``basicmodel_neutral_*.pkl`` 等）。
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core.types import CameraExtrinsics, CameraIntrinsics, Pose2D

# EasyMocap 代码的默认位置（可通过 EASYMOCAP_ROOT 覆盖）
DEFAULT_EASYMOCAP_ROOT = "/home/yby/projects/EasyMocap"

# halpe26 -> SMPL24 关键点映射：(halpe26 索引, SMPL 关节索引)。
# ⚠️ SMPL 模型文件的关节是「原生顺序」（EasyMocap 的 SMPLModel 不做重排），
# 由 kintree_table 推出：0 pelvis, 1/2 hip, 3 spine1, 4/5 knee, 6 spine2,
# 7/8 ankle, 9 spine3, 10/11 foot, 12 neck, 13/14 collar, 15 head,
# 16/17 shoulder, 18/19 elbow, 20/21 wrist, 22/23 hand。
HALPE26_TO_SMPL24: List[Tuple[int, int]] = [
    (19, 0),                 # hip(骨盆)      -> pelvis
    (11, 1), (12, 2),        # 左右髋          -> hip
    (13, 4), (14, 5),        # 左右膝          -> knee
    (15, 7), (16, 8),        # 左右踝          -> ankle
    (24, 10), (25, 11),      # 左右脚跟        -> foot（脚部只取脚跟，权重略低）
    (18, 12),                # 颈              -> neck
    (17, 15), (0, 15),       # 头顶 + 鼻子     -> head
    (5, 16), (6, 17),        # 左右肩          -> shoulder
    (7, 18), (8, 19),        # 左右肘          -> elbow
    (9, 20), (10, 21),       # 左右腕          -> wrist
]

# 每个映射的默认权重（脚跟->foot 用 0.5，其余 1.0）
_HALPE_WEIGHTS: Dict[int, float] = {24: 0.5, 25: 0.5}

# 默认拟合超参
DEFAULT_MIN_CONF = 0.3   # 2D 关键点置信度阈值，低于此值不参与拟合
DEFAULT_N_ITER = 200     # Adam 迭代次数（基础版，未做加速）
DEFAULT_W_POSE = 1e-2    # 姿态先验权重（正则到 T-pose）
DEFAULT_W_SHAPE = 1e-1   # 形状先验权重（正则到平均体型）

# 全局旋转 Rh 的初值：SMPL 模型是 Y 轴朝上，我们的桌面系是 Z 轴朝上，
# 绕 X 轴 +90° 把 Y->Z，让人体从「直立」起步（否则零初值易陷入倒立/倾斜的局部最优）。
DEFAULT_RH_INIT = (np.pi / 2, 0.0, 0.0)

# 常见 SMPL 模型文件名（在 data/bodymodels/ 下按顺序尝试）
_MODEL_CANDIDATES = [
    "SMPL_NEUTRAL.npz",
    "SMPL_MALE.npz",
    "SMPL_FEMALE.npz",
    "basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl",
    "basicmodel_m_lbs_10_207_0_v1.1.0.pkl",
    "basicmodel_f_lbs_10_207_0_v1.1.0.pkl",
]


def _resolve_model_path(model_path: Optional[str], project_data_dir: str) -> Optional[str]:
    """按优先级找 SMPL 模型文件：env > 参数 > 项目默认目录。找不到返回 None。"""
    if model_path and os.path.exists(model_path):
        return model_path
    env = os.environ.get("SMPL_MODEL_PATH")
    if env and os.path.exists(env):
        return env
    for name in _MODEL_CANDIDATES:
        cand = os.path.join(project_data_dir, "bodymodels", name)
        if os.path.exists(cand):
            return cand
    return None


def _triangulate_root(pts: List[np.ndarray], P_list: List[np.ndarray]) -> np.ndarray:
    """单点 DLT 三角化：仅用于给 SMPL 根部平移一个粗略初值（非重建本身）。

    Args:
        pts: 各视角去畸变后的 2D 关键点 ``[x, y]``（像素）。
        P_list: 各视角投影矩阵 ``(3, 4)``。
    """
    A = []
    for (x, y), P in zip(pts, P_list):
        A.append(x * P[2] - P[0])
        A.append(y * P[2] - P[1])
    A = np.asarray(A, dtype=np.float64).reshape(-1, 4)
    if A.shape[0] < 4:
        return None
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return X[:3] / X[3]


class EasymocapReconstructor:
    """EasyMocap SMPL 多视角重建器（不依赖三角测量）。

    加载 EasyMocap 的 ``SMPLModel``，把 SMPL 拟合到多视角 2D 关键点的重投影误差上。
    输出 SMPL 顶点网格 + 24 关节（桌面系，米）。
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        easymocap_root: Optional[str] = None,
        project_data_dir: Optional[str] = None,
    ) -> None:
        import torch  # noqa: F401  —— 延迟 import，未按 s 时不加载

        root = easymocap_root or os.environ.get("EASYMOCAP_ROOT") or DEFAULT_EASYMOCAP_ROOT
        if root not in sys.path:
            sys.path.insert(0, root)

        from easymocap.bodymodel.smpl import SMPLModel

        if project_data_dir is None:
            from ..core.config import project_root
            project_data_dir = os.path.join(project_root(), "data")

        path = _resolve_model_path(model_path, project_data_dir)
        if path is None:
            self._model = None
            self._error = (
                "未找到 SMPL 模型文件。请到 smpl.is.tue.mpg.de（或 smpl-x.is.tue.mpg.de）"
                "注册并下载 SMPL 中性模型，把 .npz/.pkl 放到项目的 data/bodymodels/ 下，"
                "或用环境变量 SMPL_MODEL_PATH 指定路径。推荐 .npz 格式（免 chumpy 依赖）。"
            )
            return

        # NUM_SHAPES=10：标准 SMPL 只用前 10 个形状主成分（betas 10 维），
        # 原始 shapedirs 有 300 列，需截断。
        self._model = SMPLModel(model_path=path, regressor_path=None, device=device,
                                NUM_SHAPES=10)
        self._error = None
        self._device = self._model.device
        self._faces = self._model.faces  # (13776, 3) 面索引
        self._n_joints = 24
        self._n_shapes = self._model.NUM_SHAPES   # 10
        self._n_poses = self._model.NUM_POSES     # 69（24 关节 - 根旋转）
        print(f"[EasyMocap] SMPL 模型已加载：{os.path.basename(path)}"
              f"（{self._model.nVertices} 顶点 / {self._n_joints} 关节 / "
              f"{self._n_shapes} 形状 / {self._n_poses} 姿态，device={self._model.device}）")

    @property
    def ready(self) -> bool:
        """模型是否已成功加载（False 且 _error 非空表示缺模型文件）。"""
        return self._model is not None

    @property
    def error(self) -> Optional[str]:
        return self._error

    @property
    def faces(self) -> np.ndarray:
        return self._faces

    # ------------------------------------------------------------------
    # 观测构建：halpe26 2D 关键点 -> (smpl_idx, x, y, conf) 列表
    # ------------------------------------------------------------------
    @staticmethod
    def _collect_observations(
        poses_per_cam: Dict[int, Pose2D],
        intrinsics: Dict[int, CameraIntrinsics],
        extrinsics: Dict[int, CameraExtrinsics],
        min_conf: float,
    ) -> Tuple[Dict[int, List[Tuple[int, float, float, float]]], Dict[int, np.ndarray]]:
        """把每个相机的 halpe26 关键点去畸变后，映射成 SMPL 关节观测。

        Returns:
            obs_by_cam: ``{cid: [(smpl_idx, x, y, conf), ...]}``。
            P_by_cam: ``{cid: (3,4) 投影矩阵}``。
        """
        from .triangulate import undistort_keypoints

        obs_by_cam: Dict[int, List[Tuple[int, float, float, float]]] = {}
        P_by_cam: Dict[int, np.ndarray] = {}
        for cid, pose in poses_per_cam.items():
            K = intrinsics.get(cid)
            ext = extrinsics.get(cid)
            if K is None or ext is None or pose is None:
                continue
            kp = np.asarray(pose.keypoints, dtype=np.float32)
            if kp.ndim != 2 or kp.shape[0] < 26:
                continue
            # 去畸变（与三角化同款，得到线性投影可直接用的像素坐标）
            undist = undistort_keypoints(kp, K.K, K.dist)
            obs: List[Tuple[int, float, float, float]] = []
            for halpe_idx, smpl_idx in HALPE26_TO_SMPL24:
                x, y, c = undist[halpe_idx]
                if not np.isfinite(x) or not np.isfinite(y) or c < min_conf:
                    continue
                w = _HALPE_WEIGHTS.get(halpe_idx, 1.0)
                obs.append((smpl_idx, float(x), float(y), float(c) * w))
            if obs:
                obs_by_cam[cid] = obs
                P_by_cam[cid] = K.K @ np.hstack([ext.R, ext.t.reshape(3, 1)])
        return obs_by_cam, P_by_cam

    # ------------------------------------------------------------------
    # 拟合
    # ------------------------------------------------------------------
    def reconstruct(
        self,
        poses_per_cam: Dict[int, Pose2D],
        intrinsics: Dict[int, CameraIntrinsics],
        extrinsics: Dict[int, CameraExtrinsics],
        *,
        n_iter: int = DEFAULT_N_ITER,
        min_conf: float = DEFAULT_MIN_CONF,
        w_pose: float = DEFAULT_W_POSE,
        w_shape: float = DEFAULT_W_SHAPE,
        rh_init: Tuple[float, float, float] = DEFAULT_RH_INIT,
    ) -> Optional[dict]:
        """把 SMPL 拟合到多视角 2D 关键点（重投影损失，不做三角测量）。

        Args:
            poses_per_cam: ``{cid: Pose2D}``，每个相机一个（halpe26）姿态。
            intrinsics / extrinsics: 标定内外参（桌面系）。

        Returns:
            含 ``vertices`` (6890,3)、``joints`` (24,3)、``faces`` (13776,3) 的
            dict（桌面系，米）；观测不足或模型缺失时返回 None。
        """
        import torch

        if self._model is None:
            print(f"[EasyMocap] {self._error}")
            return None

        obs_by_cam, P_by_cam = self._collect_observations(
            poses_per_cam, intrinsics, extrinsics, min_conf
        )
        if len(obs_by_cam) < 1:
            print("[EasyMocap] 无可用视角的关键点观测，跳过。")
            return None

        device = self._device
        dtype = torch.float32

        # 根部平移初值：仅用骨盆点做单点 DLT（≥2 视角时），否则给桌面中上方的默认值
        root_pts: List[np.ndarray] = []
        root_Ps: List[np.ndarray] = []
        for cid, obs in obs_by_cam.items():
            for smpl_idx, x, y, c in obs:
                if smpl_idx == 0:  # pelvis
                    root_pts.append(np.array([x, y]))
                    root_Ps.append(P_by_cam[cid])
        root = _triangulate_root(root_pts, root_Ps)
        th0 = root if root is not None else np.array([0.0, 0.0, 0.9])
        print(f"[EasyMocap] {len(obs_by_cam)} 视角观测，根部初值 Th={th0.round(2)}")

        # 可微参数：pose(1,n_poses) / shape(1,n_shapes) / Rh(1,3) / Th(1,3)
        poses = torch.zeros((1, self._n_poses), dtype=dtype, device=device, requires_grad=True)
        shapes = torch.zeros((1, self._n_shapes), dtype=dtype, device=device, requires_grad=True)
        Rh = torch.tensor(rh_init, dtype=dtype, device=device).reshape(1, 3).requires_grad_(True)
        Th = torch.tensor(th0, dtype=dtype, device=device).reshape(1, 3).requires_grad_(True)

        optimizer = torch.optim.Adam([poses, shapes, Rh, Th], lr=0.05)

        n_obs = sum(len(o) for o in obs_by_cam.values())
        for step in range(n_iter):
            optimizer.zero_grad()
            joints = self._model.forward(
                return_verts=False,
                return_smpl_joints=True,
                return_tensor=True,
                poses=poses,
                shapes=shapes,
                Rh=Rh,
                Th=Th,
            )  # (1, 24, 3)
            J = joints[0]  # (24, 3)

            reproj = torch.zeros((), dtype=dtype, device=device)
            for cid, obs in obs_by_cam.items():
                P = torch.tensor(P_by_cam[cid], dtype=dtype, device=device)
                Jh = torch.cat([J, torch.ones(24, 1, dtype=dtype, device=device)], dim=1)
                pj = (Jh @ P.T)  # (24, 3)
                pj = pj[:, :2] / pj[:, 2:3].clamp(min=1e-6)  # (24, 2)
                for smpl_idx, x, y, c in obs:
                    target = torch.tensor([x, y], dtype=dtype, device=device)
                    reproj = reproj + c * ((pj[smpl_idx] - target) ** 2).sum()

            loss = reproj / max(n_obs, 1)
            loss = loss + w_pose * (poses ** 2).sum() + w_shape * (shapes ** 2).sum()
            loss.backward()
            optimizer.step()

        # 前向得到最终顶点 + 关节（detach 到 numpy，桌面系）
        with torch.no_grad():
            verts = self._model.forward(
                return_verts=True, return_tensor=False,
                poses=poses, shapes=shapes, Rh=Rh, Th=Th,
            )  # (1, 6890, 3)
            joints_out = self._model.forward(
                return_verts=False, return_smpl_joints=True, return_tensor=False,
                poses=poses, shapes=shapes, Rh=Rh, Th=Th,
            )  # (1, 24, 3)

        return {
            "vertices": np.asarray(verts, dtype=np.float64).reshape(-1, 3),
            "joints": np.asarray(joints_out, dtype=np.float64).reshape(-1, 3),
            "faces": np.asarray(self._faces, dtype=np.int64).reshape(-1, 3),
        }
