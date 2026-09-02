"""EasyMocap 官方多视角 SMPL 重建（完整官方管线）。

背景：现有姿态重建走「RTMPose 2D → 多视角三角化」，身体某部位若不同时被
≥2 台相机看到就无法重建。EasyMocap 的思路是把 SMPL 参数化人体模型拟合到
**三角化后的 3D 关键点**（+ 多视角 2D 重投影精修），配以分阶段 LBFGS 优化和
``reg_poses_zero`` 先验——模型先验会补全「只有单视角可见」的部位。

本模块**完整走 EasyMocap 官方管线**（不做任何自己的简化拟合）：

1. ``easymocap.smplmodel.load_model`` 加载官方 ``SMPLlayer``（body25 回归器）；
2. halpe26 2D 关键点 -> body25 序（去畸变）；
3. 官方 ``batch_triangulate`` 三角化出 body25 3D 关键点；
4. 官方 ``bbox_from_keypoints`` 由 2D 关键点算包围盒；
5. 官方 ``smpl_from_keypoints3d2d`` 分阶段 LBFGS 拟合：
   ``optimizeShape``（骨长求 betas）-> ``multi_stage_optimize``
   （全局 RT -> 3D 姿态 -> 2D 重投影精修），官方损失权重
   （shape: s3d/reg_shapes；pose: k3d/reg_poses_zero/smooth_*/reg_poses/k2d）。

约定（与 :class:`~tabletennis.core.types.CameraExtrinsics` 一致）：外参 ``(R, t)``
表示「世界系（桌面系）→ 相机系」，``X_cam = R @ X_world + t``。拟合出的 SMPL 顶点 /
关节直接就是桌面系坐标（米），与 ``viewer3d`` 的世界系一致。

用法：见 ``scripts/live_control.py`` 的 ``s`` 键。模型文件：
官方 ``load_model`` 需要 ``<model_path>/smpl/SMPL_NEUTRAL.pkl`` + ``<model_path>/
J_regressor_body25.npy``，可用 ``scripts/npz_to_smpl_pkl.py`` 从项目已有 npz 生成。
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core.types import CameraExtrinsics, CameraIntrinsics, Pose2D

# EasyMocap 代码的默认位置（可通过 EASYMOCAP_ROOT 覆盖）
DEFAULT_EASYMOCAP_ROOT = "/home/yby/projects/EasyMocap"

# halpe26 -> body25 2D 关键点映射：(halpe26 索引, body25 索引)。
# body25 关节序（OpenPose）：
#   0 Nose, 1 Neck, 2 RShoulder, 3 RElbow, 4 RWrist,
#   5 LShoulder, 6 LElbow, 7 LWrist, 8 MidHip, 9 RHip,
#   10 RKnee, 11 RAnkle, 12 LHip, 13 LKnee, 14 LAnkle,
#   15 REye, 16 LEye, 17 REar, 18 LEar,
#   19 LBigToe, 20 LSmallToe, 21 LHeel, 22 RBigToe, 23 RSmallToe, 24 RHeel
# halpe26 序（rtmlib）：0 nose, 1 LEye, 2 REye, 3 LEar, 4 REar, 5/6 L/R shoulder,
#   7/8 L/R elbow, 9/10 L/R wrist, 11/12 L/R hip, 13/14 L/R knee, 15/16 L/R ankle,
#   17 head(头顶), 18 neck, 19 hip(骨盆), 20/21 L/R big toe, 22/23 L/R small toe,
#   24/25 L/R heel。
# ⚠️ halpe 17（头顶）在 body25 里没有对应关节（body25 只有 nose/eyes/ears），故不映射。
HALPE26_TO_BODY25: List[Tuple[int, int]] = [
    (0, 0),                   # nose -> Nose
    (18, 1),                  # neck -> Neck
    (6, 2), (8, 3), (10, 4),  # R 肩/肘/腕
    (5, 5), (7, 6), (9, 7),   # L 肩/肘/腕
    (19, 8),                  # hip -> MidHip
    (12, 9), (14, 10), (16, 11),  # R 髋/膝/踝
    (11, 12), (13, 13), (15, 14),  # L 髋/膝/踝
    (2, 15), (1, 16),         # 眼 -> REye/LEye
    (4, 17), (3, 18),         # 耳 -> REar/LEar
    (21, 22), (20, 19),       # 大脚趾 -> R/L
    (23, 23), (22, 20),       # 小脚趾 -> R/L
    (25, 24), (24, 21),       # 脚跟 -> R/L
]

# 每个映射的权重（默认 1.0；脚部点 conf 略低，给 0.5 免得干扰）
_HALPE_WEIGHTS: Dict[int, float] = {20: 0.5, 21: 0.5, 22: 0.5, 23: 0.5,
                                    24: 0.5, 25: 0.5}

# 默认 2D 关键点置信度阈值（低于此值的关节不参与三角化/拟合）
DEFAULT_MIN_CONF = 0.3

# SMPL 模型所在目录（load_model 的 model_path）：须含 smpl/SMPL_NEUTRAL.pkl 与
# J_regressor_body25.npy（用 scripts/npz_to_smpl_pkl.py 生成）。
_MODEL_DIR_CANDIDATES = ["bodymodels", "data/bodymodels"]


def _resolve_model_dir(project_data_dir: Optional[str]) -> Optional[str]:
    """找 load_model 的 model_path 目录（含 smpl/SMPL_NEUTRAL.pkl + J_regressor_body25.npy）。"""
    env = os.environ.get("EASYMOCAP_SMPL_DIR")
    if env and os.path.exists(os.path.join(env, "smpl", "SMPL_NEUTRAL.pkl")):
        return env
    if project_data_dir:
        cand = os.path.join(project_data_dir, "bodymodels")
        if os.path.exists(os.path.join(cand, "smpl", "SMPL_NEUTRAL.pkl")):
            return cand
        if os.path.exists(os.path.join(project_data_dir, "smpl", "SMPL_NEUTRAL.pkl")):
            return project_data_dir
    return None


def _make_args(verbose: bool = True) -> SimpleNamespace:
    """官方 ``smpl_from_keypoints3d`` 需要的 args（Config 读 verbose/model/robust3d，
    load_weight_* 读 opts）。opts 留空 = 用官方默认损失权重。"""
    return SimpleNamespace(verbose=verbose, model="smpl", robust3d=False, opts={})


class EasymocapReconstructor:
    """EasyMocap 官方多视角 SMPL 重建器。

    完整走官方管线（load_model + batch_triangulate + smpl_from_keypoints3d2d +
    分阶段 LBFGS + 官方损失权重），输出 SMPL 网格 + 关节（桌面系，米）。
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cuda",
        easymocap_root: Optional[str] = None,
        project_data_dir: Optional[str] = None,
        verbose: bool = True,
    ) -> None:
        import torch  # noqa: F401  —— 延迟 import，未按 s 时不加载

        root = easymocap_root or os.environ.get("EASYMOCAP_ROOT") or DEFAULT_EASYMOCAP_ROOT
        if root not in sys.path:
            sys.path.insert(0, root)

        from easymocap.smplmodel import load_model

        if project_data_dir is None:
            from ..core.config import project_root
            project_data_dir = os.path.join(project_root(), "data")

        model_dir = model_path or _resolve_model_dir(project_data_dir)
        if model_dir is None:
            self._model = None
            self._error = (
                "未找到 EasyMocap 需要的 SMPL 模型目录（含 smpl/SMPL_NEUTRAL.pkl + "
                "J_regressor_body25.npy）。先用 scripts/npz_to_smpl_pkl.py 从项目 "
                "data/bodymodels/SMPL_NEUTRAL.npz 生成，或用环境变量 EASYMOCAP_SMPL_DIR "
                "指定模型目录。"
            )
            return

        use_cuda = device != "cpu"
        self._model = load_model(
            gender="neutral", use_cuda=use_cuda, model_type="smpl",
            skel_type="body25", device=torch.device(device), model_path=model_dir,
        )
        self._error = None
        self._device = self._model.device
        self._faces = np.asarray(self._model.faces, dtype=np.int64)
        self._n_shapes = 10
        self._verbose = verbose
        print(f"[EasyMocap] 官方 SMPLlayer 已加载（model_dir={model_dir}，"
              f"{self._model.nVertices} 顶点 / {self._n_shapes} 形状，device={self._device}）")

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
    # 观测构建：halpe26 2D 关键点 -> body25 序（去畸变）
    # ------------------------------------------------------------------
    @staticmethod
    def _to_body25_2d(
        poses_per_cam: Dict[int, Pose2D],
        intrinsics: Dict[int, CameraIntrinsics],
        extrinsics: Dict[int, CameraExtrinsics],
        min_conf: float,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """把每相机的 halpe26 关键点去畸变后映射成 body25 序 2D 关键点。

        Returns:
            kp2d (nViews, 25, 3)：``[x, y, conf]`` 像素坐标，缺失关节 conf=0。
            bbox (nViews, 5)：官方 ``bbox_from_keypoints`` 包围盒 ``[x1,y1,x2,y2,conf]``。
            Pall (nViews, 3, 4)：``K @ [R|t]`` 投影矩阵（顺序与 kp2d 一致）。
            视角不足 2 个返回 (None, None, None)。
        """
        from .triangulate import undistort_keypoints
        from easymocap.estimator.wrapper_base import bbox_from_keypoints

        kp2d_list: List[np.ndarray] = []
        bbox_list: List[np.ndarray] = []
        P_list: List[np.ndarray] = []
        for cid in sorted(poses_per_cam.keys()):
            pose = poses_per_cam[cid]
            K = intrinsics.get(cid)
            ext = extrinsics.get(cid)
            if K is None or ext is None or pose is None:
                continue
            kp = np.asarray(pose.keypoints, dtype=np.float32)
            if kp.ndim != 2 or kp.shape[0] < 26:
                continue
            # 去畸变（与三角化同款，得到线性投影可直接用的像素坐标）
            undist = undistort_keypoints(kp, K.K, K.dist)
            out = np.zeros((25, 3), dtype=np.float32)
            for halpe_idx, b25_idx in HALPE26_TO_BODY25:
                x, y, c = undist[halpe_idx]
                if not np.isfinite(x) or not np.isfinite(y) or c < min_conf:
                    continue
                out[b25_idx] = (x, y, float(c) * _HALPE_WEIGHTS.get(halpe_idx, 1.0))
            if (out[:, 2] > 0).sum() < 3:  # 有效关键点太少，该视角丢弃
                continue
            kp2d_list.append(out)
            bbox_list.append(np.asarray(bbox_from_keypoints(out), dtype=np.float32))
            P_list.append(K.K @ np.hstack([ext.R, ext.t.reshape(3, 1)]))

        if len(kp2d_list) < 2:
            return None, None, None
        return (np.stack(kp2d_list), np.stack(bbox_list), np.stack(P_list))

    # ------------------------------------------------------------------
    # 拟合（官方管线）
    # ------------------------------------------------------------------
    def reconstruct(
        self,
        poses_per_cam: Dict[int, Pose2D],
        intrinsics: Dict[int, CameraIntrinsics],
        extrinsics: Dict[int, CameraExtrinsics],
        *,
        min_conf: float = DEFAULT_MIN_CONF,
    ) -> Optional[dict]:
        """用官方 EasyMocap 管线拟合 SMPL。

        Args:
            poses_per_cam: ``{cid: Pose2D}``，每个相机一个人（halpe26）。
            intrinsics / extrinsics: 标定内外参（桌面系）。

        Returns:
            含 ``vertices`` (6890,3)、``joints`` (24,3，SMPL 原生序，供 viewer)、
            ``joints_body25`` (25,3)、``faces`` (13776,3)、``params`` 的 dict
            （桌面系，米）；观测不足或模型缺失时返回 None。
        """
        if self._model is None:
            print(f"[EasyMocap] {self._error}")
            return None

        from easymocap.dataset import CONFIG
        from easymocap.mytools.triangulator import batch_triangulate
        from easymocap.pipeline import smpl_from_keypoints3d2d
        from easymocap.pipeline.weight import load_weight_pose, load_weight_shape
        from easymocap.smplmodel.body_param import check_keypoints

        kp2d, bboxes, Pall = self._to_body25_2d(
            poses_per_cam, intrinsics, extrinsics, min_conf
        )
        if kp2d is None:
            print("[EasyMocap] 可用视角不足 2 个，跳过拟合。")
            return None

        # 1) 官方 batch_triangulate：body25 2D -> 3D（世界系=桌面系，Z 向上）
        kp3d = batch_triangulate(kp2d, Pall, min_view=2)          # (25, 4)
        kp3d = check_keypoints(kp3d, 1, min_conf=min_conf)         # conf<0.3 置 0
        n_views = kp2d.shape[0]
        n_valid = int((kp3d[:, 3] > 0).sum())
        print(f"[EasyMocap] {n_views} 视角三角化，有效 3D 关节 {n_valid}/25")

        # 2) 官方 smpl_from_keypoints3d2d：
        #    kp3ds (1,25,4) / kp2ds (1,nViews,25,3) / bboxes (1,nViews,5) / Pall (nViews,3,4)
        args = _make_args(verbose=self._verbose)
        weight_shape = load_weight_shape("smpl", args.opts)
        weight_pose = load_weight_pose("smpl", args.opts)
        params = smpl_from_keypoints3d2d(
            self._model,
            kp3d[None],                # (1, 25, 4)
            kp2d[None],                # (1, nViews, 25, 3)
            bboxes[None],              # (1, nViews, 5)
            Pall,                      # (nViews, 3, 4)
            config=CONFIG["body25"],
            args=args,
            weight_shape=weight_shape,
            weight_pose=weight_pose,
        )
        if params is None:
            print("[EasyMocap] 官方拟合返回空，跳过。")
            return None

        # 3) 前向输出：网格 + body25 关节 + SMPL 原生 24 关节（viewer 用）
        with __import__("torch").no_grad():
            verts = self._model(
                return_verts=True, return_tensor=False, **params
            )                                    # (1, 6890, 3)
            j25 = self._model(
                return_verts=False, return_tensor=False, **params
            )                                    # (1, 25, 3)  body25
            j24 = self._model(
                return_verts=False, return_tensor=False,
                return_smpl_joints=True, **params
            )                                    # (1, 24, 3)  SMPL 原生序

        return {
            "vertices": np.asarray(verts, dtype=np.float64).reshape(-1, 3),
            "joints": np.asarray(j24, dtype=np.float64).reshape(-1, 3),
            "joints_body25": np.asarray(j25, dtype=np.float64).reshape(-1, 3),
            "faces": self._faces,
            "params": {k: np.asarray(v) for k, v in params.items()},
        }
