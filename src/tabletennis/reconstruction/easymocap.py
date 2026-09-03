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

# VPoser 预训练 checkpoint 默认位置（~2.7MB，可 VPOSER_CKPT 环境变量 / --vposer-ckpt 覆盖）
DEFAULT_VPOSER_CKPT = "/mnt/newdisk1/vposer/TR00_E096.pt"

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

# 每个映射的权重（默认 1.0）。脚部点（大脚趾/小脚趾/脚跟，halpe 20-25）conf 本就
# 偏低，若再降权会在 min_conf 门槛处被双重丢弃 → 踝关节方向失去全部 2D 约束，
# SMPL 优化器只能乱拧腿（腿部扭曲主因之一）。故脚点不再降权，噪声交给置信度
# 加权 DLT + min_conf 本身兜底（勿叠加惩罚）。
_HALPE_WEIGHTS: Dict[int, float] = {}

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


def convert_npz_to_pkl(npz_path: str, model_dir: str, easymocap_root: str) -> str:
    """npz（纯 numpy）-> 官方 load_model 可读的 pkl（纯 numpy，无 chumpy），并拷 J_regressor。

    data/ 整体被 gitignore，fresh checkout 只有 npz 没有 pkl；本函数让
    :class:`EasymocapReconstructor` 在缺模型文件时**自动生成**，无需手动跑脚本。
    scripts/npz_to_smpl_pkl.py 是它的 CLI 包装（单一实现，勿另写一套转换）。
    """
    import pickle

    d = np.load(npz_path)
    keys = ["f", "J_regressor", "v_template", "weights", "posedirs",
            "shapedirs", "kintree_table"]
    out = {k: np.ascontiguousarray(d[k]) for k in keys}
    os.makedirs(os.path.join(model_dir, "smpl"), exist_ok=True)
    with open(os.path.join(model_dir, "smpl", "SMPL_NEUTRAL.pkl"), "wb") as f:
        pickle.dump(out, f, protocol=4)
    # J_regressor_body25.npy（load_model(skel_type='body25') 依赖）
    import shutil

    src_reg = os.path.join(easymocap_root, "data", "smplx", "J_regressor_body25.npy")
    dst_reg = os.path.join(model_dir, "J_regressor_body25.npy")
    if os.path.exists(src_reg):
        os.makedirs(model_dir, exist_ok=True)
        shutil.copyfile(src_reg, dst_reg)
    return model_dir


def _ensure_model_data(project_data_dir: Optional[str],
                       easymocap_root: str) -> Optional[str]:
    """npz 在而 pkl 缺失时自动转换；成功返回 model_dir，否则 None。"""
    if not project_data_dir:
        return None
    npz = os.path.join(project_data_dir, "bodymodels", "SMPL_NEUTRAL.npz")
    model_dir = os.path.join(project_data_dir, "bodymodels")
    if not os.path.exists(npz):
        return None
    print(f"[EasyMocap] 缺 smpl/SMPL_NEUTRAL.pkl，从 {npz} 自动生成…")
    try:
        convert_npz_to_pkl(npz, model_dir, easymocap_root)
    except Exception as exc:  # noqa: BLE001
        print(f"[EasyMocap] npz->pkl 自动转换失败（{exc}），请手动跑 "
              f"scripts/npz_to_smpl_pkl.py", file=sys.stderr)
        return None
    if os.path.exists(os.path.join(model_dir, "smpl", "SMPL_NEUTRAL.pkl")):
        return model_dir
    return None


def _make_args(verbose: bool = True) -> SimpleNamespace:
    """官方 ``smpl_from_keypoints3d`` 需要的 args（Config 读 verbose/model/robust3d，
    load_weight_* 读 opts）。opts 留空 = 用官方默认损失权重。"""
    return SimpleNamespace(verbose=verbose, model="smpl", robust3d=False, opts={})


def _pose_to_body25_view(kp: np.ndarray, K: np.ndarray, dist, min_conf: float):
    """单视角 halpe26 -> body25（去畸变 + 置信度阈值 + 脚部降权）。

    返回 ``(25, 3)`` 的 ``[x, y, conf]``；有效关键点 < 3 时返回 None（该视角丢弃）。
    """
    from .triangulate import undistort_keypoints

    undist = undistort_keypoints(kp, K, dist)
    out = np.zeros((25, 3), dtype=np.float32)
    for halpe_idx, b25_idx in HALPE26_TO_BODY25:
        x, y, c = undist[halpe_idx]
        if not np.isfinite(x) or not np.isfinite(y) or c < min_conf:
            continue
        out[b25_idx] = (x, y, float(c) * _HALPE_WEIGHTS.get(halpe_idx, 1.0))
    if (out[:, 2] > 0).sum() < 3:
        return None
    return out


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
            # data/ 被 gitignore：fresh checkout 常有 npz 无 pkl -> 自动转换
            model_dir = _ensure_model_data(project_data_dir, root)
        if model_dir is None:
            self._model = None
            self._error = (
                "未找到 EasyMocap 需要的 SMPL 模型目录（含 smpl/SMPL_NEUTRAL.pkl + "
                "J_regressor_body25.npy）。项目需有 data/bodymodels/SMPL_NEUTRAL.npz "
                "（有它则按 S 会自动生成 pkl），或用环境变量 EASYMOCAP_SMPL_DIR 指定。"
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
            out = _pose_to_body25_view(kp, K.K, K.dist, min_conf)
            if out is None:
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

    # ------------------------------------------------------------------
    # 批量多帧拟合（官方视频管线：一次拟合 T 帧，激活时间平滑损失）
    # ------------------------------------------------------------------
    @staticmethod
    def _obs_to_body25_fixed(
        poses_per_cam: Dict[int, Pose2D],
        intrinsics: Dict[int, CameraIntrinsics],
        extrinsics: Dict[int, CameraExtrinsics],
        min_conf: float,
        view_ids: List[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """单帧观测 → 固定视角序的 ``(kp2d (nViews,25,3), bbox (nViews,5))``。

        与 :meth:`_to_body25_2d` 的唯一区别：视角序固定为 ``view_ids``，缺失/无效视角
        填零（kp2d 全 0、bbox conf=0），保证整批帧共享同一投影矩阵 Pall 与视角数——
        这是官方批量拟合 ``smpl_from_keypoints3d2d`` 的硬性要求。
        """
        from easymocap.estimator.wrapper_base import bbox_from_keypoints

        kp2d = np.zeros((len(view_ids), 25, 3), dtype=np.float32)
        bbox = np.zeros((len(view_ids), 5), dtype=np.float32)
        for i, cid in enumerate(view_ids):
            pose = poses_per_cam.get(cid)
            K = intrinsics.get(cid)
            ext = extrinsics.get(cid)
            if pose is None or K is None or ext is None:
                continue
            kp = np.asarray(pose.keypoints, dtype=np.float32)
            if kp.ndim != 2 or kp.shape[0] < 26:
                continue
            view = _pose_to_body25_view(kp, K.K, K.dist, min_conf)
            if view is None:
                continue
            kp2d[i] = view
            bbox[i] = np.asarray(bbox_from_keypoints(view), dtype=np.float32)
        return kp2d, bbox

    def reconstruct_batch(
        self,
        frames_obs: List[Dict[int, Pose2D]],
        intrinsics: Dict[int, CameraIntrinsics],
        extrinsics: Dict[int, CameraExtrinsics],
        *,
        min_conf: float = DEFAULT_MIN_CONF,
        view_ids: Optional[List[int]] = None,
    ) -> Optional[List[Optional[dict]]]:
        """官方多帧批量 SMPL 拟合（视频管线）。

        与单帧 :meth:`reconstruct` 的区别：把整段（或一个窗口的）帧**一次性**喂给官方
        ``smpl_from_keypoints3d2d``（``nFrames=T``），让 ``smooth_body``/``smooth_poses``/
        ``smooth_Rh`` 时间平滑损失真正生效——单视角关节不再只靠一条 2D 射线约束，而是
        由相邻帧平滑 + 模型先验补全，帧间动作自然连贯。

        Args:
            frames_obs: 每帧一个 ``{cid: Pose2D}``；空 dict 表示该帧无人（官方管线会对
                完全无观测的帧做相邻帧插值，输出仍连续）。
            view_ids: 固定相机顺序（投影矩阵 Pall 的行序）。默认 ``sorted(所有出现过的 cid)``。

        Returns:
            ``List[Optional[dict]]``，与 :meth:`reconstruct` 单帧返回格式一致（每帧一份
            vertices/joints/joints_body25/params）；首/尾完全无人帧为 ``None``（调用方
            标 no_person 且不写档）。整段无有效观测时返回 None。
        """
        if self._model is None:
            print(f"[EasyMocap] {self._error}")
            return None

        from easymocap.dataset import CONFIG
        from easymocap.mytools.triangulator import batch_triangulate
        from easymocap.pipeline import smpl_from_keypoints3d2d
        from easymocap.pipeline.weight import load_weight_pose, load_weight_shape
        from easymocap.smplmodel.body_param import check_keypoints, select_nf

        T = len(frames_obs)
        if view_ids is None:
            seen: set = set()
            for obs in frames_obs:
                seen.update(obs.keys())
            view_ids = sorted(seen)
        view_ids = sorted(view_ids)
        if len(view_ids) < 2:
            print("[EasyMocap] 批量拟合需要 ≥2 个标定视角。")
            return None

        # 固定投影矩阵（nViews, 3, 4）
        Pall = np.stack([
            intrinsics[cid].K
            @ np.hstack([extrinsics[cid].R, extrinsics[cid].t.reshape(3, 1)])
            for cid in view_ids
        ])

        # 观测 → 对齐数组
        kp2ds_list: List[np.ndarray] = []
        bboxes_list: List[np.ndarray] = []
        kp3ds_list: List[np.ndarray] = []
        for obs in frames_obs:
            kp2d, bbox = self._obs_to_body25_fixed(
                obs, intrinsics, extrinsics, min_conf, view_ids)
            kp3d = batch_triangulate(kp2d, Pall, min_view=2)      # (25, 4)
            kp3d = check_keypoints(kp3d, 1, min_conf=min_conf)
            kp2ds_list.append(kp2d)
            bboxes_list.append(bbox)
            kp3ds_list.append(kp3d)
        kp2ds = np.stack(kp2ds_list)        # (T, nViews, 25, 3)
        bboxes = np.stack(bboxes_list)      # (T, nViews, 5)
        kp3ds = np.stack(kp3ds_list)        # (T, 25, 4)

        n_valid_max = int((kp3ds[..., 3] > 0).sum(axis=1).max())
        print(f"[EasyMocap] 批量拟合 {T} 帧 × {len(view_ids)} 视角，"
              f"单帧有效 3D 关节最多 {n_valid_max}/25")

        # 掐掉首尾「完全无有效 3D 关节」的帧：官方 get_interp_by_keypoints 对首/尾
        # 空帧做相邻帧插值时会 index 越界（left=-1 / right=T）。中间空帧仍交给官方
        # 插值补全；首尾空帧返回 None，由调用方标 no_person。
        nonempty = (kp3ds[..., 3] > 0).any(axis=1)              # (T,) bool
        if not nonempty.any():
            print("[EasyMocap] 整段无有效 3D 关节，跳过批量拟合。")
            return None
        first = int(np.argmax(nonempty))
        last = int(len(nonempty) - 1 - np.argmax(nonempty[::-1]))
        if first > 0 or last < T - 1:
            print(f"[EasyMocap] 首/尾空帧修剪：保留 {first}..{last} "
                  f"（{last - first + 1}/{T} 帧）")
        kp2ds_fit = kp2ds[first:last + 1]
        bboxes_fit = bboxes[first:last + 1]
        kp3ds_fit = kp3ds[first:last + 1]
        T_fit = last - first + 1

        args = _make_args(verbose=self._verbose)
        weight_shape = load_weight_shape("smpl", args.opts)
        weight_pose = load_weight_pose("smpl", args.opts)
        params = smpl_from_keypoints3d2d(
            self._model,
            kp3ds_fit,      # (T_fit, 25, 4)
            kp2ds_fit,      # (T_fit, nViews, 25, 3)
            bboxes_fit,     # (T_fit, nViews, 5)
            Pall,           # (nViews, 3, 4)
            config=CONFIG["body25"],
            args=args,
            weight_shape=weight_shape,
            weight_pose=weight_pose,
        )
        if params is None:
            print("[EasyMocap] 官方批量拟合返回空。")
            return None

        # 一次前向出整批顶点/关节，再按帧切分（首尾被修剪的帧填 None）
        import torch
        with torch.no_grad():
            verts_all = self._model(return_verts=True, return_tensor=False, **params)  # (T_fit, 6890, 3)
            j25_all = self._model(return_verts=False, return_tensor=False, **params)   # (T_fit, 25, 3)
            j24_all = self._model(
                return_verts=False, return_tensor=False,
                return_smpl_joints=True, **params
            )                                                                          # (T_fit, 24, 3)

        results: List[Optional[dict]] = [None] * T
        for t in range(T_fit):
            p = select_nf(params, t)
            results[first + t] = {
                "vertices": np.asarray(verts_all[t], dtype=np.float64),
                "joints": np.asarray(j24_all[t], dtype=np.float64),
                "joints_body25": np.asarray(j25_all[t], dtype=np.float64),
                "faces": self._faces,
                "params": {k: np.asarray(v) for k, v in p.items()},
            }
        return results

    def reconstruct_vposer(
        self,
        frames_obs: List[Dict[int, Pose2D]],
        intrinsics: Dict[int, CameraIntrinsics],
        extrinsics: Dict[int, CameraExtrinsics],
        *,
        min_conf: float = DEFAULT_MIN_CONF,
        view_ids: Optional[List[int]] = None,
        vposer_ckpt: Optional[str] = None,
        n_iter: int = 150,
        lambda_z: float = 1e-3,
    ) -> Optional[List[Optional[dict]]]:
        """VPoser 潜空间逐帧拟合（治本）：三角化 body25 后，用 VPoser 潜变量替代
        ``smpl_from_keypoints3d2d`` 拟合姿态——姿态恒在自然流形上，从根上消掉
        腿部扭曲 / 翻转 / 反关节。

        与 :meth:`reconstruct_batch` 的区别：不做整段 batch 时间平滑，逐帧独立拟合
        （VPoser 先验本身保证姿态自然）。输出格式与 ``reconstruct_batch`` 完全一致
        （vertices/joints/joints_body25/faces/params），下游合并/存盘可直接复用。
        """
        if self._model is None:
            print(f"[EasyMocap] {self._error}")
            return None

        from easymocap.mytools.triangulator import batch_triangulate
        from easymocap.smplmodel.body_param import check_keypoints

        from .vposer import fit_frame, load_vposer

        T = len(frames_obs)
        if view_ids is None:
            seen: set = set()
            for obs in frames_obs:
                seen.update(obs.keys())
            view_ids = sorted(seen)
        view_ids = sorted(view_ids)
        if len(view_ids) < 2:
            print("[EasyMocap] VPoser 拟合需要 ≥2 个标定视角。")
            return None

        Pall = np.stack([
            intrinsics[cid].K
            @ np.hstack([extrinsics[cid].R, extrinsics[cid].t.reshape(3, 1)])
            for cid in view_ids
        ])

        ckpt = vposer_ckpt or os.environ.get("VPOSER_CKPT") or DEFAULT_VPOSER_CKPT
        vposer = load_vposer(ckpt, device=str(self._device))
        print(f"[EasyMocap] VPoser 已加载（{ckpt}），逐帧潜空间拟合 {T} 帧")

        results: List[Optional[dict]] = [None] * T
        for t, obs in enumerate(frames_obs):
            if not obs:
                continue
            kp2d, _ = self._obs_to_body25_fixed(
                obs, intrinsics, extrinsics, min_conf, view_ids)
            kp3d = batch_triangulate(kp2d, Pall, min_view=2)      # (25, 4)
            kp3d = check_keypoints(kp3d, 1, min_conf=min_conf)
            target = np.asarray(kp3d[:, :3], dtype=np.float64)     # (25, 3)
            conf = np.asarray(kp3d[:, 3], dtype=np.float64)        # (25,)
            target[conf < min_conf] = np.nan                       # 低置信度关节当缺失
            r = fit_frame(vposer, self._model, target, n_iter=n_iter, lambda_z=lambda_z)
            if r is None:
                continue
            r["faces"] = self._faces
            results[t] = r
        return results
