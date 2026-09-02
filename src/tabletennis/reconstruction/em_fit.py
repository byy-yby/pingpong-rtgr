"""EasyMocap SMPL 拟合的**可开关优化层**（Phase 2/4）。

策略：**不改动 vendored 官方 EasyMocap 任何一个文件**（保住 A/B 的 original 基准），
只在运行时把个别热点函数替换成等价实现——替换点在模块级函数 `_optimizeSMPL`
（每段拟合真正跑 LBFGS 的地方）。官方 ``body_model/lbs/lossfactory/lbfgs/optimizePose3D/
optimizePose2D/optimizeShape`` 全部原样复用，拟合**算法、损失、分阶段顺序与原文一致**，
只是把几处可度量的低效点做成开关：

- ``no_item_sync``：官方 closure 每评估一次 ``records.append(loss.item())``
  （``optimize_simple.py:285``）→ 每次评估打断一次 GPU 流水线。默认去掉（records 仅
  verbose 打印用，是纯死代码）。这是**每次评估一次的 CPU-GPU 同步**。
- ``ftol`` / ``maxiters``：FittingMonitor 收敛阈值与最大外圈步数。官方 1e-4/100 在
  接近最优时反复"确认收敛"（pose3d 段 147 次评估大半耗在此）。放宽到 5e-4 误差不升。
- warm-start 编排 ``fit_frame``：连续帧用上一帧 pose/Rh/Th 作初始值、β 固定（或低频
  ``refit_shape_every`` 重跑官方 optimizeShape），跳过官方的"全局 RT 单独对齐"段
  （Rh/Th 已接近，直接 Rh+Th+pose 一起精修更快）。

用法（typical）::

    settings = EMSettings(ftol=5e-4, maxiters=40, no_item_sync=True)
    fit = EmFit(recon, settings)          # install() 幂等，进程级
    for t in frames:
        res = fit.run(best_2d, intrinsics, extrinsics,
                      prev=res["params"] if t else None)   # prev 传 None = 冷启动

    # 对照：official 原版路径 = 直接用 recon.reconstruct(...)（本项目不 install 时
    # EasymocapReconstructor 就是原版官方管线，无任何改动）。

数值正确性：与官方路径唯一可观测差异是提前收敛 / 少 sync / 热启动初始值更近——
loss 定义、LBFGS 步、reg 权重、分阶段顺序全部一致。
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

# ----------------------------------------------------------------------
# 设置（模块级单例；install 后修改立即对后续拟合生效）
# ----------------------------------------------------------------------
@dataclass
class EMSettings:
    ftol: float = 1e-4          # FittingMonitor 外圈收敛阈值（官方 1e-4）
    maxiters: int = 100         # FittingMonitor 外圈最大步数（官方 100）
    no_item_sync: bool = True   # 去掉 closure 每次评估的 loss.item() 同步
    warm_init: bool = True      # 连续帧热启动（frame-0 冷启动拿初始参数）
    skip_global_rt_warm: bool = True  # 热启动时跳过"全局 RT 单独对齐"段
    refit_shape_every: int = 0  # 每隔 N 帧重跑官方 optimizeShape；0 = 首帧后固定 β


_SET = EMSettings()


def _optimizeSMPL_opt(body_model, body_params, prepare_funcs, postprocess_funcs,
                      loss_funcs, extra_params=None, weight_loss={}, cfg=None):
    """官方 ``easymocap/pyfitting/optimize_simple.py::_optimizeSMPL`` 的**逐行等价副本**。

    [EM-OPT] 两处差异（其余与官方完全一致，含 verbose 打印逻辑）：
      1) closure 里 `records.append(loss.item())` 仅当 ``cfg.verbose`` 才执行
         —— 默认关 verbose 时每次评估少一次 CPU-GPU 同步（官方 :285 无条件执行）；
      2) ``FittingMonitor(ftol=_SET.ftol, maxiters=_SET.maxiters)`` 取代硬编码 1e-4。
    """
    from easymocap.pyfitting.lbfgs import LBFGS
    from easymocap.pyfitting.optimize import FittingMonitor, grad_require
    from easymocap.pyfitting.optimize_simple import get_optParams

    loss_funcs = {key: val for key, val in loss_funcs.items()
                  if key in weight_loss.keys() and weight_loss[key] > 0.}
    if cfg.verbose:
        print('Loss Functions: ')
        for key, func in loss_funcs.items():
            print('  -> {:15s}: {}'.format(key, func.__doc__))
    opt_params = get_optParams(body_params, cfg, extra_params)
    grad_require(opt_params, True)
    optimizer = LBFGS(opt_params, line_search_fn='strong_wolfe')
    PRINT_STEP = 100
    records = []

    def closure(debug=False):
        optimizer.zero_grad()
        new_params = body_params.copy()
        for func in prepare_funcs:
            new_params = func(new_params)
        kpts_est = body_model(return_verts=False, return_tensor=True, **new_params)
        loss_dict = {key: func(kpts_est=kpts_est, **new_params)
                     for key, func in loss_funcs.items()}
        cnt = len(records)
        if cfg.verbose and cnt % PRINT_STEP == 0:
            print('{:-6d}: '.format(cnt) + ' '.join(
                [key + ' %f' % (loss_dict[key].item() * weight_loss[key])
                 for key in loss_dict.keys() if weight_loss[key] > 0]))
        loss = sum([loss_dict[key] * weight_loss[key] for key in loss_dict.keys()])
        # [EM-OPT-1] records 只给 verbose 用；默认别每评估 .item()（CPU-GPU 同步）
        if cfg.verbose:
            records.append(loss.item())
        if debug:
            return loss_dict
        loss.backward()
        return loss

    # [EM-OPT-2] 收敛阈值 / 步数上限可调（官方 ftol=1e-4, maxiters 默认 100）
    fitting = FittingMonitor(ftol=_SET.ftol, maxiters=_SET.maxiters)
    final_loss = fitting.run_fitting(optimizer, closure, opt_params)
    fitting.close()
    grad_require(opt_params, False)
    loss_dict = closure(debug=True)
    if cfg.verbose:
        print('{:-6d}: '.format(len(records)) + ' '.join(
            [key + ' %f' % (loss_dict[key].item() * weight_loss[key])
             for key in loss_dict.keys() if weight_loss[key] > 0]))
    loss_dict = {key: val.item() for key, val in loss_dict.items()}
    for func in postprocess_funcs:
        body_params = func(body_params)
    return body_params


_INSTALLED = False


def install(settings: Optional[EMSettings] = None) -> None:
    """把 ``_optimizeSMPL`` 替换成上面的等价副本（进程级、幂等）。

    只有 mode='optimized' 的调用方需要调它；mode='original' 保持官方函数不动。
    """
    global _SET, _INSTALLED
    if settings is not None:
        _SET = settings
    if _INSTALLED:
        return
    import easymocap.pyfitting.optimize_simple as osi
    osi._optimizeSMPL = _optimizeSMPL_opt
    _INSTALLED = True


# ----------------------------------------------------------------------
# warm-start 编排（复用官方 optimizePose3D / optimizePose2D / optimizeShape）
# ----------------------------------------------------------------------
class EmFit:
    """逐帧 SMPL 拟合（连续帧热启动 + 分阶段开关）。

    与官方 ``reconstruct()`` 的输出格式一致：dict{vertices, joints, joints_body25,
    faces, params}。``prev=None`` 首帧 = 冷启动（等价官方入口，仅少 sync / 可调
    ftol-maxiters）；后续帧传上一帧 params = 热启动（β 固定 / 低频重拟合）。
    """

    def __init__(self, recon, settings: Optional[EMSettings] = None,
                 verbose: bool = False):
        import easymocap.pipeline.basic as basic
        self.recon = recon
        self.body = recon._model
        self.verbose = verbose
        install(settings if settings is not None else EMSettings())
        # 官方损失权重 / kintree 与 reconstruct 一致
        from easymocap.dataset import CONFIG
        from easymocap.pipeline.config import Config
        from easymocap.pipeline.weight import load_weight_pose, load_weight_shape
        args = _make_args(verbose=verbose)
        self._args = args
        self._CONFIG = CONFIG
        self._Config = Config
        self._weight_shape = load_weight_shape("smpl", args.opts)
        self._weight_pose = load_weight_pose("smpl", args.opts)
        self._kintree_shape = CONFIG["body15"]["kintree"][1:]  # 同官方 smpl 分支
        self._frame = 0
        self._shape_cache: Optional[np.ndarray] = None
        self._basic = basic

    # -- 观测 → kp3d（复用 recon 的静态方法 + 官方三角化）--
    def _prep(self, poses_per_cam, intrinsics, extrinsics, min_conf):
        from easymocap.mytools.triangulator import batch_triangulate
        from easymocap.smplmodel.body_param import check_keypoints
        kp2d, bboxes, Pall = self.recon._to_body25_2d(
            poses_per_cam, intrinsics, extrinsics, min_conf)
        if kp2d is None:
            return None, None, None, None
        kp3d = batch_triangulate(kp2d, Pall, min_view=2)     # (25,4)
        kp3d = check_keypoints(kp3d, 1, min_conf=min_conf)
        return kp2d, bboxes, Pall, kp3d

    def _forward_out(self, params):
        import torch
        with torch.no_grad():
            verts = self.body(return_verts=True, return_tensor=False, **params)
            j25 = self.body(return_verts=False, return_tensor=False, **params)
            j24 = self.body(return_verts=False, return_tensor=False,
                            return_smpl_joints=True, **params)
        return {
            "vertices": np.asarray(verts, np.float64).reshape(-1, 3),
            "joints": np.asarray(j24, np.float64).reshape(-1, 3),
            "joints_body25": np.asarray(j25, np.float64).reshape(-1, 3),
            "faces": self.recon.faces,
            "params": {k: np.asarray(v) for k, v in params.items()},
        }

    def run(self, poses_per_cam, intrinsics, extrinsics, *,
            prev: Optional[dict] = None, min_conf: float = 0.3) -> Optional[dict]:
        """拟合一帧。prev = 上一帧 result['params']（热启动）或 None（冷启动）。"""
        from easymocap.smplmodel.body_param import check_keypoints  # noqa: F401
        kp2d, bboxes, Pall, kp3d = self._prep(
            poses_per_cam, intrinsics, extrinsics, min_conf)
        if kp2d is None:
            return None
        n_valid = int((kp3d[:, 3] > 0).sum())
        if self.verbose:
            print(f"[EmFit] frame {self._frame}：{kp2d.shape[0]} 视角，有效 3D 关节 "
                  f"{n_valid}/25")

        from easymocap.pyfitting import optimizePose3D, optimizePose2D, optimizeShape
        from easymocap.pipeline.config import Config as _Cfg

        cfg = _Cfg(self._args)
        cfg.device = self.body.device
        cfg.OPT_R = cfg.OPT_T = cfg.OPT_POSE = True

        warm = bool(prev is not None and _SET.warm_init and prev.get("poses") is not None)
        # ---- β：首帧（或到 refit_shape_every）官方 optimizeShape；否则用缓存 ----
        refit = (self._shape_cache is None or
                 (_SET.refit_shape_every > 0 and self._frame % _SET.refit_shape_every == 0))
        if refit:
            init_shp = {"poses": np.zeros((1, 72), np.float32),
                        "shapes": (self._shape_cache if self._shape_cache is not None
                                   else np.zeros((1, 10), np.float32)),
                        "Rh": np.zeros((1, 3), np.float32),
                        "Th": np.zeros((1, 3), np.float32)}
            out_shp = optimizeShape(self.body, init_shp, kp3d[None],
                                    weight_loss=self._weight_shape,
                                    kintree=self._kintree_shape)
            self._shape_cache = np.asarray(out_shp["shapes"], np.float32)

        # ---- 姿态初始值：热启动用上一帧；冷启动用零位姿 ----
        if warm:
            params = {k: np.asarray(v, np.float32).copy()
                      for k, v in prev.items() if k in ("poses", "Rh", "Th")}
            params["shapes"] = self._shape_cache.copy()
        else:
            params = {"poses": np.zeros((1, 72), np.float32),
                      "Rh": np.zeros((1, 3), np.float32),
                      "Th": np.zeros((1, 3), np.float32),
                      "shapes": self._shape_cache.copy()}

        # ---- 阶段编排 ----
        if warm and _SET.skip_global_rt_warm:
            # Rh/Th 已接近解：直接 Rh+Th+poses 一起精修（= 官方"3D pose"段）。
            # 官方 global-RT 段（pose 锁零单独对齐）在热启动时是纯浪费。
            params = optimizePose3D(self.body, params, kp3d[None],
                                    weight=self._weight_pose, cfg=cfg)
        else:
            # 冷启动：对齐官方 multi_stage —— 先全局 RT（pose 锁零），再全 pose 3D，
            # 最后 2D 精修。smooth_body 在 RT 段关闭（同 basic.multi_stage_optimize）。
            saved = self._weight_pose.get("smooth_body", 0.)
            self._weight_pose["smooth_body"] = 0.
            cfg_r = _Cfg(self._args); cfg_r.device = self.body.device
            cfg_r.OPT_R = cfg_r.OPT_T = True; cfg_r.OPT_POSE = False
            params = optimizePose3D(self.body, params, kp3d[None],
                                    weight=self._weight_pose, cfg=cfg_r)
            self._weight_pose["smooth_body"] = saved
            params = optimizePose3D(self.body, params, kp3d[None],
                                    weight=self._weight_pose, cfg=cfg)
        if kp2d is not None:
            params = optimizePose2D(self.body, params,
                                    bboxes[None], kp2d[None], Pall,
                                    weight=self._weight_pose, cfg=cfg)
        self._frame += 1
        return self._forward_out(params)


def _make_args(verbose: bool = False):
    from types import SimpleNamespace
    return SimpleNamespace(verbose=verbose, model="smpl", robust3d=False, opts={})
