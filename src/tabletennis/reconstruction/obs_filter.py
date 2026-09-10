"""按「多视角 3D 共识」治理低质量 2D 关键点（P0-1）。

误差来源实测（``docs/error_budget_report.md``，三段视频）：

- 2D 关键点的**系统性偏差**占拟合误差的 39~48%（最大项），而随机抖动只占 0.5%
  （帧间 σ 0.9~1.5px，但两视角对同一关节的分歧有 5~7.6px）→ 是**每个视角各自偏**
  的偏差，不是噪声；
- 误差与置信度强相关：conf 0.9~1.0 的关节误差 2.2cm，conf 0.3~0.5 到 14.9cm，
  conf<0.3 到 21cm。而 ``--fit-conf`` 默认 0.15 把这些最差的关节也放进了拟合。

本模块做两件事：

1. **共识降权**（:func:`filter_obs_by_consensus`）：用**鲁棒三角化的 3D 关键点**
   （阶段 D RANSAC，本身已抗离群）当参照，把每个视角的 2D 观测按重投影偏差降权
   ——``w = 1/(1+(r/σ)²)``（Cauchy），偏差超 ``max_px`` 直接置 0。SMPL 的两项损失
   （``loss_kp3d`` / ``loss_kp2d``）都以关键点 conf 为权重，所以只要改 conf 就能
   同时影响「3D 目标」与「2D 重投影精修」两条通路，无需改官方代码。

2. **阈值抬升**：由调用方传更高的 ``--fit-conf``（0.15 → 0.3~0.5），把最差的关节
   整条剔掉——SMPL 先验 + 时间平滑会补全它们，比让它们拽着整个身体走更准。

⚠️ 参照物是「2D 自己算出来的 3D」，若某关节在多数视角上都一致地偏，共识会跟着偏。
这是多视角方法的固有限制（少数服从多数），但对「单视角外推/遮挡」这类**局部**离群
是有效的。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.types import CameraExtrinsics, CameraIntrinsics, Pose2D
from .easymocap import HALPE26_TO_BODY25, _pose_to_body25_view

__all__ = ["ConsensusConfig", "filter_obs_by_consensus"]


@dataclass
class ConsensusConfig:
    """共识降权参数。

    Attributes:
        sigma_px: Cauchy 尺度——偏差 = σ 时权重降到 0.5。默认 10px（≈ 地面地板
            6px 的 1.7 倍，留出正常视差/标定残差的余量）。
        max_px: 偏差超过它就完全丢弃该视角该关节（权重 0）。默认 30px。
        min_conf: 只对原始 conf ≥ 该值的观测做降权（低于它的本来就要被
            ``--fit-conf`` 门槛丢掉，没必要再算）。0 = 全部。
    """

    sigma_px: float = 10.0
    max_px: float = 30.0
    min_conf: float = 0.0

    @property
    def enabled(self) -> bool:
        return self.sigma_px > 0


def _proj(P: np.ndarray, X: np.ndarray) -> np.ndarray:
    """``(N,3)`` 世界点经投影矩阵 ``P (3,4)`` → ``(N,2)`` 像素。"""
    c = np.hstack([X, np.ones((len(X), 1))]) @ P.T
    z = c[:, 2:3]
    z = np.where(np.abs(z) < 1e-12, np.nan, z)
    return c[:, :2] / z


def filter_obs_by_consensus(
    frames_obs: Sequence[Dict[int, Pose2D]],
    kp3ds: Optional[np.ndarray],
    intrinsics: Dict[int, CameraIntrinsics],
    extrinsics: Dict[int, CameraExtrinsics],
    cfg: ConsensusConfig,
) -> Tuple[List[Dict[int, Pose2D]], dict]:
    """按 3D 共识给 2D 观测的 conf 乘权重。

    Args:
        frames_obs: 逐帧 ``{cid: Pose2D}``（halpe26，原始畸变像素）。
        kp3ds: ``(T, 25, 4)`` 鲁棒三角化的 body25 3D 关键点（世界系，第 4 列 conf）。
            None / 全 0 的帧跳过（该帧无共识参照，观测原样保留）。
        cfg: 见 :class:`ConsensusConfig`。

    Returns:
        ``(frames_obs_new, stats)``；``stats`` 含
        ``n_obs``（参与统计的观测数）/ ``n_zero``（被丢）/ ``w_sum`` / ``r_px``（偏差样本）。
        原 ``Pose2D`` 不被修改（改的是副本）。
    """
    stats = {"n_obs": 0, "n_zero": 0, "w_sum": 0.0, "r_px": []}
    if not cfg.enabled or kp3ds is None:
        return list(frames_obs), stats

    # halpe26 -> body25 的逐关节映射（halpe 17「头顶」不在 body25，保持原 conf）
    h2b = {h: b for h, b in HALPE26_TO_BODY25}
    out: List[Dict[int, Pose2D]] = []
    for t, obs in enumerate(frames_obs):
        if not obs or t >= len(kp3ds):
            out.append(obs)
            continue
        kp3d = np.asarray(kp3ds[t], dtype=np.float64)
        ref_ok = np.isfinite(kp3d[:, :3]).all(axis=1) & (kp3d[:, 3] > 0)
        if not ref_ok.any():
            out.append(obs)
            continue
        new_obs: Dict[int, Pose2D] = {}
        for cid, pose in obs.items():
            K = intrinsics.get(cid)
            ext = extrinsics.get(cid)
            kp = np.asarray(pose.keypoints, dtype=np.float64)
            if K is None or ext is None or kp.ndim != 2 or kp.shape[0] < 26:
                new_obs[cid] = pose
                continue
            view = _pose_to_body25_view(kp.astype(np.float32), K.K, K.dist, 0.0)
            if view is None:
                new_obs[cid] = pose
                continue
            P = K.K @ np.hstack([ext.R, ext.t.reshape(3, 1)])
            uv = _proj(P, kp3d[:, :3])
            r = np.linalg.norm(uv - view[:, :2], axis=1)      # (25,)
            w = np.ones(25, dtype=np.float64)
            m = ref_ok & np.isfinite(r) & (view[:, 2] > 0)
            if m.any():
                w[m] = 1.0 / (1.0 + (r[m] / cfg.sigma_px) ** 2)
                w[m & (r > cfg.max_px)] = 0.0
                stats["r_px"].extend(r[m].tolist())
            # 写回 halpe26 序（同一 body25 关节的多个 halpe 点共享权重）
            kp_new = kp.copy()
            for h in range(min(26, kp.shape[0])):
                b25 = h2b.get(h)
                if b25 is None or not m[b25]:
                    continue
                if kp_new[h, 2] < cfg.min_conf:
                    continue
                stats["n_obs"] += 1
                if w[b25] <= 0.0:
                    stats["n_zero"] += 1
                stats["w_sum"] += float(w[b25])
                kp_new[h, 2] *= float(w[b25])
            new_obs[cid] = Pose2D(
                camera_id=pose.camera_id, keypoints=kp_new,
                score=pose.score,
                bbox=(None if pose.bbox is None else np.asarray(pose.bbox).copy()),
                skeleton=pose.skeleton)
        out.append(new_obs)
    return out, stats


def format_stats(stats: dict, name: str = "共识降权") -> str:
    """把 :func:`filter_obs_by_consensus` 的 stats 拼成一行日志。"""
    n = int(stats.get("n_obs", 0))
    if n == 0:
        return f"{name}：无有效观测（未生效）"
    r = np.asarray(stats.get("r_px", []), dtype=np.float64)
    med = float(np.median(r)) if r.size else float("nan")
    mean_w = float(stats["w_sum"]) / n
    return (f"{name}：{n} 个观测，丢 {stats['n_zero']} "
            f"({100.0 * stats['n_zero'] / n:.1f}%)，平均权重 {mean_w:.3f}，"
            f"3D 共识偏差中位 {med:.1f}px")
