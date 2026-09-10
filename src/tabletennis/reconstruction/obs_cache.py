"""Pass-1 观测缓存：把「检测 + 身份关联」的产物存盘，供 Pass-2（SMPL 拟合）反复重跑。

动机：`reconstruct_video.py` 的 batch 档里，检测占大头（实测三段视频 86~303s），
SMPL 拟合只要 62~90s。调拟合侧参数（`--fit-conf` / 置信度治理 / 损失权重）时，
每次重跑检测是纯浪费——把 Pass 1 的产物缓存下来，Pass 2 就能秒级重跑，
且**保证 A/B 的 2D 输入逐位一致**（否则两轮检测的细微差异会污染对比）。

缓存内容（每人一组）：
    ``obs_{g}``    (T, n_cids, 26, 3) float32 —— 该人每帧每相机的 halpe26 关键点
    ``bbox_{g}``   (T, n_cids, 4)    float32 —— 检测框（``select_fit_views`` 判裁边要用）
    ``robust_{g}`` (T, 25, 4)        float64 —— 阶段 D 鲁棒三角化的 body25 3D 关键点
    ``gate_{g}``   (n_cids,)         int32   —— 下半身门命中帧数（仅用于打印）
公共：
    ``cids``    (n_cids,) —— 视角列序
    ``indices`` (T,)      —— 主时钟帧号
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..core.types import Pose2D

__all__ = ["save_pass1", "load_pass1"]


def _pack_obs(frames_obs: Sequence[Dict[int, Pose2D]], cids: List[int],
              n_kpts: int = 26) -> Tuple[np.ndarray, np.ndarray]:
    """``[{cid: Pose2D}]`` → ``(kp (T,nC,K,3), bbox (T,nC,4))``（缺的填 0）。"""
    T, nC = len(frames_obs), len(cids)
    col = {c: i for i, c in enumerate(cids)}
    kp = np.zeros((T, nC, n_kpts, 3), dtype=np.float32)
    bb = np.zeros((T, nC, 4), dtype=np.float32)
    for t, obs in enumerate(frames_obs):
        for cid, pose in obs.items():
            i = col.get(cid)
            if i is None:
                continue
            k = np.asarray(pose.keypoints, dtype=np.float32)
            if k.ndim == 2:
                n = min(n_kpts, k.shape[0])
                kp[t, i, :n] = k[:n, :3]
            b = getattr(pose, "bbox", None)
            if b is not None:
                b = np.asarray(b, dtype=np.float32).ravel()
                if b.size >= 4 and np.isfinite(b[:4]).all():
                    bb[t, i] = b[:4]
    return kp, bb


def _unpack_obs(kp: np.ndarray, bb: np.ndarray, cids: List[int]
                ) -> List[Dict[int, Pose2D]]:
    """``(kp, bbox)`` → ``[{cid: Pose2D}]``（全 0 的格子 = 该相机该帧无观测）。"""
    out: List[Dict[int, Pose2D]] = []
    for t in range(kp.shape[0]):
        obs: Dict[int, Pose2D] = {}
        for i, cid in enumerate(cids):
            k = kp[t, i]
            if not (k[:, 2] > 0).any():
                continue
            b = bb[t, i]
            obs[cid] = Pose2D(
                camera_id=cid, keypoints=k.copy(),
                score=float(k[k[:, 2] > 0, 2].max()),
                bbox=(b.copy() if b.size == 4 and (b > 0).any() else None),
                skeleton="halpe26")
        out.append(obs)
    return out


def save_pass1(path: str,
               frames_obs_by_pid: Dict[int, List[Dict[int, Pose2D]]],
               frames_robust_by_pid: Dict[int, List[Optional[object]]],
               indices: Sequence[int],
               cids: Sequence[int],
               lower_gate: Optional[Dict[int, Dict[int, int]]] = None,
               robust_override: Optional[Dict[int, np.ndarray]] = None,
               n_kpts: int = 26) -> None:
    """存 Pass 1 观测。

    Args:
        frames_robust_by_pid: 每帧的 ``TrackFrameResult`` 或 None（只用它算 override）。
        robust_override: 已算好的 ``{pid: (T,25,4)}``；给了就不再看 ``frames_robust_by_pid``。
    """
    cids = sorted(cids)
    payload: Dict[str, np.ndarray] = {
        "cids": np.asarray(cids, dtype=np.int32),
        "indices": np.asarray(indices, dtype=np.int64),
        "n_people": np.asarray([len(frames_obs_by_pid)], dtype=np.int32),
    }
    for g, frames in frames_obs_by_pid.items():
        kp, bb = _pack_obs(frames, cids, n_kpts=n_kpts)
        payload[f"obs_{g}"] = kp
        payload[f"bbox_{g}"] = bb
        ov = None if robust_override is None else robust_override.get(g)
        if ov is None and frames_robust_by_pid:
            ov = _robust_override_from_results(frames_robust_by_pid.get(g, []), frames)
        payload[f"robust_{g}"] = (np.asarray(ov, dtype=np.float64)
                                  if ov is not None
                                  else np.zeros((len(frames), 25, 4)))
        gate = (lower_gate or {}).get(g, {}) or {}
        payload[f"gate_{g}"] = np.asarray(
            [int(gate.get(c, 0)) for c in cids], dtype=np.int32)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **payload)
    os.replace(tmp, path)


def _robust_override_from_results(frames_robust, frames_obs) -> Optional[np.ndarray]:
    """``[TrackFrameResult|None]`` → ``(T,25,4)``（与 reconstruct_video 同口径）。

    与 ``scripts/reconstruct_video.py::_robust_kp3ds_override`` 是同一套逻辑，
    放在这里是为了让缓存自身可独立读写（脚本侧仍调用它，避免两套实现漂移）。
    """
    from .easymocap import HALPE26_TO_BODY25

    T = len(frames_obs)
    out = np.zeros((T, 25, 4), dtype=np.float64)
    any_ok = False
    for t in range(T):
        res = frames_robust[t] if t < len(frames_robust) else None
        if res is None:
            continue
        kp3 = np.asarray(res.skeleton.keypoints, dtype=np.float64)
        conf3 = np.asarray(res.skeleton.confidence, dtype=np.float64)
        obs = frames_obs[t] if t < len(frames_obs) else {}
        for hi, b25 in HALPE26_TO_BODY25:
            if hi >= len(kp3) or not np.isfinite(kp3[hi]).all() or conf3[hi] <= 0:
                continue
            cs = [float(np.asarray(p.keypoints, np.float64)[hi, 2])
                  for p in obs.values()
                  if np.asarray(p.keypoints, np.float64).ndim == 2
                  and hi < len(p.keypoints)
                  and np.asarray(p.keypoints, np.float64)[hi, 2] >= 0.2]
            out[t, b25, :3] = kp3[hi]
            out[t, b25, 3] = float(np.mean(cs)) if cs else float(conf3[hi])
            any_ok = True
    return out if any_ok else None


def load_pass1(path: str) -> dict:
    """读回 Pass 1 缓存。

    Returns:
        ``{"cids", "indices", "frames_obs_by_pid", "kp3ds_by_pid", "lower_gate",
        "n_people"}``；``kp3ds_by_pid`` 是 ``{pid: (T,25,4)}``（无有效关节的帧全 0）。
    """
    d = np.load(path)
    cids = [int(c) for c in d["cids"]]
    indices = d["indices"].tolist()
    n_people = int(d["n_people"][0])
    frames_obs_by_pid: Dict[int, List[Dict[int, Pose2D]]] = {}
    kp3ds_by_pid: Dict[int, np.ndarray] = {}
    lower_gate: Dict[int, Dict[int, int]] = {}
    for g in range(n_people):
        frames_obs_by_pid[g] = _unpack_obs(d[f"obs_{g}"], d[f"bbox_{g}"], cids)
        kp3ds_by_pid[g] = np.asarray(d[f"robust_{g}"], dtype=np.float64)
        gate = np.asarray(d[f"gate_{g}"], dtype=np.int64)
        lower_gate[g] = {c: int(n) for c, n in zip(cids, gate) if n > 0}
    return {"cids": cids, "indices": indices,
            "frames_obs_by_pid": frames_obs_by_pid,
            "kp3ds_by_pid": kp3ds_by_pid,
            "lower_gate": lower_gate, "n_people": n_people}
