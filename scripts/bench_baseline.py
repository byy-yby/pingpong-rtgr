#!/usr/bin/env python3
"""Phase-1 基线 benchmark：官方 EasyMocap 逐帧**冷启动**全量离线拟合。

被测链路（= live_control EM 线程逐帧行为，EasyMocap 代码零改动）：
    GT SMPL 人体(N 帧连续动作) → 真实内参/外参投影成 4 视角 halpe26 2D 观测(含噪声)
    → EasymocapReconstructor.reconstruct()
        ├─ _to_body25_2d        halpe26→body25, 去畸变, conf 过滤
        ├─ batch_triangulate    多视角 DLT
        ├─ check_keypoints
        └─ smpl_from_keypoints3d2d
             ├─ optimizeShape        (β, LBFGS max_iter=10, FittingMonitor ftol=1e-4)
             └─ multi_stage_optimize
                  ├─ Optimize global RT  (Rh+Th)      ← optimizePose3D 第 1 次
                  ├─ Optimize 3D Pose    (Rh+Th+poses)← optimizePose3D 第 2 次
                  └─ Optimize 2D Pose    (Rh+Th+poses)  LBFGS strong_wolfe max_iter=20
        → SMPL forward(return_verts=True) 出网格

输出（每帧 + 汇总 JSON 落到 --out，便于后续 original vs optimized A/B）：
    整帧 wall / 各阶段 wall / 各阶段 SMPL 前向调用次数 / 关节误差(body25 中位 mm)
    / GPU 峰值显存 / 平均 GPU util
  注：2D 检测段（RTMPose detect_batch ~10ms）不属于 EasyMocap，另见 optimization_report。

用法:
  python scripts/bench_baseline.py --frames 3 [--noise 0.5] [--out data/bench/baseline.json]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
_SRC = os.path.join(_ROOT, "src")
for _p in (_SRC, "/home/yby/projects/EasyMocap"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch

from tabletennis.reconstruction.easymocap import (
    EasymocapReconstructor,
    HALPE26_TO_BODY25,
)
from tabletennis.core.types import Pose2D
from tabletennis.reconstruction.triangulate import load_camera_rig

import easymocap.pipeline.basic as basic  # noqa: E402  (monkeypatch 目标)

# 全局（main 里赋初值，make_person/observe 要用）
recon = None
cids = []
intrinsics = extrinsics = None


# ---------------------------------------------------------------- GT 序列
def make_person(rng, pos=(1.0, 1.0)):
    """构造一个站立的 GT 人（params，Th 已校准让脚落桌面 z=-0.76）。"""
    body = recon._model
    betas = rng.uniform(-1.0, 1.0, size=10).astype(np.float32)
    betas[0] = rng.uniform(0.5, 1.2)
    poses = np.zeros((1, 72), np.float32)
    poses[0, 3 * 3:4 * 3] = [0, 0, 0.15]      # 盆骨前倾
    poses[0, 6 * 3:7 * 3] = [0, 0, 0.10]      # 脖子
    poses[0, 18 * 3:19 * 3] = [0, 0.5, 0]     # 左肘
    poses[0, 4 * 3:5 * 3] = [0.15, 0, 0]      # 左膝
    poses[0, 5 * 3:6 * 3] = [-0.15, 0, 0]     # 右膝
    Rh = np.array([np.pi / 2, 0, 0], np.float32)   # SMPL Y-up → 世界 Z-up
    Th = np.array([pos[0], pos[1], 0.386], np.float32)
    p = {"poses": poses, "shapes": betas[None], "Rh": Rh[None], "Th": Th[None]}
    with torch.no_grad():
        v = body(return_verts=True, return_tensor=False, **p)[0]
    zoff = -0.76 - v[:, 2].min()
    p["Th"] = (Th + np.array([0, 0, zoff])).astype(np.float32)[None]
    return p


def frame_params(base, t, n_frames):
    """第 t 帧的平滑动作：真实人体相邻帧参数变化很小（Phase-2 warm-start 的依据）。"""
    w = t / max(n_frames - 1, 1)
    p = {k: v.copy() for k, v in base.items()}
    ps = p["poses"].copy()
    ps[0, 4 * 3:5 * 3] += 0.30 * w          # 左膝渐弯
    ps[0, 5 * 3:6 * 3] -= 0.30 * w          # 右膝
    ps[0, 3 * 3:4 * 3] = [0, 0, 0.15 + 0.15 * w]
    ps[0, 18 * 3:19 * 3] = [0, 0.5 + 0.4 * w, 0]   # 左肘抬
    ps[0, 19 * 3:20 * 3] = [0, 0.3 * w, 0]         # 右肘
    p["poses"] = ps
    p["Th"] = p["Th"] + np.array([[0.05 * w, 0.02 * w, 0]], np.float32)  # 缓慢平移
    return p


def observe(j25_world, sigma_px):
    """把 body25 关节投影成 4 视角 halpe26 观测（映射关节加高斯噪声）。"""
    best = {}
    for cid in cids:
        K = intrinsics[cid].K
        P = K @ np.hstack([extrinsics[cid].R, extrinsics[cid].t.reshape(3, 1)])
        Xh = np.hstack([j25_world, np.ones((25, 1))])
        c = Xh @ P.T
        p2d = c[:, :2] / c[:, 2:3]
        halpe = np.zeros((26, 3), np.float32)
        for hi, b25 in HALPE26_TO_BODY25:
            x, y = p2d[b25]
            if sigma_px:
                x += np.random.normal(0, sigma_px)
                y += np.random.normal(0, sigma_px)
            halpe[hi] = (x, y, 1.0)
        best[cid] = Pose2D(camera_id=cid, keypoints=halpe, score=1.0, skeleton="halpe26")
    return best


# ---------------------------------------------------------------- 仪器
class FwdCounter:
    """包住 body.forward，统计拟合期 SMPL 前向调用次数与纯前向墙钟。

    注意不能用实例 `__call__` 覆盖（特殊方法走类型查找，nn.Module.__call__
    不会被实例属性拦截）；forward 是普通方法，实例属性会生效。
    """

    def __init__(self, body):
        self.body = body
        self.n = 0
        self.t = 0.0
        self.orig = body.forward  # 绑定方法（尚未被实例属性遮蔽）

    def __enter__(self):
        ctr = self

        def wrapped(*a, **kw):
            t0 = time.perf_counter()
            r = ctr.orig(*a, **kw)
            ctr.t += time.perf_counter() - t0
            ctr.n += 1
            return r
        self.body.forward = wrapped
        return self

    def __exit__(self, *a):
        self.body.forward = self.orig


class GpuSampler:
    """后台每 200ms 采样 nvidia-smi 的 GPU util / mem%，均值留档。"""

    def __init__(self):
        self.util, self.mem = [], []
        self._stop = False
        self._th = None

    def __enter__(self):
        self._th = threading.Thread(target=self._run, daemon=True)
        self._th.start()
        return self

    def _run(self):
        try:
            while not self._stop:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5)
                parts = out.stdout.strip().split(",")
                if len(parts) >= 2:
                    self.util.append(float(parts[0].strip()))
                    self.mem.append(float(parts[1].strip()))
                time.sleep(0.2)
        except Exception:
            pass

    def __exit__(self, *a):
        self._stop = True
        if self._th:
            self._th.join(timeout=2)
        return self

    @property
    def util_pct(self):
        return float(np.mean(self.util)) if self.util else None

    @property
    def mem_gb(self):
        return float(np.mean(self.mem) / 1024) if self.mem else None


def median_joint_err_mm(pred_j25, gt_j25):
    return float(np.median(np.linalg.norm(pred_j25 - gt_j25, axis=1)) * 1000)


# ---------------------------------------------------------------- main
def main() -> None:
    global recon, cids, intrinsics, extrinsics
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=int, default=3)
    ap.add_argument("--noise", type=float, default=0.5, help="2D 观测高斯噪声 px")
    ap.add_argument("--out", default="data/bench/baseline.json")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "需要 GPU"
    intrinsics, extrinsics = load_camera_rig()
    cids = sorted(extrinsics)
    recon = EasymocapReconstructor(verbose=False)
    body = recon._model

    # ---- 按段包住官方阶段函数（patch basic 命名空间的全局名即可被内部查到）----
    originals = {nm: getattr(basic, nm) for nm in
                 ("optimizeShape", "optimizePose3D", "optimizePose2D")}
    call_log = []
    fw = FwdCounter(body)

    def _log(name):
        def w(*a, **k):
            t0 = time.perf_counter()
            n0 = fw.n
            r = originals[name](*a, **k)
            call_log.append((name, time.perf_counter() - t0, fw.n - n0))
            return r
        return w

    basic.optimizeShape = _log("optimizeShape")
    basic.optimizePose3D = _log("optimizePose3D")
    basic.optimizePose2D = _log("optimizePose2D")

    rng = np.random.default_rng(0)
    base = make_person(rng)
    rows = []
    for t in range(args.frames):
        p_gt = frame_params(base, t, args.frames)
        with torch.no_grad():
            j_gt = body(return_verts=False, return_tensor=False, **p_gt)[0]
        best = observe(j_gt, args.noise)

        call_log.clear()
        torch.cuda.reset_peak_memory_stats()
        gpu = GpuSampler()
        with gpu, fw:
            fw.n = fw.t = 0
            t0 = time.perf_counter()
            res = recon.reconstruct(best, intrinsics, extrinsics)
            dt = time.perf_counter() - t0

        # call_log 顺序: [optimizeShape] [optimizePose3D(globalRT)] [optimizePose3D(3D)] [optimizePose2D]
        per = {"frame": t, "wall_s": dt, "fwd_total": fw.n,
               "fwd_pure_s": fw.t,
               "err_mm": median_joint_err_mm(np.asarray(res["joints_body25"]), j_gt),
               "mem_peak_mb": torch.cuda.max_memory_allocated() / 1e6,
               "gpu_util_pct": gpu.util_pct, "gpu_mem_gb": gpu.mem_gb,
               "stages": {"shape": None, "globalRT": None, "pose3d": None,
                          "pose2d": None, "other": None}}
        fit_sum = 0.0
        fit_fwd = 0
        pos3d_seen = False
        for name, s, nf in call_log:
            key = name
            if name == "optimizePose3D":
                key = "globalRT" if not pos3d_seen else "pose3d"
                pos3d_seen = True
            elif name == "optimizeShape":
                key = "shape"
            else:
                key = "pose2d"
            per["stages"][key] = {"s": s, "fwd": nf}
            fit_sum += s
            fit_fwd += nf
        mesh_fwd = fw.n - fit_fwd
        # other = 拟合前(2D 组装+三角化) + 拟合后(网格前向+收尾)
        per["stages"]["other"] = {"s": dt - fit_sum, "fwd": mesh_fwd}
        per["fwd_fit"] = fit_fwd
        rows.append(per)
        _pr(per)

    # ---- 汇总 ----
    stage_names = ["shape", "globalRT", "pose3d", "pose2d", "other"]
    g = np.median
    summary = {
        "bench": "baseline-cold", "n_frames": len(rows), "noise_px": args.noise,
        "gpu": torch.cuda.get_device_name(0),
        "per_frame_median_ms": float(g([r["wall_s"] for r in rows]) * 1000),
        "fwd_per_frame": float(g([r["fwd_total"] for r in rows])),
        "fwd_fit_per_frame": float(g([r["fwd_fit"] for r in rows])),
        "fwd_pure_ms_per_frame": float(g([r["fwd_pure_s"] for r in rows]) * 1000),
        "err_mm_median": float(g([r["err_mm"] for r in rows])),
        "mem_peak_mb_median": float(g([r["mem_peak_mb"] for r in rows])),
        "gpu_util_pct_avg": float(np.mean([r["gpu_util_pct"] for r in rows
                                           if r["gpu_util_pct"] is not None])),
        "stages_ms_median": {k: float(g([r["stages"][k]["s"] for r in rows]) * 1000)
                             for k in stage_names},
        "stages_fwd_median": {k: float(g([r["stages"][k]["fwd"] for r in rows]))
                              for k in stage_names},
        "frames": rows,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== 基线汇总 (N={len(rows)} 帧冷启动) ===")
    print(f"单帧全量拟合    : {summary['per_frame_median_ms']:7.0f} ms")
    print(f"SMPL 前向/帧    : {summary['fwd_per_frame']:.0f} 次"
          f"  (纯前向合计 {summary['fwd_pure_ms_per_frame']:.0f} ms)")
    print(f"关节误差(body25) : {summary['err_mm_median']:.1f} mm 中位")
    print(f"GPU 峰值显存    : {summary['mem_peak_mb_median']:6.0f} MB")
    print(f"GPU util 平均   : {summary['gpu_util_pct_avg']:.0f} %")
    print("阶段(s中位) 耗时 / SMPL 前向次数:")
    for k in stage_names:
        print(f"  {k:9s}: {summary['stages_ms_median'][k]:7.0f} ms   前向 {summary['stages_fwd_median'][k]:4.0f} 次")
    print(f"JSON -> {args.out}")


def _pr(per):
    toks = []
    for k, v in per["stages"].items():
        s = "-" if v is None else "%dms/%d" % (v["s"] * 1000, v["fwd"])
        toks.append(k + "=" + s)
    parts = "  ".join(toks)
    print(f"frame {per['frame']}: wall={per['wall_s']*1000:.0f}ms fwd={per['fwd_total']} "
          f"err={per['err_mm']:.1f}mm mem={per['mem_peak_mb']:.0f}MB gpu={per['gpu_util_pct']}%")
    print(f"    {parts}")


if __name__ == "__main__":
    main()
