#!/usr/bin/env python3
"""Phase-2 A/B：官方原版 vs 逐项可开关优化（同一合成序列、同一噪声，逐配置跑）。

配置（每个 = 一个独立可开关的优化子集）：
  A original  : recon.reconstruct() 官方入口，零改动          （基准）
  B sync      : EmFit，只开 no_item_sync（ftol 仍 1e-4）      （同步消除，单独量化）
  C loose     : EmFit，sync + ftol=5e-4 + maxiters=40（冷启动）  （收敛/步数策略）
  D warm      : EmFit，loose + 热启动(warm_init, skip_global_rt_warm) + β 首帧后固定
                稳态帧=1..N-1（第 0 帧冷启动拿初始参数）

输出：每配置每帧 wall/err/SMPL 前向次数，汇总 original vs 各项（中位帧 / 稳态帧）。
用法:
  python scripts/bench_optimized.py --frames 4 [--noise 0.5] [--out data/bench/optimized.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
for _p in (_ROOT + "/src", "/home/yby/projects/EasyMocap"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch

from tabletennis.reconstruction.easymocap import (
    EasymocapReconstructor,
    HALPE26_TO_BODY25,
)
from tabletennis.reconstruction.em_fit import EmFit, EMSettings
from tabletennis.core.types import Pose2D
from tabletennis.reconstruction.triangulate import load_camera_rig


# ---------------------------------------------------------------- 合成序列
def make_person(recon, rng, pos=(1.0, 1.0)):
    body = recon._model
    betas = rng.uniform(-1.0, 1.0, size=10).astype(np.float32)
    betas[0] = rng.uniform(0.5, 1.2)
    ps = np.zeros((1, 72), np.float32)
    ps[0, 3 * 3:4 * 3] = [0, 0, 0.15]; ps[0, 6 * 3:7 * 3] = [0, 0, 0.10]
    ps[0, 18 * 3:19 * 3] = [0, 0.5, 0]; ps[0, 4 * 3:5 * 3] = [0.15, 0, 0]
    ps[0, 5 * 3:6 * 3] = [-0.15, 0, 0]
    p = {"poses": ps, "shapes": betas[None],
         "Rh": np.array([[np.pi / 2, 0, 0]], np.float32),
         "Th": np.array([[pos[0], pos[1], 0.386]], np.float32)}
    with torch.no_grad():
        v = body(return_verts=True, return_tensor=False, **p)[0]
    p["Th"] = p["Th"] + np.array([[0, 0, -0.76 - v[:, 2].min()]], np.float32)
    return p


def frame_params(base, t, n):
    w = t / max(n - 1, 1)
    p = {k: v.copy() for k, v in base.items()}
    ps = p["poses"].copy()
    ps[0, 4 * 3:5 * 3] += 0.30 * w; ps[0, 5 * 3:6 * 3] -= 0.30 * w
    ps[0, 3 * 3:4 * 3] = [0, 0, 0.15 + 0.15 * w]
    ps[0, 18 * 3:19 * 3] = [0, 0.5 + 0.4 * w, 0]; ps[0, 19 * 3:20 * 3] = [0, 0.3 * w, 0]
    p["poses"] = ps
    p["Th"] = p["Th"] + np.array([[0.05 * w, 0.02 * w, 0]], np.float32)
    return p


def observe(intr, ext, cids, jw, sigma_px, rng):
    best = {}
    for cid in cids:
        K = intr[cid].K
        P = K @ np.hstack([ext[cid].R, ext[cid].t.reshape(3, 1)])
        c = np.hstack([jw, np.ones((25, 1))]) @ P.T
        p2d = c[:, :2] / c[:, 2:3]
        halpe = np.zeros((26, 3), np.float32)
        for hi, b25 in HALPE26_TO_BODY25:
            x, y = p2d[b25]
            if sigma_px:
                x += rng.normal(0, sigma_px); y += rng.normal(0, sigma_px)
            halpe[hi] = (x, y, 1.0)
        best[cid] = Pose2D(camera_id=cid, keypoints=halpe, score=1.0, skeleton="halpe26")
    return best


def errmm(res, jgt):
    if res is None:
        return float("nan")
    return float(np.median(np.linalg.norm(np.asarray(res["joints_body25"]) - jgt, axis=1)) * 1000)


# ---------------------------------------------------------------- 计数器
class FwdCounter:
    def __init__(self, body):
        self.body = body
        self.n = 0
        self.orig = body.forward

    def __enter__(self):
        ctr = self

        def wrapped(*a, **kw):
            ctr.n += 1
            return ctr.orig(*a, **kw)
        self.body.forward = wrapped
        return self

    def __exit__(self, *a):
        self.body.forward = self.orig


# ---------------------------------------------------------------- 配置
def cfg_sync():
    return EMSettings(ftol=1e-4, maxiters=100, no_item_sync=True, warm_init=False)


def cfg_loose():
    return EMSettings(ftol=5e-4, maxiters=40, no_item_sync=True, warm_init=False)


def cfg_warm():
    return EMSettings(ftol=5e-4, maxiters=40, no_item_sync=True,
                      warm_init=True, skip_global_rt_warm=True, refit_shape_every=0)


def cfg_warm_strict():          # E：warm + 官方收敛阈值（隔离 warm 本身的收益）
    return EMSettings(ftol=1e-4, maxiters=100, no_item_sync=True,
                      warm_init=True, skip_global_rt_warm=True, refit_shape_every=0)


def cfg_warm_mid():             # F：warm + ftol 3e-4（误差/速度折中）
    return EMSettings(ftol=3e-4, maxiters=60, no_item_sync=True,
                      warm_init=True, skip_global_rt_warm=True, refit_shape_every=0)


def cfg_stream():               # G：streaming 激进档（Phase 4 目标帧率档）
    return EMSettings(ftol=1.5e-3, maxiters=25, no_item_sync=True,
                      warm_init=True, skip_global_rt_warm=True, refit_shape_every=0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--noise", type=float, default=0.5)
    ap.add_argument("--out", default="data/bench/optimized.json")
    ap.add_argument("--configs", default="A,B,C,D")
    args = ap.parse_args()
    want = args.configs.split(",")

    intr, ext = load_camera_rig()
    cids = sorted(ext)
    recon = EasymocapReconstructor(verbose=False)
    body = recon._model

    # ---- 预生成序列（噪声一致，供各配置复用）----
    rng = np.random.default_rng(0)
    base = make_person(recon, rng)
    seq = []
    for t in range(args.frames):
        p = frame_params(base, t, args.frames)
        with torch.no_grad():
            j = body(return_verts=False, return_tensor=False, **p)[0]
        seq.append((p, j, observe(intr, ext, cids, j, args.noise, rng)))

    results = {}
    order = {"A": "original", "B": "sync", "C": "loose", "D": "warm",
             "E": "warm_strict", "F": "warm_mid", "G": "stream"}

    def run_official(obs):
        return recon.reconstruct(obs, intr, ext)

    # A：original 必须先跑（此时 optimize_simple 尚未被 install 改写）
    if "A" in want:
        print("\n=== A original（官方入口，零改动）===")
        rows = []
        for t, (_, jgt, obs) in enumerate(seq):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with FwdCounter(body) as fc:
                res = run_official(obs)
            torch.cuda.synchronize()
            rows.append({"frame": t, "wall_s": time.perf_counter() - t0,
                         "err_mm": errmm(res, jgt), "fwd": fc.n})
            print(f"  frame {t}: {rows[-1]['wall_s']*1000:.0f}ms err={rows[-1]['err_mm']:.1f}mm")
        results["A_original"] = rows

    # B/C/D：EmFit（install 幂等；settings 由构造传入）
    def run_emfit(settings, warm):
        fit = EmFit(recon, settings, verbose=False)
        rows = []
        prev = None
        for t, (_, jgt, obs) in enumerate(seq):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with FwdCounter(body) as fc:
                res = fit.run(obs, intr, ext, prev=prev)
            torch.cuda.synchronize()
            rows.append({"frame": t, "wall_s": time.perf_counter() - t0,
                         "err_mm": errmm(res, jgt), "fwd": fc.n})
            prev = res["params"] if res is not None else prev
            print(f"  frame {t}: {rows[-1]['wall_s']*1000:.0f}ms err={rows[-1]['err_mm']:.1f}mm "
                  f"fwd={rows[-1]['fwd']}")
        return rows

    if "B" in want:
        print("\n=== B sync（no_item_sync，ftol 同官方）===")
        results["B_sync"] = run_emfit(cfg_sync(), warm=False)
    if "C" in want:
        print("\n=== C loose（+ftol5e-4/maxiters40 冷启动）===")
        results["C_loose"] = run_emfit(cfg_loose(), warm=False)
    if "D" in want:
        print("\n=== D warm（+热启动，β 固定，ftol5e-4）===")
        results["D_warm"] = run_emfit(cfg_warm(), warm=True)
    if "E" in want:
        print("\n=== E warm_strict（warm，ftol 同官方 1e-4）===")
        results["E_warm_strict"] = run_emfit(cfg_warm_strict(), warm=True)
    if "F" in want:
        print("\n=== F warm_mid（warm，ftol3e-4）===")
        results["F_warm_mid"] = run_emfit(cfg_warm_mid(), warm=True)
    if "G" in want:
        print("\n=== G stream（warm + ftol1.5e-3/maxiters25）===")
        results["G_stream"] = run_emfit(cfg_stream(), warm=True)

    # ---- 汇总 ----
    def med(arr):
        return float(np.median([r["wall_s"] for r in arr]))

    def med_err(arr):
        return float(np.median([r["err_mm"] for r in arr]))

    cold = med(results.get("A_original", [{"wall_s": float("nan")}]))
    summary = {"frames": args.frames, "noise_px": args.noise, "per_config": {}}
    for key, rows in results.items():
        steady = [r for r in rows if r["frame"] > 0]
        entry = {"all_frames_median_ms": med(rows) * 1000,
                 "steady_median_ms": med(steady) * 1000 if steady else None,
                 "err_mm_median": med_err(rows),
                 "speedup_vs_original": (cold / med(rows)) if med(rows) else None,
                 "fwd_median": float(np.median([r["fwd"] for r in rows]))}
        summary["per_config"][key] = entry

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== 汇总 ===")
    print(f"{'config':14s} {'中位帧ms':>9s} {'稳态帧ms':>9s} {'err mm':>8s} {'提速':>6s}")
    for key, rows in results.items():
        e = summary["per_config"][key]
        sp = "" if e["speedup_vs_original"] is None else f"x{e['speedup_vs_original']:.2f}"
        st = "-" if e["steady_median_ms"] is None else f"{e['steady_median_ms']:.0f}"
        print(f"{key:14s} {e['all_frames_median_ms']:9.0f} {st:>9s} "
              f"{e['err_mm_median']:8.1f} {sp:>6s}")
    print(f"JSON -> {args.out}")


if __name__ == "__main__":
    main()
