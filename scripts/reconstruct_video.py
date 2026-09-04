#!/usr/bin/env python3
"""离线用 EasyMocap 重建某次四路录像：``data/video/<session>/cam{0..3}.mp4``。

流程（与 live_control 在线 EasyMocap 同一套检测 + 拟合，只是输入换成视频文件）
  1) 读 session 文件夹里的 ``cam{cid}.mp4``（cid = 标定相机号）＋ ts 副产物；
  2) 按设备时间戳把四路重新对齐到主时钟相机（允许编码丢帧，见 video_source 文档）；
  3) 每个主时钟帧：各相机对齐帧 → 人检测(yolo11n-gray) + 姿态模型（默认 RTMPose
     halpe26，``--pose-model vitpose-b-coco_25`` 切 ViTPose，输出同样重排成 halpe26）→
     每相机取最高置信度的人 → SMPL 拟合（默认官方多帧批量，激活帧间平滑）；
  4) 球轨迹（独立一遍，默认开启）：逐帧各相机球检测（经典 / YOLO）→ DLT 三角化
     → 3D 球心，写 ``ball_trajectory.npz``（供 visualize_recon.py 渲染红球 + 轨迹线）；
  5) 输出：逐帧 ``npz`` + ``recon_index.npz`` + ``recon_meta.json``。

用法
  python scripts/reconstruct_video.py data/video/20260902_180000 \
      [--config batch] [--stride 1] [--ref-cam 0] [--out ...]

拟合档位 --config
  batch    = 官方多帧批量拟合（多人）：整段一次 smpl_from_keypoints3d2d，激活时间平滑（默认）
  official = 官方 cold ``reconstruct()``，每帧冷启动（最慢，行为=原版）
  warm     = EmFit 热启动 ftol 5e-4 / maxiters 40（约 x1.8，误差≈官方）
  stream   = EmFit 热启动 ftol 1.5e-3 / maxiters 25（约 x2.3，误差略优于官方）

多人识别：``--person-groups "[[0,2],[1,3]]"`` 每个组=一个人（组内相机拍同一个人，
组下标=身份，仿 live_control 按 P 的 match_people_fixed）。单人用 ``--person-groups
"[[1,3]]"``（只放拍人的那组相机）。``--det-conf`` 人检测阈值（默认 0.2）、
``--fit-conf`` 拟合关键点阈值（默认 0.15）——黑衣服/低亮度人关键点置信度偏低，
阈值太高会把 2D 已检出的关节在 3D 拟合里丢掉，致其退化成均值模型。

调试选项
  --fake-poses  不跑检测，注入一个合成站姿人观测 → 专用于验证 录制→对齐→重建→存档
                整条管道（视频内容无关），或给重建流程计时。
  --stride N    主时钟每 N 帧重建 1 帧（默认 1 全量）。
  --max-frames  最多处理前 N 个主时钟帧（调试/计时用，0 = 全部）。

球重建选项（默认开启）
  --no-ball          跳过球轨迹重建。
  --ball-detector    球检测路线：classical（默认）/ yolo。
  --ball-model       显式指定 YOLO 球 ONNX 路径（覆盖 yolo 的自动查找）。
  --ball-min-conf    球三角化最低置信度（默认 0.3）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
sys.path.insert(0, os.path.join(_ROOT, "src"))

import numpy as np


def _add_easymocap_path(root: str) -> None:
    if os.path.isdir(root) and root not in sys.path:
        sys.path.insert(0, root)


def _median(x):
    x = [v for v in x if v is not None and not (isinstance(v, float) and np.isnan(v))]
    return float(np.median(x)) if x else float("nan")


# ----------------------------------------------------------------------
# 合成站姿观测（--fake-poses）：用同一套 2D 投影生成一个固定人，模拟每帧检测结果
# ----------------------------------------------------------------------
_FAKE_PS = None   # lazy: SMPL 参数（站立在球桌中心附近）


def fake_obs_for(recon, intrinsics, extrinsics, cids, rng=None):
    """给 cids 每台相机一份同一站姿人的 halpe26 Pose2D（噪声可选）。"""
    from tabletennis.core.types import Pose2D
    from tabletennis.reconstruction.easymocap import HALPE26_TO_BODY25
    global _FAKE_PS
    if _FAKE_PS is None:
        body = recon._model
        rng0 = np.random.default_rng(0)
        betas = rng0.uniform(-0.5, 0.5, size=10).astype(np.float32)
        betas[0] = 1.0
        ps = np.zeros((1, 72), np.float32)
        ps[0, 3 * 3:4 * 3] = [0, 0, 0.15]
        ps[0, 18 * 3:19 * 3] = [0, 0.5, 0]
        ps[0, 4 * 3:5 * 3] = [0.15, 0, 0]
        ps[0, 5 * 3:6 * 3] = [-0.15, 0, 0]
        p = {"poses": ps, "shapes": betas[None],
             "Rh": np.array([[np.pi / 2, 0, 0]], np.float32),
             "Th": np.array([[1.0, 1.0, 0.386]], np.float32)}
        with __import__("torch").no_grad():
            v = body(return_verts=True, return_tensor=False, **p)[0]
        p["Th"] = p["Th"] + np.array([[0, 0, -0.76 - v[:, 2].min()]], np.float32)
        with __import__("torch").no_grad():
            j25 = body(return_verts=False, return_tensor=False, **p)[0]
        _FAKE_PS = j25
    j25 = _FAKE_PS
    if rng is None:
        rng = np.random.default_rng(1)
    best: Dict[int, object] = {}
    for cid in cids:
        K = intrinsics[cid].K
        P = K @ np.hstack([extrinsics[cid].R, extrinsics[cid].t.reshape(3, 1)])
        c = np.hstack([j25, np.ones((25, 1))]) @ P.T
        p2d = c[:, :2] / c[:, 2:3]
        halpe = np.zeros((26, 3), np.float32)
        for hi, b25 in HALPE26_TO_BODY25:
            x, y = p2d[b25]
            x += rng.normal(0, 0.3); y += rng.normal(0, 0.3)
            halpe[hi] = (x, y, 1.0)
        best[cid] = Pose2D(camera_id=cid, keypoints=halpe, score=1.0, skeleton="halpe26")
    return best


def _proj_err_px(Pall: np.ndarray, kp2d: np.ndarray, j25: np.ndarray) -> tuple:
    """拟合出的 body25 关节（世界系）投影回各视角 vs 观测 2D 的重投影误差 (mean, max) px。"""
    errs = []
    for i in range(kp2d.shape[0]):
        c = np.hstack([j25, np.ones((25, 1))]) @ Pall[i].T      # (25,3) 相机系
        uv = c[:, :2] / c[:, 2:3]
        m = kp2d[i, :, 2] > 0
        if m.sum() == 0:
            continue
        errs.append(np.linalg.norm(uv[m] - kp2d[i, m, :2], axis=1))
    if not errs:
        return float("nan"), float("nan")
    all_err = np.concatenate(errs)
    return float(np.mean(all_err)), float(np.max(all_err))


# ----------------------------------------------------------------------
def build_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session_dir", help="data/video/<session> 文件夹（含 cam*.mp4）")
    ap.add_argument("--config", choices=["official", "warm", "stream", "batch"], default="batch")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=0, help="0=全部")
    ap.add_argument("--ref-cam", type=int, default=None)
    ap.add_argument("--min-cams", type=int, default=2)
    ap.add_argument("--det-conf", type=float, default=0.2,
                    help="人检测置信度阈值（yolo11n bbox；黑衣服/低亮度人易被漏检，离线降到 0.2 少丢人）")
    ap.add_argument("--fit-conf", type=float, default=0.15,
                    help="SMPL 拟合关键点置信度阈值（halpe26 关键点低于此被丢弃；黑衣服人关键点偏低，降到 0.15 保留更多约束）")
    ap.add_argument("--pose-model", default="rtmpose-x-halpe26",
                    help="姿态模型：rtmpose-x-halpe26（默认，384×288 更高精度，脚部更准）/ "
                         "rtmpose-l-halpe26（256×192 更快）/ vitpose-b-coco_25 等（ViTPose，"
                         "coco_25 25 点含脚，输出重排成 halpe26；vitpose-s 最快 / b 甜点 / l 最强）")
    ap.add_argument("--pose-input-size", type=int, nargs=2, default=(288, 384),
                    metavar=("H", "W"), help="姿态模型输入尺寸 (H, W)，默认 288 384（对应 384×288）")
    ap.add_argument("--vposer", action="store_true",
                    help="用 VPoser 潜空间拟合姿态（治本：姿态恒在自然流形，消腿扭/翻转），替代官方 batch 拟合")
    ap.add_argument("--vposer-ckpt", default=None,
                    help="VPoser checkpoint 路径（默认 VPOSER_CKPT 环境变量或 easymocap.DEFAULT_VPOSER_CKPT）")
    ap.add_argument("--person-groups", default="[[0,2],[1,3]]",
                    help="相机分组 JSON：每个组=一个人，组内相机拍同一个人（仿 live_control P）")
    ap.add_argument("--out", default=None, help="输出目录（默认 <session>/recon）")
    ap.add_argument("--root", default=None,
                    help="项目根（读 data/calibration 与 data/extrinsics，默认自动探测）")
    ap.add_argument("--fake-poses", action="store_true")
    ap.add_argument("--em-ftol", type=float, default=None)
    ap.add_argument("--em-maxiters", type=int, default=None)
    ap.add_argument("--progress", type=int, default=10)
    ap.add_argument("--em-verbose", action="store_true")
    ap.add_argument("--easymocap-root", default="/home/yby/projects/EasyMocap")
    ap.add_argument("--no-ball", action="store_true", help="跳过球轨迹重建（默认开启）")
    ap.add_argument("--ball-detector", choices=["classical", "yolo"], default="yolo",
                    help="球检测路线：yolo（默认，onnx，与 live_control 一致）/ classical（背景减除+帧差，先验弱易误检）")
    ap.add_argument("--ball-model", default=None,
                    help="YOLO 球 ONNX 模型路径（给定则覆盖 --ball-detector yolo 的自动查找）")
    ap.add_argument("--ball-imgsz", type=int, default=1280, help="YOLO 球检测输入分辨率")
    ap.add_argument("--ball-min-conf", type=float, default=0.15, help="球三角化最低置信度")
    ap.add_argument("--ball-radius-min", type=float, default=5.0, help="经典检测球半径像素下限")
    ap.add_argument("--ball-radius-max", type=float, default=15.0, help="经典检测球半径像素上限")
    return ap.parse_args()


def run_batch_mode(args, src, intrinsics, extrinsics, cids_ok, recon, detector, out_dir):
    """官方多帧批量拟合（多人）：按相机分组识别身份 → 逐人批量拟合。

    跨相机身份关联仿 live_control 按 P 的实时重建：``match_people_fixed`` 固定相机
    分组（每组相机拍同一个人，组下标=身份）。对每个身份把整段帧一次性喂给官方
    ``smpl_from_keypoints3d2d``（nFrames=T），激活 smooth_body/smooth_poses/smooth_Rh
    时间平滑；单视角关节由相邻帧约束 + 模型先验补全。
    """
    from tabletennis.reconstruction.associate import match_people_fixed
    from tabletennis.reconstruction.obs2d import pose_to_dict, save_pose2d
    from tabletennis.reconstruction.triangulate import MultiViewTriangulator

    # 过滤掉未标定的相机，组内相机数 <2 则该身份无法三角化（后面跳过）
    person_groups = [[c for c in g if c in cids_ok] for g in args.person_groups]
    n_people = len(person_groups)
    triangulator = MultiViewTriangulator(intrinsics, extrinsics)
    print(f"  相机分组（{n_people} 人）：{person_groups}")

    indices = list(range(0, src.n_ref, max(1, args.stride)))
    if args.max_frames > 0:
        indices = indices[: args.max_frames]

    # ---- Pass 1：检测 + 分组匹配，按身份（组下标）收集每帧观测 ----
    frames_obs_by_pid = {g: [] for g in range(n_people)}   # pid -> [ {cid:Pose2D} per frame ]
    n_seen_by_pid = {g: 0 for g in range(n_people)}
    pose2d_by_frame = {}                                    # 全部检出 2D 姿态（存盘供回放叠加）
    t_global0 = time.time()
    t0 = time.time()
    for k in indices:
        if args.fake_poses:
            # 合成：每组各造一个固定站姿人（同一个人投到该组相机，测管道用）
            for g, group in enumerate(person_groups):
                obs = fake_obs_for(recon, intrinsics, extrinsics, group)
                frames_obs_by_pid[g].append(obs)
                n_seen_by_pid[g] += 1
            continue
        frames_k = src.frames_for_ref(k)
        frames_k = {cid: f for cid, f in frames_k.items() if cid in cids_ok}
        if frames_k:
            items = sorted(frames_k.items())
            plist = detector.detect_batch([f for _, f in items])
            poses_per_cam = {cid: pl for (cid, _f), pl in zip(items, plist) if pl}
        else:
            poses_per_cam = {}
        if poses_per_cam:
            pose2d_by_frame[str(k)] = {
                str(cid): [pose_to_dict(p) for p in pl] for cid, pl in poses_per_cam.items()
            }
        for g, group in enumerate(person_groups):
            matched = match_people_fixed(poses_per_cam, triangulator, groups=[group])
            obs = matched[0] if matched else {}
            frames_obs_by_pid[g].append(obs)
            if obs:
                n_seen_by_pid[g] += 1
    det_wall = time.time() - t0
    stats = []
    for g in range(n_people):
        seen = frames_obs_by_pid[g]
        n_seen = sum(1 for obs in seen if obs)
        avg_views = float(np.mean([len(obs) for obs in seen if obs])) if n_seen else 0.0
        stats.append(f"p{g}={n_seen}帧/均{avg_views:.1f}视角")
    print(f"  检测 {len(indices)} 帧耗时 {det_wall:.1f}s | 各身份被看到：{', '.join(stats)}"
          + "（均<2 视角=常单视角，重建会退化成均值模型）")
    if pose2d_by_frame:
        save_pose2d(out_dir, pose2d_by_frame)
        print(f"  2D 姿态检测已存 pose2d.json（{len(pose2d_by_frame)} 帧）")

    # ---- Pass 2：逐人批量拟合 ----
    t0 = time.time()
    results_by_pid = {}
    for g, group in enumerate(person_groups):
        if len(group) < 2:
            print(f"  ⚠ p{g} 组内相机数 {len(group)} < 2，跳过（无法三角化）")
            results_by_pid[g] = None
            continue
        if n_seen_by_pid[g] == 0:
            print(f"  p{g}：整段未匹配到人，跳过")
            results_by_pid[g] = None
            continue
        if args.vposer:
            results_by_pid[g] = recon.reconstruct_vposer(
                frames_obs_by_pid[g], intrinsics, extrinsics,
                min_conf=args.fit_conf, view_ids=sorted(group),
                vposer_ckpt=args.vposer_ckpt)
            if results_by_pid[g] is None:
                print(f"  ⚠ p{g} VPoser 拟合失败")
        else:
            results_by_pid[g] = recon.reconstruct_batch(
                frames_obs_by_pid[g], intrinsics, extrinsics,
                min_conf=args.fit_conf, view_ids=sorted(group))
            if results_by_pid[g] is None:
                print(f"  ⚠ p{g} 批量拟合失败")
    fit_wall = time.time() - t0
    print(f"  批量拟合 {n_people} 人耗时 {fit_wall:.1f}s")

    # ---- 逐帧合并多人结果 + 写档 ----
    n_ok = n_gap = 0
    err_mean_arr, err_worst_arr = [], []
    idx_arr, status_arr, wall_arr = [], [], []
    code = {"ok": 0, "no_person": 1, "fit_failed": 2, "error": 3}
    per_frame_ms = (det_wall + fit_wall) / max(1, len(indices)) * 1000.0
    for t, k in enumerate(indices):
        people = []
        for g in range(n_people):
            r = results_by_pid[g]
            if r is not None and r[t] is not None:
                people.append(r[t])
        if not people:
            n_gap += 1
            idx_arr.append(k); status_arr.append(code["no_person"]); wall_arr.append(per_frame_ms)
            err_mean_arr.append(float("nan")); err_worst_arr.append(float("nan"))
            continue
        # 重投影误差：逐人算，跨人取均值/最差
        errs = []
        for g in range(n_people):
            r = results_by_pid[g]
            if r is None or r[t] is None:
                continue
            obs = frames_obs_by_pid[g][t]
            if not obs:
                continue
            cids_g = sorted(person_groups[g])
            kp2d, _ = recon._obs_to_body25_fixed(obs, intrinsics, extrinsics, 0.0, cids_g)
            Pall_g = np.stack([intrinsics[c].K
                               @ np.hstack([extrinsics[c].R, extrinsics[c].t.reshape(3, 1)])
                               for c in cids_g])
            e = _proj_err_px(Pall_g, kp2d, np.asarray(r[t]["joints_body25"]))
            if np.isfinite(e[0]):
                errs.append(e)
        em = float(np.mean([e[0] for e in errs])) if errs else float("nan")
        ew = float(np.max([e[1] for e in errs])) if errs else float("nan")
        err_mean_arr.append(em); err_worst_arr.append(ew)
        idx_arr.append(k); status_arr.append(code["ok"]); wall_arr.append(per_frame_ms)
        n_ok += 1
        ensure_faces(out_dir, people[0])
        save_one_multi(out_dir, k, people, per_frame_ms, em, ew)
        if (t + 1) % max(1, args.progress) == 0 or t == len(indices) - 1:
            print(f"  写档 {t+1}/{len(indices)} (主时钟 {k}/{src.n_ref}) | ok={n_ok} gap={n_gap}")

    src.close()

    # ---- 汇总 / 存档 ----
    wall_s = time.time() - t_global0
    np.savez(os.path.join(out_dir, "recon_index.npz"),
             ref_frame=np.asarray(idx_arr, np.int64),
             status=np.asarray(status_arr, np.int8),
             wall_ms=np.asarray(wall_arr, np.float64),
             err_mean_px=np.asarray(err_mean_arr, np.float64),
             err_worst_px=np.asarray(err_worst_arr, np.float64))
    index_data = {
        "session_dir": args.session_dir, "out_dir": out_dir,
        "config": args.config, "stride": args.stride,
        "ref_cam": src.ref_cam,
        "n_ref_frames": src.n_ref, "processed": len(indices),
        "ok": n_ok, "no_person_gap": n_gap, "failed": 0,
        "wall_s": round(wall_s, 3),
        "detect_wall_s": round(det_wall, 3),
        "fit_wall_s": round(fit_wall, 3),
        "person_groups": person_groups, "n_people": n_people,
        "reproj_err_mean_px_median": _median(err_mean_arr),
        "source": src.summary(),
    }
    with open(os.path.join(out_dir, "recon_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(index_data, fh, ensure_ascii=False, indent=2)
    print("=== 完成 ===")
    print(f"  处理 {len(indices)} 帧（ok={n_ok}, 无人缺口={n_gap}）耗时 {wall_s:.1f}s")
    print(f"  检测 {det_wall:.1f}s + 批量拟合 {fit_wall:.1f}s（{n_people} 人）")
    print(f"  重投影误差中位 {_median(err_mean_arr):.2f}px |  → {out_dir}")


def run_ball_recon(args, src, intrinsics, extrinsics, cids_ok, out_dir):
    """离线球轨迹重建：逐主时钟帧各相机球检测 → 置信度加权 DLT 三角化 → 存轨迹。

    与姿态重建独立跑一遍（球检测 ~3.6ms/相机，解码开销远小于 SMPL 拟合）。球检测
    默认经典路线（背景减除 + 帧差 + 尺寸先验，有状态、按相机分背景）；``--ball-detector
    yolo`` / ``--ball-model`` 走 onnxruntime YOLO。结果写 ``out_dir/ball_trajectory.npz``
    （每帧 ``ref_frame`` + 3D 球心 ``X``，失败帧 ``X=NaN``），供 visualize_recon.py 回放渲染。
    """
    from tabletennis.reconstruction.ball import triangulate_ball
    from tabletennis.reconstruction.obs2d import ball_to_dict, save_ball2d
    from tabletennis.reconstruction.triangulate import MultiViewTriangulator
    from tabletennis.vision.ball import ClassicalBallDetector, YoloBallDetector
    from tabletennis.vision.detector import create_detector

    triangulator = MultiViewTriangulator(intrinsics, extrinsics)
    cids = sorted(set(triangulator.cameras) & set(cids_ok))
    if len(cids) < 2:
        print("✗ 标定视角不足 2 个，跳过球轨迹重建。")
        return None

    # 检测器：YOLO（显式路径 / 自动找权重）或经典（每相机一个，背景模型独立）
    ball_det = None
    if args.ball_model:
        if os.path.exists(args.ball_model):
            ball_det = YoloBallDetector(args.ball_model, imgsz=args.ball_imgsz)
        else:
            print(f"⚠ --ball-model 指定模型不存在：{args.ball_model}，回退经典检测。")
    elif args.ball_detector == "yolo":
        ball_det = create_detector("ball_yolo")
        if ball_det is None:
            print("⚠ YOLO 球权重缺失（runs/detect/*/weights/best.onnx），回退经典检测。")
    if ball_det is not None:
        print(f"球检测：YOLO（backend={getattr(ball_det, 'actual_provider', '?')}，"
              f"imgsz={getattr(ball_det, 'imgsz', args.ball_imgsz)}）")
    else:
        dets = {cid: ClassicalBallDetector(
            radius_px=(args.ball_radius_min, args.ball_radius_max)) for cid in cids}
        print(f"球检测：经典（背景减除+帧差+尺寸先验，{len(dets)} 相机各一实例）")

    indices = list(range(0, src.n_ref, max(1, args.stride)))
    if args.max_frames > 0:
        indices = indices[: args.max_frames]

    ref_arr, X_arr = [], []
    conf_arr, err_arr, nv_arr, ang_arr = [], [], [], []
    ball2d_by_frame = {}
    t0 = time.time()
    n_ball = 0
    for k in indices:
        frames_k = {cid: f for cid, f in src.frames_for_ref(k).items() if cid in cids}
        balls = {}
        ball2d_k = {}
        if frames_k:
            if ball_det is not None:
                items = sorted(frames_k.items())
                if hasattr(ball_det, "detect_batch"):
                    outs = ball_det.detect_batch([f for _, f in items])
                else:
                    outs = [ball_det.detect(f) for _, f in items]
                for (cid, _f), bl in zip(items, outs):
                    if bl:
                        balls[cid] = bl[0]  # 单球，取最高置信者（YOLO 已按 conf 排序）
                        ball2d_k[str(cid)] = [ball_to_dict(b) for b in bl]
            else:
                for cid, f in frames_k.items():
                    d = dets[cid].detect(f)
                    if d:
                        balls[cid] = d[0]
                        ball2d_k[str(cid)] = [ball_to_dict(b) for b in d]
        if ball2d_k:
            ball2d_by_frame[str(k)] = ball2d_k
        res = triangulate_ball(balls, triangulator, min_conf=args.ball_min_conf) if balls else None
        ref_arr.append(k)
        if res is not None:
            X, conf, err, nv, ang = res
            X_arr.append(np.asarray(X, np.float64).reshape(3))
            conf_arr.append(float(conf)); err_arr.append(float(err))
            nv_arr.append(int(nv)); ang_arr.append(float(ang))
            n_ball += 1
        else:
            X_arr.append(np.full(3, np.nan))
            conf_arr.append(0.0); err_arr.append(np.nan)
            nv_arr.append(0); ang_arr.append(0.0)
        if len(ref_arr) % max(1, args.progress) == 0:
            print(f"  球重建 {len(ref_arr)}/{len(indices)}（主时钟 {k}/{src.n_ref}，"
                  f"已成功 {n_ball}）")
    wall = time.time() - t0
    src.close()

    np.savez(
        os.path.join(out_dir, "ball_trajectory.npz"),
        ref_frame=np.asarray(ref_arr, np.int64),
        X=np.asarray(X_arr, np.float64),
        conf=np.asarray(conf_arr, np.float64),
        reproj_err=np.asarray(err_arr, np.float64),
        n_views=np.asarray(nv_arr, np.int32),
        angle_deg=np.asarray(ang_arr, np.float64),
    )
    if ball2d_by_frame:
        save_ball2d(out_dir, ball2d_by_frame)
    with open(os.path.join(out_dir, "ball_meta.json"), "w", encoding="utf-8") as fh:
        json.dump({
            "detector": "yolo" if ball_det is not None else "classical",
            "min_conf": args.ball_min_conf,
            "n_frames": len(indices), "ok": n_ball,
            "wall_s": round(wall, 3),
            "ref_cam": src.ref_cam,
        }, fh, ensure_ascii=False, indent=2)
    print(f"球轨迹：{n_ball}/{len(indices)} 帧三角化成功，耗时 {wall:.1f}s"
          f" → {os.path.join(out_dir, 'ball_trajectory.npz')}")
    return {"n_frames": len(indices), "ok": n_ball, "wall_s": round(wall, 3)}


def main() -> None:
    args = build_args()
    _add_easymocap_path(args.easymocap_root)

    from tabletennis.core.types import Frame  # noqa: F401
    from tabletennis.reconstruction.easymocap import EasymocapReconstructor
    from tabletennis.reconstruction.em_fit import EmFit, EMSettings
    from tabletennis.reconstruction.triangulate import load_camera_rig
    from tabletennis.reconstruction.video_source import VideoSource

    session_dir = args.session_dir
    if not os.path.isdir(session_dir):
        print(f"✗ 找不到文件夹：{session_dir}")
        sys.exit(2)

    out_dir = args.out or os.path.join(session_dir, "recon")
    os.makedirs(out_dir, exist_ok=True)

    # ---- 输入：四路视频 + 对齐 ----
    src = VideoSource(session_dir, ref_cam=args.ref_cam)
    print("=== 视频源 ===")
    for cid in src.cids:
        print(f"  cam{cid}: {src.n_frames_by_cam[cid]} 帧，"
              f"脉冲周期 {src.period_sec(cid)*1000:.2f}ms"
              + ("（有 ts）" if cid in src.ts_by_cam else "（无 ts，按帧号对齐）"))
    print(f"  主时钟 cam{src.ref_cam} → 对齐后 {src.n_ref} 帧")

    # ---- 标定 ----
    intrinsics, extrinsics = load_camera_rig(args.root)
    cids_ok = [c for c in src.cids if c in intrinsics and c in extrinsics]
    missing = [c for c in src.cids if c not in cids_ok]
    if missing:
        print(f"  ⚠ 以下相机无标定，重建时跳过：{missing}")

    # ---- 模型 / 检测 / 拟合器 ----
    recon = EasymocapReconstructor(verbose=False)
    if not recon.ready:
        print(f"✗ {recon.error}")
        sys.exit(1)
    if args.fake_poses:
        detector = None
    elif args.pose_model.startswith("vitpose"):
        # ViTPose：easy_ViTPose coco_25（25 点含脚）ONNX，输出重排成 halpe26。
        from tabletennis.vision.pose.vitpose_pose import ViTPosePoseDetector
        detector = ViTPosePoseDetector(model=args.pose_model, device="cuda",
                                       backend="cuda", score_thr=args.det_conf)
        print(f"  人检测置信度阈值：{args.det_conf} | 姿态模型：{args.pose_model}（ViTPose）")
    else:
        from tabletennis.vision.pose.rtmpose_pose import RTMPoseDetector
        detector = RTMPoseDetector(device="cuda", backend="tensorrt",
                                   score_thr=args.det_conf,
                                   model=args.pose_model,
                                   input_size=tuple(args.pose_input_size))
        print(f"  人检测置信度阈值：{args.det_conf} | 姿态模型：{args.pose_model} "
              f"输入 {tuple(args.pose_input_size)}")
    if detector is None and not args.fake_poses:
        print("✗ 姿态检测器未就绪")
        sys.exit(1)

    # ---- 球轨迹重建（独立一遍，先跑，写 ball_trajectory.npz；失败不阻断姿态）----
    if not args.no_ball:
        print("=== 球轨迹重建 ===")
        try:
            src_ball = VideoSource(session_dir, ref_cam=args.ref_cam)
            run_ball_recon(args, src_ball, intrinsics, extrinsics, cids_ok, out_dir)
        except Exception as exc:  # noqa: BLE001 —— 球失败不影响姿态重建
            print(f"✗ 球轨迹重建失败（姿态重建继续）：{exc}")
    # 解析相机分组（每个组=一个人）
    try:
        args.person_groups = json.loads(args.person_groups)
    except Exception as exc:  # noqa: BLE001
        print(f"✗ --person-groups 解析失败：{exc}")
        sys.exit(2)
    if (not isinstance(args.person_groups, list) or not args.person_groups
            or not all(isinstance(g, list) and g for g in args.person_groups)):
        print("✗ --person-groups 须为非空列表的列表，如 '[[0,2],[1,3]]'")
        sys.exit(2)

    if args.config == "batch":
        run_batch_mode(args, src, intrinsics, extrinsics, cids_ok, recon, detector, out_dir)
        return

    if args.config == "official":
        fit = None
        print("档位：official（官方 cold reconstruct，每帧冷启动）")
    else:
        s = EMSettings(
            ftol=args.em_ftol if args.em_ftol is not None else (1.5e-3 if args.config == "stream" else 5e-4),
            maxiters=args.em_maxiters if args.em_maxiters is not None else (25 if args.config == "stream" else 40),
            no_item_sync=True, warm_init=True, skip_global_rt_warm=True, refit_shape_every=0,
        )
        fit = EmFit(recon, s, verbose=args.em_verbose)
        print(f"档位：{args.config}（EmFit 热启动 ftol={s.ftol} maxiters={s.maxiters}）")

    # ---- 主循环 ----
    def run_one(k: int):
        nonlocal prev, detector, fit
        frames_k = src.frames_for_ref(k)
        frames_k = {cid: f for cid, f in frames_k.items() if cid in cids_ok}
        if not frames_k:
            return None, {"status": "no_person"}
        if args.fake_poses:
            best = fake_obs_for(recon, intrinsics, extrinsics, list(frames_k.keys()))
        else:
            items = sorted(frames_k.items())
            poses = detector.detect_batch([f for _, f in items])
            best = {}
            for (cid, _f), pl in zip(items, poses):
                if pl:
                    best[cid] = max(pl, key=lambda p: p.score)
        if len(best) < args.min_cams:
            return None, {"status": "no_person"}
        if fit is not None:
            res = fit.run(best, intrinsics, extrinsics, prev=prev)
        else:
            res = recon.reconstruct(best, intrinsics, extrinsics)
        if res is None:
            return None, {"status": "fit_failed"}
        prev = res["params"]
        # 重投影误差（拟合关节 vs 观测，px）
        kp2d, _, Pall = recon._to_body25_2d(best, intrinsics, extrinsics, min_conf=0.0)
        if kp2d is not None:
            em, ew = _proj_err_px(Pall, kp2d, np.asarray(res["joints_body25"]))
        else:
            em = ew = float("nan")
        return res, {"status": "ok", "cam_idx": {c: src.maps[k][c] for c in best},
                     "err_mean": em, "err_worst": ew}

    prev = None
    n_total = src.n_ref
    indices = list(range(0, n_total, max(1, args.stride)))
    if args.max_frames > 0:
        indices = indices[: args.max_frames]
    t0 = time.time()
    n_ok = n_gap = n_fail = 0
    wall_ms_list = []
    err_list = []

    # 汇总数组（与 indices 一一对应，供 recon_index.npz 快速画图/分析）
    idx_arr, status_arr, wall_arr, err_mean_arr, err_worst_arr = [], [], [], [], []
    code = {"ok": 0, "no_person": 1, "fit_failed": 2, "error": 3}

    for i, k in enumerate(indices):
        t = time.perf_counter()
        try:
            res, extra = run_one(k)
        except Exception as exc:  # noqa: BLE001 —— 单帧异常不中断整段
            print(f"  [帧 {k}] ✗ 异常：{exc}")
            res, extra = None, {"status": "error"}
        dt = (time.perf_counter() - t) * 1000
        status = extra.get("status", "?")
        idx_arr.append(k)
        status_arr.append(code.get(status, 3))
        wall_arr.append(dt)
        if status == "ok":
            n_ok += 1
            wall_ms_list.append(dt)
            ensure_faces(out_dir, res)
            save_one(out_dir, k, res, dt, extra.get("err_mean"),
                     extra.get("err_worst"), extra.get("cam_idx"))
            err_mean_arr.append(extra.get("err_mean"))
            err_worst_arr.append(extra.get("err_worst"))
        else:
            err_mean_arr.append(float("nan"))
            err_worst_arr.append(float("nan"))
            if status == "no_person":
                n_gap += 1
            else:
                n_fail += 1
        if (i + 1) % max(1, args.progress) == 0 or i == len(indices) - 1:
            el = time.time() - t0
            print(f"  帧 {i+1}/{len(indices)} (主时钟 {k}/{n_total}) | "
                  f"{dt:6.0f}ms/帧 | 已过 {el:6.1f}s | ok={n_ok} gap={n_gap} fail={n_fail}")

    src.close()

    # ---- 汇总 / 存档 ----
    wall_s = time.time() - t0
    np.savez(os.path.join(out_dir, "recon_index.npz"),
             ref_frame=np.asarray(idx_arr, np.int64),
             status=np.asarray(status_arr, np.int8),
             wall_ms=np.asarray(wall_arr, np.float64),
             err_mean_px=np.asarray(err_mean_arr, np.float64),
             err_worst_px=np.asarray(err_worst_arr, np.float64))
    index_data = {
        "session_dir": session_dir, "out_dir": out_dir,
        "config": args.config, "stride": args.stride,
        "ref_cam": src.ref_cam,
        "n_ref_frames": n_total, "processed": len(indices),
        "ok": n_ok, "no_person_gap": n_gap, "failed": n_fail,
        "wall_s": round(wall_s, 3),
        "recon_per_frame_ms_median": _median(wall_ms_list),
        "reproj_err_mean_px_median": _median(err_mean_arr),
        "source": src.summary(),
    }
    with open(os.path.join(out_dir, "recon_meta.json"), "w", encoding="utf-8") as fh:
        json.dump(index_data, fh, ensure_ascii=False, indent=2)
    print("=== 完成 ===")
    print(f"  处理 {len(indices)} 帧（ok={n_ok}, 无人缺口={n_gap}, 失败={n_fail}）耗时 {wall_s:.1f}s")
    print(f"  平均 {wall_s/max(1, len(indices))*1000:.0f}ms/帧 | 单帧中位 {_median(wall_ms_list):.0f}ms")
    print(f"  重投影误差中位 {_median(err_mean_arr):.2f}px |  → {out_dir}")


def ensure_faces(out_dir: str, res: dict) -> None:
    """SMPL 网格拓扑（13776 面）只在输出目录写一次，供 3D 回放查看器重建网格。

    逐帧 npz 只存顶点以省空间；拓扑对所有帧相同，所以单独存 ``recon_faces.npy``。
    """
    path = os.path.join(out_dir, "recon_faces.npy")
    if os.path.exists(path):
        return
    faces = res.get("faces")
    if faces is None:
        return
    np.save(path, np.asarray(faces, dtype=np.int64))
    print(f"  ✓ 已存 SMPL 网格拓扑 {path}（{np.asarray(faces).shape[0]} 面）")


def save_one(out_dir: str, k: int, res: dict, dt_ms: float,
             err_mean=None, err_worst=None, cam_idx=None) -> None:
    """把一帧拟合结果存 npz（含网格/关节/SMPL 参数/参与视角/重投影误差）。"""
    np.savez_compressed(
        os.path.join(out_dir, f"frame_{k:06d}.npz"),
        ref_frame=k,
        joints=np.asarray(res["joints"], dtype=np.float32),
        joints_body25=np.asarray(res["joints_body25"], dtype=np.float32),
        vertices=np.asarray(res["vertices"], dtype=np.float32),
        params_poses=res["params"]["poses"].reshape(-1),
        params_shapes=res["params"]["shapes"].reshape(-1),
        params_Rh=res["params"]["Rh"].reshape(-1),
        params_Th=res["params"]["Th"].reshape(-1),
        wall_ms=dt_ms,
        err_mean_px=err_mean, err_worst_px=err_worst)
    if cam_idx:
        cids_a = np.asarray(list(cam_idx.keys()), np.int32)
        idx_a = np.asarray(list(cam_idx.values()), np.int64)
        with open(os.path.join(out_dir, f"frame_{k:06d}_cams.json"), "w") as fh:
            json.dump({str(c): int(i) for c, i in zip(cids_a, idx_a)}, fh)


def save_one_multi(out_dir: str, k: int, people: List[dict], dt_ms: float,
                   err_mean=None, err_worst=None) -> None:
    """把一帧多个人的拟合结果存进同一 npz。

    person 0 用无后缀键（与旧单帧格式兼容），person p≥1 用 ``_p`` 后缀；
    外加 ``n_people`` 标量供回放查看器知道本帧有几个人。
    """
    payload = {"ref_frame": k, "n_people": len(people),
               "wall_ms": dt_ms, "err_mean_px": err_mean, "err_worst_px": err_worst}
    for p, res in enumerate(people):
        suf = "" if p == 0 else f"_{p}"
        payload[f"vertices{suf}"] = np.asarray(res["vertices"], dtype=np.float32)
        payload[f"joints{suf}"] = np.asarray(res["joints"], dtype=np.float32)
        payload[f"joints_body25{suf}"] = np.asarray(res["joints_body25"], dtype=np.float32)
        payload[f"params_poses{suf}"] = np.asarray(res["params"]["poses"]).reshape(-1)
        payload[f"params_shapes{suf}"] = np.asarray(res["params"]["shapes"]).reshape(-1)
        payload[f"params_Rh{suf}"] = np.asarray(res["params"]["Rh"]).reshape(-1)
        payload[f"params_Th{suf}"] = np.asarray(res["params"]["Th"]).reshape(-1)
    np.savez_compressed(os.path.join(out_dir, f"frame_{k:06d}.npz"), **payload)


if __name__ == "__main__":
    main()
