#!/usr/bin/env python3
"""合成数据端到端验证官方 EasyMocap 管线。

流程：
  1. 官方 load_model 载入 SMPLlayer；
  2. 生成已知 GT 人：随机体型 betas + 合理站姿 poses + Rh 使人在 **Z-up 桌面系**
     站立（模板本身是 Y-up，Rh=Rx(+90°) 把 Y->Z），Th 放到桌面附近；
  3. 用项目真实外参（data/extrinsics/table_extrinsics.yaml）+ 合成内参，
     把 body25 3D 关节投影到 4 相机 -> body25 2D，反映射成 halpe26 2D（加高斯噪声）；
  4. 走 EasymocapReconstructor.reconstruct（官方 batch_triangulate + smpl_from_keypoints3d2d）；
  5. 对比拟合 body25 关节与 GT 的误差、重投影误差，检查网格是否在 Z-up 世界直立。

用法：
  cd 项目根目录 && python scripts/test_official_smpl.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

# 项目 src + EasyMocap 都塞进 sys.path（与 live_control 一致）
_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
for p in (os.path.join(_ROOT, "src"), "/home/yby/projects/EasyMocap"):
    if p not in sys.path:
        sys.path.insert(0, p)

from tabletennis.core.types import CameraExtrinsics, CameraIntrinsics, Pose2D
from tabletennis.reconstruction.easymocap import HALPE26_TO_BODY25, EasymocapReconstructor


# ----------------------------------------------------------------------
# 合成内参（真实外参配合；无真实内参文件，用合理默认）
# ----------------------------------------------------------------------
def synth_intrinsics(cid: int) -> CameraIntrinsics:
    # fx=1000（HFOV≈72°）：配合真实外参让站在桌旁地板上的人全高可见。
    # （真实内参文件未提交；测试目的是验证拟合正确性，相机几何只需自洽即可。）
    fx = fy = 1000.0
    w, h = 1440, 1080
    K = np.array([[fx, 0, w / 2], [0, fy, h / 2], [0, 0, 1]], dtype=np.float64)
    return CameraIntrinsics(width=w, height=h, K=K, dist=np.zeros(5, dtype=np.float64))


def load_table_extrinsics() -> dict:
    from tabletennis.calibration.extrinsics import load_extrinsics
    return load_extrinsics(os.path.join(_ROOT, "data", "extrinsics", "table_extrinsics.yaml"))


def load_rig() -> tuple:
    """读真实内外参（live_control 同款 load_camera_rig），无内参则回退合成。"""
    from tabletennis.reconstruction.triangulate import load_camera_rig
    intrinsics, extrinsics = load_camera_rig()
    if intrinsics:
        print("使用真实内参（data/calibration/cam_*.yaml）")
    else:
        print("无真实内参，回退合成内参 fx=1000")
        cids = sorted(extrinsics.keys())
        intrinsics = {cid: synth_intrinsics(cid) for cid in cids}
    return intrinsics, extrinsics


# ----------------------------------------------------------------------
# 合成 GT 人：body25 3D（桌面系） + 4 相机 2D
# ----------------------------------------------------------------------
def make_gt_person(body, rng: np.random.Generator, pos=(2.0, 2.0)) -> dict:
    """随机体型 + 站姿，Rh=Rx(+90°) 使人在 Z-up 桌面系站立。"""
    import torch

    n_shapes = 10
    # 随机但合理的 betas（±1.5 内，防止体型爆炸）
    betas = rng.uniform(-1.2, 1.2, size=n_shapes)
    betas[0] = rng.uniform(0.5, 1.5)   # 身高略正向
    # 站姿：肘/膝微屈，身体略前倾
    # SMPL native 关节序：0 pelvis, 1 Lhip, 2 Rhip, 3 spine1, 4 Lknee, 5 Rknee,
    #   6 spine2, 7 Lankle, 8 Rankle, 9 spine3, 10 Lfoot, 11 Rfoot, 12 neck,
    #   13 Lcollar, 14 Rcollar, 15 head, 16 Lshoulder, 17 Rshoulder, 18 Lelbow,
    #   19 Relbow, 20 Lwrist, 21 Rwrist, 22 Lhand, 23 Rhand；pose 切片 = [3j:3j+3]
    poses = np.zeros((1, 72), dtype=np.float32)
    poses[0, 3 * 3:4 * 3]  = [0.0, 0.0, 0.15]    # spine1 前倾
    poses[0, 6 * 3:7 * 3]  = [0.0, 0.0, 0.10]    # spine2 前倾
    poses[0, 9 * 3:10 * 3] = [0.0, 0.0, 0.08]    # spine3 前倾
    poses[0, 18 * 3:19 * 3] = [0.0, 0.5, 0.0]    # Lelbow 弯曲 ~30°
    poses[0, 19 * 3:20 * 3] = [0.0, -0.5, 0.0]   # Relbow 弯曲
    poses[0, 4 * 3:5 * 3]  = [0.15, 0.0, 0.0]    # Lknee 微屈
    poses[0, 5 * 3:6 * 3]  = [-0.15, 0.0, 0.0]   # Rknee 微屈
    # 全局：Y-up 模板 -> Z-up 世界；人站在球桌旁地板上（脚底 z=-0.76，桌高 0.76m）
    Rh = np.array([np.pi / 2, 0.0, 0.0], dtype=np.float32)
    feet_z = -0.76
    Th = np.array([pos[0], pos[1], 0.0], dtype=np.float32)
    params = {"poses": poses, "shapes": betas[None].astype(np.float32),
              "Rh": Rh[None], "Th": Th[None]}
    with torch.no_grad():
        verts = body(return_verts=True, return_tensor=False, **params)[0]  # (6890,3)
    # 校正：让脚底恰好落在 feet_z（取最小 z 平移）
    z_off = feet_z - verts[:, 2].min()
    params["Th"] = (Th + np.array([0, 0, z_off])).astype(np.float32)[None]
    with torch.no_grad():
        j25 = body(return_verts=False, return_tensor=False, **params)[0]
        verts = body(return_verts=True, return_tensor=False, **params)[0]
    return {"params": params, "j25": j25, "verts": verts}


def project_to_cameras(j25_world, intrinsics, extrinsics) -> dict:
    """body25 3D（桌面系）-> 各相机 body25 2D 像素。"""
    out = {}
    for cid, K in intrinsics.items():
        ext = extrinsics[cid]
        P = K.K @ np.hstack([ext.R, ext.t.reshape(3, 1)])
        Xh = np.concatenate([j25_world, np.ones((j25_world.shape[0], 1))], axis=1)
        cam = Xh @ P.T                      # (25, 3)
        p2d = cam[:, :2] / cam[:, 2:3]      # (25, 2)
        in_img = ((p2d[:, 0] >= 0) & (p2d[:, 0] < K.width) &
                  (p2d[:, 1] >= 0) & (p2d[:, 1] < K.height))
        out[cid] = (p2d, in_img)
    return out


def body25_to_halpe26(p2d, in_img, rng: np.random.Generator, sigma: float) -> np.ndarray:
    """body25 2D -> halpe26 2D（反映射 HALPE26_TO_BODY25，未覆盖的 halpe 关节置 0）。"""
    halpe26 = np.zeros((26, 3), dtype=np.float32)
    for halpe_idx, b25_idx in HALPE26_TO_BODY25:
        if not in_img[b25_idx]:
            continue
        x, y = p2d[b25_idx]
        x += rng.normal(0, sigma)
        y += rng.normal(0, sigma)
        halpe26[halpe_idx] = (x, y, 1.0)
    return halpe26


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def main() -> None:
    rng = np.random.default_rng(42)
    intrinsics, extrinsics = load_rig()
    cids = sorted(extrinsics.keys())
    use_real = any(cid in intrinsics for cid in cids)
    # 站位：真实内参视野较窄，取 (1.0,1.0) 覆盖最好；合成内参用 (2.0,2.0) 全可见
    pos = (1.0, 1.0) if use_real else (2.0, 2.0)
    print(f"相机：{cids}；测试站位 {pos}")

    recon = EasymocapReconstructor(verbose=False)
    assert recon.ready, recon.error
    body = recon._model

    for sigma in (0.5, 1.5):
        gt = make_gt_person(body, rng, pos=pos)
        j25_gt = gt["j25"]
        cam2d = project_to_cameras(j25_gt, intrinsics, extrinsics)

        # 组装 halpe26 Pose2D（供 reconstruct 用）
        best = {}
        for cid in cids:
            p2d, in_img = cam2d[cid]
            halpe26 = body25_to_halpe26(p2d, in_img, rng, sigma)
            best[cid] = Pose2D(camera_id=cid, keypoints=halpe26, score=1.0,
                               skeleton="halpe26")

        result = recon.reconstruct(best, intrinsics, extrinsics)
        assert result is not None, "重建返回 None"
        j25_fit = result["joints_body25"]

        # 逐关节误差（只统计有效 GT 关节）
        err = np.linalg.norm(j25_fit - j25_gt, axis=1)
        valid = np.linalg.norm(j25_gt, axis=1) > 1e-6
        # 可见性：某关节被 ≥2 台相机看到才算「真看到」（三角化需要 ≥2 视角）
        seen = np.zeros(25, dtype=bool)
        for cid in cids:
            _, in_img = cam2d[cid]
            seen |= in_img
        n_seen = np.zeros(25, dtype=int)
        for cid in cids:
            _, in_img = cam2d[cid]
            n_seen += in_img
        seen_ge2 = n_seen >= 2
        names = ["Nose","Neck","RSh","RElb","RWr","LSh","LElb","LWr","MidHip","RHip",
                 "RKnee","RAnk","LHip","LKnee","LAnk","REye","LEye","REar","LEar",
                 "LBigToe","LSmallToe","LHeel","RBigToe","RSmallToe","RHeel"]
        print(f"\n===== sigma={sigma}px =====")
        print(f"三角化有效 3D 关节: {int(seen_ge2.sum())}/25（其余由 reg_poses_zero 先验补全）")
        for mask, tag in ((seen_ge2, "真看到(≥2视角)"), (~seen_ge2 & valid, "先验补全")):
            idx = np.where(mask)[0]
            if len(idx) == 0:
                continue
            e = err[idx]
            print(f"[{tag}] 误差(mm): 均值 {e.mean()*1e3:6.1f}  中位 {np.median(e)*1e3:6.1f}  最大 {e.max()*1e3:6.1f}")
        bad = np.argsort(err[valid])[::-1][:3]
        idxs = np.where(valid)[0]
        for i in bad:
            tag = "seen" if seen_ge2[idxs[i]] else "prior"
            print(f"  最差关节 {names[idxs[i]]:8s}({tag:5s}) {err[idxs[i]]*1e3:6.1f}mm")

        # 直立性检查：脚底在地板附近（z≈-0.76），身高 ≈1.7m（Z 向上）
        zmin, zmax = result["vertices"][:, 2].min(), result["vertices"][:, 2].max()
        height = zmax - zmin
        print(f"网格 z 范围: {zmin:.2f} ~ {zmax:.2f}（脚底≈-0.76 地板，身高 {height:.2f}m，Z 向上）")
        assert abs(zmin - (-0.76)) < 0.2, f"网格未落在桌面高度 (zmin={zmin})"
        assert height > 1.4, "网格身高异常（未直立 / 塌陷）！"
        # 重投影误差（拟合关节 -> 相机）
        rep_errs = []
        for cid in cids:
            p2d_gt, _ = cam2d[cid]
            P = intrinsics[cid].K @ np.hstack([extrinsics[cid].R,
                                               extrinsics[cid].t.reshape(3, 1)])
            Xh = np.concatenate([j25_fit, np.ones((25, 1))], axis=1)
            cam = Xh @ P.T
            p2d_fit = cam[:, :2] / cam[:, 2:3]
            rep_errs.append(np.linalg.norm(p2d_fit - p2d_gt, axis=1))
        rep = np.concatenate(rep_errs)
        print(f"重投影误差(px): 均值 {rep.mean():.2f}  中位 {np.median(rep):.2f}")

    print("\n[PASS] 官方管线端到端验证通过")


if __name__ == "__main__":
    main()
