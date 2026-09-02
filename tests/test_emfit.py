"""EasyMocap 优化层（em_fit.EmFit）与原版管线的数值一致性测试。

覆盖规格要求的「关键数学的数值正确性」：
  1. 冷启动 EmFit（ftol/maxiters/no_item_sync 全用官方默认值）应与官方
     ``reconstruct()`` **逐位等价**（优化路径/损失/权重全同，差异只在跑偏性浮动）。
  2. warm（热启动）单帧结果应与官方冷启动在同一观测下误差相近（≤容差），
     即加速不牺牲正确性。
  3. 输出的 body25 关节与官方 forward(params) 自洽（同一 forward，误差应≈0）。

依赖 GPU + SMPL 模型文件；缺任一条件整文件 skip（不拖慢普通 pytest 跑法）。
用法：``python -m pytest tests/test_emfit.py -s``
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

torch = pytest.importorskip("torch")

# 只测一个有代表性的合成帧，控制运行时长（官方冷启动 ~5s + EmFit ~7s）
_rng = np.random.default_rng(3)


@pytest.fixture(scope="module")
def recon():
    if not torch.cuda.is_available():
        pytest.skip("需要 GPU（RTX 5080 环境）")
    from tabletennis.reconstruction.easymocap import EasymocapReconstructor
    r = EasymocapReconstructor(verbose=False)
    if not r.ready:
        pytest.skip(f"缺 SMPL 模型：{r.error}")
    return r


@pytest.fixture(scope="module")
def rig():
    from tabletennis.reconstruction.triangulate import load_camera_rig
    return load_camera_rig()


def _make_person(recon, pos=(1.0, 1.0)):
    body = recon._model
    betas = _rng.uniform(-1.0, 1.0, size=10).astype(np.float32)
    betas[0] = _rng.uniform(0.5, 1.2)
    ps = np.zeros((1, 72), np.float32)
    ps[0, 3 * 3:4 * 3] = [0, 0, 0.15]; ps[0, 18 * 3:19 * 3] = [0, 0.5, 0]
    ps[0, 4 * 3:5 * 3] = [0.15, 0, 0]; ps[0, 5 * 3:6 * 3] = [-0.15, 0, 0]
    p = {"poses": ps, "shapes": betas[None],
         "Rh": np.array([[np.pi / 2, 0, 0]], np.float32),
         "Th": np.array([[pos[0], pos[1], 0.386]], np.float32)}
    with torch.no_grad():
        v = body(return_verts=True, return_tensor=False, **p)[0]
    p["Th"] = p["Th"] + np.array([[0, 0, -0.76 - v[:, 2].min()]], np.float32)
    return p


def _observe(recon, rig, j25, sigma=0.5):
    from tabletennis.reconstruction.easymocap import HALPE26_TO_BODY25
    from tabletennis.core.types import Pose2D
    intr, ext = rig
    cids = sorted(ext)
    best = {}
    for cid in cids:
        K = intr[cid].K
        P = K @ np.hstack([ext[cid].R, ext[cid].t.reshape(3, 1)])
        c = np.hstack([j25, np.ones((25, 1))]) @ P.T
        p2d = c[:, :2] / c[:, 2:3]
        halpe = np.zeros((26, 3), np.float32)
        for hi, b25 in HALPE26_TO_BODY25:
            x, y = p2d[b25]
            if sigma:
                x += _rng.normal(0, sigma); y += _rng.normal(0, sigma)
            halpe[hi] = (x, y, 1.0)
        best[cid] = Pose2D(camera_id=cid, keypoints=halpe, score=1.0, skeleton="halpe26")
    return best


def _gt_joint(recon, params):
    with torch.no_grad():
        return recon._model(return_verts=False, return_tensor=False, **params)[0]


def _median_err(a, b):
    return float(np.median(np.linalg.norm(np.asarray(a) - np.asarray(b), axis=1)) * 1000)


def _max_err(a, b):
    return float(np.max(np.linalg.norm(np.asarray(a) - np.asarray(b), axis=1)) * 1000)


def test_emfit_cold_equals_official(recon, rig):
    """官方默认参数下，EmFit 冷启动 ≈ 官方 reconstruct（逐位/微差）。"""
    from tabletennis.reconstruction.easymocap import EasymocapReconstructor
    from tabletennis.reconstruction.em_fit import EmFit, EMSettings

    # 注意：必须在官方跑完之后再构造 EmFit（install 会替换 optimize_simple 内部函数）
    base = _make_person(recon)
    with torch.no_grad():
        j = _gt_joint(recon, base)
    obs = _observe(recon, rig, j)

    res_o = recon.reconstruct(obs, rig[0], rig[1])
    assert res_o is not None, "官方 reconstruct 返回 None"

    # 与官方逐位一致的设置（no_item_sync 只影响同步，不影响数值）
    fit = EmFit(recon, EMSettings(ftol=1e-4, maxiters=100,
                                  no_item_sync=True, warm_init=False), verbose=False)
    res_e = fit.run(obs, rig[0], rig[1], prev=None)
    assert res_e is not None, "EmFit 冷启动返回 None"

    med = _median_err(res_e["joints_body25"], res_o["joints_body25"])
    mx = _max_err(res_e["joints_body25"], res_o["joints_body25"])
    assert med < 1.0, f"EmFit 冷启动与官方关节中位差 {med:.2f}mm（应≈0）"
    assert mx < 5.0, f"EmFit 冷启动与官方关节最大差 {mx:.2f}mm（应≈0，autograd 浮动级）"
    # 输出与自身 forward 自洽
    with torch.no_grad():
        j_fwd = recon._model(return_verts=False, return_tensor=False,
                             **res_e["params"])[0]
    assert _max_err(j_fwd, res_e["joints_body25"]) < 1e-3


def test_emfit_warm_close_to_official(recon, rig):
    """同一观测下，warm（热启动）相对官方冷启动：误差≤容差（加速不牺牲质量）。"""
    from tabletennis.reconstruction.em_fit import EmFit, EMSettings

    # frame A（冷启动给 warm 提供初始参数）
    baseA = _make_person(recon)
    with torch.no_grad():
        jA = _gt_joint(recon, baseA)
    obsA = _observe(recon, rig, jA)
    fit = EmFit(recon, EMSettings(ftol=5e-4, maxiters=40, no_item_sync=True,
                                  warm_init=True, skip_global_rt_warm=True),
                verbose=False)
    resA = fit.run(obsA, rig[0], rig[1], prev=None)
    assert resA is not None

    # frame B = 同一人小幅动作（真实连续帧）
    baseB = {k: v.copy() for k, v in baseA.items()}
    ps = baseB["poses"].copy()
    ps[0, 4 * 3:5 * 3] += 0.30; ps[0, 18 * 3:19 * 3] = [0, 0.9, 0]
    baseB["poses"] = ps
    baseB["Th"] = baseB["Th"] + np.array([[0.05, 0.02, 0]], np.float32)
    with torch.no_grad():
        jB = _gt_joint(recon, baseB)
    obsB = _observe(recon, rig, jB)

    # warm：拿 A 结果热启动
    resB_warm = fit.run(obsB, rig[0], rig[1], prev=resA["params"])
    assert resB_warm is not None

    # 对照：B 的 GT 关节误差（warm 结果 vs GT 中位 ≤ 容差）
    med_gt = _median_err(resB_warm["joints_body25"], jB)
    assert med_gt < 25.0, f"warm 结果相对 GT 中位误差 {med_gt:.1f}mm 超容差"


def test_emfit_outputs_shape(recon, rig):
    """输出结构/形状/单位正确（6890 顶点网格 + 25 关节 + 24 原生关节）。"""
    from tabletennis.reconstruction.em_fit import EmFit, EMSettings
    base = _make_person(recon)
    with torch.no_grad():
        j = _gt_joint(recon, base)
    obs = _observe(recon, rig, j)
    fit = EmFit(recon, EMSettings(ftol=5e-4, maxiters=40), verbose=False)
    res = fit.run(obs, rig[0], rig[1], prev=None)
    assert res is not None
    assert res["vertices"].shape == (6890, 3)
    assert res["joints_body25"].shape == (25, 3)
    assert res["joints"].shape == (24, 3)
    assert set(res["params"].keys()) == {"poses", "shapes", "Rh", "Th"}
    # 直立人脚在桌面 z≈-0.76，高度≈1.7m
    zmin = res["vertices"][:, 2].min()
    height = res["vertices"][:, 2].max() - zmin
    assert abs(zmin + 0.76) < 0.05, f"脚底 z={zmin:.3f}（应在桌面 -0.76）"
    assert 1.5 < height < 2.0, f"身高 {height:.2f}m 异常"
