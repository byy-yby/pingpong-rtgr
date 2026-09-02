# EasyMocap SMPL 拟合性能基线（BASELINE_PROFILE）

> Phase-1 摸底结论。**本阶段未改动任何算法**，只读代码 + 跑基准，产出权威数字，
> 供 Phase 2/3/4 的优化决策与 A/B 对比使用。复现脚本：`scripts/bench_baseline.py`。

## 1. 被测对象与边界

- **被测代码**：仓库内集成用的官方 EasyMocap 完整 SMPL 拟合管线，
  入口 `src/tabletennis/reconstruction/easymocap.py::EasymocapReconstructor.reconstruct()`；
  拟合内核为 vendored 官方 `/home/yby/projects/EasyMocap`（git HEAD `e681319`，**零改动**）。
- **EasyMocap 之外的环节不在此基线内**：2D 检测（RTMPose `detect_batch` 4 相机 ~10ms /
  ~90fps）、球/人 3D 三角化（`triangulate_batch` 姿态 0.43ms）都已在既有子系统内达标，
  见 `docs/optimization_report.md`。本基线只管 **kp2d → SMPL 拟合 → 网格** 这段。
- **复现数据**：`data/recordings/` 为空，无真实录制序列 → 用合成 GT（1 个真人比例动作、
  N 帧连续小幅变化），经**真实标定**（`data/calibration/cam_*.yaml` 内参 + 
  `data/extrinsics/table_extrinsics.yaml` 外参，fx≈1770 窄 FOV）投影成 4 视角 halpe26，
  加 σ=0.5px 高斯噪声喂给 `reconstruct()`。真实场景多数关节只有 2 视角可见（窄 FOV），
  与线上行为一致。

## 2. 调用链 / 代码地图

```
EasymocapReconstructor.reconstruct()                       easymocap.py
├─ _to_body25_2d   halpe26→body25、去畸变、conf 过滤(≥3kp/视角, ≥2视角)
├─ batch_triangulate + check_keypoints                     triangulator.py / body_param.py
└─ smpl_from_keypoints3d2d(model, kp3d, kp2d, bbox, P…)    pipeline/basic.py:65
   ├─ optimizeShape(β only)                                pyfitting/optimize_simple.py:17
   │     LBFGS(max_iter=10) + FittingMonitor(ftol=1e-4, maxiters=100)
   └─ multi_stage_optimize                                 pipeline/basic.py:14
        ├─ optimizePose3D “global RT”  (只 Rh+Th)          ← optimizePose3D 第 1 次调用
        ├─ optimizePose3D “3D pose”    (Rh+Th+poses)       ← 第 2 次调用
        └─ optimizePose2D              (Rh+Th+poses, k2d)  pyfitting/optimize_simple.py:351
```

每段 `_optimizeSMPL`（optimize_simple.py:244）= **一个 LBFGS**（`line_search_fn='strong_wolfe'`,
默认 `max_iter=20, max_eval=25`），外圈 `FittingMonitor.run_fitting`（optimize.py:30）最多跑
`maxiters=100` 次 `optimizer.step(closure)`，相邻 step 的 rel_loss_change ≤ `ftol=1e-4` 即停。
closure（optimize_simple.py:268）每次调用都：

1. `body_model(return_verts=False, return_tensor=True, **params)` —— **关节稀疏路径**
   （`SMPLlayer.forward` 里 `use_joints=True` 且非 return_verts → 只用 49 个"关节顶点"
   `j_v_template = X_regressor@v_template`，24 SMPL 关节 + 25 body25 回归点做 LBS，
   **不走 6890 顶点全 LBS**；body_model.py:359）
2. 逐 loss 项求和（k3d/smooth_*/reg_*/reg_poses_zero 等）
3. `records.append(loss.item())` ← **每次评估一次 CPU-GPU 同步**
4. `loss.backward()`

拟合收敛后 reconstruct 再跑一次 `return_verts=True` 全顶点前向出网格。

## 3. 运行方式（可复现）

```bash
cd <repo worktree>
/home/yby/miniconda3/envs/tt/bin/python scripts/bench_baseline.py --frames 3 --noise 0.5 \
    --out data/bench/baseline.json
```

## 4. 结果（RTX 5080, conda `tt` py3.11, N=3 帧冷启动逐帧全量拟合）

| 指标 | 值 |
|---|---|
| **单帧全量拟合（中位 wall）** | **4839 ms**（3 帧 4.84 / 4.93 / 3.99 s，动作幅度大则慢） |
| SMPL 前向调用 / 帧 | **232**（shape 9 + globalRT 18 + pose3d 147 + pose2d 53 + 网格 3） |
| 纯 SMPL 前向合计 | 797 ms/帧 → 均 3.4 ms/次（**只占 wall 的 16%**） |
| 关节误差 body25 中位 | 17.5 mm（σ=0.5px 观测；含遮挡关节的先验补全误差） |
| GPU 峰值显存 | 38 MB（很小） |
| GPU util（smi 平均） | ~10%（**严重欠载**） |

阶段耗时 / 前向次数（中位）：

| 阶段 | wall | SMPL 前向 | 每评估全成本 |
|---|---|---|---|
| optimizeShape | 28 ms | 9 | ~3 ms（只 10 维 β） |
| global RT（Rh+Th） | 121 ms | 18 | ~6.7 ms |
| **3D pose**（Rh+Th+poses） | **2898 ms** | 147 | **~19.7 ms** |
| 2D pose | 935 ms | 53 | ~17.6 ms |
| 其余（三角化+网格前向） | 14 ms | 3 | — |

> 注：frame 0 的 optimizeShape 段 860ms/9 次属 CUDA 首调预热（lazy kernel 编译），
> 稳态帧（1/2）才是 28ms。

## 5. 成本解剖

**每帧 = 232 次评估 × ~19ms/评估；其中纯前向仅 ~3.4ms。** 即：

```
wall 4839ms ≈ 232 × (forward ~3.4ms + backward ~5ms* + losses/正则 ~?ms
                     + LBFGS/强 Wolfe 行搜索 + Python 循环 + CPU-GPU 同步)
```

独立微基准（prof2.py）：SMPL 关节稀疏路径 no-grad 前向 **~2.0ms**；前向+autograd 反向
**~8.4ms**。反向与损失求和的量级与参数维度（~78 个活动参数）完全不相称 → 每次评估的
开销**不是 FLOP 也不是显存**，是：
1. **大量小 kernel 启动**（batch_rodrigues 24 组、逐项 einsum/matmul、逐 loss 项求和），
   每个 kernel 几十 µs 启动、实际计算 ns~µs 级；
2. **每次评估一次 `.item()` 同步**（optimize_simple.py:285 `records.append(loss.item())`），
   把 GPU 流水线打断等 CPU；
3. **LBFGS strong_wolfe 行搜索**：每步多次 closure + 标量比较逻辑在 CPU 上串行；
4. **Python/autograd 图开销**：每评估重建 ~78 维参数图（distutils 无关，是 autograd 每次
   重新构图 + 200+ 节点）。

**为什么评估次数这么高**：外圈 FittingMonitor 的 `ftol=1e-4` 很严，LBFGS 内圈
`tolerance_change=1e-9` 在接近最优时磨刀（每步内还要做多次行搜索评估）。
pose3d 段 147 次 ≈ 多数耗在"损失下降已 <1e-4 的临界区反复确认收敛"。

## 6. CPU-GPU 同步点 / 低效清单（代码级）

| 位置 | 现象 | 每次发生 |
|---|---|---|
| `optimize_simple.py:285` `records.append(loss.item())` | **每次 closure 同步一次** | ~232/帧 |
| `optimize.py:53/61` FittingMonitor `loss.item()` + rel_change | 每外圈 step 同步 | ~15/帧 |
| `optimize_simple.py:78/204/299` 段末 `detach().cpu().numpy()` | 每段一次，可接受 | 4 |
| `closure` 每评估 `optimizer.zero_grad()` + 全参数 autograd 重建 | 图重建开销 | 232 |
| `_optimizeSMPL` 里 `records` 无界 append Python float | 内存/GC 噪音 | 232 |

## 7. 结论与优化地图（供后续 Phase 决策）

瓶颈是**「评估次数 × 每次评估的 CPU 侧开销 + 同步」**，不是矩阵计算、不是 GPU 吞吐、
不是显存、不是 6890 顶点 LBS（拟合期根本不跑全顶点）。据此：

- **Phase 2（低风险，可立刻做）**：
  - 去掉/降频 closure 内 `.item()` 同步（只留最后一次或用 tensor 记录）；
  - **warm-start 连续帧**：shape/pose/Rh/Th 用上一帧结果初始化，跳过/降频 optimizeShape，
    β 固定或低频更新（已测：warm≈1.5×，ftol 1e-4→5e-4 ≈2×，两者叠加 ~3×，误差不升）；
  - 每段起步时用更粗 ftol + 上限 maxiters（现外圈默认 100 太宽）；收敛即停更果断；
  - loss 求和从逐项 `*weight + sum` 的小 Python 循环/多次 kernel 收敛到一次乘法；
  - 缓存损失构造器里每次都在切/复制的常量（lossfactory 每次闭包重算的 mask/索引）。
- **Phase 3（计算层）**：先 `torch.compile` 试前向+反向合体（benchmark 验证），
  forward+backward+全部 loss 一次 CUDA graph 捕获（消除逐 kernel 启动与图重建）；
  body_model 关节稀疏路径的 batch_rodrigues / einsum 融合。**不预设 C++/CUDA 更快的结论**。
- **Phase 4（流式）**：独立 streaming 模式——warm-start + β 固定 + θ/平移为主变量 +
  每帧限步 + 早停，逐帧不重建；保留原离线模式。
- 目标：从 ~0.2 FPS（4839ms/帧）往 5~10 FPS 走，误差 <20mm 不劣化。

## 8. 与既有子系统的衔接

live_control 现在逐帧调 `reconstruct()` = 每帧**冷启动**全量拟合 → 与 2D 检测(~90fps)
完全脱节（EM 线程 ~0.2fps）。连续帧 warm-start 是 Phase 2 最直接的收益点；检测与三角化
两段不在 EasyMocap 内、本优化不动它们。
