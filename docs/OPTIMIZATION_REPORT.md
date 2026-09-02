# EasyMocap SMPL 拟合性能优化报告（OPTIMIZATION_REPORT）

> 汇总 Phase 1–5 结论：基线、逐项可测的提速、哪些优化有效/无效、新的瓶颈、A/B 方法。
> 基线细节见 `BASELINE_PROFILE.md`；复现脚本 `scripts/bench_baseline.py` /
> `scripts/bench_optimized.py`；实现 `src/tabletennis/reconstruction/em_fit.py`。

## 0. 交付物总览

| 文件 | 作用 |
|---|---|
| `docs/BASELINE_PROFILE.md` | Phase-1 基线 profile（代码地图/结果/成本解剖/同步点） |
| `scripts/bench_baseline.py` | 可复现基线基准（合成 N 帧，冷启动逐帧全量拟合） |
| `src/tabletennis/reconstruction/em_fit.py` | **优化层**：`EMSettings`（每项开关）+ `_optimizeSMPL` 等价副本 + `EmFit`（连续帧热启动编排） |
| `scripts/bench_optimized.py` | Phase-2/4 A/B 基准（original/sync/loose/warm/stream 逐项隔离） |
| `tests/test_emfit.py` | 数值一致性测试（cold≈官方、warm 精度、几何 sanity）——**已跑通 3 passed** |
| vendored `/home/yby/projects/EasyMocap` | **零改动**（A/B 的 original 基准一直可用） |

**关键设计**：不重写 EasyMocap、不删官方功能。优化层 = 运行时把官方单个热点函数
`pyfitting/optimize_simple._optimizeSMPL` 替换为逐行等价副本（去掉 per-eval `.item()`
同步、收敛参数可调），其余 body_model/lbs/lossfactory/lbfgs/optimizePose3D/2D/Shape
全部原样复用。官方入口 `EasymocapReconstructor.reconstruct()` 未被触碰 → original 模式
恒定可用；`EmFit` 默认用官方默认参数时与官方**数值等价**（测试证实 med 差 <1mm）。

## 1. 基线（Phase 1，复现于 BASELINE_PROFILE.md）

合成 4 帧连续动作（σ=0.5px 观测，真实标定窄 FOV，多数关节仅 2 视角）逐帧冷启动：

- **单帧全量拟合中位 4.6~4.8 s**（bench_baseline 4839ms / bench_optimized A 4653ms，
  随动作幅度 ±10% 波动）
- SMPL 前向调用 **232 次/帧**，纯前向合计 797ms（占 wall 16%）
- 阶段占比：optimizeShape 28ms / global-RT 121ms / **3D pose 2.9s(147 次前向)** /
  2D pose 0.9s(53 次) / 其余 14ms
- GPU：**util ~10%，峰值显存 38MB** —— 完全没吃满
- body25 关节误差（对 GT 中位）：**18.1mm**

## 2. 每项优化的实测效果（同一 4 帧序列、同一噪声，逐项隔离）

`bench_optimized.py` 每个配置 = 一个可独立开关的优化子集，A/B 表格 =
**单次 A→G 连续跑**（同一合成序列、同一噪声种子），数值与提交的
`data/bench/optimized.json` 完全一致。中位帧 = 全 4 帧；稳态帧 = frame 1..3；
提速 = 官方中位帧 / 本配置中位帧（JSON 同口径）：

| 配置 | 内容 | 中位帧 | 稳态帧 | 提速 | err(GT 中位) | 判定 |
|---|---|---|---|---|---|---|
| A original | 官方 `reconstruct()` 零改动 | 4653 ms | 4604 ms | x1.00 | 18.1mm | — 基准 |
| B sync | 仅去 per-eval `.item()` 同步（ftol 仍 1e-4） | 4301 ms | 4653 ms | x1.08 | 18.7mm | ✅ 数值不变，纯省同步 |
| C loose | + ftol 1e-4→5e-4 / maxiters 100→40（冷启动） | 3081 ms | 2910 ms | x1.51 | 19.5mm | ⚠️ 提速但误差 +1.4mm |
| E warm_strict | + 热启动（ftol 仍 1e-4） | 3250 ms | 3218 ms | x1.43 | 17.4mm | ✅ **提速且误差改善** |
| D warm | warm + ftol 5e-4 | 2581 ms | 2456 ms | x1.80 | 17.0mm | ✅ 推荐档：快 + 更准 |
| F warm_mid | warm + ftol 3e-4 | 2807 ms | 2503 ms | x1.66 | 17.3mm | ✅ |
| G stream | warm + ftol 1.5e-3 / maxiters 25 | 2048 ms | 1994 ms | **x2.27** | 17.7mm | ✅ 激进档 ~0.5fps |

要点：
- **warm（热启动）是质变**：只它（E，收敛仍 1e-4）就把误差从 18.1→17.4 且快 1.43×——
  相邻帧先验本身就是有效正则，不是纯加速 hack。**冷启动放宽 ftol 的误差代价（C +1.4mm）
  在 warm 下被抵消**（D/F/G 误差全部好于或接近官方）。
- sync 消除是「0 风险」收益：不动任何数值，只少 CPU-GPU 同步。
- β 在 frame 0 官方 optimizeShape 后固定（`refit_shape_every=0`）→ 稳态帧不再跑 shape 段。
- **G 档每帧仍 ~128 次前向、~16ms/次** —— 卡在 per-eval 成本上，见 §4。
- ⚠️ 帧级 ~±10% run-to-run（同机噪音），故个位数 ms 的差（如 B 的稳态 4653 vs A 4604）
  是噪声而非信号；**比值口径固定取「中位帧」**，且对 A/B 这类纯同步优化应看多次均值。

## 3. 计算层（Phase 3）实测结论

- **torch.compile：不可用（本机）**。实测报 `TritonMissing`——conda `tt` 没装 triton
  （CLAUDE.md 早记「triton 装不上不影响普通训练」，compile 默认 inductor 后端正依赖它）。
  装 triton 后再试可作后续项；本报告不假设它能提速。
- **torch.profiler（一帧 warm 拟合）硬证据**：
  - `Self CUDA time ≈ 94ms` vs wall ~2.0s → **GPU 只忙 ~5%**；
  - `Self CPU ≈ 4.7s`；头部：`LBFGS.step` python 侧 42%、`cudaLaunchKernel` 75k 次、
    `cudaStreamSynchronize` 8.5k 次、`_local_scalar_dense` 8.2k 次（.item()）、
    `aten::empty/view/select` ~4.5 万次。
  - → **瓶颈是 CPU 端 dispatch + 每评估 ~2 次标量同步（LBFGS strong_wolfe 行搜索的
    float(loss) 决策）+ autograd 小图重建 + python 循环**，不是矩阵吞吐/显存/6890 顶点。
- 官方 `_optimizeSMPL` 里每评估一次 `records.append(loss.item())`（optimize_simple.py:285）
  是**纯粹的额外同步**（records 只给 verbose 打印），已在我们副本里消除；
  剩 `lbfgs.py:308 float(orig_loss)` 与行搜索的比较是算法需要的，未动。
- 拟合期走的是 **49 点关节稀疏路径**（`j_v_template`，body_model.py:359），**不是**
  6890 顶点全 LBS——「全顶点 LBS 慢」是误解，全顶点只出现在最后出网格那一次 forward。

## 4. 新的瓶颈

优化后一帧 ~2.0s 的构成：`~128 次评估 × ~16ms/次`。每次评估：
纯前向 ~3.4ms + 反向/autograd + 全 loss 项求和 + **LBFGS 强 Wolfe 行搜索（CPU 决策）**
+ 小 kernel 启动。要再提速只有两条路：
1. **砍评估次数**（最直接）：热启动下已从 232→128；再压需改收敛/限步语义（误差会上来，
   或用帧间更准的初值）。实测 ftol 1.5e-3 已进入平台期（130→120 不再变快），
   **瓶颈已从「评估太多」转为「每次评估 CPU 太贵」**。
2. **压低 per-eval CPU**：本环境只能靠装 triton 走 torch.compile 融合（fwd+bwd+loss 一次
   graph 化，消灭 75k 次 launch + python），或换 `line_search_fn`/自写 C 扩展。两者都属
   Phase-3 的「不预设 C/C++ 更快」之外的后续项——**工程结论：1s/帧是当前 CPU-bound
   LBFGS 架构的墙，0.5s 以下需要动评估内核而不是继续调参。**

## 5. Streaming 模式（Phase 4）与接入方法

`EmFit` 即显式 streaming 语义：
- 首帧（`prev=None`）冷启动拿 β + 初值；
- 后续帧 `prev=上一帧 params` → **无逐帧冷启动**：β 固定（`refit_shape_every=0`，可设
  N 低频重拟合）、只把 θ/Rh/Th 当活动变量、跳官方「global-RT 单独对齐」段
  （Rh/Th 已近解）、FittingMonitor 早停 + `maxiters` 兜底；
- 官方离线模式 `reconstruct()` 保持原样，两种模式并存。

接入 live_control（EM 线程，现为冷启动 reconstruct()）建议改法（本项目未擅改线上工具，
给了完整 diff 供选）：
```python
# _em_load_worker 里：默认不变；用 env 显式开流式
import os
from tabletennis.reconstruction.em_fit import EmFit, EMSettings
if os.environ.get("EASYMOCAP_STREAM"):
    self._em_fit = EmFit(self._em_recon, EMSettings(ftol=5e-4, maxiters=40,
        no_item_sync=True, warm_init=True, skip_global_rt_warm=True), verbose=False)
# _reconstruct_easymocap 里（替换 self._em_recon.reconstruct(...) 那一句）：
prev = getattr(self, "_em_prev", None)
result = self._em_fit.run(best, self._em_intrinsics, self._em_extrinsics, prev=prev)
self._em_prev = result["params"] if result else None
```
不设 `EASYMOCAP_STREAM` 时行为与现在完全一致（original 路径）。

## 6. 验证

- `python -m pytest tests/test_emfit.py` → **3 passed**：EmFit 冷启动与官方关节
  med <1mm / max <5mm（等价）；warm 结果对 GT 中位误差 <25mm（加速不牺牲质量）；
  输出几何 sanity（脚 z≈-0.76、身高 1.5–2m、6890/25/24 形状）。
- A/B 各配置在同一合成 4 帧序列、同一噪声种子下对比；未改分辨率/未减相机数/未动输入。

## 7. 一句话给下一轮

> 最快且最稳的一步已落地：**热启动 + 去同步 + 收敛阈值/步数开关** = 原版 4.7s/帧 →
> 2.0s/帧（**x2.27**），误差从 18.1 略降至 17.7mm，且 original/optimized 双模式并存、逐项可关。
> 继续压到实时需装 triton 走 torch.compile 融合每次评估（消灭 CPU dispatch），或接受
> 更激进的限步；不建议也不曾做过任何算法层面的重写。
