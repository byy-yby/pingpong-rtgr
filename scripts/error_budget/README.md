# error_budget — 重投影误差来源分析工具

结论与全部实测数字见 `docs/error_budget_report.md`。这里只记怎么跑。

全部脚本 `sys.path.insert(0, 'src')`，**必须从仓库根目录**运行，用 conda 环境 `tt`。

```bash
PY=/home/yby/miniconda3/envs/tt/bin/python
S=data/video/20260908_161147
T=scripts/error_budget
```

| 脚本 | 作用 | 用法 |
|---|---|---|
| `errbudget.py` | 主分析：逐帧/逐人/逐相机/逐关节残差、地板 `r_in`、预测残差、Sampson 极线误差、时间分布、置信度分箱 | `$PY $T/errbudget.py $S out.npz` |
| `shares.py` | 由 `errbudget.py` 的 npz 算**来源占比**（2D 白噪声 / 2D 系统偏差 / 标定 / SMPL） | `$PY $T/shares.py out.npz` |
| `aruco_gt.py` | **独立真值**：桌面 4 角 ArUco 刚体的尺度、跨视角残差、逐相机 LOO、残差-像面半径依赖、静止散布、共面性 | `$PY $T/aruco_gt.py $S 10` |
| `aruco_ba.py` | 标记上的**交替 BA**（三角化 ↔ 逐相机 PnP）重估外参，可选存修正外参 | `$PY $T/aruco_ba.py $S 10 5 fix.npz` |
| `floor_fixed.py` | 把修正外参代回去重算**姿态**地板，验证外参修正是否真有收益 | `$PY $T/floor_fixed.py $S fix.npz` |
| `trend.py` | 误差随时间（10 段）/置信度/深度代理 `mpp` 的变化 | `$PY $T/trend.py out.npz` |

## 口径（改前先读）

- **像素不可跨视角比**：近端视角 ~3.6mm/px、远端 ~5.7-6.2mm/px，同一 px 值物理误差差 1.7 倍。
- **`r_in`（地板）只统计「有效关节」**：被下半身门掩码的关节、裁边视角都不进分母 →
  它是**自洽性**不是真值；别拿「掩码后重投影 18.7→13.3px」当提升证据。
- **中位数不可加、RMS 被离群点支配** → `shares.py` 用「逐样本方差占比的中位数」，
  且过滤 `r_fit ≤ 1px` 的退化样本（否则比值爆表，曾得到 626%/147% 的假占比）。
- ArUco 真值基准：黑方块**实测** 0.1751m（配置写 0.18m，是打印缩放）；
  中心间距真值 = 桌面尺寸 − 2×(边长/2 + 白边 0.02)。
- `aruco_ba.py` 修的是**外参**（内参固定）；只对桌面平面上的点做过拟合验证，
  故必须跑 `floor_fixed.py` 确认对**离面**的姿态点也有效（实测地板 −12%）。
