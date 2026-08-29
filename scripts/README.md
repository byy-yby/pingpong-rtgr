# scripts — 命令行启动脚本

## 干什么

每个脚本是一个**独立的功能入口**，覆盖「枚举相机 → 预览 → 调参 → 同步验证 → 姿态识别」。
所有脚本顶部用 `sys.path.insert(0, "<project>/src")` 引导 import，因此**要在项目根目录外也能直接
`python scripts/xxx.py` 运行**，不用先 pip install 包。

| 脚本 | 功能 |
|---|---|
| `list_cameras.py` | 枚举在线相机，打印型号 / 序列号（配 `cameras.yaml` 前先跑这个）。 |
| `grab_preview.py` | 单相机实时预览（验证触发、调曝光/增益/伽马）。 |
| `grab_sync.py` | 四机同步采集验证（观察各机帧号 / 设备时间戳是否对齐）。 |
| `query_parameters.py` | 打印一台相机的所有图像参数 + 取值范围（研究「亮度/曝光补偿句柄」用）。 |
| `live_control.py` | **交互式四机控制**：2×2 平铺 + trackbar 调曝光/增益/伽马 + 键盘切检测 + 触发信号自检。 |
| `pose_preview.py` | **RTMPose-l 姿态识别实时预览**：自动开相机 → 检测 → 骨架叠加。 |
| `calibrate_intrinsics.py` | **相机内参标定 GUI**（PySide6, ChArUco）：四路预览 + 点击选中 + Enter 采集 + 计算并检验（RMS/主点/焦距判定）。 |
| `calibrate_extrinsics.py` | **多相机外参标定 GUI**（PySide6，ChArUco）：四路预览 + Enter 四机同时拍照 + 检测/位姿 + 求外参保存。 |

## 怎么用

```bash
conda activate tt

python scripts/list_cameras.py
python scripts/grab_preview.py --camera-id 0
python scripts/live_control.py                          # 外部触发（默认）
python scripts/live_control.py --trigger continuous     # 无信号发生器时自由采集
python scripts/pose_preview.py                          # 单相机自由采集 + 姿态
python scripts/pose_preview.py --all-cameras            # 四路平铺姿态
python scripts/grab_sync.py                             # 四机同步验证
python scripts/query_parameters.py --camera-id 0
python scripts/calibrate_intrinsics.py                  # 内参标定 GUI
python scripts/calibrate_extrinsics.py                  # 外参标定 GUI（ChArUco）
python scripts/calibrate_extrinsics.py --generate-board board.png   # 只生成打印板
```

`live_control.py` 键盘：`[p]` 姿态、`[b]` 球、`[t]` 球桌、`[q]`/`[ESC]` 退出；
外部触发模式下收不到帧会在画面顶部报「NO TRIGGER SIGNAL」并提示检查信号发生器 / Line0。

> 所有脚本运行前先关 MVS 客户端（避免设备被独占报 `0x80000203`）。

## 还没做完

- **缺 pipeline 编排脚本**：目前是「单功能脚本」，还没有一个把
  「采集 → 多路检测 → 2D→3D 重建 → 3D 可视化」串起来的整体入口（等重建 / 3D 可视化模块就绪）。
- **缺录制 / 回放脚本**：没有把四机同步帧序列存盘、再从盘上回放的脚本（`data/recordings/` 目前空置）。
- `live_control.py` 的球 / 球桌开关现在点了只会提示「接口已定义，算法待实现」，等 `vision` 注册后自动生效。
