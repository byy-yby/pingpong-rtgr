# 乒乓球场景重建系统

用 **4 台海康 MV-CS016-10UM**（USB3 黑白工业相机）做多视角实时重建，目标产物：

- **球轨迹重建**：三角化出乒乓球的 3D 轨迹
- **人体动作重建**：多视角 2D 关键点 → 3D 骨架
- **桌面位置还原**：拟合出球台平面的位姿

相机由**信号发生器统一外部触发**实现四机同步曝光。

## 硬件

| 项 | 型号 / 说明 |
|---|---|
| 相机 ×4 | 海康机器人 MV-CS016-10UM（USB3，**黑白**，160 万像素面阵） |
| 触发 | 信号发生器，输出接每台相机的 **Line0**（6-pin I/O） |
| SDK | 海康 MVS V4.8.0（已装 `/opt/MVS`，含 Python 绑定） |
| GPU | GT 1030 2GB（弱，深度学习跑 CPU 轻量模型） |

## 目录结构

```
dataCollection/
├── config/            # cameras.yaml（触发/曝光/序列号映射）、calibration.yaml（标定板参数）
├── src/tabletennis/
│   ├── core/          # 跨模块共享数据类型（Frame/Pose2D/…）+ 配置加载
│   ├── camera/        # 相机控制：开关/触发/图像参数/多机管理（★已完成）
│   │   └── mv_import/     # vendor 的 MVS Python 绑定（生成代码，勿手改）
│   ├── calibration/   # 相机标定：内参标定（棋盘格张正友法）
│   ├── vision/        # 视觉识别：姿态已实现（RTMPose-l），球/球桌待实现
│   │   └── pose/          # RTMPose 2D 姿态检测器
│   └── visualization/ # 可视化：2D 骨架叠加已实现，3D（Open3D）待实现
├── scripts/           # 各功能的启动脚本（见下）
├── tests/             # pytest 测试
└── data/              # 标定图像、标定结果、录制序列
```

> **相机内参标定**分两步：先用 PySide6 图形界面（4 路实时预览 + 点击选中 + Enter 纯拍照）采集照片，
> 入口在 [`scripts/calibrate_intrinsics.py`](scripts/calibrate_intrinsics.py)；角点检测与内参计算在
> [`src/tabletennis/calibration/`](src/tabletennis/calibration/)（待接入 GUI）。

每个模块目录下都有各自的 `README.md`，说明「这个模块干什么 / 怎么用 / 还没做完什么」，详见
[模块文档](#模块文档)。

## 环境安装

本机 base 是 Python 3.14（太新，open3d/torch 等可能缺 wheels），建议新建 3.11 环境：

```bash
conda env create -f environment.yml
conda activate tt
```

（或 `pip install -r requirements.txt`。相机控制阶段只需 `numpy` + `pyyaml`；内参标定 GUI 需
`opencv-python` + `PySide6`；姿态识别需 `onnxruntime` + `rtmlib`。）

相机 SDK 无需安装，已随系统配好环境变量：

- `MVCAM_COMMON_RUNENV=/opt/MVS/lib`
- `LD_LIBRARY_PATH=/opt/MVS/lib/64:...`

代码里也做了 `setdefault` 兜底，直接运行即可。

## 快速上手

```bash
# 1. 确认 4 台相机在线，拿到序列号
python scripts/list_cameras.py

# 2. 单相机实时预览（先连续采集看画面）
python scripts/grab_preview.py --camera-id 0

# 3. 交互式控制：四机 2×2 平铺 + trackbar 调曝光/增益/伽马 + 检测开关（推荐）
python scripts/live_control.py                       # 外部触发（默认）
python scripts/live_control.py --trigger continuous  # 自由采集（无信号发生器）

# 4. 人体姿态识别实时预览（相机 + RTMPose-l + 骨架叠加）
python scripts/pose_preview.py                       # 单相机自由采集
python scripts/pose_preview.py --all-cameras         # 四路平铺，每路都跑检测
python scripts/pose_preview.py --trigger external    # 外部触发

# 5. 四机同步验证（观察各机帧号/时间戳是否对齐）
python scripts/grab_sync.py

# 6. 查看这台相机支持哪些图像参数、取值范围
python scripts/query_parameters.py --camera-id 0

# 跑测试
pytest tests/ -v
```

> **重要**：运行前先关闭 MVS 客户端（`/opt/MVS/bin/MVS`）。它以独占方式打开相机，
> 否则程序会得到 `0x80000203`（设备无访问权限）。

## 触发接线（信号发生器）

1. 信号发生器输出（BNC/TTL）**并联**到 4 台相机的 I/O 输入线 **Line0**。
2. 每台相机配置（代码 `trigger.py` 已封装，`config/cameras.yaml` 里 `trigger.mode: external`）：
   - `TriggerMode = On`
   - `TriggerSource = Line0`
   - `TriggerActivation = RisingEdge`（上升沿）
   - `LineDebouncerTime ≈ 50us`（防抖动误触发）
3. 信号发生器频率 = 相机采集帧率。外部触发下四机曝光对齐，用帧里的
   `device_timestamp` / `nFrameNum` 做同步判断。

**没接信号发生器时**：`trigger.mode: software` 逐帧软触发，或 `trigger.mode: continuous`
自由采集，用于调试。

## 调亮度 / 曝光补偿（关键）

本机是**黑白相机**，没有白平衡/饱和度，画面亮度靠四者组合：

| 参数 | GenICam 节点 | 说明 | 代码入口 |
|---|---|---|---|
| 曝光时间 | `ExposureTime` | 单位 μs，先关 `ExposureAuto` | `controls.set_exposure_time_us()` |
| 增益 | `Gain` | 单位 dB，先关 `GainAuto` | `controls.set_gain_db()` |
| 黑电平 | `BlackLevel` | 暗部偏移（类似曝光补偿） | `controls.set_black_level()` |
| 伽马 | `Gamma` | 非线性亮度曲线 | `controls.set_gamma()` |
| 亮度 | `Brightness` | Integer 节点，**部分机型不存在** | `controls.set_brightness()` |
| 对比度 | `Contrast` | Float 节点 | `controls.set_contrast()` |

> **实测 MV-CS016-10UM**：没有 `BlackLevel` / `Brightness` / `Contrast` 节点，实际可调只有
> `ExposureTime`(15μs~10s)、`Gain`(0~17dB)、`Gamma`(0~4)。快速调亮优先加曝光（运动模糊小），
> 不够再加增益（会放大噪点）。

每个参数都能查合法范围（`get_exposure_range_us()` / `get_gain_range_db()` 等），
`scripts/query_parameters.py` 会一次性打印当前值 + 范围。

## 姿态识别（RTMPose-l / Halpe-26）

2D 人体姿态用 **RTMPose-l**（top-down：YOLOX 检测人 + RTMPose 关键点，onnxruntime
推理），默认输出 **Halpe-26**（26 个关键点：COCO-17 + 头顶/颈/骨盆 + 足趾/脚跟），
已与相机启动流程打通（`vision/pose/rtmpose_pose.py`）。本机 GT 1030 太弱，默认跑 CPU；
换好 GPU 后 `--device cuda`。实时预览建议「单路 + 降分辨率 + 隔帧」，见
`scripts/pose_preview.py` 的 `--stride` / `--max-side`。

模型权重首次运行自动下载到 `~/.cache/rtmlib/hub/checkpoints`。

## 路线图

- [x] **阶段 1~2**：脚手架 + 相机控制模块（开关、外部触发、图像参数、四机同步）
- [x] **交互控制**：`live_control.py` 四机平铺 + trackbar 调参 + 检测开关 + 触发信号自检
- [x] **姿态 2D**：RTMPose-l-halpe26（26 点）人体关键点检测 + 骨架叠加（`pose_preview.py`）
- [x] **相机内参标定 · 照片采集**：[`scripts/calibrate_intrinsics.py`](scripts/calibrate_intrinsics.py)（PySide6 GUI 纯采集；检测/标定逻辑已就绪，待接入）
- [ ] **球检测**（`vision/ball/`）：接口已定义，算法待实现
- [ ] **球桌检测**（`vision/table/`）：接口已定义，算法待实现
- [ ] **多相机外参标定**：四机位姿（世界系对齐）
- [ ] **2D→3D 重建**：球轨迹 / 人体骨架 / 桌面三角化
- [ ] **阶段 5**：Open3D 3D 实时可视化

## 模块文档

| 模块 | 说明 | 文档 |
|---|---|---|
| core | 共享数据类型 + 配置加载 | [`src/tabletennis/core/README.md`](src/tabletennis/core/README.md) |
| camera | 相机控制（SDK 封装） | [`src/tabletennis/camera/README.md`](src/tabletennis/camera/README.md) |
| calibration | 相机标定（内参，棋盘格） | [`src/tabletennis/calibration/README.md`](src/tabletennis/calibration/README.md) |
| vision | 视觉识别（RTMPose 姿态） | [`src/tabletennis/vision/README.md`](src/tabletennis/vision/README.md) |
| visualization | 2D 叠加可视化 | [`src/tabletennis/visualization/README.md`](src/tabletennis/visualization/README.md) |
| scripts | 启动脚本 | [`scripts/README.md`](scripts/README.md) |
| tests | 测试 | [`tests/README.md`](tests/README.md) |
| config | 配置文件 | [`config/README.md`](config/README.md) |

更多开发细节（SDK 用法、踩坑、约定）见 [CLAUDE.md](CLAUDE.md)。
