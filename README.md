# 乒乓球场景重建系统

用 **4 台海康 MV-CS016-10UM**（USB3 黑白工业相机）做多视角实时重建，目标产物：

- **球轨迹重建**：三角化出乒乓球的 3D 轨迹
- **人体动作重建**：多视角 2D 关键点 → 3D 骨架
- **桌面位置还原**：拟合出球台平面的位姿

相机由**信号发生器统一外部触发**实现四机同步曝光（实测同步误差 ~1µs、出帧率 100Hz）。

## 硬件

| 项 | 型号 / 说明 |
|---|---|
| 相机 ×4 | 海康机器人 MV-CS016-10UM（USB3，**黑白** Mono8，1440×1080，最高 249fps） |
| 触发 | 信号发生器，输出接每台相机的 **Line0**（6-pin I/O） |
| SDK | 海康 MVS V4.8.0（已装 `/opt/MVS`，含 Python 绑定） |
| GPU | **RTX 5080 16GB**（Blackwell，onnxruntime-gpu + TensorRT FP16） |

> ⚠️ 4 相机满速 ~12.4Gbps，请 **2+2 分开插**（2 台 CPU 直连 USB + 2 台芯片组），别插 USB2 口（会掉到 ~25fps）。

## 当前能力（已全链路跑通）

- **球检测**：三条路线——经典 CV（无训练）、YOLO（yolov8n，mAP50 0.961）、灰度 yolo11n 1ch（mAP50 0.968）；`refine_ball_center` 亚像素精修到 0.2px。
- **球 3D 重建**：多视角 DLT 三角化 + 红球实时渲染，`live_control.py` 按 **b** 开启。
- **人体姿态**：RTMPose-l-halpe26（26 点）+ **yolo11n 灰度人检测**（默认），2D→3D 骨架重建，按 **P** 开启。
- **TensorRT 加速**：姿态 YOLOX 11.3→2.26ms、RTMPose 3.9→1.08ms；球 YOLO 4 相机 batch ~7ms。
- **相机标定**：内参（棋盘格）+ 外参（ChArUco 板）+ 桌面定原点（4 大 ArUco 标记），PySide6 GUI。
- **三角化已向量化**：姿态 6.5→0.43ms/人（~12×）。

性能优化全景见 [`docs/optimization_report.md`](docs/optimization_report.md)。

## 目录结构

```
dataCollection/
├── config/            # cameras.yaml（触发/像素格式/序列号映射）、calibration.yaml
├── src/tabletennis/
│   ├── core/          # 共享数据类型（Frame/Pose2D/Skeleton3D/…）+ 配置加载
│   ├── camera/        # 相机控制：开关/触发/图像参数/多机管理（★完成）
│   │   └── mv_import/     # vendor 的 MVS Python 绑定（生成代码，勿手改）
│   ├── calibration/   # 相机标定：内参（棋盘格）+ 外参（ChArUco）+ 桌面定位
│   ├── vision/        # 视觉：姿态(RTMPose + yolo11n 人检测)、球(经典/YOLO/灰度)、球桌
│   │   ├── pose/          # RTMPose 姿态 + yolo11n 人检测
│   │   ├── ball/          # 球检测三路线 + 亚像素精修
│   │   └── table/         # 球桌识别（4 大 ArUco 标记）
│   ├── reconstruction/# 三角化(DLT 向量化) + 跨视角匹配 + 轨迹跟踪
│   └── visualization/ # 2D 叠加 + Open3D 3D 场景（骨架/球/桌面/相机视锥）
├── scripts/           # 各功能启动脚本（见下）
├── tests/             # pytest 测试
├── data/              # 标定图像、标定结果、录制序列、训练数据集
├── runs/              # 训练产物（权重/ONNX，gitignored）
└── docs/              # 性能优化报告等
```

## 环境安装

本机 base 是 Python 3.14（太新），务必用 **Python 3.11 环境 `tt`**：

```bash
conda env create -f environment.yml
conda activate tt
# 深度推理/训练另需 onnxruntime-gpu + torch（详见 CLAUDE.md「torch 训练栈」）
```

相机 SDK 无需安装，环境变量已配好（`MVCAM_COMMON_RUNENV=/opt/MVS/lib`、`LD_LIBRARY_PATH`），代码里也有 `setdefault` 兜底。

## 快速上手

```bash
# 1. 确认 4 台相机在线，拿到序列号
python scripts/list_cameras.py

# 2. 交互式控制：四机 2×2 平铺 + trackbar 调参 + 检测开关（推荐）
python scripts/live_control.py                       # 外部触发（默认）
python scripts/live_control.py --trigger continuous  # 自由采集（无信号发生器）

# 3. 人体姿态实时预览（相机 + RTMPose + 骨架叠加）
python scripts/pose_preview.py --all-cameras

# 4. 球 3D 重建（合成自检 / 真机）
python scripts/reconstruct_ball.py --synthetic --no-display   # 无硬件自检
python scripts/reconstruct_ball.py                            # 真机

# 5. 姿态 2D→3D 重建
python scripts/reconstruct_pose.py --synthetic --no-display

# 6. 四机同步验证 / 参数查询
python scripts/grab_sync.py
python scripts/query_parameters.py --camera-id 0

# 跑测试
pytest tests/ -v
```

> **重要**：运行前先关闭 MVS 客户端（`/opt/MVS/bin/MVS`），它以独占方式打开相机，否则报 `0x80000203`（设备无访问权限）。

### live_control 按键

| 键 | 功能 |
|---|---|
| **b** | 球追踪（YOLO/经典 → 亚像素精修 → DLT → Open3D 红球） |
| **p** | 姿态 2D→3D 重建（右上角显示 3D FPS） |
| **t** | 桌面定位（4 大 ArUco 标记定桌面世界系） |
| **r** | 录制球检测数据（3s 倒计时 → 存图+预标注 → 再按 r 停） |
| **R** | viewer3d 视角复位；W/A/S/D 平移、方向键旋转、+/− 缩放 |

## 触发接线（信号发生器）

1. 信号发生器输出（BNC/TTL）**并联**到 4 台相机的 I/O 输入线 **Line0**。
2. 每台相机（`config/cameras.yaml` 里 `trigger.mode: external`）：`TriggerMode=On`、`TriggerSource=Line0`、`TriggerActivation=RisingEdge`、`LineDebouncerTime≈50us`。
3. 信号发生器频率 = 采集帧率（实测 100Hz）。外部触发下四机曝光对齐，用 `device_timestamp`/`nFrameNum` 做同步判断。

没接信号发生器时：`trigger.mode: software` 逐帧软触发，或 `trigger.mode: continuous` 自由采集（调试用）。

## 调亮度 / 曝光补偿

本机是**黑白相机**，没有白平衡/饱和度，实际可调只有 **曝光 `ExposureTime`**(15µs~10s)、**增益 `Gain`**(0~17dB)、**伽马 `Gamma`**(0~4)。快速调亮优先加曝光（运动模糊小），不够再加增益（会放大噪点）。参数范围见 `scripts/query_parameters.py`。

> 球高速（10m/s）时曝光需压到 ≤100µs（配合补光）否则拖影几十像素；默认 `camera_settings.json` 曝光 10832µs 仅适合静态调试。

## 标定流程

1. **内参**：`scripts/calibrate_intrinsics.py`（PySide6 GUI，四路预览 + Enter 采集棋盘格）。
2. **外参**：`scripts/calibrate_extrinsics.py`（ChArUco 板：Enter 拍不同板位姿、C 求相机间相对外参）。
3. **桌面定原点**：`calibrate_extrinsics.py` 按 **T**（或 `live_control.py` 按 **t**），用 4 个大 ArUco 标记定桌面世界系，写 `data/extrinsics/table_extrinsics.yaml`（X 短边/Y 长边/Z 向上）。

## 球检测数据 / 训练管线

录制 → 标注 → 归一化 → 增强 → 训练 → 导出：

```bash
python scripts/capture_ball.py --prelabel      # 或 live_control 按 r 录制
python scripts/label_ball.py                   # 拖框人工校正
python scripts/normalize_ball_boxes.py         # 手绘框统一为正方形
python scripts/augment_ball.py                 # 数据增强
python scripts/train_ball.py                   # yolov8n@1280（或 train_ball_gray.py 训 yolo11n 1ch）
python scripts/fix_onnx_dynamic.py             # 固化 h/w（否则 TRT 静默回退 CUDA）
```

训练产物到 `runs/detect/*/weights/best.onnx`，TRT 引擎缓存自动按 onnx 内容哈希分目录（`~/.cache/tabletennis/trt_engines/`）。

## 路线图

- [x] **阶段 1~2**：脚手架 + 相机控制模块（开关、外部触发、图像参数、四机同步）
- [x] **交互控制**：`live_control.py` 四机平铺 + 调参 + 检测开关 + 触发自检
- [x] **姿态 2D**：RTMPose-l-halpe26 + yolo11n 灰度人检测
- [x] **相机标定**：内参 + 外参 + 桌面定位（PySide6 GUI）
- [x] **球检测**：经典 CV + YOLO（yolov8n）+ 灰度 yolo11n 三路线
- [x] **球 3D 重建**：DLT 三角化 + 亚像素精修 + Open3D 红球渲染
- [x] **姿态 2D→3D 重建**：向量化 DLT + 跨视角匹配 + 身份跟踪
- [x] **TensorRT 加速**：姿态 YOLOX + RTMPose + 球 YOLO
- [ ] **冲 100fps**：960 球检测重训 / 姿态 Fixed ROI / YOLOX decode 上 GPU（见优化报告）

## 模块文档

| 模块 | 说明 | 文档 |
|---|---|---|
| core | 共享数据类型 + 配置加载 | `src/tabletennis/core/README.md` |
| camera | 相机控制（SDK 封装） | `src/tabletennis/camera/README.md` |
| calibration | 相机标定（内参/外参/桌面） | `src/tabletennis/calibration/README.md` |
| vision | 视觉识别（姿态/球/球桌） | `src/tabletennis/vision/README.md` |
| reconstruction | 三角化/匹配/跟踪 | `src/tabletennis/reconstruction/README.md` |
| visualization | 2D/3D 可视化 | `src/tabletennis/visualization/README.md` |
| scripts | 启动脚本 | `scripts/README.md` |
| tests | 测试 | `tests/README.md` |

更多开发细节（SDK 用法、时间戳坑、踩坑、约定）见 [CLAUDE.md](CLAUDE.md)。
