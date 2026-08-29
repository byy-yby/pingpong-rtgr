# CLAUDE.md — 乒乓球场景重建系统（开发经验收集）

本文件供 Claude Code 阅读，记录项目结构、SDK 用法、踩坑与约定。README.md 是给人看的，这里记开发时需要的硬事实。

## 项目现状

- **已完成**：
  - 相机控制模块（`src/tabletennis/camera/`）——开关、外部/软件触发、图像参数、
    多机管理，脚本 `list_cameras` / `grab_preview` / `grab_sync` / `query_parameters`、测试、文档。
  - 交互式控制 `scripts/live_control.py`——四机 2×2 平铺 + trackbar 调曝光/增益/伽马 +
    键盘切检测开关 + 外部触发信号自检。
  - 视觉模块 `vision/`——**RTMPose-l-halpe26 2D 姿态（26 点）已实现**；球只有抽象接口。
  - 可视化模块 `visualization/`——2D 骨架叠加（`overlay2d.py`）+ Open3D 3D 场景
    （`viewer3d.py`：相机视锥 + 标准尺寸球桌）。
  - 相机内参标定——`src/tabletennis/calibration/`（棋盘格张正友标定）+ `scripts/calibrate_intrinsics.py`
    （PySide6 GUI），已并入本包。
  - 多相机外参标定——`src/tabletennis/calibration/extrinsics.py`（ChArUco 板，
    `compute_relative_extrinsics` 求相机间相对外参，世界系=参考相机）+ `scripts/calibrate_extrinsics.py`
    （PySide6 GUI：四路预览 + Enter 拍不同板位姿 + C 求相对外参 + Open3D 相机位置；T 桌面定原点 + 3D 球桌）。
  - 球桌识别 + 场景可视化——`vision/table/`（两个大 ArUco 标记定桌面世界系）+
    `live_control.py` 按 T：四机视角画桌面边框 + 生成 Open3D 3D 场景（相机 + 标准尺寸球桌）。
  - 人体姿态 2D→3D 重建——`reconstruction/`（`triangulate.py` 置信度加权多视角 DLT +
    去畸变/交会角质量；`associate.py` 跨视角实例匹配；`load_camera_rig` 读标定）+ 3D 骨架
    实时渲染（`viewer3d.py` 新增 `add_skeleton_layer` / `set_skeletons`）+ 入口
    `scripts/reconstruct_pose.py`（`--synthetic` 无硬件自检）。世界系用
    `data/extrinsics/table_extrinsics.yaml`（桌面系，与 3D 场景一致）。
- **未开始**：球检测、球 3D 轨迹重建、严格按时间戳对齐的通用组帧（`pipeline`）。
  `pipeline` 目录尚未建；`reconstruction` 已建（姿态部分完成）。

## 标定工具的归属（易混）

标定在本包内，分两层（都复用 `camera/mv_import/`，只吃灰度图）：

- **内参** `src/tabletennis/calibration/intrinsics.py`：棋盘角点 + 张正友标定，产出 `CameraIntrinsics`；
  GUI `scripts/calibrate_intrinsics.py`（四路预览 + 点击选中 + Enter 采集）。普通棋盘格
  8×12 内角 / 30mm/格，见 `config/calibration.yaml`。
- **外参** `src/tabletennis/calibration/extrinsics.py`：ChArUco 检测 + solvePnP 求「板→相机」位姿，
  `compute_relative_extrinsics` 由多组板位姿求相机间相对外参（世界系=参考相机），产出 `CameraExtrinsics`；
  GUI `scripts/calibrate_extrinsics.py`（四路预览 + Enter 拍不同板位姿 + C 求相对外参 + Open3D 相机位置；
  T 桌面定原点 + Open3D 球桌 + 四路画桌面边框/球网）。外参与桌面定位**完全分开**。
  参数见 `config/extrinsics.yaml`（含 `reference_camera`）。

早前曾有独立的 `/home/yby/camera_calibration/` 桌面工具与 ChArUco 占位，现已统一到本包。

### ChArUco 外参的坑（OpenCV 5.0）

- 本机 OpenCV 是 **5.0.0**，旧 aruco API（`detectMarkers` / `interpolateCornersCharuco` /
  `estimatePoseCharucoBoard`）**已移除**，只能走 `CharucoBoard` + `CharucoDetector`：
  `detectBoard` → `matchImagePoints` → `solvePnP`（`extrinsics.py` 已封装）。
- 标定板 13x9 有 **58 个标记**，字典必须 ≥58（`DICT_*_50` 只有 50 个不够，`create_board` 会报错）。
- `square_length_m` / `marker_length_m` / `dictionary` / `legacy_pattern` 必须与实际打印板一致；
  `--generate-board` 可生成与配置一致的板，重新打印保证匹配。
- 球桌定原点（`scripts/calibrate_extrinsics.py` 的 `T` 键）：两个大 ArUco 标记（如
  `DICT_5X5_50` 的 ID0/ID1）平放对角线两角，`extrinsics.build_table_frame` 用两标记
  平均法向/平均 X 轴构造桌面系（原点=标记0左上角，**X 沿短边、Y 沿长边、Z 向上**，
  即标记 X 轴做桌面 Y、桌面 X 用 Y×Z 导出），写 `table_extrinsics.yaml`。
  `--generate-markers` 生成带朝向标注的大标记图。
- **单标记 pose 别用 `SOLVEPNP_IPPE_SQUARE`**：它要求对象点以标记**中心**为原点
  （`[-L/2,L/2,0]...`）；本包 `estimate_marker_pose` 用**角点原点**对象点，配 `SOLVEPNP_IPPE`
  （平面 4 点、自动消歧）。用错 IPPE_SQUARE 会得到 z≈0 的错误位姿。
- **桌面大标记 vs 标定板字典冲突**：桌面标记用 `DICT_5X5_50` 的 ID0/ID1，标定板用
  `DICT_5X5_250`——前者是后者的**子集**，两者 0/1 号图案完全相同。标定板若放在桌上，
  `detect_markers(DICT_5X5_50)` 会把板上的 0/1 号格子误当成桌面标记。已用**几何校验**
  兜底：两个大标记必须相距≈桌面对角线（`expected_marker_distance_m`，默认 3.136m，
  容差 `marker_distance_tol_m`=0.5m），不符即拒绝（`TableDetector` 与
  `calibrate_extrinsics.py` 的 `_register_table` 都已加）。彻底解法是重打一套与板
  字典不重叠的桌面标记（如 `DICT_6X6_*`）。

## 硬件 / 环境事实

- 相机：4 × 海康 **MV-CS016-10UM**，USB3，**黑白**（Mono），1.6MP 面阵。
  `lsusb` 显示 `2bdf:0001 Hikrobot`。
- SDK：海康 **MVS V4.8.0**，装于 `/opt/MVS`。
  - `.so`：`/opt/MVS/lib/64/libMvCameraControl.so`（软链 → `.so.4.8.0.3`）。
  - Python 绑定源码：`/opt/MVS/Samples/64/Python/MvImport/`（已 vendor 进 `camera/mv_import/`）。
  - Python 开发指南：`/opt/MVS/doc/工业相机Linux SDK开发指南V4.8.0（Python）/html`。
- 关键环境变量（shell profile 已配好）：
  - `MVCAM_COMMON_RUNENV=/opt/MVS/lib` —— **决定 .so 加载路径**，缺失会 TypeError。
  - `LD_LIBRARY_PATH=/opt/MVS/lib/64:/opt/MVS/lib/32:...`
  - `MVCAM_SDK_PATH=/opt/MVS`
- Python：base 是 **3.14.6**（miniconda）。太新，open3d/torch/mediapipe 可能缺 wheels，
  **务必用 Python 3.11 环境**（见 `environment.yml`，conda env 名 `tt`）。
- GPU：已换 **RTX 5080**（Blackwell sm_120，驱动 580 / CUDA 13.0，16GB）。姿态估计已部署
  GPU：onnxruntime-gpu 1.26（**最后一个支持 CUDA 12 的版本**，1.27 起切 CUDA 13）+ pip 的
  `nvidia-*-cu12` 运行库（CUDA 12.9 / cuDNN 9.25）。`device` 默认 `cuda`，`backend` 支持
  `tensorrt`（TensorrtExecutionProvider FP16，需 TensorRT 10.x 运行库，装
  `tensorrt-cu12-libs==10.14.1.48`，其 wheel **3.96GB**、安装时从 pypi.nvidia.com 现下）。
  **CUDA 13 的 nvidia pip wheel 尚未发布**（PyPI 上是 0.0.0a0 占位），所以别用 cu13。
  实测：YOLOX TRT 11.3→2.26ms、RTMPose TRT 3.9→1.08ms、单相机 detect 端到端 14.6→8.0ms。
  **坑**：mmpose SDK 的 YOLOX 烤入 EfficientNMS，预 NMS TopK K=5000 超 TensorRT 上限 3840，
  走 TRT 会报 `K exceeds the maximum value allowed (3840)`，`_patch_yolox_for_trt` 把 K 改 3000
  解决。详见 `vision/gpu_env.py` 与 `vision/pose/rtmpose_pose.py`。
- **torch 训练栈（RTX 5080 用 CUDA 12.8）**：Blackwell sm_120 没有 cu121/cu124 的 wheel，
  必须 `torch==2.11.0+cu128` + `torchvision==0.26.0+cu128`（cp311，2026-08 实测可用）。
  网络是国内环境：**download.pytorch.org 被墙**（直连 ~247B/s），本地代理对国内镜像反而
  拖慢，装包前必须 `export http_proxy= https_proxy= HTTP_PROXY= HTTPS_PROXY= all_proxy= ALL_PROXY=`
  清空代理。torch/torchvision 走阿里云 wheel 目录（扁平目录**不是**合法 simple-index，要用
  `--find-links https://mirrors.aliyun.com/pytorch-wheels/cu128/`），其余依赖走
  `--index-url https://mirrors.aliyun.com/pypi/simple`（完整 pypi 镜像）。tuna/阿里云对
  >90MB 大文件偶发断连，小包可靠。
  **省流量技巧**：环境里已有 onnxruntime-gpu 的 `nvidia-*-cu12` 运行库（cudnn 9.25/cublas
  12.9 比 torch pin 的新版但 ABI 兼容），可 `--no-deps` 只装 torch+torchvision，再用 `ldd`
  枚举 `torch/lib/libtorch_cuda.so` 缺的库逐个补：
  - `libcusparseLt.so.0`→`nvidia-cusparselt-cu12==0.7.1`
  - `libnccl.so.2`→`nvidia-nccl-cu12==2.28.9`、`libnvshmem_host.so.3`→`nvidia-nvshmem-cu12==3.4.5`、
    `libcupti.so.12`→`nvidia-cuda-cupti-cu12`
  - `libcufile.so.0`→**`nvidia-cufile-cu12`**（torch 的 `cuda-toolkit[cufile]` extra 映射到它，
    不是 `nvidia-cuda-cufile-cu12` 也不是 `cuda-cufile-12-8`，后两者在 PyPI 上是 404）
  triton（torch 的硬依赖，188MB）装不上也不影响普通训练，只有 `torch.compile` 才需要。
  **磁盘告急**：多次装大 wheel 会把 `~/.cache/pip` 撑爆（曾到 7.1G，触发 Errno 28 磁盘满），
  先 `pip cache purge`。训练脚本 `scripts/train_ball.py` 已加 `--patience` 早停。

## SDK 用法（已踩平的关键点）

### import 机制（`camera/sdk.py` 里集中处理）
1. MVS 绑定在 import 时执行 `check_sys_and_update_dll()`，用
   `os.getenv('MVCAM_COMMON_RUNENV') + "/64/libMvCameraControl.so"` 拼路径，
   所以 **import 之前必须 setdefault 该变量**（sdk.py 已做）。
2. `MvCameraControl_class.py` 内部是 `from PixelType_header import *` 这种
   **绝对 import**，所以 **mv_import 目录必须先加进 sys.path**（sdk.py 已做）。
3. 其它模块一律 `from .sdk import ...`，不要直接碰 `mv_import`。

### 生命周期（顺序不能乱）
```
MvCamera.MV_CC_Initialize()                     # 全局初始化一次
MvCamera.MV_CC_EnumDevices(ALL_LAYER_TYPES, devList)
cam = MvCamera(); cam.MV_CC_CreateHandle(devInfo.raw); cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
  ... 配置参数 / MV_CC_StartGrabbing() ...
cam.MV_CC_StopGrabbing(); cam.MV_CC_CloseDevice(); cam.MV_CC_DestroyHandle()
MvCamera.MV_CC_Finalize()
```

### 抓帧：用轮询，不用回调
- 每相机一个线程 `MV_CC_GetImageBuffer(stFrame, timeout_ms)`（timeout 用 1000）。
- 成功（ret==0）且 `stFrame.pBufAddr` 非空时：`ctypes.string_at(pBufAddr, nFrameLen)`
  拷成 bytes → `np.frombuffer` → `.copy()`（**必须 copy**，SDK Free 后会复用缓冲）。
- 有界队列 `queue.Queue(maxsize≈10)`，满则丢最旧（`camera.py::_put`）。
- 回调式（`RegisterImageCallBackEx`）在 Python 里受 GIL 影响大，不推荐多机实时用。

### 触发（信号发生器，`camera/trigger.py`）
外部触发接相机 **Line0**，标准 GenICam 节点串：
```python
SetEnumValueByString("TriggerMode", "On")
SetEnumValueByString("TriggerSource", "Line0")     # 低延迟可换 Line2
SetEnumValueByString("TriggerActivation", "RisingEdge")
SetFloatValue("TriggerDelay", 0.0)
SetBoolValue("TriggerCacheEnable", False)          # 关闭触发缓存，防攒帧
SetEnumValueByString("LineSelector", "Line0")
SetIntValueEx("LineDebouncerTime", 50)             # us，防误触发
```
- 软件触发调试：`TriggerSource=Software` + `SetCommandValue("TriggerSoftware")`。
- 连续采集：`TriggerMode=Off`。

### 图像参数句柄（曝光/亮度/曝光补偿，`camera/parameter.py`）
所有参数是 GenICam 节点，字符串 key + `MV_CC_Set/Get{Float,Int,Enum,Bool}Value`：

| 参数 | 节点 | 类型 | 备注 |
|---|---|---|---|
| 曝光时间 | `ExposureTime` | Float(us) | 先关 `ExposureAuto` |
| 自动曝光 | `ExposureAuto` | Enum(Off/Once/Continuous) | 手动前设为 Off |
| 增益 | `Gain` | Float(dB) | 先关 `GainAuto` |
| 自动增益 | `GainAuto` | Enum | |
| 黑电平 | `BlackLevel` | Float | "曝光补偿"的暗部偏移 |
| 伽马 | `Gamma` | Float | |
| 亮度 | `Brightness` | Integer | **部分机型无此节点** |
| 对比度 | `Contrast` | Float | |
| 帧率 | `AcquisitionFrameRate` / `ResultingFrameRate` | Float(Hz) | 后者只读 |
| 像素格式 | `PixelFormat` | Enum | 黑白固定 `Mono8` |

- **黑白相机没有白平衡/饱和度**；调亮度 = 曝光 + 增益 + 黑电平 + 伽马。
- **实测 MV-CS016-10UM**（2026-08-25 验证）：**没有 `BlackLevel` / `Brightness` / `Contrast` 节点**
  （读回均为 None）；实际可调只有 `ExposureTime`(15μs~10s)、`Gain`(0~17dB)、`Gamma`(0~4)。
  调亮度就用这三者；最高帧率 ~165Hz @1440×1080。
- Get 返回的 `MVCC_FLOATVALUE{fCurValue,fMax,fMin}` / `MVCC_INTVALUE{nCurValue,nMax,nMin,nInc}`
  自带范围，`parameter.py` 已封装成 `get_*_range()`。
- 设置枚举用 `SetEnumValueByString`（比传 int 可读）；SDK 绑定方法内部已做
  `strKey.encode('ascii')` / `byref(stValue)` / `c_float()` 等转换，**调用方直接传 Python 值**即可。

### 像素 / 帧结构
- 黑白相机用 `Mono8`（值 `17301505`），帧缓冲 = H×W 字节。
- `MV_FRAME_OUT.pBufAddr` + `stFrameInfo`（`MV_FRAME_OUT_INFO_EX`）含：
  `nWidth/nHeight/nFrameNum/nDevTimeStampHigh/nDevTimeStampLow/nHostTimeStamp/nFrameLen/enPixelType`。
- 设备时间戳 = `(nDevTimeStampHigh << 32) | nDevTimeStampLow`，用于四机同步判断。

## 约定

- 分层：`core`（数据类型）→ `camera` / `vision` / `calibration` / `reconstruction` /
  `visualization` → `pipeline`（编排）。下层不依赖上层，跨层共享类型放 `core/types.py`。
- 脚本用 `sys.path.insert(0, "<project>/src")` 引导 import；测试用 `tests/conftest.py`。
- 相机模块所有对 SDK 的 import 都收敛在 `camera/sdk.py`。
- 视觉检测器走**注册工厂**：`vision/detector.py` 里的 `register_detector(kind, factory)` /
  `create_detector(kind)`。上层（`live_control.py`）按名字取，未注册时 `create_detector`
  返回 `None`，调用方据此提示「接口已定义、算法待实现」。新算法实现后只需 `register_detector`
  一行，脚本无需改。当前已注册 `pose`（RTMPose-l-halpe26，26 点）与 `table`
  （球桌识别，`TableDetector.load_default()` 自动加载标定数据），`ball` 待注册。
- 每个模块目录都有 `README.md`（作用 / 用法 / 未完成），改完模块记得同步更新它。

## 踩坑记录 / TODO

- [ ] **Python 3.14 太新**：装 open3d/mediapipe 前先建 3.11 环境。
- [ ] **udev 权限**：若 `list_cameras.py` 枚举为 0，大概率是 USB 设备无访问权限，
  需装海康 udev 规则（`/opt/MVS/driver`）或 `sudo chmod`，见验证章节。
- [ ] **四机同步**：`get_latest_bundle` 仍只取各机最新帧；外参标定已改用
  `get_synchronized_bundle`（清队列 + 各取下一帧 = 同一触发周期）。通用的按
  `device_timestamp` 最近邻配对仍待 pipeline 阶段。
- [ ] **USB3 带宽**：4 × 1.6MP 高帧率同时出图可能撞带宽墙，必要时降帧率或
  用 `TriggerDelay` 错峰。
- [ ] **单通道喂模型**：本机是黑白 Mono8，RTMPose 训练在 RGB 上，`rtmpose_pose.py` 里把灰度
  复制成 3 通道再送模型（存在 domain gap，靠固定短曝光 + 补光缓解）。
- [x] **RTMPose 性能**：已上 TensorRT——YOLOX 11.3→2.26ms、RTMPose 3.9→1.08ms、单相机
  detect 端到端 14.6→8.0ms（GPU 不再瓶颈，剩余是 CPU 预处理/NMS 开销）。YOLOX 因烤入
  NMS 的 TopK-5000 走 TRT 需先 patch 成 3000（`_patch_yolox_for_trt`）。要再提速：
  ① 换 RTMO（one-stage，砍掉 YOLOX+逐人 RTMPose）；② YOLOX 重新导出成动态 batch 以批处理。
- [ ] **球检测**：接口已在 `vision/detector.py`（`BallDetector`），经典 CV 路线（阈值/连通域）
  待接入 `register_detector("ball", ...)`。（球桌已实现并注册。）
- [ ] **YOLO 微调训练**（2026-08-29 进行中）：数据集已统一框 + 增强 3000→11944，
  用 `scripts/train_ball.py`（yolov8n.pt 预训练迁移，imgsz 1280，single_cls，--patience 早停）
  训练单类 ball。torch 栈见「torch 训练栈」节。
- [ ] 后续模块目录待建：`pipeline/`（`reconstruction/` 已建，姿态三角化 + 匹配完成）。
- [ ] **姿态重建精度受相机距离限制**：相机距桌面约 3~6m（球在画面 ~12~24px），
  合成 1.5px 噪声下 3D 关节误差约 cm 级；更精确需更高分辨率或更近的机位。
