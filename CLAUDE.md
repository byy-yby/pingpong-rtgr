# CLAUDE.md — 乒乓球场景重建系统（开发经验收集）

本文件供 Claude Code 阅读，记录项目结构、SDK 用法、踩坑与约定。README.md 是给人看的，这里记开发时需要的硬事实。

## 项目现状（截至 2026-09-01，已全链路跑通）

- **相机控制** `src/tabletennis/camera/` —— 开关、外部/软件触发、图像参数、多机管理。
  脚本 `list_cameras` / `grab_preview` / `grab_sync` / `query_parameters`。
- **交互式控制** `scripts/live_control.py` —— 四机 2×2 平铺 + trackbar 调参 + 检测开关 + 外部触发自检。
- **录像 + 离线重建（EasyMocap 放弃实时后的主路线）** —— live_control 按 `v`（或点面板「录像」）
  四路 mp4 录到 `data/video/<YYYYmmdd_HHMMSS>/cam{cid}.mp4` + `cam{cid}_ts.npy` + `meta.json`
  （`camera/recorder.py::SessionVideoRecorder`）；`scripts/reconstruct_video.py <session>`
  离线 EasyMocap 重建（`reconstruction/video_source.py` 对齐 + `em_fit.py` 热启动流式拟合）。
- **姿态 2D** `vision/pose/rtmpose_pose.py` —— RTMPose-l-halpe26（26 点），top-down。
  **人检测默认换 `yolo11n-gray`**（1ch 灰度原生），见「人检测」节。
- **球检测** `vision/ball/` —— 三条路线都实现：经典 CV、YOLO（yolov8n）、灰度 yolo11n（1ch）。
- **球 3D 重建** `reconstruction/ball.py` + `scripts/reconstruct_ball.py` + live_control 按 b —— 三角化 + 红球渲染。
- **姿态 2D→3D 重建** `reconstruction/triangulate.py`（置信度加权多视角 DLT，已向量化）+ `associate.py`（跨视角匹配）+ `pose_track.py`（PoseTracker 身份跟踪）。
- **相机标定** 内参（棋盘格张正友）+ 外参（ChArUco 板 `compute_relative_extrinsics`）+ 桌面定原点（4 大 ArUco 标记）。
- **球桌识别 + 场景可视化** `vision/table/` + `viewer3d.py`（相机视锥 + 球桌 + 骨架 + 球图层）。
- **TensorRT 全链路部署** 姿态 YOLOX + RTMPose 已上 TRT FP16，球 YOLO 也上 TRT。
- **性能优化报告** `docs/optimization_report.md`（球+姿态 profiling 与加速方案，权威数字在此）。

无 `pipeline/` 包——项目走**脚本式编排**（`scripts/reconstruct_pose.py` / `reconstruct_ball.py` 里各自一个 `Reconstruct*` 类），不另建 pipeline 包。分层：`core` → `camera` / `vision` / `calibration` / `reconstruction` / `visualization`，下层不依赖上层。

## 视觉 / 重建子系统（本项目的核心）

### 球检测：三条路线

注册在 `vision/detector.py`：`register_detector("ball", ...)`（经典）、`register_detector("ball_yolo", ...)`（YOLO）。

1. **经典路线** `vision/ball/classical_ball.py::ClassicalBallDetector`
   —— 运行均值背景减除 + 相邻帧差取并集 → 尺寸先验（半径 5~15px，面积 78~707px²）滤候选
   → `refine_ball_center` 亚像素质心。无模型权重、纯 numpy+cv2、CPU 可跑、零标注。
   有状态（跨帧维护背景），`_bg/_prev` 按 `camera_id` 分 dict，单实例可跨 4 相机复用（曾因不按相机分背景导致串扰）。局限：先验弱，手指/拍边/反光/阴影易误检。
2. **YOLO 路线** `vision/ball/yolo_ball.py::YoloBallDetector`
   —— onnxruntime 加载训练导出的单类 `best.onnx`（输出 `(1,5,N)`），bbox 后接 `refine_ball_center` 亚像素精修。
   backend 默认 `"auto"`→TRT FP16，`"tensorrt"/"cuda"/"cpu"` 可选，预分配 buffer 复用。
   无 torch 运行时依赖（onnxruntime 即可）。
3. **灰度 yolo11n（1ch）路线** `scripts/train_ball_gray.py` 训练
   —— yolo11n-grayscale 1 通道单类微调，**通道自适应**（`yolo_ball.py` 按 ONNX 输入通道数 1ch/3ch 自动适配，1ch 不再复制成 3 通道，输入 tensor 18.75MB→6.27MB）。

**权重与缓存**：
- yolov8n@1280（3ch）：`runs/detect/ball/weights/best.onnx`（12.8MB），mAP50 **0.961** / mAP50-95 **0.849**；真图抽检 76/76 检出、球心中位误差 0.2px、0 误检。
- 灰度 yolo11n 1ch：`runs/detect/ball_gray/weights/best.onnx`，mAP50 **0.9677** / mAP50-95 **0.8538**（略优）。
- 训练权重 `best.pt` 在各 `runs/detect/*/weights/` 下；`*.pt`/`*.onnx`/`runs/` 均 gitignored，需重训/重导。
- TRT 引擎缓存 `~/.cache/tabletennis/trt_engines/<onnx内容哈希>/`（按 onnx 内容哈希分目录，换权重/换输入形状必重建，杜绝静默复用旧引擎）。

**推理配置现状**：imgsz **1280 矩形**（长边 1280、短边按 4:3 → 1280×960，去 padding），conf 阈值 **0.2**。
960 矩形（~145FPS）因小球缩到 8~16px（YOLO 最小尺度 stride=8）会间歇性单相机漏检致三角化失败，已回退 1280（球 10.7~21px，~90FPS）。要兼得速度需重训 960（`train_ball_gray.py` 改 imgsz=960）。

**部署链路**（live_control 按 b）：后台线程 `_ball_load_worker` 建 session（TRT 引擎缓存命中秒开）+ 首帧热启动 → 逐相机**顺序** detect（4 路 ~75ms≈13FPS；`detect_batch` 4 相机批处理与顺序持平且相机丢帧会触发新形状引擎重建 ~55s，故热路径用顺序）→ `triangulate_ball` DLT → `viewer3d` 红球。`BALL_ONNX` 环境变量可覆盖模型路径。

**数据/训练管线**：live_control 按 r 录制（或 `capture_ball.py`，存 PNG + 可选经典检测预标注）→ `label_ball.py` 拖框标注 → `normalize_ball_boxes.py` 手绘框统一为正方形（refine 精修球心+半径、边长 2.4×r）→ `augment_ball.py` 增强（3000→12000，几何+光度）→ `train_ball.py` 训练（yolov8n）/`train_ball_gray.py`（yolo11n 1ch）→ `fix_onnx_dynamic.py` 修 h/w → 部署。

### 球重建线程（`_ball_recon_loop`）

`live_control.py` 球开启时：独立后台线程 `_ball_recon_loop` 负责「逐相机**非阻塞**取帧 → detect_batch → DLT 三角化 → 更新 `self._latest`」，主循环只读它做 2D 显示 + 姿态重建 + 交互。**无卡尔曼**（逐帧纯 DLT，用户要求；早前的卡尔曼曾因 dt 写死 0.01 假设 100FPS 导致快速球被 0.5m 门限误判冻结）。球 3D 更新 ~140Hz（受相机 100Hz 出帧约束）。三角化 `min_conf=0.15`（曾 0.3 与检测器 conf 0.25 不匹配导致 conf 0.25~0.3 的球 2D 有 3D 被滤）。球轨迹 LineSet 已移除（200 点淹没 2cm 小球），只留红球。`[球诊断]` 每 2 秒打印重建速率与单次 detect_batch+DLT 耗时。

> ⚠️ 曾试 `get_synchronized_bundle`（清队列+阻塞取下一帧）反而引入 ~10ms 开销致 <50Hz，已回退逐相机非阻塞取帧。

### 人检测：yolo11n 灰度（默认）

`vision/pose/rtmpose_pose.py` 的 `RTMPoseDetector` 默认 `det="yolo11n-gray"`（`vision/pose/yolo11_person.py::Yolo11PersonDetector`），替换 rtmlib YOLOX(humanart)：
- **1ch 灰度原生**（第一层卷积 1→16，COCO-80 person=类0，2.62M 参数），消除 gray→3ch 复制与 domain gap；TRT FP16 + 4 相机 batch。实测人检测 **12.8→6.7ms/轮**。
- 三处提速：① `conf_thresh` 0.35→**0.5**（对齐旧 YOLOX，0.35 太松致 RTMPose 裁剪数翻倍）；② RTMPose 预处理归一化 float64→**float32 就地算**（省 ~1ms）；③ 人检测 imgsz **640→416**（人够大，检测段 5.1→2.5ms）。
- ONNX 在 `~/.cache/tabletennis/yolo11n_grayscale_person_416.onnx`（`scripts/export_yolo11_person.py` 从 `data/weights/gray/yolo11n-grayscale.pt` 导出，`fix_onnx_dynamic.py --ch 1` 修 h/w；gitignored 需重导）。`det="yolox-tiny"` 分支保留作回退。
- 真实含人帧 `detect_batch`(4 相机) 优化后 ~10.0ms + 重建 1ms ≈ **~90fps**。

### 三角化：已向量化

`reconstruction/triangulate.py::MultiViewTriangulator`：
- `triangulate_batch`（姿态/球统一走它）——26 关节堆成 `(26,8,4)` 一次性 4×4 gram `np.linalg.eigh` 最小特征向量（等价 DLT 最小奇异解），重投影/交会角/置信度全向量化。姿态 **6.5→0.43ms/人（~12×）**，球 0.33ms（单点无回归）。
- 最差视角重投影 >12px 时逐点回退到 `triangulate_point`（保证与旧输出一致，坐标差 1e-12mm 级）。
- `mean_conf` 曾用 `conf.sum()` 把遮挡相机置信度也算进去 → 改只对有效视角取均值（差分测试抓到的真 bug）。
- 单球无需跨视角匹配（每相机最多 0/1 检测，直接 `{cam_id: Ball2D}` 喂三角化）；`associate.py::match_people` 只用于多人姿态。

### 可视化 `viewer3d.py`（Open3D）

- 默认视角：`up=(0,0,1)`（Z 竖直）、`set_front(front=[1,1,0.9])`——**`set_front` 传的是「从 lookat 指向相机」的方向**，`[1,1,0.9]`（+Z 朝上）= 相机在 +X+Y+Z 斜上方俯瞰球桌（此前 `[-1,-1,-0.9]` 会让相机跑到桌面下方仰视）。R 复位即回此视角。
- 键盘：`VisualizerWithKeyCallback` + **W/A/S/D 平移、方向键旋转、+/− 缩放、R 复位**。**translate 的 +y 才是「向上」**（W=`translate(0,+step)`、S=`translate(0,-step)`，曾写反）。
- 图层：相机视锥（`build_cameras_scene`）、球桌、骨架（`add_skeleton_layer`/`set_skeletons`）、红球（`add_ball_layer`/`set_ball`，跨线程传球心加锁）。
- 每个方法里自己 `o3d = _o3d()` 惰性 import（`_update_ball_geometry` 曾漏写致渲染线程 NameError 窗口退出）；Open3D 窗口必须在主线程开，后台线程只加载模型。

<<<<<<< HEAD
### IMU 姿态（维特智能 WT9011DCL-BT5.0，按 i）

`scripts/live_control.py` 按 **i** 读插在拍柄末端的 IMU：3D 窗口里球拍**朝向来自 IMU**（notify 线程 100Hz 直接推 `viewer3d.set_imu_orientation`，绕开主循环 ~20FPS），**位置绑到检测到的右手腕**（主循环每帧 `right_wrist_anchor` 选「离桌面原点最近的有效右手腕」→ `set_imu_anchor`，两人时即按 i 放拍在原点那一侧）。代码在 `src/tabletennis/imu/`：
- `witmotion.py`：0x55 协议解析 + 角度/四元数 -> 旋转矩阵（纯 numpy，无 I/O）。WT9011DCL 走**蓝牙 5.0 (BLE)**，默认 **0x61 组合包**（加速度6B+角速度6B+角度6B）@10Hz（可 0.2~200Hz），角度 int16/32768×180°、欧拉 Z-Y-X；兼容经典 0x53 角度 / 0x59 四元数。
- **帧格式 UART≠BLE（本模块最大坑，已实测）**：UART 是 `0x55|Flag|Data|Checksum` 共 21B 带校验（校验和 = 0x55 起到数据末之和低 8 位）；**BLE 流（WT901BLE5.0, MTU 23）0x61 包只有 `0x55|0x61|18B` 共 20B 无校验**，且多包塞进一条 notify（80B=4 包、40B=2 包）。解析器必须 `WitMotionParser(checksum=False)`；按 21B 解析每个包都失败 → 有效包只剩 1/256 运气值 → **实测表现 = 能连上但姿态几乎不动/偶发跳一下**。
- `reader.py`：`ImuReader` 后台 BLE 线程（`bleak`，已装进 `tt`）。GATT 服务 `0000ffe5` / notify 收数据 `0000ffe4` / 写命令 `0000ffe9`（UUID 来自官方 SDK `WitBluetooth_BWT901BLE5_0` 的 `BleUUID.java`）；扫描广播名含 "WT" 的模块。
- **坑**：BLE 流**无校验字节**（20B 包，见上）；上报率默认 10Hz 已由 reader 连上后自动提 100Hz（官方 5 字节写命令 `FF AA 69 88 B5` 解锁 → `FF AA 03 <val> 00` RATE → `FF AA 00 00 00` 保存）；模块上电可能被手机/上位机占用（BLE 一般单连接），用 `--imu-mac` 直接指定 MAC 更稳。
- **参考姿态初始化（已解决安装角问题，无需知道 IMU 轴方向）**：按 i 后把球拍平放于**桌面坐标系原点**（桌角原点标记，正面朝上、点口端/手柄朝桌面 −Y；表系 X=短边/Y=长边 2.74m/Z 向上），live_control 采集 `_IMU_INIT_N=20` 个静止样本（窗内角偏差 ≤5°）锁成 `R_home`，之后每个读数经 `imu_to_paddle_world(R, R_home, R_ref)`（`witmotion.py`，= `R@R_home.T@R_ref`）换算成球拍桌面系世界朝向，`R_ref=_IMU_REF_YZ=Rz(-90°)`（mesh 手柄 +X→桌面 −Y、拍面 +Z→+Z）——固定安装角 A 被消掉、参考时刻输出恰为 `R_ref`，挥拍时贴合真实世界朝向。**⚠️ 别用 `R_home.T@R_imu`（共轭旋转），一般三维运动屏幕朝向会整体错位**（有回归测试 `test_imu_to_paddle_world_*` 锁定）。若实测手柄反了 180° 把 `_IMU_REF_YZ` 反号即可。运动时不锁（滚动窗+限频提示），重按 i 重锁。
- **磁力计航向锚定（消 yaw 漂移，2026-09-02 加，09-02 晚改寄存器读）**：症状「挥拍几圈
  摆回参考姿态，拍面平对但手柄航向对不上」= 模块片上 Kalman 的 yaw 在没有可用磁场基准时
  退化成纯陀螺积分（模块默认只报 0x61 acc+gyro+角度，**磁力计根本没吐**）。主机侧修法
  （无需上位机「转 8 字」）——**官方 BWT901BLE5.0 固件不上报磁力计数据流**（RRST 寄存器
  0x02 写 0x0F 请求 0x54 补报实测无效），正确做法是**主动读寄存器**：`ImuReader
  (request_mag=True)` 提好速率后 `_probe_mag` 连发读命令 `FF AA 27 3A 00`（`_read_reg_cmd`，
  寄存器 0x3A = 磁力计 HX 起点），模块以 **0x71 响应帧**回传 HX/HY/HZ(0x3A/B/C)，收到即开
  `_mag_poll_loop` ~20Hz 轮询保持快照（**纯读不写、不 SAVE、不依赖固件**）；锁参考那一刻
  把当前磁航向记进 `MagYawLock`（`witmotion.py`；纯函数 `tilt_compensated_mag_heading`/
  `wrap_pi` 可单测），之后每包在模块**平放且静止**（|roll|/|pitch|≤12°、|gyro|≤40°/s）时
  把显示航向锚回磁场（世界竖直修正 `R_disp = Rz(δ) @ R_disp0`），快速倾斜挥拍/转动时冻结
  修正、回落模块 yaw。绝对磁场方向在「heading 相对参考相减」里被消掉 → 不依赖 IMU 轴方向、
  不需要任何磁场校准。验证看 `[IMU] 磁力计读取可用（寄存器 0x3A / 0x71 响应）` 与锁参考
  后的 `磁锚定` 日志；单测 `test_mag_*` / `test_imu_yaw_lock_*` / `test_parse_reg_0x71*` /
  `test_reader_mag_snapshot_*` 在 `tests/test_imu.py`。
  **⚠️ 若 0x3A 读无响应**（日志 `[IMU] 模块未回传磁力计（读寄存器 0x3A 无响应）`，固件读
  不了该寄存器），磁锚定不启用。此时走降级：① `WorldHeadingHold` 静止冻结（消「不动也慢
  漂」：模块真静止时把显示航向冻住，静止期零偏不进显示，转动不跳变）；② `live_control.
  _maybe_auto_relock` 原点自动重锁（**须开 P 有真实右手腕**：手腕在桌面原点水平 0.45m
  内 + 平放 + 静止持续 ~1.2s → 重锁参考清零，须先挪开再回原点才再触发）；③ 参考锁定等
  `ImuReader.on_ready`（配置完成后才锁，避免锁到过渡期低速流——曾见 0.5 包/秒时已锁）。
  相关单测 `test_world_heading_hold_*` / `test_reader_ready_*` / `test_handle_bearing_*`。
- `right_wrist_anchor` / `so3_project` / `imu_to_paddle_world` 逻辑可测（`viewer3d.right_wrist_anchor` 按骨架名找 `right_wrist`，halpe26=idx10；锚点默认原点 = 桌面原点 (0,0,0)）。
- 四元数默认不上报（0x59 需 `FF AA 27 51 00` 寄存器读，当前用角度即可）。
=======
### 录像 + 离线重建（EasyMocap 的主路线）

录制端 `camera/recorder.py`：
- **帧走旁路 sink 不进主循环**：`Camera.set_frame_sink(fn)` 在采集线程入主队列前把帧送给录制
  回调（须快进快出，只入队）；`SessionVideoRecorder` 每台相机一个后台编码线程
  `_CameraWriter` 消费队列写 `mp4v`→`.mp4`，**编码跟不上丢最旧帧不阻塞抓帧**——写进文件
  的每一帧都 append 设备时间戳，收尾 `np.save cam{cid}_ts.npy`。
- **编解码实测**（本机 OpenCV 5.0）：1440×1080 Mono8 下 `mp4v` ~8ms/帧（100fps 预算内可行）；
  MJPG 26ms 太慢；`avc1`(h264) 无 v4l2 设备打不开。`fps` 只写 mp4 头（播放速度），重建读
  帧序号 + ts 副产物，不受影响。
- 命名按**逻辑相机号** `cam{cid}.mp4`（cid=标定 `cam_{cid}.yaml` 的号），与在线 EasyMocap
  一致；live_control 按 `v` / 面板「录像」按钮启停（`fps=100` 外部触发 / `30` 自由采集）。

离线端 `reconstruction/video_source.py` + `scripts/reconstruct_video.py`：
- **跨相机对齐绝不用「减绝对 ts 差」（各机时钟基准偏移秒级不可比）**，改为**脉冲号对齐**：
  每台相机内部对相邻 ts 差/周期取整 → 丢几拍加几号（`_pulse_ids`），主时钟每帧的脉冲号
  用 `searchsorted` 在目标相机的脉冲号序列里找同号帧（单调 → 天然不倒退）；目标相机那拍被
  编码丢掉就没帧给（该视角缺帧）。录制 sink 挂在同一触发抓帧流上、四台从同一拍起写，
  所以「脉冲号相同 ⇒ 同一物理触发」成立。
- `np.savez_compressed` 返回 None（不是 file 对象），别调 `.close()`。
- 默认档位 `--config stream`（EmFit 热启动 ~2.1s/帧）；`--fake-poses` 注入合成站姿人跑
  整条 录制→对齐→检测→拟合→存档 管道，无硬件/无真人视频也能验证与计时。
- 实测单帧真机开销：检测不在此列；EmFit stream ~2.1s、official cold ~6.2s（GPU 5080）——
  即**放弃实时（100fps 视频离线跑）是必然选择**。
>>>>>>> worktree-easymocap-recon
- 四元数默认不上报（0x51 寄存器读响应同 0x71 帧；当前用角度 + 磁力计寄存器读即可）。

## 标定工具的归属（易混）

标定在本包内，分两层（都复用 `camera/mv_import/`，只吃灰度图）：

- **内参** `src/tabletennis/calibration/intrinsics.py`：棋盘角点 + 张正友标定，产出 `CameraIntrinsics`；GUI `scripts/calibrate_intrinsics.py`。普通棋盘格 8×12 内角 / 30mm/格，见 `config/calibration.yaml`。
- **外参** `src/tabletennis/calibration/extrinsics.py`：ChArUco 检测 + solvePnP 求「板→相机」位姿，`compute_relative_extrinsics` 由多组板位姿求相机间相对外参（世界系=参考相机），产出 `CameraExtrinsics`；GUI `scripts/calibrate_extrinsics.py`（Enter 拍板位姿 / C 求相对外参 / **T 桌面定原点**）。
- **桌面定位与外参完全分开**：桌面用 **4 个不同 ID 大标记**（ID0/1/2/3，18cm 黑块 + 2cm 白边）放四角，`build_table_frame_from_corners` **只用角点位置**（不依赖标记朝向），`fuse_marker_poses` 三角化射线求交消除单标记弱深度。`localize_table_bundle` 为标定脚本与 live_control 共用的单一实现，按 T/t 都写 `table_extrinsics.yaml`（世界系=桌面，X 短边/Y 长边/Z 向上）。

### ChArUco 外参的坑（OpenCV 5.0）

- 本机 OpenCV **5.0.0**，旧 aruco API 已移除，只能 `CharucoBoard` + `CharucoDetector`：`detectBoard` → `matchImagePoints` → `solvePnP`。
- 标定板 13x9 有 **58 个标记**，字典必须 ≥58（`DICT_*_50` 不够会报错）。
- `marker_length_m` **不影响检测**（只影响物理尺度换算）；决定检测成败的是 `dictionary` / `legacy_pattern` / `squares_x/y`。`legacy_pattern: true` 用于 OpenCV <4.6 生成的板。
- `calibrateCamera` 要求物点/像点 **float32**（Point3f/Point2f），传 float64 报错。
- **单标记 pose 别用 `SOLVEPNP_IPPE_SQUARE`**（要求对象点以标记中心为原点）；本包用**角点原点**对象点配 `SOLVEPNP_IPPE`。用错得到 z≈0 错误位姿。
- **桌面「上」要取反**：`estimate_marker_pose` 对象点「角点原点 + Y 向下」→ 标记 Z=X×Y 指向标记内部，平放时朝下，`build_table_frame` 必须取反 Z（否则球网朝下、桌腿朝上）。
- **大标记需要白边（quiet zone）**：黑方块四周必须被白包围否则检测不到；`white_border_m` 让原点落在白边外角（`t` 往标记 -X/-Y 各退白边宽）。
- **桌面大标记 vs 标定板字典冲突**：桌面标记 `DICT_5X5_50` 的 ID0-3 是标定板 `DICT_5X5_250` 的**子集**，图案逐像素相同 → 桌角大标记会污染 `detectBoard`，致 Enter 找不到板 / 外参旋转错（相机平面被「掰斜 60°」）。已用**几何校验**兜底（两标记相距≈桌面对角线 3.136m，容差 0.5m），标外参时盖住大标记；彻底解法重打 `DICT_6X6_*`。且错误结果会被按 T **写进 `table_extrinsics.yaml` 持久化**（「重新打开位置不变」不是缓存，是读同一个写错的 yaml）。
- **单标记弱深度**：8.4cm 标记在 3m 外 ~50px，深度误差放大 ~6%（十几厘米）系统偏差 → 桌面原点错，需三角化射线求交（每标记 ≥2 台相机看到才准）。

## 硬件 / 环境事实

- 相机：4 × 海康 **MV-CS016-10UM**，USB3，**黑白**（Mono8），Sony IMX273 全局快门 1.6MP，1440×1080。最大 **249.1fps @ Mono8**，单帧 1.483MiB，单台满速 ~369MiB/s ≈ **3.1Gbps（一个 USB3 口）**。`lsusb` 显示 `2bdf:0001 Hikrobot`。
- **触发同步**：信号发生器接相机 6-pin I/O **Line0**，实测出帧率 **100Hz**，四机同步误差 **~1µs 量级**（cam1/cam2 亚微秒；**cam3（SN DB1719717）最差** std 0.57~0.96µs、band 最大 3.2µs）。
- **时间戳（重要，SDK 文档撒谎）**：
  - `nHostTimeStamp` 实测是**毫秒级**（Unix epoch ms，分辨率 1ms），不是文档写的 µs；µs 级同步测量要用 `CLOCK_MONOTONIC` 在 `GetImageBuffer` 返回时自己打点。
  - 设备时间戳是**原始 tick，约 10ns/tick（100MHz 计数器）**，跨相机绝对大小不可比（各机基准偏移 ~1.49s/0.79s/2.28s），只能比扣除基准后的 jitter/std。
  - `nTriggerIndex` 恒为 0（未实现）；`TimestampReset` 节点不存在（报 `0x80000109` = `MV_E_GC_NODE_NOT_FOUND`）。
  - **硬触发跨脉冲抓帧伪误差**：串行抓 4 台 ~255ms 会让各机抓到不同脉冲，主机到达差被混入 ±10ms 整周期；唯一可靠法是对实测周期**取模对齐**剥掉整周期假误差。
  - 错误码：`0x80000203` = `MV_E_ACCESS_DENIED`（MVS 客户端独占相机时）；`0x80000109` = 节点不存在。
- **USB3 拓扑（关键）**：4 相机满速 ~12.4Gbps。芯片组控制器聚合 ~800-900MiB/s 且 PCIe 3.0×4 与 SATA/网卡共享，只能稳妥挂 2 台。**推荐 2+2 分开插**（2 台 CPU 直连 + 2 台芯片组）；后置口：`SS`=芯片组 5G、`VR Ready SS`=CPU 直连 5G、`SS 10`(白框)=CPU 直连 10G、Flash BIOS 旁=USB2.0（**别插相机**）。USB2 口会把相机拖到 ~25fps 且整组同步掉速。4 台重复复位（`usb reset` 10-20 分钟成组）是 USB 链路不稳信号，指向带宽争抢或 12V 供电波动。
- SDK：海康 **MVS V4.8.0**，装于 `/opt/MVS`。`.so` 在 `/opt/MVS/lib/64/libMvCameraControl.so`；Python 绑定 vendor 进 `camera/mv_import/`。关键环境变量 `MVCAM_COMMON_RUNENV=/opt/MVS/lib`（缺了 TypeError）、`LD_LIBRARY_PATH`、`MVCAM_SDK_PATH`。
- Python：base **3.14.6** 太新，**务必用 conda 3.11 环境 `tt`**（`/home/yby/miniconda3/envs/tt/bin/python`）。
- **相机参数持久化**：`config/camera_settings.json`（gitignored）保存曝光/增益/伽马，`core/config.py` 提供 `load/save/resolve_camera_settings`，优先级 **命令行 > 设置文件 > 默认（5000us/0dB/1.0）**。所有开相机脚本都从它读；**引入后 `cameras.yaml` 的 exposure/gain 段已成死代码**（只剩 trigger/pixel_format 有效）。
- GPU：**RTX 5080**（Blackwell sm_120，16GB，驱动 580/CUDA 13.0）。onnxruntime-gpu **1.26**（最后一个支持 CUDA 12 的版本，1.27 起切 CUDA 13）+ `nvidia-*-cu12` 运行库（CUDA 12.9/cuDNN 9.25）。`device` 默认 cuda，`backend` 支持 `tensorrt`（TensorrtExecutionProvider FP16，需 TensorRT 10.x，装 `tensorrt-cu12-libs==10.14.1.48`）。**CUDA 13 的 nvidia pip wheel 未发布**（PyPI 是 0.0.0a0 占位）。实测：YOLOX TRT 11.3→2.26ms、RTMPose TRT 3.9→1.08ms、单相机 detect 端到端 14.6→8.0ms。
  - **坑**：mmpose YOLOX 烤入 EfficientNMS，预 NMS TopK K=5000 超 TRT 上限 3840 → `_patch_yolox_for_trt` 改 3000。
  - **坑**：TRT 引擎构建首次 30~60s（动态 batch 曾到 103s），进度提示要用 **print 而非 logger.info**（默认 logging 级别 WARNING 吞掉提示，用户误以为卡死）；缓存命中后 session 秒开。
  - **坑**：onnxruntime provider 传 `get_available_providers()` 全量会静默走 TRT EP 建引擎 ~52s 卡主线程——要显式限定 provider + 模型创建/首帧热启动挪后台线程。
  - **坑**：RTMPose 的 `_trt_session` 若未配动态 batch profile（`trt_profile_min/opt/max_shapes` 1/4/8）+ 预热只喂 batch=1 → 人数 1→2 时 batch 4→8 触发 TRT 引擎重建 30~40s 卡死。
- **torch 训练栈（RTX 5080 用 CUDA 12.8）**：`torch==2.11.0+cu128` + `torchvision==0.26.0+cu128`。网络是国内环境：download.pytorch.org 被墙；装包前 `export http_proxy= https_proxy= ... all_proxy=` 清空代理（本机 SOCKS 127.0.0.1:7897 会让 huggingface_hub 崩溃）。torch 走 `--find-links https://mirrors.aliyun.com/pytorch-wheels/cu128/`（扁平目录不是 simple-index），其余 `--index-url` 阿里云 pypi。缺库用 `ldd libtorch_cuda.so` 枚举补齐（libcusparseLt/nccl/nvshmem/cupti/cufile，其中 cufile 正确包名是 `nvidia-cufile-cu12`）。triton 装不上不影响普通训练。**磁盘告急**：大 wheel 撑爆 `~/.cache/pip`（曾 7.1G 触发 Errno 28），先 `pip cache purge`。
- 网络/数据集备选：公开乒乓球数据集（Roboflow `yolov8bigdataset`、Kaggle `ketzoomer/table-tennis-ball-position-detection-dataset`、HF `weslien/topspin-opentt-ball-subset`——HF 是 parquet 后端不能直接 clone 图、且 CC BY-NC-SA 非商用）；本机 SOCKS 代理对 httpx 崩溃，最终用户用自己录的数据。

## SDK 用法（已踩平的关键点）

### import 机制（`camera/sdk.py` 里集中处理）
1. MVS 绑定在 import 时执行 `check_sys_and_update_dll()`，用 `os.getenv('MVCAM_COMMON_RUNENV') + "/64/libMvCameraControl.so"` 拼路径，**import 前必须 setdefault 该变量**（sdk.py 已做）。
2. `MvCameraControl_class.py` 内部是绝对 import，**mv_import 目录必须先加进 sys.path**（sdk.py 已做）。
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
- 每相机一个线程 `MV_CC_GetImageBuffer(stFrame, timeout_ms=1000)`。
- 成功且 `pBufAddr` 非空：`ctypes.string_at(pBufAddr, nFrameLen)` → `np.frombuffer` → `.copy()`（**必须 copy**，SDK Free 后复用缓冲）。
- 有界队列 `queue.Queue(maxsize≈10)`，满则丢最旧。回调式受 GIL 影响大，不推荐多机实时用。
- `--no-display` 下别在循环里 `drain()` 追生产速度清队列（会 688% CPU）；有界队列本来丢最旧帧。

### 触发（信号发生器，`camera/trigger.py`）
外部触发接 **Line0**，标准 GenICam 节点串：`TriggerMode=On` / `TriggerSource=Line0` / `TriggerActivation=RisingEdge` / `TriggerDelay=0` / `TriggerCacheEnable=False` / `LineSelector=Line0` + `LineDebouncerTime=50`(us)。软件触发调试：`TriggerSource=Software` + `SetCommandValue("TriggerSoftware")`。连续采集：`TriggerMode=Off`。

### 图像参数句柄（`camera/parameter.py`）
所有参数是 GenICam 节点，字符串 key + `MV_CC_Set/Get{Float,Int,Enum,Bool}Value`。**实测 MV-CS016-10UM 没有 `BlackLevel`/`Brightness`/`Contrast` 节点**（读回 None），实际可调只有 `ExposureTime`(15µs~10s)、`Gain`(0~17dB)、`Gamma`(0~4)。黑白相机无白平衡/饱和度；调亮度=曝光+增益+伽马。Get 返回 `MVCC_FLOATVALUE{fCurValue,fMax,fMin}` 自带范围。枚举用 `SetEnumValueByString`。

### 像素 / 帧结构
- 黑白用 `Mono8`（值 17301505），帧缓冲 = H×W 字节。
- `MV_FRAME_OUT.pBufAddr` + `stFrameInfo`（`MV_FRAME_OUT_INFO_EX`）含 `nWidth/nHeight/nFrameNum/nDevTimeStampHigh/nDevTimeStampLow/nHostTimeStamp/nFrameLen/enPixelType`。设备时间戳 = `(nDevTimeStampHigh << 32) | nDevTimeStampLow`（10ns/tick，见上）。

## 约定

- 分层：`core` → `camera` / `vision` / `calibration` / `reconstruction` / `visualization`。下层不依赖上层，跨层共享类型放 `core/types.py`。无 `pipeline/` 包，脚本式编排。
- 脚本用 `sys.path.insert(0, "<project>/src")` 引导 import；测试用 `tests/conftest.py`。
- 相机模块所有对 SDK 的 import 收敛在 `camera/sdk.py`。
- 视觉检测器走**注册工厂**：`vision/detector.py` 的 `register_detector(kind, factory)` / `create_detector(kind)`。已注册 `pose`（RTMPose-l-halpe26 + yolo11n-gray 人检测）、`table`（`TableDetector.load_default()` 自动加载标定）、`ball`（经典）、`ball_yolo`（YOLO）。未注册返回 None，调用方提示「接口已定义、算法待实现」。
- 每个模块目录都有 `README.md`（作用/用法/未完成），改完模块同步更新。

## 踩坑记录 / TODO

### 时间戳 / 同步
- [x] `nHostTimeStamp` 是 ms 非 µs；设备时间戳是 10ns/tick 原始计数；`nTriggerIndex` 恒 0；`TimestampReset` 不存在——同步测量靠 CLOCK_MONOTONIC + 取模对齐。
- [x] **离线跨相机对齐用「脉冲号」**：设备 ts 只在单机内部相减、丢几拍加几号，再按主时钟脉冲号 searchsorted 同号帧（`reconstruction/video_source.py`，绝对 ts 不可比）。见「录像 + 离线重建」节。
- [ ] 实时按 `device_timestamp` 最近邻组帧仍待 pipeline 阶段；当前 `get_latest_bundle` 取各机最新帧，外参标定用 `get_synchronized_bundle`（清队列+各取下一帧）。

### 相机 / 环境
- [ ] **Python 3.14 太新**：装 open3d/torch 前先建 3.11 环境 `tt`。
- [ ] **udev 权限**：枚举为 0 大概率是 USB 无权限，装海康 udev 规则或 `sudo chmod`。
- [ ] **USB3 带宽**：4 × 1.6MP 高帧率撞带宽墙；2+2 分开插、必要时降帧率/TriggerDelay 错峰。
- [x] `config/camera_settings.json` 共享参数（`cameras.yaml` exposure/gain 已失效）。

### 标定
- [x] OpenCV 5.0 ChArUco 新 API、IPPE 坑、DICT 子集冲突、4 标记桌面定位——见「标定工具的归属」节。
- [ ] cam_3 内参重投影 RMS 0.56px 偏高（远端 12px 处精度最差），可考虑重标；标定尺度基准（内参焦距/板格边长）偏导致 ~5.8% 尺度误差。

### 视觉 / 球
- [x] 球检测三路线 + 训练管线 + TRT 缓存哈希分目录 + 后台线程加载——见「球检测」节。
- [ ] **960 重训**（~3.5h，`train_ball_gray.py` imgsz=960）：唯一能兼得 ~145FPS + 可靠检测的办法。
- [ ] 录制 100fps 未做（PNG 编码 26ms/帧是瓶颈，改 JPG + 去预标注 + 录制剥离主循环）。
- [ ] 多球检测未做（`label_ball.py` 只支持单框、跨视角多球关联未实现）。
- [x] conf 阈值不匹配（detector 0.25 vs `min_conf` 0.3）→ `min_conf` 降到 0.15，靠重投影/交会角兜底。
- [x] 卡尔曼 dt 写死 0.01 假设 100FPS 致快速球误判冻结 → 用实际帧间隔，最终整体移除卡尔曼走逐帧纯 DLT。

### 视觉 / 姿态
- [x] 人检测 yolo11n 灰度（`det="yolo11n-gray"` 默认）+ 三处提速 + conf 0.5 + float32 归一化——见「人检测」节。
- [x] RTMPose 预处理瓶颈是**归一化**（uint8→float64 占 3.35ms 的 60%），不是 warp（0.09ms）；float32 就地算省 ~1ms。
- [ ] RTMPose `_trt_session` 加动态 batch profile(1/4/8) + 预热 batch 8（避免人数变化触发引擎重建）。
- [ ] YOLOX decode+NMS 上 GPU（`_yolox_decode_batch` numpy CPU 3.7ms → <0.5ms，零精度损失）。
- [ ] 姿态最大机会：Fixed ROI（半固定机位砍掉整个 YOLOX 段 ~12ms + 删 match_people）未做；RTMO one-stage 候选。
- [x] YOLOX 动态 batch 重导出（`export_yolox_dynamic_batch.py` 纯 PyTorch 重建，输出 `(B,3549,85)` 不烤 NMS）；三个坑：不烤 /255（humanart 训练吃 0-255）、decode 用 cell 左上角（center=(delta+grid)×stride 不加 0.5）、TRT profile min1/opt4/max8。
- [ ] 录像离线重建目前每相机取最高置信度一人（`reconstruct_video.py`，与 live_control 在线一致）；多人离线（match_people 或逐人 EmFit）未串。

### 录像 / 重建相关已踩坑
- [x] `np.savez_compressed` 返回 None（无 `.close()`）；编解码实测 mp4v≈8ms 可用、MJPG 太慢、h264 无 v4l2 打不开——见「录像 + 离线重建」节。
- [ ] 录像 meta/fps 头为标称值；若以后要按真实触发率校准视频时间轴，读 `cam{cid}_ts.npy` 中位差即可（离线已如此，见 video_source.period_sec）。

### 可视化
- [x] `set_front` 是「lookat→相机」方向（front=[1,1,0.9] 俯瞰）；translate +y 才是上（W/S 方向）。见「可视化」节。

### 球 / 三角化
- [x] 三角化向量化（`triangulate_batch`，姿态 12×、球持平）+ `mean_conf` 只对有效视角取均值——见「三角化」节。
- [x] 单球无需跨视角匹配；依赖只有 numpy+opencv（无 scipy/torch/filterpy）。
- [ ] 精度预算：相机距桌面 3~6m、球 12~24px，mm/px ≈ 1.7~3.3mm；合成 0.3px 噪声下 3D 球中位误差 ~0.68mm，姿态 cm 级。毫米级靠「短曝光(≤100µs)+补光+亚像素精修+4 视角过定 DLT」。

### 训练 / 数据
- [x] 训练管线（录制→标注→归一化→增强→训练→导出→fix h/w）；yolov8n mAP50 0.961、yolo11n-gray 0.9677。
- [x] 增强旋转方向：cv2 旋转矩阵是 `[alpha beta; -beta alpha]`（y-down），与数学 CCW 相反；光度增强对 12~24px 小球要保守（强模糊/伽马/噪声会抹掉球）。
- [x] ultralytics `dynamic=True` 导出 h/w 全动态 → TRT EP 静默回退 CUDA 不建引擎 → `fix_onnx_dynamic.py`（`--ch` 参数支持 1 通道）固化 h/w 仅 batch 动态。

### GPU / 部署
- [x] TensorRT 全链路（姿态 YOLOX+RTMPose、球 YOLO）；onnxruntime-gpu 1.26(CUDA12)；TopK patch；引擎缓存哈希分目录。
- [ ] 实时 GPU 争用待确认：若 detect_batch 实测 ~14ms（而非基准 6.9ms）说明 Open3D 渲染和 TRT 抢 GPU，需调低 3D 渲染频率或 `nvidia-smi -lgc` 锁频消降频抖动。
- [x] 性能优化全景见 `docs/optimization_report.md`（球 4 项 + 姿态 3 项，12FPS → 40~90FPS 路径）。
