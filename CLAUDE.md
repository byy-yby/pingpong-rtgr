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
  离线 `reconstruct_video.py` 走 **ViTPose**（`vision/pose/vitpose_pose.py`，coco_25 25 点含脚
  → 重排成 halpe26），默认 **`vitpose-h-coco_25`**；见「离线 2D 模型」节。
- **球检测** `vision/ball/` —— 三条路线都实现：经典 CV、YOLO（yolov8n）、灰度 yolo11n（1ch）。
- **球 3D 重建** `reconstruction/ball.py` + `scripts/reconstruct_ball.py` + live_control 按 b —— 三角化 + 红球渲染。
- **姿态 2D→3D 重建** `reconstruction/triangulate.py`（置信度加权多视角 DLT，已向量化）+ `associate.py`（跨视角匹配）+ `person_track.py`（**阶段 A–D 多人跟踪/身份**，宽视野一相机多人，见「多人跟踪」节）。`pose_track.py::PoseTracker` 是旧的单相机时序身份跟踪，已被 person_track 取代。
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

### 离线 2D 模型 + 拟合置信度阈值（P0 落地，2026-09-10）

**离线不追推理速度 → 默认换最强 2D 模型**。`scripts/reconstruct_video.py --pose-model`
默认 **`vitpose-h-coco_25`**（`vision/pose/vitpose_pose.py`，coco_25 25 点含脚 → halpe26）。

**端到端总账（3 段视频，脚本自报「重投影误差中位」）**：改前 b+fit-conf0.15 均值
**13.61px** → 只改 fit-conf0.5 **13.70px（+0.09，没变好）** → 再换 vitpose-h
**11.59px（12.94/14.60/13.30 → 11.27/11.96/11.54，−15%）**。
即**降重投影误差基本全靠换 2D 模型，`fit-conf` 贡献 ≈0**（它换的是稳定性）。
- 模型放 **`/mnt/newdisk1/vitpose/`**（新硬盘），项目内软链 `data/weights/vitpose`。
  档位 b/s 单文件；**l = 1.23GB 单文件；h = 226KB 图 + 394 个外挂分片同目录（~2.55GB）**
  ——分片靠相对路径解析，**别单独拷 .onnx**。`vitpose_pose.py::_download_h` 从
  `JunkyByte/easy_ViTPose` 拉（走 HTTP 代理 `http://127.0.0.1:7897`；SOCKS 会让 hf_hub 崩）。
- **跨模型公平 A/B**（`scripts/error_budget/ab_pose_model.py`，同一批 2D 点、除模型外参数全同，
  两段视频）：h/l 比原默认 vitpose-b 重投影中位 **−0.77/−0.92px** 与 **−1.09px**，
  收益集中在下身/脚（−1.4~−2.0px）；**h vs l 头对头**：一段打平（−0.03px）、
  一段 h 好 0.79px。检测段耗时 161014：l 254.9s / h 323.5s（2015 帧）。
- **根关节抖动（2D-free 指标，2 段合并）**：b→c50→l→h 的 p1 max = **14.79→7.68→3.69→3.86cm**，
  p1 中位 0.32→0.28→0.25→0.22 —— **换模型对追踪稳定性的贡献比调 fit-conf 更大**。
  「根关节」= 骨盆（body25 #19，3D 骨架锚点）；「跳变」= 相邻帧（100fps ⇒ 10ms）重建出的
  骨盆位移，人走路 10ms 也就 ~1cm。它**不经过 2D 置信度、不当分母**，所以没法靠改口径刷好看。
- 要速度用 `--pose-model vitpose-l-coco_25`（精度基本持平、快 ~2.8×）。

**`--fit-conf` 默认 0.15 → 0.5**：低置信度（遮挡外推）关节不参与 SMPL 拟合。
它**不改「重投影误差中位」的分母**（掩码是硬编码的 conf>0），实测对脚本自报中位是
**+0.03~+0.14px（略差）**——它换的是稳定性不是精度。**必须用固定观测集评**的是那些
「把观测 conf 置 0」的开关（`--consensus-sigma`、下半身门）：掉出分母 = 定义变窄。
`ab_eval.py` 用同一批 (帧,人,视角,关节) 同时算 conf>0.5 与 conf>0 两组。
8 配置 × 3 段视频（5462 帧）：c50 在 fit/outfit/conf>0 四个口径上**全部不劣**，
根关节 p1 最坏跳变 **14.79→7.68cm（−48%）**。ViTPose conf 中位 0.87~0.90，0.5 不误杀。

**多视角 3D 共识降权（`reconstruction/obs_filter.py`，默认关）**：Cauchy
`w=1/(1+(r/σ)²)`，r = 2D 观测与阶段 D 鲁棒 3D 的重投影距离，> `--consensus-max` 直接丢。
**在 b 和 h 两个基线上各测一遍，结论一致：脚本自报的重投影中位会掉 ~1.1px，但那是分母
变窄**（不拟合 30px 外的观测 ⇒ 它们也不进误差统计）——固定观测集上拟合视角反而 **+1.3px**，
2D-free 的根关节抖动机一好一坏（p1 max 3.86→3.44 好、p0/p1 中位 0.15/0.22→0.19/0.27 差）。
故默认 `--consensus-sigma 0`；要「压住最坏那一下跳变」时才开（代价是典型帧变糙）。
`--save-pass1/--load-pass1` 可缓存 Pass 1 观测秒级重跑拟合。
权威数字见 `docs/error_budget_report.md` §6/§7/§8。

### 三角化：已向量化

`reconstruction/triangulate.py::MultiViewTriangulator`：
- `triangulate_batch`（姿态/球统一走它）——26 关节堆成 `(26,8,4)` 一次性 4×4 gram `np.linalg.eigh` 最小特征向量（等价 DLT 最小奇异解），重投影/交会角/置信度全向量化。姿态 **6.5→0.43ms/人（~12×）**，球 0.33ms（单点无回归）。
- 最差视角重投影 >12px 时逐点回退到 `triangulate_point`（保证与旧输出一致，坐标差 1e-12mm 级）。
- `mean_conf` 曾用 `conf.sum()` 把遮挡相机置信度也算进去 → 改只对有效视角取均值（差分测试抓到的真 bug）。
- 单球无需跨视角匹配（每相机最多 0/1 检测，直接 `{cam_id: Ball2D}` 喂三角化）；`associate.py::match_people` 只用于多人姿态。

### 多人跟踪 / 身份：`person_track.py`（阶段 A–D，2026-09-08 起默认）

换广角镜头后**一台相机能同时拍到近/远两人**，「每相机只拍一个人」的旧前提失效，
`match_people_fixed`（固定分组 `[[0,2],[1,3]]`）逐帧挑错致 3D 骨架身份闪跳。
现走 `MultiPersonTracker.step(frame_idx, frames, time_s)` 四阶段（离线
`reconstruct_video.py --assoc tracker` 默认、在线 live_control 同一实现；
`--assoc fixed` 保留旧路线回退）：

- **阶段 A（单相机局部跟踪）**：每相机独立，IoU + 质心最近邻接本帧检测到该相机
  已有 `local_track_id`（命中重置 `missed_frames`）；新检测开新 id，漏检轨迹
  `missed_frames += 1`，超 `LOCAL_TRACK_MAX_MISSED=10` 删除。**ROI 引导重检测**
  只对「漏检的活跃轨迹」触发：用其卡尔曼预测 3D 位置投到该相机
  （**`project_distorted`**，原始像素）开小窗，边长 = 该人最近 bbox 对角线 ×2
  （缺失时 300px），窗内以 `ROI_REDETECT_CONF=0.15` 重跑人检测；命中→全图坐标 +
  `source="roi_redetect"`，未命中→`source="predicted_only"`（**不参与三角化**）。
  每 `FULL_DETECT_INTERVAL=30` 帧强制全图检测。同一相机本帧的多个小窗结果**合并成
  一次** `LocalCameraTracker.update`（逐轨迹分别调会把别人的局部轨迹当漏检反复计数）。
- **阶段 B（世界系 3D 卡尔曼）**：`x=[p,v]∈R⁶` 匀速模型，观测 = 本帧鲁棒三角化的
  **根关节**，`R` = 三角化协方差 + 基础观测噪声（上限 `obs_max_std`）。
- **阶段 C（跨相机身份关联）**：人少时**枚举**分配方案，代价
  `E = Σ_轨迹 Σ_分配到该轨迹的相机 w_cam·‖重投影根 − 检测中心‖²`，
  `w_cam = bbox_conf × mean(kpt_conf)`；只有 `E(best) < E(上一帧)·(1−SWITCH_MARGIN)`
  才切换（`SWITCH_MARGIN=0.2`），抑身份抖动。
- **阶段 D（逐关键点鲁棒三角化）**：`triangulate_point_robust`（RANSAC + 加权 DLT）。

**三条实测踩过的红线**（有回归测试锁定，改前先看）：

1. **绝不用「卡尔曼预测框」当伪检测**。非全图帧曾把预测框喂人检测器 → 图像里未必
   有那个人，姿态凭空生成 → 三角化又「确认」预测 → **3D→框→姿态→3D 自激回路**，
   实测根关节 10 帧漂到 1.5m 外、z 到 −0.58m（地下）。现在帧间观测**只**来自 ROI
   小窗里真实图像上的 YOLO 检测（`dets[cid]=[]`，见 `step()`）。
2. **根关节绝不回退到四肢**。骨盆（halpe26 #19）三角化失败 → 用**双髋中点**合成
   骨盆（同语义点，不跳变）；双髋也没有 → `root_index=-1`（本帧不更新观测、纯预测、
   `misses+=1` 触发全图检测）。曾用「置信度最高的关节」兜底 → 选到**右手腕(10)** →
   卡尔曼观测瞬移 0.5m、速度反向（`triangulate.py::triangulate_pose_robust` 里的
   `_HIP_INDEX=(11,12)`）。
3. **畸变口径别混**：`MultiViewTriangulator._dlt/reproj/project` 吃**无畸变**像素；
   ROI 开窗、`anchor_2d`/`box_offset`/`last_bbox` 是**原始畸变**像素
   （`project_distorted`）。用错会致 ROI 偏出真实人（本机 1440×1080 畸变 ~20px 量级）。

**RANSAC 兜底（阶段 D 补丁）**：N≥3 时旧逻辑要求「某视角对 ≥2 内点」，但真实像素噪声下
两个好视角本身可互差 ~8px → **一对都满足不了 → 整条关节被丢弃**（实测 20260908_161147
f17/f18 骨盆：c0/c3 互差 8.3/6.9px 一致、c1 偏 65px）。现退化取「最优视角对」——
两视角残差最大值最小、且 ≤2×阈值（`2*ransac_thresh_px`=16px）才接受，随后正常精修。

**实测（session 20260908_161147，200 帧 2 人，4 相机）**：`root_index==19` 196/199 帧
（其余 `-1`）、相邻帧根跳变中位 **0.40/1.96cm**、max 5.0/12.1cm、**>30cm 的帧数 0**
（伪框时代是 29.8~33.7cm）；观测来源 **100% `roi_redetect`/`detect`**（无 `predicted`）。
残留的大骨盆残差集中在 c1/c2 的 24~58px——是**姿态质量**问题（人在桌边、下半身被桌子
挡住），不是追踪/标定：同一人在空旷处走（f150）四相机骨盆一致到 3.3~5.8px。
`roi_gate_px=120, roi_gate_ratio=0.0` 为当前默认（**注意**：早期 A/B 曾判它「更差」，
那次对比被伪框路径污染；新架构下 PX120 的跳变统计与 c0 骨盆残差都更好）。

**下半身可信度门（`lower_body_gate`，默认开，2026-09-09）**：隔球桌看远端的人时，
ViTPose 对膝/踝/脚尖/脚跟是**外推**出来的点（同视角上半身 kp 置信 ~0.9，下半身中位
0.24~0.40），旧版照样喂进三角化，把 SMPL 的腿整体拉歪。现在每视角逐帧判
`lower_body_unreliable`（下半身中位置信 < `lower_body_conf_ratio=0.5` × 上半身中位
**且** < `lower_body_conf_abs=0.5`），命中则 `mask_lower_body` 把该视角的
`LOWER_BODY_HALPE26=(13,14,15,16,20,21,22,23,24,25)` 置信置 0 再进阶段 D；
**髋 11/12/19 绝不在掩码内**（阶段 B/C 的根关节靠它，且髋几乎不被桌子挡住）。
`TrackFrameResult.raw_obs` 保留原姿态供 2D 叠加显示、`lower_body_masked` 记命中视角。
CLI：`--no-lower-body-gate` / `--lower-body-conf-ratio` / `--lower-body-conf-abs`，
以及 `--upper-body-only PID [PID ...]`（整段只用上身，连好视角的腿也丢）。

实测（session 20260908_161147，719 帧 2 人）：门命中 p0 c1×568帧/c3×261帧、
p1 c0×398帧/c2×289帧/c1×9帧。**别拿「重投影误差 18.70→13.27px」当证据**——那个指标
只统计有效关节，掩码后坏视角的下半身压根不进分母，是假提升。同 2D、同关节集拆开比
才是真的：上身 9.5→9.7 / 12.1→12.2px（无损伤）、**下身·好视角 p1 20.4→14.8px**
（p0 10.8→10.4）、下身·坏视角 49.8→51.4 / 50.7→62.1px（**故意变差**：不再追那些
错点——它们与好视角的 3D 共识本就差 50~60px）。

**「远端连髋也丢」是开关不是默认（`mask_hips_when_unreliable`，默认关，2026-09-09）**：
用户要求「远端的髋也不要，就要上半身」，但实测**远端的髋是全视角里最准的**——同 2D、同
口径拆组（`HIP25={8,9,12}`）：髋·远端 5.7px=**3.5cm**、置信 0.84，髋·近端 13.8px=5.5cm、
置信 0.86；上半身也是远端更准（4.0px=2.5cm vs 11.5px=4.6cm）。真正烂的是**膝以下**
（踝 27cm、脚 42~49cm，置信 0.29），已由下半身门覆盖。丢髋的代价是**削掉拟合的根约束**
（阶段 B/C 的根关节就靠髋），A/B（240 帧，同配置只切该开关）实测追踪侧全面退化：
`root_index==19` 帧数 235→226 / 232→**192**，骨盆内点数 2.49→1.95 / 2.07→1.61，
p1 相邻帧根跳变 max 12.07→**28.32cm**。故**默认关**，开关留给现场 A/B：
`TrackConfig.mask_hips_when_unreliable` / `--mask-hips-when-unreliable`，
`mask_lower_body(pose, include_hips=True)` 是底层实现（`HIP_HALPE26=(11,12,19)` /
`HIP_BODY25={8,9,12}`）。`--upper-body-only PID` 语义就是「只要上半身」，**已改成连髋一起丢**。
端到端 A/B（同 240 帧，只切该开关）：报告口径重投影中位 11.46→**11.64px**（略差）、
髋·远端 3.2→5.0cm（**故意变差**，不再拟合那些点）、**留出视角**（p0 的 c3 / p1 的 c2）
上身 8.7→8.3 / 14.5→14.7cm、髋 4.6→5.0cm、下身 57.4→54.4cm——即丢髋**没有**换来更好的
3D，只是少了一个根约束。结论：默认关。

**「某相机某帧没框」的成因（实测 515 个「人×相机」帧无框）**：A 预测位置投影出画幅
151（那人不在该相机视野里）· C 全图只看到**另一个人** 300（被球桌/另一人挡住或太小）·
D 全图也没检出人 28 · **B ROI 小窗漏检 36（7%，唯一可修的）**——全图能检出、就在预测
位置 200px 内（中位 110px），却被 ROI 门限拒掉（`roi_gate_px=120` 距离门 + 「框心必须
落在小窗内」+ 尺寸门）。**预测绝不填框**（红线①）：预测只用来开搜索窗，凭空造框会让
图像里没人的地方长出骨架。故 2D 叠加里没框的相机画**灰色虚线 `pred pN` 框**
（`obs2d.save_pred_boxes` → `pred_boxes.json` → `overlay2d.draw_pred_box`，**纯显示、
重建不读**），让「这帧这台相机没观测」一眼可见、又不会误认成实测框。

**SMPL 拟合视角要去掉「画面边缘只露半截」的相机**（`person_track.select_fit_views`，
`reconstruct_video.py` tracker 模式调用）：某人只在某相机边缘露出一条时，人检测框被
边界裁掉一半，姿态模型对看不见的另一半**外推**出完整骨架（中位 kp 置信 0.21/0.34、
bbox 高仅 128~168px vs 正常 190~405px，投回该视角差 95~290px），喂进多视角拟合会明显
拉偏。判据：该相机上此人 bbox 贴边（任一边距边界 ≤3px）的帧数 ≥50% 出场帧即丢弃；
筛完不足 2 台则回退出场最多的 2 台。实测（20 帧 batch）：p0 丢 c3、p1 丢 c2 后
重投影误差中位 **36.7px → 18.7px**，且每人仍有 3 视角（`--assoc fixed` 只用 2 视角
得 10.5px，但那是欠约束的 2 视角拟合，不是更准）。

### 可视化 `viewer3d.py`（Open3D）

- 默认视角：`up=(0,0,1)`（Z 竖直）、`set_front(front=[1,1,0.9])`——**`set_front` 传的是「从 lookat 指向相机」的方向**，`[1,1,0.9]`（+Z 朝上）= 相机在 +X+Y+Z 斜上方俯瞰球桌（此前 `[-1,-1,-0.9]` 会让相机跑到桌面下方仰视）。R 复位即回此视角。
- 键盘：`VisualizerWithKeyCallback` + **W/A/S/D 平移、方向键旋转、+/− 缩放、R 复位**。**translate 的 +y 才是「向上」**（W=`translate(0,+step)`、S=`translate(0,-step)`，曾写反）。
- 图层：相机视锥（`build_cameras_scene`）、球桌、骨架（`add_skeleton_layer`/`set_skeletons`）、红球（`add_ball_layer`/`set_ball`，跨线程传球心加锁）。
- 每个方法里自己 `o3d = _o3d()` 惰性 import（`_update_ball_geometry` 曾漏写致渲染线程 NameError 窗口退出）；Open3D 窗口必须在主线程开，后台线程只加载模型。

### 离线重建回放 `recon_player.py` + `scripts/visualize_recon.py`

- `scripts/reconstruct_video.py` 每帧存 `frame_NNNNNN.npz`（vertices/joints/…）时**多写一次 `recon_faces.npy`**（13776×3 SMPL 拓扑，整段一次，`ensure_faces`）。
- `scripts/visualize_recon.py <recon目录>` 弹出 Open3D 新渲染器窗口逐帧回放（`--watch` 重建进行中追帧；`--render T out.png` EGL 出 PNG；`--root` 指定含标定的项目根）。t 是主时钟帧号：no_person/失败帧清空人体，被 `--stride` 跳过帧保持上一姿态（`ReconTimeline`）。
- **「重投影误差」怎么读（2026-09-10 起两口径、中位与平均都报）**：口径 = 每帧每人取
  **拟合视角**（已剔裁边）里 conf>0 的关节，SMPL 的 body25 关节投影 vs 2D 观测的像素距离。
  `_proj_err_multi` 一次算两个口径（各自走下面的三层聚合成**每帧一个数**，再对帧取
  中位/平均/p90，写进 `recon_index.npz` 的 `err_mean_px`/`err_best_px` 与 `recon_meta`
  的 `reproj_err_{mean,best}_px_{median,mean,p90}`）：
  - **all（全部视角）**——所有观测视角**等权**，就是拟合目标函数的口径。
  - **best（最高置信视角）**——**每个关节只取置信度最高的那台相机**再平均。低置信度视角的
    2D 本就是姿态模型**外推**出来的（置信 0.9-1.0 的关节实测 2.2cm、0.3-0.5 的 14.9cm，
    见 error_budget_report §1），拿它们当基准会污染指标；best 更接近「和 Ground Truth 比」。
    **但它只是更好的代理、不是真值**：conf 是模型自报的，高分但位置错的检测照样进分母。
  **两个口径必须都看**：合并 5455 帧 all 的平均/中位比 **1.11**、best 只有 **1.02**——
  all 的长尾几乎全部来自低置信度视角。**被掩码的下半身与裁边视角都不进分母，是自洽性不是真值**。
  **算式的三层聚合别记错**（`_proj_err_px` @136 + 调用点 @663/699）：
  ① 一颗「钉子」= (帧, 人, 相机, 关节) —— SMPL 的 25 个 body25 关节（世界系米）用
  `P=K·[R|t]` 投回像素，与该帧该相机 2D 观测的欧氏距离（px）；**只统计 conf>0 的钉子**
  （掩码在 `_obs_to_body25_fixed(..., 0.0, ...)` 里硬编码）；
  ② 每帧每人把所有钉子取**均值**；一帧多人再对**人取均值** → **每帧一个数**；
  ③ 全段几百上千帧的这串数取**中位数** = `recon_meta.reproj_err_mean_px_median`。
  **中位数与平均数都要报**：中位数 = 「典型的一帧」（一半帧比它好、一半比它差），不看长尾；
  平均数 = 把长尾算进去（最能反映平均每帧吃多少误差）。**两者比值就是长尾的度量**。
  > 🚫 别把中位数当唯一权威。少数被遮挡/漏检的帧误差可达均值的几倍，光看中位数会漏掉它们。
  **分母能不能刷**：把观测 conf 置 0 的开关（`--consensus-sigma`、下半身门）会让这些钉子
  掉出分母 ⇒ 数字变小可能只是定义变窄（共识降权实测如此，见 error_budget_report §8）；
  **`--fit-conf` 不属于这一类**（它只改拟合输入、不改分母，对中位的影响是 +0.03~+0.14px）。
  标尺（20260908_161147）：全体中位 **4.97mm/px → 13px ≈ 5cm**（近端视角 3.6~4.2mm/px、
  远端 5.7~6.2 → 同一 px 值物理误差差 1.7 倍，**px 跨视角不可比**）。
  **地板** = 同一批 2D 直接逐关节三角化后的跨视角一致性：中位 6.0px≈2.6cm（还被最小
  二乘吸收 ~30%：3 视角 6 方程 3 未知 → 残差≈0.71σ，真实分歧 ≈8.5px≈3.7cm）→ 拟合
  13px 只有地板的 **~1.5×**。2D 检测器自身抖动单轴 σ **0.4~1.1px/帧** → 13px 不是检测
  噪声，主要来自「同一关节在不同相机上定位不同」（标定/遮挡/姿态模型偏差）。分人
  8.1px(3.6cm)/12.6px(5.2cm)、p90 11.5/18.5cm；最差关节 骨盆中 7.1cm、R髋 5.9、R踝 5.4，
  最好 L肩 2.7、眼 2.8。**结论：偏大但正常，对姿态够用（肩宽 40cm 的 1/8），对球远远不够。**
- **误差在帧间的分布（2026-09-10，3 段 5455 帧，h + fit-conf 0.5，`reproj_stats.py`）**：
  all 中位 **11.60** / 平均 **12.89** / p90 15.46 / p99 37.23 / max 182.88px；
  best 中位 **8.87** / 平均 **9.05** / p90 10.59 / p99 13.06px。
  **⚠️「视频开头/结尾动作怪 → 误差大」这个猜测被证伪——开头反而是最好的一段**：
  合并 2s 头 **10.71** < 中间 **11.62** < 尾 **15.00**（单段 10 等分表见下）；
  161014 头 10.92/中 11.25/尾 12.14、161102 头 10.75/中 11.86/尾 19.62。
  **误差也不随动作剧烈度上升**：按根关节逐帧位移分箱基本是平的（0-0.2 cm/帧 11.1~11.8、
  0.2-0.5 → 11.4~12.1、2-5 cm/帧 14.2~14.9 但只有 22~55 帧）；只取「稳定帧」
  （<0.5cm/帧，占 77%）all 中位 **11.53** ≈ 全体 11.60 → **动作稳定与否不是误差的主因**。
  **长尾是少数帧的、不是首尾整段的**（all p99 37px vs 中位 11.6px；最坏单帧 161147
  frame 210 all=182.9px）。**best 口径下长尾基本消失**（同一批帧 best max 仍 179.9 但 p99
  只有 13.1）⇒ 坏帧的成因是**某个视角的点偏了**，不是整具身体拟合崩了。
  10 等分/首尾/稳定帧/最差帧一键出：`scripts/error_budget/reproj_stats.py <recon目录>`。
  **别用 `pose2d.json` 复算**——它存的是**未掩码**原始姿态，复算会把下半身门外推的点算进来
  （实测 161147 all 中位 11.54 → 虚高 15.2px），必须读 `recon_index.npz`。
- **球的重投影误差（`ball_reproj_errors`，2026-09-10）**：同样两口径、同样报中位/平均，
  但**含义和姿态不一样**——球的 3D 由各视角 2D **直接加权 DLT** 得到、**没有拟合步骤**，
  所以这个误差就是**三角化自身的残差**（跨视角一致性），**不含「和真值差多少」**。
  参与视角判定与 `triangulate_ball` 一致（`conf>=min_conf` 且去畸变成功）。写进
  `ball_trajectory.npz` 的 `reproj_err_all`/`reproj_err_best` + `ball_meta.json`。
  `reconstruct_video.py <session> --ball-only --ball-model runs/detect/ball_gray/weights/best.onnx
  --out <同一个 out 目录>` 只补跑球、不动姿态结果（球很快：719 帧 18s、2728 帧 70s）。
  实测（三段，yolo11n-gray@1280，`--ball-min-conf 0.15`）：

  | session | 三角化成功 | all 中位/平均 | best 中位/平均 |
  |---|---|---|---|
  | 161147 | 467/719 (65%) | 3.61 / **164.88** | 2.15 / 261.42 |
  | 161014 | 1508/2015 (75%) | 3.69 / 4.72 | 2.08 / 2.93 |
  | 161102 | 2146/2728 (79%) | 4.31 / 4.82 | 3.19 / 3.41 |

  **161147 的平均值是坏的（164.88 vs 中位 3.61，比值 45.7）**——6 帧（1.3%）出现
  1e4~2.8e4px 的残差把均值拉飞。根因已定位：**cam0 有个静态误检**，球心恒在像素
  (319,226) 一动不动（`ball2d.json` 里 6 次出现全在同一像素，同一时段 cam2 在
  (872,836)→(880,881) 平滑跟球），于是「真球 + 误检」两条**根本不相交**的射线被
  DLT 硬解出一个点。已排除数值退化：那两条射线的 gram 条件数只有 51，朴素 SVD 与
  批量路径给出同样的烂解 ⇒ **是观测本身矛盾，不是求解器坏了**。
- **读球的数必须先按「参与视角数」拆开**：2 视角**欠约束**（4 方程 3 未知，DLT 总能
  给出一个点，哪怕两射线不相交）⇒ 它的**中位数反而最小**（161147 2 视角中位 1.67px
  vs 4 视角 3.60px），只看中位数会得出「视角越少越准」的错觉。只取 **≥3 视角**时
  三段一致且干净：all 中位 3.63~4.40、平均 4.16~4.86、p99 14.9~24.5px、max 25~64px。
  `reproj_stats.py` 已内置这张按视角数的表。
  ⚠️ **球不要照搬姿态的「best 更接近真值」结论**：姿态的 3D 来自带强先验的 SMPL 拟合，
  挑最高置信视角能甩掉模型被坏观测带偏的部分；球的 3D 是对 2D 的最小二乘插值，两个口径
  都只是**同一个拟合**的残差，挑一台相机≠挑到更准的基准（实测 4 视角帧 best 4.40 >
  all 3.60）。球的真值得靠刚性靶标那类外部参照。
- **按 `v` 的真实画面叠加（`Recon2DOverlay`）**：从重建目录读 `pose2d.json`（**原始**
  2D 姿态，下半身门不改它）/ `ball2d.json` / `pred_boxes.json`，配 `VideoSource` 取主时钟
  帧，2×2 平铺。三件事：① **固定 4 格 + 每格标 `camN`**（缺帧画 `no frame (drop/misalign)`
  占位块，见「录像」节 Q3）——否则缺一路后面几格前移，看着像相机接错；② 该相机该帧
  **没检出人**时画**灰色虚线 `pred pN` 框**（卡尔曼预测位置，`draw_pred_box`；**纯显示**，
  重建不读，红线①）；③ 实测框/骨架仍按原样画在**原始分辨率**图上再缩放。
- **「看不出体型」不是渲染 bug（Q1 实测，2026-09-09）**：SMPL β **每人整段只估一次**
  （top-5 置信帧，`select_shape_frames`；`OPT_SHAPE=False`，逐帧只拟合 θ/t），实测
  2~718 帧 β 完全恒定、用存档 β 重算顶点与 npz 差 0.000mm。β 模长只有
  **0.63(p0)/0.24(p1)**——25 个稀疏关键点只约束**肢体长度**，对胖瘦/肩宽/胸廓几乎无
  约束（β 前几维才管体型），所以 mesh 与 points 两条路都只能给出「标准身材的这个人」。
  要看出体型得加约束（轮廓项 / VPoser·HuMoR 密度先验），不是渲染端能修的。
- **播放观感（三处曾踩的坑，2026-09）**：① 播放速度按**内容帧边界 pacing**——目标 = 录像真实出帧率（meta 里 period 反推，100fps 录制=实时），标题栏实时显示 `≈N帧/秒`；渲染跟不上自动掉帧不慢放，别用固定 sleep。② **必须 `widget.force_redraw()`**（每次换帧后）——否则场景变了事件流不保证重绘，观感一卡一卡/跳帧。③ 100Hz 拟合常单帧/两三帧 no_person 抖动 → `--hold-gaps`（默认 3≈30ms）：连续 no_person ≤N 帧保持上一姿态不清空（`ReconTimeline.hold_gaps`），否则人体高频闪没。
- **GUI 播放 = 点云模式（2026-09-04 重构；用户拍板「只要点云就行」）**：Open3D 0.19 Filament 的 `Scene.update_geometry` **只收 PointCloud**；三角网格逐帧动画只能 remove+add，每次重挂留不可回收引擎级残留、~1 万次（实测 8k~11.5k ops）即段错误——旧的限频/对象池只能推迟到崩溃点（「播一段就停 / 干脆不播」的根因，GPU 渲染一直在跑、不是 CPU 软渲染）。修法：`ReconScene(mode=...)` 两条路——`mode="mesh"`（平滑三角网格）**只**用于 `render_still`（每次新建场景加一次、不累积）；`mode="points"`（`play_gui` 用，**mesh 默认参数不用动**）把人体表面（SMPL 顶点+每面质心 ≈20666 点）、地板影（顶点沿光水平投影、不透明深色点）、骨骼（关节连线等分点）、球（单点）、轨迹（逐采样点）全做成 `t.geometry.PointCloud`，**add 一次后每帧 `update_geometry` 原地改顶点缓冲 + `show_geometry` 切可见性**，全程 0 次 remove+add。实测 8 遍×714 帧播放（5712 次换帧）实体数恒 8（+0 churn）、RSS 平稳、中位换帧 7.7ms、无段错误 → 可无限长播。**⚠️ 别给场景里的 pcd 设 `.point.normals`**（normals 只 CPU 侧算法线烘焙用；设了会把云注册进别的低层入口 → update 报 `_Map_base::at`）。
- **人体凸凹明暗 = 顶点色烘焙 `bake_body_shading`（defaultUnlit），不走 Filament 实时光照**：逐顶点 Lambert（`ambient=_BAKE_AMBIENT`+`key=_BAKE_KEY`×`max(n·光向,0)`），光从上方略偏左前来（`_KEY_LIGHT_FROM`，与假影太阳同侧）。朝光面≈肤色顶格、颌下/腋下/腹股沟等凹处法线背光自然暗一档 → 一眼看出身体起伏；纯 numpy 可单测。原因（EGL 实测，2026-09）：Filament **cast_shadows 不产真影**、fill/IBL 低强度无效；高 lux（~1e5）方向光**能出强漫反射**——旧结论「太阳定向光不生效」实为强度 1000 太小 + fill 用了 0.19 旧版参数序（color 位塞了方向向量），非平台限制。真影不可靠故地上阴影仍用程序化假影，明暗走烘焙，双保险。**影子两层都画在地板平面 `floor_z`（不是脚平面）**：重建 SMPL 脚底常悬空几 cm（142020 实测 median -0.713 vs floor -0.76），贴脚画盘会悬在地板上方成脱开深斑。① 接触椭圆 `contact_shadow_planes`（脚底核心深影）② 整身投影软影 `project_floor_shadow`+`convex_hull2d`（**纯 numpy 2D monotone-chain**，投影点全在同平面——open3d `compute_convex_hull` 会因退化抛 QH6154/每帧打 QH7089 精度告警刷屏；质心 apex 三角扇 CCW → 法线 +Z）——垂直俯视桌顶时人影被桌面挡住看不见（取景限制，侧视/低视角可见），这是物理遮挡不是 bug。透明材质必须显式 `shader="defaultLitTransparency"`——`base_color` alpha 默认不混合。
- 键盘：Space 播放/暂停、←/→ 步进、Home/End、R 复位、-/= 调速、Esc 退出。

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
### 录像 + 离线重建（EasyMocap 的主路线）

录制端 `camera/recorder.py`：
- **帧走旁路 sink 不进主循环**：`Camera.set_frame_sink(fn)` 在采集线程入主队列前把帧送给录制
  回调（须快进快出，只入队）；`SessionVideoRecorder` 每台相机一个后台编码线程
  `_CameraWriter` 消费队列，灰度帧在 Python 侧转 yuv420p 喂 **ffmpeg 子进程**
  （默认 GPU `h264_nvenc`，见 `_resolve_encoder`）→`.mp4`，**编码跟不上丢最旧帧不阻塞抓帧**
  ——写进文件的每一帧都 append 设备时间戳，收尾 `np.save cam{cid}_ts.npy`。
- **编码器选型（第一性原则，2026-09-03）**：mp4v 是丢帧根因——本机实测 1440×1080
  噪声内容 mp4v **~44fps/路**（131812 会话四路各丢 30-38%）；MJPG 26ms/帧、`avc1`(h264)
  无 v4l2 设备打不开（OpenCV 无 libx264）。改走 **NVENC**：自编 ffmpeg 位于
  `/home/yby/tools/ffmpeg-nvenc/bin/ffmpeg`（源码+头文件也在 `/home/yby/tools/nvcodec`；
  自带/系统 ffmpeg 无 nvenc，BtbN latest 的 nvenc 请求 API 13.1 > 驱动 580 的 13.0 打不开，
  故从 ffmpeg 7.0.2 + nv-codec-headers `n13.0.19.1` 源码自编）。`recorder.py` 探测链
  `$TT_FFMPEG` > 该路径 > PATH；编码器 `h264_nvenc` → `libx264` → cv2 mp4v 兜底，
  `$TT_RECORDER_CODEC=auto|nvenc|x264|cv2` 可强制。`fps` 只写 mp4 头（播放速度），
  重建读帧序号 + ts 副产物，不受影响。
- **两个关键实测结论**（均 1440×1080 Mono8 最坏=随机噪声内容）：
  ① 直接喂 yuv420p 比喂 gray 快 ~2 倍（4 路 gray ~93fps/路 → **yuv420p ~168fps/路**）——
  gray 会让 ffmpeg 每帧软件上采样（swscale gray→yuv420p 就是那堵墙）；灰度图没颜色，
  U/V 恒 128，编码线程每帧只多一次 Y 拷贝 + 两个 `os.write`。② 4 路 100fps 噪声满压 3s：
  **编码丢 0**（喂 293 → 写 293 全落盘，cv2 读回帧数一致）——对比 mp4v 同场景丢 385/路。
- **首次录像冷启动丢帧（20260903_141311 实例）**：同一 live_control 进程里**第一次**录像
  四路开头各丢 ~44 帧（读 `cam{cid}_ts.npy` 定位：缺口全在 ~5% 处一处集中爆发，后面只有
  零星 1-2 拍 = 正常 USB 传输丢 <1%）——`_resolve_encoder` 的探测（含一次冷 NVENC 实编码）
  留到编码线程收到首帧才触发，lru_cache 锁让 4 个线程全堵住、队列（128）溢出丢最旧；
  live_control 满 GPU（TRT 推理）时该探测可达 1s+。**第二次**录像探测已缓存 → 编码丢 0
  （141353：10.3s 喂=写=1028，仅 4-6 拍传输层单缺）。修法：`recorder.start()` 在挂 sink 前
  先 `_resolve_encoder()`（冷启动代价移到开录前、无害）。**若再见「短录丢/长录不丢」，先问
  是不是该进程第一次录像**，别往编码吞吐上想（长短录的编码器吞吐一样）。
- **每相机诊断**：`SessionVideoRecorder.stop()` 现在逐相机打印
  `喂 {n_fed} → 写 {frames} 帧（编码丢 {n_dropped}）· 实测 {fps} fps / 100 目标`，
  并把 `fed_per_cam` / `encoder_dropped_per_cam` / `measured_period_s` 写进
  `meta.json`。**先看时长口径**：`duration_s` 是从 start() 到 stop() **返回**（含 4 路
  编码线程排空 backlog + mp4 落盘 + meta 写入，实测 ~1-2s），**不是真实录制长度**；
  真实长度是 `capture_s`（=用户按停瞬间）。meta 的 `feed_diag_per_cam[*].last_feed_s`
  就是真实录制终点。**帧数不足分三层**：① 生产侧（USB 抓帧，100Hz 触发但主队列
  丢帧/取帧慢）② 编码侧（`_CameraWriter` 队列满丢最旧 = `encoder_dropped`）。换 nvenc 后
  `encoder_dropped` 应恒 ~0（~168fps/路 ≫ 100 目标）——若又非零，先怀疑 GPU/驱动或
  ffmpeg 子进程异常，再看每行的「喂」少不少（生产侧）还是「写 < 喂 - 丢」（编码侧）。
  ③ **四路同时整段静默 = 进程级/总线级一次性停供**
  （`recorder.py::_feed_gap_report` 对每帧入队墙钟找 >60ms 空档 + `camera.py` 记
  GetImageBuffer 超时墙钟 → meta 写 `feed_diag_per_cam`，stop 打印判定）：
  静默区里超时 >0 → 「相机/总线停供（抓帧线程在超时轮询）」；=0 → 「进程冻结
  （抓帧线程没来取，GIL/阻塞调用）」。
  **判定例（曾误读，勿再犯）**：3.2s 会话四路各 186 帧、ts 全 10.000ms 连续、跨度
  仅 1.85s，当时被判成「~1.3s 全局停供」——**错的**。`feed_diag` 证明 last_feed
  ≈ 按停瞬间、入队全程无 >60ms 空档、GetImageBuffer 超时 0：那段 "缺失" 只是
  stop() 收尾 ~1.3s。同理 122139：5.0s=3.1s 录制+1.9s 收尾，USB 传输丢帧 <1%，
  真损失是编码侧 cam0 丢 23/cam3 丢 18（~5-7%，集中在 1.3~1.9s 一次 CPU 抢占，
  cam1/cam2 为 0）——逐帧 ts 的 >1 拍间隙是**编码丢最旧造成的写入断档**，不等于
  USB 丢脉冲。**离线重建不受影响**：脉冲号对齐天然容忍缺帧。
- **20260908_161147 的逐相机缺帧已定位（Q3，2026-09-09）**：719 主时钟帧里
  cam0/1/2/3 各有 0/24/22/28 帧缺帧（`fed_per_cam` 719/715/715/713，`encoder_dropped`
  全 0、入队无 >35ms 空档、GetImageBuffer 超时 0）→ **是采集侧单拍丢**（USB/触发丢
  脉冲），不是编码侧。逐相机相邻 ts 差/周期取整直方图几乎全是 1 拍（偶有 2~3 拍），
  脉冲号累加与绝对取整零不一致；cam1 实测直接跳过脉冲 89（间隔正好 2.0000 周期）。
  共 **51/719 拍至少缺一台相机**（7%）。回放里这表现为「某格空白」——
  `Recon2DOverlay.tile()` 已改成**按 `src.cids` 固定 4 格**、缺帧画 `camN / no frame
  (drop/misalign)` 占位块、每格左上角标 `camN`，否则缺一路会让后面几格整体前移、
  看起来像「相机接错了」。
- 命名按**逻辑相机号** `cam{cid}.mp4`（cid=标定 `cam_{cid}.yaml` 的号），与在线 EasyMocap
  一致；live_control 按 `v` / 面板「录像」按钮启停（`fps=100` 外部触发 / `30` 自由采集）。

离线端 `reconstruction/video_source.py` + `scripts/reconstruct_video.py`：
- **跨相机对齐绝不用「减绝对 ts 差」（各机时钟基准偏移秒级不可比）**，改为**脉冲号对齐**：
  每台相机内部对相邻 ts 差/周期取整 → 丢几拍加几号（`_pulse_ids`），主时钟每帧的脉冲号
  用 `searchsorted` 在目标相机的脉冲号序列里找同号帧（单调 → 天然不倒退）；目标相机那拍被
  编码丢掉就没帧给（该视角缺帧）。录制 sink 挂在同一触发抓帧流上、四台从同一拍起写，
  所以「脉冲号相同 ⇒ 同一物理触发」成立。
- `np.savez_compressed` 返回 None（不是 file 对象），别调 `.close()`。
- **依赖 EasyMocap 源码**（默认 `/home/yby/projects/EasyMocap`，`EASYMOCAP_ROOT` /
  `--easymocap-root` 可覆盖；SMPL 模型在项目 `data/bodymodels/`，缺 `smpl/SMPL_NEUTRAL.pkl`
  时自动从 `SMPL_NEUTRAL.npz` 生成）。误删后恢复：
  `git clone --depth 1 https://github.com/zju3dv/EasyMocap /home/yby/projects/EasyMocap`
  （2026-09-09 实测该源 master 带 mv1p / `smooth_Rh` 补丁，不是纯官方 upstream）。
- 默认档位 `--config stream`（EmFit 热启动 ~2.1s/帧）；`--fake-poses` 注入合成站姿人跑
  整条 录制→对齐→检测→拟合→存档 管道，无硬件/无真人视频也能验证与计时。
- **多人身份默认走阶段 A–D 跟踪**（`--assoc tracker`，见「多人跟踪 / 身份」节）；
  `--person-groups` 退化为「人数 + 每人的 home 视角（拟合投影用）」，不再当身份依据。
  tracker 模式把人**曾出现过的全部相机**都作为该人的拟合视角。
- **SMPL β 只用置信度最高的 `--shape-top-k` 帧**（默认 5，`select_shape_frames`）——
  整段一次估计，之后逐帧只拟合姿态/平移；`0` = 官方原版用全部帧。
- 实测单帧真机开销：检测不在此列；EmFit stream ~2.1s、official cold ~6.2s（GPU 5080）——
  即**放弃实时（100fps 视频离线跑）是必然选择**。
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
- [x] 离线 2D 模型默认换 **vitpose-h**（比 vitpose-b 重投影 −0.8~−0.9px、根关节最坏跳变 14.8→3.9cm）+ `--fit-conf` 0.5；共识降权实现但默认关——见「离线 2D 模型」节。
- [ ] 在线 `live_control.py` 的姿态仍走 RTMPose-l + TRT（追实时，没跟离线一起换 ViTPose-h）；若要在线也升级需另测吞吐。
- [x] 人检测 yolo11n 灰度（`det="yolo11n-gray"` 默认）+ 三处提速 + conf 0.5 + float32 归一化——见「人检测」节。
- [x] RTMPose 预处理瓶颈是**归一化**（uint8→float64 占 3.35ms 的 60%），不是 warp（0.09ms）；float32 就地算省 ~1ms。
- [ ] RTMPose `_trt_session` 加动态 batch profile(1/4/8) + 预热 batch 8（避免人数变化触发引擎重建）。
- [ ] YOLOX decode+NMS 上 GPU（`_yolox_decode_batch` numpy CPU 3.7ms → <0.5ms，零精度损失）。
- [ ] 姿态最大机会：Fixed ROI（半固定机位砍掉整个 YOLOX 段 ~12ms + 删 match_people）未做；RTMO one-stage 候选。
- [x] YOLOX 动态 batch 重导出（`export_yolox_dynamic_batch.py` 纯 PyTorch 重建，输出 `(B,3549,85)` 不烤 NMS）；三个坑：不烤 /255（humanart 训练吃 0-255）、decode 用 cell 左上角（center=(delta+grid)×stride 不加 0.5）、TRT profile min1/opt4/max8。
- [x] 录像离线重建的多人身份：`reconstruct_video.py --assoc tracker`（默认）走 `person_track.py` 阶段 A–D；每人的 SMPL 拟合视角 = 「曾看到该人的全部相机」（tracker 模式覆盖 `--person-groups`，后者只给人数/`--assoc fixed` 用）。
- [x] 伪预测框自激漂移、根关节落到手腕、RANSAC 视角对兜底——见「多人跟踪 / 身份」节三条红线。
- [ ] `person_track` 仍只跟踪**人数上限内**的人（`--max-people` / `--person-groups` 组数），新人进入不自动开新身份；`LOCAL_TRACK_MAX_MISSED`/`ROI_REDETECT_CONF` 未做参数化扫描。
- [ ] ROI 小窗漏检（Q4 实测 36/515 无框帧，占人×相机帧 ~0.6%）：全图能检出、离预测 110px 中位，被 `roi_gate_px=120` 距离门 + 「框心必须落在小窗内」+ 尺寸门拒掉。放宽前**先做干净 A/B**（早期放宽被伪框路径污染、结论作废）；影响面小，暂不改默认。
- [ ] 离线 tracker 模式的 EmFit 仍逐人顺序拟合；多人并行/共享初值未做。

### 录像 / 重建相关已踩坑
- [x] `np.savez_compressed` 返回 None（无 `.close()`）；编码 mp4v 跟不上 100fps（噪声 ~44fps/路）→ 换 ffmpeg h264_nvenc（yuv420p 直喂免 swscale，4 路 ~168fps/路、满压丢 0）——见「录像 + 离线重建」节。
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
