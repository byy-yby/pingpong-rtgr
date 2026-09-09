# reconstruction —— 多视角 2D → 3D 重建

把各相机视角的 2D 检测结果 + 标定内外参转成 3D 世界坐标。当前实现**人体姿态**的
跨视角匹配与三角化；球轨迹 / 桌面拟合留待后续。

## 文件

- `triangulate.py` —— 置信度加权的多视角 DLT 三角化（`MultiViewTriangulator`），
  内含关键点去畸变、重投影误差 / 交会角质量评估、多视角外点剔除；`load_camera_rig`
  从 `data/calibration/cam_N.yaml` 与 `data/extrinsics/table_extrinsics.yaml` 读标定。
- `associate.py` —— 跨视角**实例级**球员匹配（`match_people`）：几何锚点 +
  两两三角化重投影门限 + 并查集合并 + 一致性精化。运动场景队服相同，不用外观。
- `person_track.py` —— **宽视野一相机多人的跟踪/身份（阶段 A–D）**：
  `MultiPersonTracker` 逐帧串联「单相机局部跟踪 → 世界系 3D 卡尔曼 →
  枚举分配 + 重投影代价最小的跨相机关联 → 逐关键点鲁棒三角化」。
  离线 `reconstruct_video.py --assoc tracker`（默认）与在线 `live_control.py` 都用它；
  旧的 `match_people_fixed`（每相机只留 bbox 最大的一个人）只作 `--assoc fixed` 回退。
- `easymocap.py` —— **EasyMocap SMPL 多视角重建（不依赖三角测量）**：把 SMPL
  参数化人体模型直接拟合到多视角 2D 关键点的重投影误差上，模型先验补全「只有
  单视角可见」的部位。基础版单人 / SMPL-24 关节（无手脸细节），依赖 EasyMocap
  的 `easymocap.bodymodel.smpl.SMPLModel` + SMPL 身体模型文件（见下）。
- `em_fit.py` —— 优化层（`EmFit` + `EMSettings`）：对官方 `reconstruct()` 的
  热启动 / 去同步 / 收敛开关封装，`run(..., prev=上帧 params)` 帧间连续拟合
  （离线脚本默认档位，约 x2.3 且误差≈官方）。
- `video_source.py` —— **离线输入**：把 `data/video/<session>/cam*.mp4` + ts
  副产物读成对齐的多视角灰度帧流（`VideoSource.frames_for_ref`）。四路各自丢帧时
  用**脉冲号对齐**把帧绑回同一触发（详见模块文档）。
- `obs2d.py` —— 2D 观测的 JSON 存读：`pose2d.json`（每帧每相机的**原始** 2D 姿态）、
  `ball2d.json`、`pred_boxes.json`（**纯显示**的卡尔曼预测框，回放叠加画灰色虚线用，
  **重建不读**——预测只用来开 ROI 搜索窗，伪造成检测会引发自激回路）。

## 数据流

```
每相机 RTMPose -> {cam_id: [Pose2D]}          (Pose2D.keypoints = (26,3) [x,y,conf])
    |  match_people(poses_per_cam, triangulator)
    v
[{cam_id: Pose2D}, ...]                        (每个人在各相机的观测，>=2 视角)
    |  triangulator.triangulate_pose(obs)
    v
[Skeleton3D, ...]                              (keypoints (26,3) 米；NaN=未重建)
```

## 关键设计

- **去畸变**：RTMPose 的关键点在畸变后的原始像素系，先 `cv2.undistortPoints(P=K)`
  换成无畸变像素，再与线性投影矩阵 `P = K[R|t]` 一致地三角化。
- **置信度加权**：DLT 每一行按该视角关节置信度缩放；综合置信度 = 视角置信度均值 ×
  交会角因子 × 重投影误差衰减。
- **相机位置**：交会角（两视角光心→3D 点射线的夹角）越小深度越不稳，低于阈值放弃。
- **2 视角的固有歧义**：两人若落在同一极线上，仅靠 2 视角无法区分——真实场景里
  球员分居球桌两侧（空间上远离）通常不触发；若要彻底消歧，需 3+ 视角、外观或时序。

## 用法

```python
from tabletennis.reconstruction import MultiViewTriangulator, match_people, load_camera_rig

intrinsics, extrinsics = load_camera_rig()
tri = MultiViewTriangulator(intrinsics, extrinsics)

people = match_people(poses_per_cam, tri)       # -> [{cam_id: Pose2D}, ...]
for obs in people:
    skel = tri.triangulate_pose(obs)            # -> Skeleton3D
```

完整实时入口见 `scripts/reconstruct_pose.py`（`--synthetic` 可无硬件自检）。

## 多人跟踪 / 身份（`person_track.py`，阶段 A–D）

换广角镜头后一台相机能同时拍到近/远两人，「每相机只拍一个人」的前提失效，
身份必须靠**世界系几何 + 时序**决定。`MultiPersonTracker.step(frame_idx, frames, time_s)`
每帧走四步：

- **阶段 A（单相机局部跟踪）**：每台相机独立，IoU + 质心最近邻匹配本帧检测到
  该相机已有的 `local_track_id`（命中则重置 `missed_frames`）；未匹配的检测开新
  id，未匹配的局部轨迹 `missed_frames += 1`，超过 `LOCAL_TRACK_MAX_MISSED` 删除。
  **ROI 引导重检测**：某活跃轨迹在某相机本帧没匹配上检测时，用它卡尔曼预测的
  3D 位置投到该相机（`project_distorted`）开小窗，边长 = 该人最近一次 bbox 对角线
  ×2（缺失时 300px），窗内以 `ROI_REDETECT_CONF`(0.15) 重跑人检测；命中则坐标映射
  回全图并标 `source="roi_redetect"`，未命中标 `source="predicted_only"`（**不参与
  三角化**）。每 `FULL_DETECT_INTERVAL`(30) 帧强制全图检测。
- **阶段 B（世界系 3D 卡尔曼）**：状态 `x=[p,v]∈R⁶`，匀速模型；观测 = 本帧鲁棒
  三角化的**根关节**（骨盆）；`R` 由三角化协方差 + 基础观测噪声给出。
- **阶段 C（跨相机身份关联）**：给定各相机检测与各轨迹预测，**枚举**分配方案
  （人少时穷举），代价 `E = Σ 轨迹 Σ 分配到该轨迹的相机 w_cam·‖重投影根 − 检测中心‖²`，
  `w_cam = bbox_conf × mean(kpt_conf)`；只有当 `E(best) < E(上一帧方案)·(1−SWITCH_MARGIN)`
  时才切换（`SWITCH_MARGIN=0.2`），抑制身份抖动。
- **阶段 D（逐关键点鲁棒三角化）**：见 `triangulate.py::triangulate_point_robust`。

**三条不能踩的线**（都实测踩过，见 CLAUDE.md）：

1. **不用「卡尔曼预测框」当伪检测**。非全图帧曾经把预测框喂给人检测器 → 图像里
   未必有那个人，姿态凭空生成 → 三角化又「确认」了预测 → 3D→框→姿态→3D 自激回路，
   根关节 10 帧漂到 1.5m 外。现在帧间观测**只**来自 ROI 小窗里真实图像上的检测。
2. **根关节绝不回退到四肢**。骨盆（halpe26 #19）三角化失败时用**双髋中点**合成，
   双髋也没有才置 `root_index=-1`（本帧不更新观测、走纯预测并计 `misses`）。
   曾经用「置信度最高的关节」兜底，结果选到右手腕 → 卡尔曼观测瞬移 0.5m。
3. **畸变口径**：`MultiViewTriangulator` 的 `_dlt/reproj/project` 吃**无畸变**像素；
   ROI 开窗要投到**原始畸变**像素（`project_distorted`），两者不可混用。

**拟合视角选择 `select_fit_views`**：某人只在某台相机的画面边缘露出半截时，人检测框
被边界裁掉一半，姿态模型会把看不见的另一半**外推**出来（实测 kp 置信 0.2~0.35、
bbox 高 128~168px），把关节投回该视角差 95~290px。该函数按「bbox 贴边帧 ≥50% 出场帧」
剔掉这类相机，不足 2 台时回退出场最多的 2 台。离线 20 帧实测重投影误差中位
36.7px → 18.7px。

**下半身可信度门 `lower_body_gate`（默认开）**：隔球桌看远端的人时，膝/踝/脚尖/脚跟
是姿态模型**外推**的点（同视角上半身 kp 置信 ~0.9，下半身中位 0.24~0.40），旧版照样
喂进三角化会把 SMPL 的腿拉歪。每视角逐帧判 `lower_body_unreliable`（下半身中位置信
< 0.5 × 上半身中位 **且** < 0.5 绝对值），命中则 `mask_lower_body` 把
`LOWER_BODY_HALPE26` 的置信置 0 再进阶段 D；**髋 11/12/19 不掩码**（根关节靠它）。
`TrackFrameResult.raw_obs` 保留原姿态供 2D 叠加、`lower_body_masked` 记命中视角。
CLI `--no-lower-body-gate` / `--lower-body-conf-ratio` / `--lower-body-conf-abs`，
`--upper-body-only PID...` 整段只用上身（含髋）。实测 20260908_161147：**下身·好视角**
重投影中位 p1 20.4→14.8px、p0 10.8→10.4px，上身不变（9.5→9.7 / 12.1→12.2）。

**「远端连髋也丢」`mask_hips_when_unreliable`（默认关）**：`mask_lower_body(pose,
include_hips=True)` 连 `HIP_HALPE26=(11,12,19)` 一起置 0，即被判不可信的视角只留上半身。
默认关是因为**远端髋实测比近端还准**（髋·远端 3.5cm/置信 0.84 vs 髋·近端 5.5cm/0.86），
而髋是阶段 B/C 的根关节：A/B 实测追踪退化（`root_index==19` 232→192 帧、骨盆内点
2.07→1.61、根跳变 max 12.07→28.32cm）。CLI `--mask-hips-when-unreliable`。

## EasyMocap（按 S）

```python
from tabletennis.reconstruction import EasymocapReconstructor, load_camera_rig

intrinsics, extrinsics = load_camera_rig()
rec = EasymocapReconstructor()          # 模型路径优先级：SMPL_MODEL_PATH env > 参数 > data/bodymodels/
result = rec.reconstruct(poses_per_cam, intrinsics, extrinsics)
# result = {vertices (6890,3), joints (24,3), faces (13776,3)}（桌面系，米）
```

- **前提**：需 SMPL 身体模型文件（`SMPL_NEUTRAL.npz` 或 `basicmodel_neutral_*.pkl`）。
  到 smpl.is.tue.mpg.de 注册下载，放进项目 `data/bodymodels/`（推荐 `.npz`，免 chumpy）。
- **无三角测量**：拟合只重投影 SMPL 关节到各相机 2D 关键点；单视角可见的部位由
  SMPL 姿态/形状先验 + 骨长结构补全。三角化仅用于给根部平移一个粗略初值。
- **依赖**：EasyMocap 代码（`easymocap.bodymodel.smpl`，路径 `EASYMOCAP_ROOT` 或默认
  `/home/yby/projects/EasyMocap`），纯 torch+numpy，无需 smplx/chumpy。
- 实时入口在 `scripts/live_control.py` 按 S。

## 离线：录像 → EasyMocap（EasyMocap 无法实时时的替代路线）

```bash
# 1) live_control 按 v（或面板「录像」按钮）录四路 → data/video/<时间戳>/cam{0..3}.mp4 + ts
# 2) 离线重建该文件夹：
python scripts/reconstruct_video.py data/video/<时间戳> [--config stream] [--stride 1]
#    → <时间戳>/recon/：逐帧 frame_NNNNNN.npz + recon_index.npz + recon_meta.json
#    --fake-poses 无硬件/视频内容验证整条管道（注入合成站姿人，跑检测器时不读画面）
```

- `--config`：`official`（官方 cold `reconstruct()`，每帧冷启动，最慢）、
  `warm` / `stream`（`em_fit.EmFit` 热启动，前者准、后者快，默认 `stream`）。
- 丢帧不影响正确性：录制端把设备时间戳写 `cam{cid}_ts.npy`，离线按**脉冲号**对齐
  （各机时钟绝对基准不可比，见 CLAUDE.md 时间戳节）。
