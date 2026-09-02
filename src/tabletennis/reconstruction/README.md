# reconstruction —— 多视角 2D → 3D 重建

把各相机视角的 2D 检测结果 + 标定内外参转成 3D 世界坐标。当前实现**人体姿态**的
跨视角匹配与三角化；球轨迹 / 桌面拟合留待后续。

## 文件

- `triangulate.py` —— 置信度加权的多视角 DLT 三角化（`MultiViewTriangulator`），
  内含关键点去畸变、重投影误差 / 交会角质量评估、多视角外点剔除；`load_camera_rig`
  从 `data/calibration/cam_N.yaml` 与 `data/extrinsics/table_extrinsics.yaml` 读标定。
- `associate.py` —— 跨视角**实例级**球员匹配（`match_people`）：几何锚点 +
  两两三角化重投影门限 + 并查集合并 + 一致性精化。运动场景队服相同，不用外观。
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
