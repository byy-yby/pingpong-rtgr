# reconstruction —— 多视角 2D → 3D 重建

把各相机视角的 2D 检测结果 + 标定内外参转成 3D 世界坐标。当前实现**人体姿态**的
跨视角匹配与三角化；球轨迹 / 桌面拟合留待后续。

## 文件

- `triangulate.py` —— 置信度加权的多视角 DLT 三角化（`MultiViewTriangulator`），
  内含关键点去畸变、重投影误差 / 交会角质量评估、多视角外点剔除；`load_camera_rig`
  从 `data/calibration/cam_N.yaml` 与 `data/extrinsics/table_extrinsics.yaml` 读标定。
- `associate.py` —— 跨视角**实例级**球员匹配（`match_people`）：几何锚点 +
  两两三角化重投影门限 + 并查集合并 + 一致性精化。运动场景队服相同，不用外观。

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
