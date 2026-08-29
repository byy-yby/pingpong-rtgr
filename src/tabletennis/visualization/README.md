# visualization — 可视化模块

## 干什么

把检测结果叠加到相机画面上（2D），以及用 Open3D 做 3D 场景渲染（`viewer3d`）。

`overlay2d.py` 提供的函数：

- `gray_to_bgr(image)`：黑白灰度图 → 3 通道 BGR（复制通道，不污染原数据）。
- `draw_pose(image, pose)`：把单个 `Pose2D` 的骨骼连线 + 关键点画到 BGR 图上。
- `draw_ball` / `draw_table`：画球（圆）/ 球桌（2D 角点多边形）。
- `draw_table_model(image, table, R, t, K, dist)`：把标准尺寸球桌的 3D 线框（桌面边框 + 4 腿 + 球网）投影到图像。
- `annotate_frame(image, poses, title=...)`：把一帧所有姿态画到灰度图，返回带标题的 BGR 图。
- `tile_images(images, cols=2)`：把多张 BGR 图平铺成一格（4 路 → 2×2），缺处补黑。

`viewer3d.py` 提供 `SceneViewer3D`（Open3D 后台线程）：`build_scene()` 构建「相机视锥 +
标准尺寸球桌 + 地面网格」，`start()` / `close()` 控制独立 3D 窗口。

骨架连线/关键点定义从 `vision.skeleton.get_skeleton()` 取，保证画图和检测用的同一套索引。

## 怎么用

```python
from tabletennis.visualization.overlay2d import annotate_frame, tile_images

bgr = annotate_frame(frame.image, poses, title="cam0")
```

实际使用见 `scripts/live_control.py`（四机平铺 + 状态条）与 `scripts/pose_preview.py`（单机骨架叠加）。

## 还没做完

- **3D 场景已实现（`viewer3d`）**：Open3D 后台线程渲染「相机视锥 + 标准尺寸球桌」，
  由 `live_control.py` 按 T 触发。骨架 / 球轨迹的 3D 渲染仍待重建模块。
- `draw_ball` 已实现但**暂时没被调用**——球检测器还没写；`draw_table` / `draw_table_model`
  已由球桌检测接入。
- 没有封装交互式/可持久化的可视化组件（如录像、保存带标注的图片）。
