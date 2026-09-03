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
`viewer3d` 也提供 SMPL 层（`add_smpl_layer` / `set_smpl`，在线拟合时实时显示网格）。

`recon_player.py` 提供离线重建结果的 **3D 回放**（`ReconTimeline` 时间线 +
`ReconScene` 场景 + `render_still` EGL 出图 + `play_gui` 交互窗口）：

- 逐帧回放 SMPL 网格（主时钟 t，no_person/失败清空、被 stride 跳过帧保持上姿态）。
  播放按**内容帧边界 pacing**：目标速度 = 录像真实出帧率（`recon_meta.json` 的
  period 反推，100fps 录制 → 实时回放），标题栏实时显示 `≈N帧/秒`，渲染跟不上
  自动掉帧不慢放。**`--hold-gaps N`**（默认 3≈30ms）：连续 no_person ≤ N 帧保持
  上一姿态不清空，100Hz 拟合单帧抖动不会把人体闪没。
- 人体 **凸凹明暗烘焙在顶点色**（`recon_player.bake_body_shading` 逐顶点 Lambert，
  `defaultUnlit` 显示）——朝光面亮、颌下/腋下/腹股沟等凹处自然变暗，一眼看出身体
  起伏。地上阴影是程序化假影（Filament cast_shadows 在本机 EGL 下实测不产真影，
  见模块 docstring），**两层都画在地板平面**：① 两片脚下接触椭圆（核心深影）
  ② **整身投影软影**（所有 SMPL 顶点沿光水平方向投到地面 → 凸包填充盘，带身形且随
  姿态伸长，默认开、`--no-cast` 关）。重建的 SMPL 脚底常悬空几 cm，影子钉在地板
  才是真「地上影」（贴脚画会悬空）。
- 交互：Space 暂停/播放、←/→ 步进、Home/End、R 复位、-/= 调速、Esc 退出。
- 用法：`scripts/visualize_recon.py <recon目录>`（加 `--watch` 在重建进行中追帧，
  `--render T out.png` EGL 出图验证）。

骨架连线/关键点定义从 `vision.skeleton.get_skeleton()` 取，保证画图和检测用的同一套索引。

## 怎么用

```python
from tabletennis.visualization.overlay2d import annotate_frame, tile_images

bgr = annotate_frame(frame.image, poses, title="cam0")
```

实际使用见 `scripts/live_control.py`（四机平铺 + 状态条）与 `scripts/pose_preview.py`（单机骨架叠加）。

## 还没做完

- **3D 场景已实现（`viewer3d`）**：Open3D 后台线程渲染「相机视锥 + 标准尺寸球桌」，
  由 `live_control.py` 按 T 触发。在线 SMPL 网格层（`add_smpl_layer`）与离线回放
  （`recon_player`，带程序化接触阴影）已就绪；多人与球轨迹的实时 3D 仍未接入主循环。
- `draw_ball` 已实现但**暂时没被调用**——球检测器还没写；`draw_table` / `draw_table_model`
  已由球桌检测接入。
- 没有封装交互式/可持久化的可视化组件（如录像、保存带标注的图片）——回放窗口见
  `scripts/visualize_recon.py`（重建结果已在录像/离线链里）。
