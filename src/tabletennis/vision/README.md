# vision — 视觉识别模块（2D 检测）

## 干什么

对相机帧做 2D 视觉检测。已实现**人体姿态**（RTMPose-l / Halpe-26）与**球桌识别**
（两个大 ArUco 标记定桌面世界系），球检测只有抽象接口。

| 文件 | 职责 |
|---|---|
| `detector.py` | 检测器抽象接口 `Detector` / `PoseDetector` / `BallDetector` / `TableDetector` + 注册工厂 `register_detector` / `create_detector`。 |
| `pose/rtmpose_pose.py` | **RTMPose** top-down 2D 姿态检测器（YOLOX 检测人 + RTMPose 关键点，默认 Halpe-26，onnxruntime）。 |
| `skeleton.py` | 骨架定义（关键点名 + 骨骼连线），内置 COCO-17 与 Halpe-26。 |
| `table/table_detector.py` | **球桌识别**：检测两个大 ArUco 标记（ID0/ID1）解「桌面→相机」位姿，失败回退已存外参，投影标准尺寸球桌。 |

约定：上层脚本 / pipeline 只依赖抽象，通过 `create_detector(kind)` 按名字取检测器；
未实现返回 `None`。新算法实现后 `register_detector` 一行接入，脚本不用改。当前已注册 `pose` 与 `table`。

## 怎么用

命令行（推荐，自动走相机启动流程）：

```bash
python scripts/pose_preview.py                     # 单相机自由采集 + Halpe-26 骨架叠加
python scripts/pose_preview.py --all-cameras       # 四路平铺，每路都跑检测
python scripts/live_control.py                     # 四机平铺，按 [p] 开关姿态检测
```

直接 API：

```python
from tabletennis.vision.pose.rtmpose_pose import RTMPoseDetector

det = RTMPoseDetector(model="rtmpose-l-halpe26", device="cpu", score_thr=0.5)  # 首次运行自动下模型
poses = det.detect(frame)        # frame: tabletennis.core.types.Frame -> List[Pose2D]
for p in poses:
    print(p.keypoints.shape)     # (26, 3)  [x, y, confidence]，Halpe-26 顺序
```

模型说明：姿态默认 **rtmpose-l-halpe26**（Halpe-26，256×192，26 点），人体检测默认
**yolox-m**。不带 `-halpe26` 后缀的模型（如 `rtmpose-l`）输出 COCO-17。权重自动下载到
`~/.cache/rtmlib/hub/checkpoints`。本机 GT 1030 太弱默认跑 CPU，换好 GPU 后 `device="cuda"`。
黑白帧会复制成 3 通道再喂模型（RTMPose 训练在 RGB）。

## 还没做完

- **球检测未实现**：`BallDetector` 接口已定义，`ball/` 目录与算法待补（未注册，`create_detector("ball")` 返回 None）。
- **球桌检测已实现**：`TableDetector`（`table/`）用两个大 ArUco 标记定桌面世界系并投影标准尺寸球桌；`create_detector("table")` 可用，`live_control.py` 按 T 接入。
- **WholeBody-133 未加**：`skeleton.py` 里有 COCO-17 / Halpe-26，WholeBody-133（手/脸/脚）未加。
- **2D→3D 不在本模块**：多视角关键点三角化（`Skeleton3D`）属于未来的重建模块，本模块只负责单帧 2D。
- `to_openpose=True` 时输出 OpenPose 序，但 `skeleton.py` 没有对应连线定义，用时会缺连线。
