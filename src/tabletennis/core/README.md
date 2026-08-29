# core — 共享数据类型与配置加载

## 干什么

跨模块共享的**数据契约**和配置工具。项目分层约定里 `core` 是最底层，其它模块（`camera` /
`vision` / `reconstruction` / `visualization`）都依赖它、且**不反向依赖**。跨层传数据一律用
这里定义的 dataclass，避免各模块各自定义重复的类型。

- `types.py`：共享数据类型
  - `Frame` / `FrameBundle`：相机模块产出的帧 + 同一触发时刻的多机帧组。
  - `Pose2D`：单帧 2D 人体姿态（关键点 `(N,3)` + 置信度 + bbox + 骨架名）。
  - `Skeleton3D`：多视角三角化后的 3D 骨架（占位，重建阶段用）。
  - `Ball2D` / `Table2D`：球 / 球桌的 2D 检测结果。
  - `CameraIntrinsics` / `CameraExtrinsics`：内参（K、畸变）/ 外参（R、t），附投影矩阵工具。
- `config.py`：YAML 配置加载与路径工具
  - `load_yaml` / `load_yaml_optional` / `load_default_config`
  - `project_root()` / `config_dir()`

## 怎么用

```python
from tabletennis.core.types import Frame, Pose2D, Ball2D, Table2D
from tabletennis.core.config import load_yaml, project_root

cfg = load_yaml(f"{project_root()}/config/cameras.yaml")
```

数据流约定：`camera` 产出 `Frame` → `vision` 消费 `Frame` 产出 `Pose2D` / `Ball2D` / `Table2D`
→（未来的）重建模块消费多路 `Pose2D` 产出 `Skeleton3D`。新增跨模块共享类型时加进 `types.py`
并同步 `__init__.py` 的导出，不要在某个上层模块里私自定义。

## 还没做完

- `Skeleton3D` / `CameraExtrinsics` 目前只是**占位类型**，还没有模块真正生产或消费它们
  （2D→3D 重建、外参标定都未开始）。`CameraIntrinsics` 已由 `calibration` 模块生产 / 读回
  （`calibrate_camera` / `load_intrinsics`）。
- 缺 3D 层面的球轨迹、桌面平面类型（当前只有 `Ball2D` / `Table2D` 的 2D 版本）。
