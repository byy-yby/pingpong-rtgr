# config — 配置文件

## 干什么

集中放运行参数，代码用 `tabletennis.core.config.load_yaml` 读取。

| 文件 | 用途 |
|---|---|
| `cameras.yaml` | 触发模式 / 图像 / 曝光 / 增益 + 相机序列号 → 逻辑索引映射。 |
| `calibration.yaml` | 棋盘格内参标定参数（`scripts/calibrate_intrinsics.py` 使用）。 |
| `extrinsics.yaml` | ChArUco 外参标定参数（`scripts/calibrate_extrinsics.py` 使用）。 |

## 怎么用

1. 先跑 `python scripts/list_cameras.py` 拿到 4 台相机的**序列号**。
2. 把序列号填进 `cameras.yaml` 的 `cameras` 列表（`serial:` 字段）；留空则按枚举顺序绑定 `0..N-1`。
3. 改 `trigger.mode`（`external` / `software` / `continuous`）、`exposure.time_us`、`gain.db` 等，
   再运行 `scripts/live_control.py` / `grab_sync.py`（它们默认读这份配置）。

关键字段：

```yaml
trigger:
  mode: external          # external=信号发生器接 Line0 | software | continuous
  source: Line0           # 低延迟可换 Line2
  debouncer_us: 50        # Line0 滤波时间，防误触发
exposure:
  auto: false
  time_us: 5000.0
gain:
  auto: false
  db: 0.0
cameras:
  - { index: 0, serial: "xxx" }   # 填实际序列号
  ...
```

## 还没做完

- 缺**视觉 / 重建参数**（检测阈值、骨架类型、三角化参数等）的集中配置。
