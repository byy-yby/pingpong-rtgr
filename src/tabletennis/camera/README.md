# camera — 相机控制模块（海康 MVS SDK 封装）

## 干什么

对海康 MVS SDK 做薄封装，提供「四台 MV-CS016-10UM 统一开关、触发、取帧、调参」的能力。
对外只暴露两个稳定入口 `CameraManager`（多机）和 `Camera`（单机），SDK 细节收在内部。

| 文件 | 职责 |
|---|---|
| `sdk.py` | SDK 生命周期 + 设备枚举。集中处理 `.so` 加载路径、`sys.path` 引导（**所有 SDK import 收敛于此**）。 |
| `frame.py` | SDK 帧缓冲 `MV_FRAME_OUT` → numpy 灰度图 → `Frame`（**必须 copy**，SDK 会复用缓冲）。 |
| `camera.py` | 单相机 `open/start/stop/close` + 采集线程 + 有界队列（满则丢最旧帧）。 |
| `camera_manager.py` | 多相机统一管理：`setup/start/stop/close`、按序列号/索引选相机、`get_latest_bundle`。 |
| `trigger.py` | 外部触发（信号发生器 Line0）/ 软件触发 / 连续采集的 GenICam 配置。 |
| `parameter.py` | `ImageControl`：曝光/增益/黑电平/伽马/亮度/对比度/帧率/像素格式的读写 + 范围查询。 |
| `recorder.py` | **四路同步录像**（`SessionVideoRecorder`）：每相机独立后台编码线程 + 帧 sink 旁路取帧 + 设备时间戳副产物，离线重建用。见下方「离线录像」。 |
| `mv_import/` | vendor 的海康 MVS Python 绑定（`MvCameraControl_class.py` 等）。**生成代码，勿手改**。 |

## 怎么用

典型「相机启动流程」（`with` 块会自动 `setup` / `close`）：

```python
from tabletennis.camera import CameraManager

with CameraManager(trigger_mode="external", exposure_us=5000, gain_db=5) as mgr:
    mgr.start()
    bundle = mgr.get_latest_bundle()          # FrameBundle，每台相机一帧
    for cid, frame in bundle.frames.items():
        print(cid, frame.frame_num, frame.device_timestamp)
```

单机直接控制（枚举 → 打开 → 采帧）：

```python
from tabletennis.camera import Camera, enumerate_devices, initialize_sdk, finalize_sdk

initialize_sdk()
dev = enumerate_devices()[0]
cam = Camera(dev, logical_id=0, trigger_mode="continuous", exposure_us=5000)
cam.open()
cam.start()
frame = cam.get_latest_frame(timeout=1.0)     # None 表示超时
cam.controls.set_gamma(1.0)                   # open 之后才能调参
cam.close()
finalize_sdk()
```

命令行入口见 [`scripts/README.md`](../../../scripts/README.md)：`list_cameras.py`（枚举）、
`grab_preview.py`（单机预览）、`grab_sync.py`（四机同步验证）、`query_parameters.py`（参数范围）。

> 运行前先关 MVS 客户端，否则 `OpenDevice` 报 `0x80000203`（设备被独占）。

## 离线录像（camera/recorder.py）

`Camera.set_frame_sink(sink)` 把每帧**旁路**交给录制回调（采集线程里同步调用，须快进快出），
`SessionVideoRecorder` 给每台相机开一个后台编码线程消费该回调，写
`data/video/<YYYYmmdd_HHMMSS>/cam{cid}.mp4`（mp4v；编码跟不上时丢最旧帧不阻塞抓帧）。
每个写进文件的帧都记下设备时间戳 → `cam{cid}_ts.npy`，离线脚本靠它把丢帧后的四路
重新对齐到同一触发脉冲。

```python
from tabletennis.camera import CameraManager
from tabletennis.camera.recorder import SessionVideoRecorder

mgr = CameraManager(trigger_mode="external"); mgr.start()
rec = SessionVideoRecorder(mgr.cameras, fps=100.0)   # fps 只影响播放速度
rec.start(); ...; rec.stop()                          # stop() 返回 meta（帧数/时长/相机序列号）
```

入口：`live_control.py` 按 `v`（或点面板「录像」按钮）；重建端见
`reconstruction/video_source.py` + `scripts/reconstruct_video.py`。

## 还没做完

- **严格帧对齐未做**：`get_latest_bundle` 只取各机「各自最新帧」，未按 `device_timestamp`
  做最近邻配对（外部硬触发下曝光已对齐，实时预览够用；严格配对留给 `pipeline` 阶段）。
- **Mono10 / Mono12 Packed 解包未实现**：`frame.py` 对这两种打包格式直接 `raise
  NotImplementedError`，本机用 `Mono8` 无需处理，未来换彩色/高位深相机要补。
- **只有轮询取帧**，没有 `RegisterImageCallBackEx` 回调路径（Python 里受 GIL 影响，暂不推荐多机实时用）。
- **触发细节未穷尽**：`TriggerDelay` 错峰、`TriggerCacheEnable` 等已封装但未在脚本里充分暴露。
- 黑白相机实际可调参数只有曝光/增益/伽马（无 `BlackLevel`/`Brightness`/`Contrast` 节点），
  `parameter.py` 对这些做了存在性探测，`get` 返回 `None` 属正常。
