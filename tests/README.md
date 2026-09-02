# tests — 测试

## 干什么

用 pytest 验证相机模块的基础行为。`conftest.py` 把 `src/` 加进 `sys.path`，测试里直接
`from tabletennis.camera import ...`。

`test_camera.py` 覆盖：

- 设备枚举不抛异常（0 台 / 4 台都算正常）。
- `Camera.open/close` 生命周期。
- 图像参数设置 + 读回（像素格式应为 Mono8、`list_supported()` 不抛异常）。
- 软件触发端到端取帧（`start` → `get_latest_frame` 拿到灰度图、尺寸正确）。

无相机硬件时，依赖硬件的用例用 `pytest.mark.skipif` 自动跳过（模块加载时枚举一次判断）。

## 怎么用

```bash
conda activate tt
pytest tests/ -v
```

## 还没做完

`test_video_offline.py` 覆盖：

- 跨相机**脉冲号对齐**（`_pulse_ids` / `align_maps`）：各相机丢不同帧时仍把四路逐拍
  绑回同一触发；无 ts 时退化为帧号对齐。
- **录制往返**：假相机（只实现 `logical_id / serial / set_frame_sink`）驱动
  `SessionVideoRecorder` 录 3 路 mp4 → `VideoSource` 按 ts 副产物重新对齐（无相机硬件）。

`test_emfit.py` 覆盖 EasyMocap 优化层与原版的数值一致性（GPU + SMPL 模型，缺则整文件 skip）。

## 还没做完

- `core` / `vision` / `visualization` 都没有单测。
- **vision 检测器无测试**：RTMPose 依赖模型下载 + 较慢，还没写含固定输入/期望输出的单测（可考虑
  用 mock 帧或离线小图做冒烟测试）。
- 需要硬件的测试在无相机环境会整体 skip，**CI/无硬件机器上跑不到真正的抓帧路径**；
  若要做无相机回归，需给 `Camera` 注入 mock SDK。
