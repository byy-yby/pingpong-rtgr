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

- **只有 camera 模块有测试**：`core` / `vision` / `visualization` 都没有单测。
- **vision 检测器无测试**：RTMPose 依赖模型下载 + 较慢，还没写含固定输入/期望输出的单测（可考虑
  用 mock 帧或离线小图做冒烟测试）。
- 需要硬件的测试在无相机环境会整体 skip，**CI/无硬件机器上跑不到真正的抓帧路径**；
  若要做无相机回归，需给 `Camera` 注入 mock SDK。
