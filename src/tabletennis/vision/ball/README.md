# vision/ball — 乒乓球 2D 检测

- `refine.py`：`refine_ball_center` —— 粗定位附近强度加权质心 + 二阶矩求亚像素球心/半径/置信度（纯 numpy，无 scipy）。对 12~24px 的模糊小球有效；不用霍夫圆。
- `classical_ball.py`：`ClassicalBallDetector` —— 背景减除 + 帧差 + 尺寸先验的经典路线，无模型，`register_detector("ball")`。
- `yolo_ball.py`：`YoloBallDetector` —— onnxruntime 推理训练导出的 `best.onnx`（单类 ball，输出 `(1,5,N)`），bbox 后接 `refine_ball_center` 精修，`register_detector("ball_yolo")`。无 torch 依赖，跨相机复用。

## YoloBallDetector 后端（backend 参数）

- `"auto"`（默认）：有 TensorRT 则走 **TRT FP16**，否则 CUDA，再否则 CPU。
- `"tensorrt"` / `"cuda"` / `"cpu"`：固定后端。
- TRT 引擎缓存到 `~/.cache/tabletennis/trt_engines`（首次构建 ~30-60s，之后复用；构建失败自动回退 CUDA）。

实测性能（RTX 5080，1440×1080→1280×1280，单路）：

| 项 | 耗时 |
|---|---|
| CUDA 推理（FP32） | ~26.9 ms |
| TRT 推理（FP16） | ~8.8 ms |
| 预处理（原：4 次临时数组分配） | ~27 ms |
| 预处理（现：预分配 buffer 复用） | ~8.4 ms |
| **单路 detect 合计（TRT + 优化预处理）** | **~18.5 ms**（4 路顺序 ≈75ms → ~13 FPS） |

**关键教训：这活儿 CPU 预处理和 GPU 各占一半。** 只换 TRT 不动预处理，速度不变
（CPU 瓶颈）；只优化预处理不动 TRT 也只快一半。两个都要做。

## 坑（务必记住）

1. **provider 列表不能传 `ort.get_available_providers()` 全量**。本机装了
   `tensorrt-cu12-libs`，全量传会让 CUDA 时也先去建 TRT 引擎（实测 ~52s 阻塞主线程）。
   必须显式限定，或显式 TRT（见上）。
2. **`dynamic=True` 导出的 onnx 必须把 h/w 固化成静态**，否则 TRT EP **静默回退 CUDA**
   （不建引擎、不报错、白跑）。用 `scripts/fix_onnx_dynamic.py` 处理成「仅 batch 动态」。
3. **live_control 主线程不能同步建引擎 / 首跑热启动**（~55s）——模型创建 + 首帧
   热启动已挪到后台线程（`_ball_load_worker`）。
4. **batch 推理（`detect_batch`）在 4 路时无收益**（TRT batch-4 ≈30ms ≈ 4×batch-1，几乎
   线性），且相机丢帧会触发 batch-3 引擎重建卡死。live_control 热路径用顺序 `detect`，
   `detect_batch` 仅作 API 保留。
