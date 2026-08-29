# vision/ball — 乒乓球 2D 检测

- `refine.py`：`refine_ball_center` —— 粗定位附近强度加权质心 + 二阶矩求亚像素球心/半径/置信度（纯 numpy，无 scipy）。对 12~24px 的模糊小球有效；不用霍夫圆。
- `classical_ball.py`：`ClassicalBallDetector` —— 背景减除 + 帧差 + 尺寸先验的经典路线，无模型，`register_detector("ball")`。
- `yolo_ball.py`：`YoloBallDetector` —— onnxruntime 推理训练导出的 `best.onnx`（单类 ball，输出 `(1,5,N)`），bbox 后接 `refine_ball_center` 精修，`register_detector("ball_yolo")`。无 torch 依赖，跨相机复用。

## 坑（务必记住）

**provider 列表不能传 `ort.get_available_providers()` 全量**。本机装了 `tensorrt-cu12-libs`，
`TensorrtExecutionProvider` 会在可用列表里，全量传会让 `InferenceSession` 初始化先去建 TRT 引擎
（实测 **~52s**），live_control 主线程按 b 会直接卡死。必须显式
`["CUDAExecutionProvider", "CPUExecutionProvider"]`（无 CUDA 时退 `["CPUExecutionProvider"]`）。
显式 CUDA 后 session 创建 ~1s；CUDA EP 首次推理还有进程级惰性初始化（实测 ~1-6s），
live_control 里已用后台线程加载 + 热启动吸收，勿再同步阻塞主循环。
