# 全系统性能优化报告（球检测 + 姿态检测）

> 日期：2026-08-31　|　硬件：RTX 5080 (16GB) + onnxruntime 1.26（TensorRT/CUDA）
> 数据来源：`profile_system.py` / `bench_ball_variants.py` 实测（相机用 1440×1080 Mono8 合成帧模拟）。

---

## 0. 基线（现状，实测）

| 模块 | 现状 | 说明 |
|---|---|---|
| **球检测** | **~20–23 ms/相机**，4 相机**串行** → 整轮 ~80–92 ms ≈ **12 FPS** | 最大瓶颈，本报告重点 |
| 姿态 YOLOX 段 | ~12 ms/轮（4 帧 batch） | 已上 TensorRT，较优 |
| 姿态 RTMPose 段 | ~1.8 ms/人（batch 后 ~0.3 ms/人推理） | 已上 TensorRT + batch，基本到位 |
| DLT 三角化 | 姿态 6.5 ms/人、球 0.33 ms | 纯 numpy，可向量化 |
| CPU↔GPU 传输 | 球 2.2 ms(H2D)、姿态 0.66 ms | **非瓶颈** |

球检测慢的根源：① 预处理里反复整图拷贝（8–11 ms）；② 1280 大输入 + fp32 推理（~6 ms）；③ 4 相机串行没批处理。

---

## 优化 ①　球检测：融合灰度→3 通道预处理（零风险，先做）

### 改哪里
`src/tabletennis/vision/ball/yolo_ball.py` → `detect()` 里这两行（约 105–106 行）。

### 改动前后

```python
# —— 现在（5 步，反复新建整张大数组）——
inp = cv2.cvtColor(lb, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0
inp = inp.transpose(2, 0, 1)[None].astype(np.float32)

# —— 改成（3 步，只搬一次）——
f = lb.astype(np.float32)                       # 灰度→浮点（一次）
f *= (1.0 / 255.0)                              # 就地归一化，不新建数组
inp = np.empty((1, 3, self.imgsz, self.imgsz), np.float32)
inp[:] = f                                      # 一次广播：复制成 3 通道 + 加 batch 维
```

### 原理
`lb` 是 1280×1280 灰度图（1.6 MB）。原代码里 `cvtColor`→`astype`→`/255`→`transpose` 每步都把 18.75 MB 的 float32 数组**再读一遍写一遍**，共搬 ~75 MB。改后：`*=` 就地归一化、`inp[:] = f` 用 numpy 广播一步完成"1 通道→3 通道 + batch"，共搬 ~30 MB。

### 效果 / 影响
- **8–11 ms → 预期 3–4 ms**（约 3×）。
- **对检测结果零影响**：喂进模型的仍是同样形状 `(1,3,1280,1280)`、同样 [0,1] 数值，只差浮点最后一位舍入。

---

## 优化 ②　球检测：分辨率 1280 → 640（速度↑、精度需验证，是个取舍）

### 改哪里（两处）
1. **重导出模型文件**（不改代码）：
   ```python
   from ultralytics import YOLO
   YOLO("best.pt").export(format="onnx", imgsz=640, simplify=True)
   ```
2. `yolo_ball.py` → `YoloBallDetector.__init__` 默认参数 `imgsz: int = 1280`（约 70 行）改为 `640`，让 `_letterbox` 缩放到 640。

### 效果
1280×1280=164 万像素 → 640×640=41 万像素（**4× 更少**）。推理 **6 → 2.6 ms**（CUDA fp32，实测），预处理再降 4×。

### 对算法的影响（重点）
`letterbox` 是等比缩放后补边。球在原图只有 12–24 px，缩放后：

| 输入 | 缩放比 | 球在模型眼里 |
|---|---|---|
| 1280（训练/现用） | 0.889 | 10.7–21 px |
| 960 | 0.667 | 8–16 px |
| 640 | 0.444 | **5.3–10.7 px** |

YOLOv8 最小检测尺度是 **8 px**（stride=8）。640 下大半球落到 8px 以下 → **小尺寸/高速球漏检率上升**。模型是在 1280 上训练的，640 是没训练过的尺度。
**折中：用 960**（球 8–16 px，仍高于底线），速度提 ~1.8×，风险小得多。**改 640 前必须先跑验证集对比 recall。**

---

## 优化 ③　球检测：TensorRT FP16（零精度损失，引擎构建放后台）

### 改哪里
`yolo_ball.py` → `__init__` 里创建 session 的这段（约 88–96 行）。

### 改动前后

```python
# —— 现在（只用 CUDA fp32）——
if "CUDAExecutionProvider" in ort.get_available_providers():
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
else:
    providers = ["CPUExecutionProvider"]
self.session = ort.InferenceSession(model_path, providers=providers)

# —— 改成（优先 TensorRT fp16，失败回退 CUDA）——
import os
cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "tabletennis", "ball_trt")
os.makedirs(cache_dir, exist_ok=True)
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
if "TensorrtExecutionProvider" in ort.get_available_providers():
    providers = [
        ("TensorrtExecutionProvider", {
            "device_id": 0,
            "trt_fp16_enable": True,
            "trt_engine_cache_enable": True,          # 引擎只建一次，之后秒开
            "trt_engine_cache_path": cache_dir,
        }),
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
elif "CUDAExecutionProvider" in ort.get_available_providers():
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
else:
    providers = ["CPUExecutionProvider"]
self.session = ort.InferenceSession(model_path, sess_options=so, providers=providers)
```

### 为什么现在没上、以及怎么补
现在 `yolo_ball.py` **故意避开了 TensorRT**（注释里写了：怕全量 provider 触发 TRT 引擎构建 ~52 s，主线程按 b 直接卡死 UI）。正确做法不是避开，而是：**引擎构建放后台线程**——`live_control.py` 里已经有 `_ball_load_worker` 后台加载线程（第 513 行），把模型创建 + 首帧热启动挪到那里即可，UI 不卡。

### 效果 / 影响
FP16 精度下降可忽略（实测对检测结果无影响）。推理：
- 1280：**6 → 3.4 ms**（实测）
- 640：**2.6 → 0.96 ms**（实测）

---

## 优化 ④　球检测：4 相机批处理（零精度损失）

### 改哪里（两处）
1. `yolo_ball.py` 新增 `detect_batch(frames)`：把 4 台相机的灰度图各自 letterbox 后堆成 `(4,3,imgsz,imgsz)` **一次 `session.run`**，再按相机拆回结果。
2. `live_control.py` → `_reconstruct_ball_frame`（约 576–580 行）：把
   ```python
   for cid, f in frames_items:
       balls_per_cam[cid] = detector.detect(f)      # 现在的串行
   ```
   换成 `balls_per_cam = detector.detect_batch([f for _, f in frames_items])`。

### 效果
合并 4 次 `session.run` 的固定开销（H2D 建立、kernel 启动）为 1 次。实测：
- 1280 fp32：4×6=24 ms → ~19 ms（批处理省 ~20%，主要省固定开销）
- 1280 **TRT**：4×3.4=13.6 → **11.5 ms**；640 **TRT**：4×0.96=3.8 → **3.2 ms**

### 前提
模型 ONNX 需要**动态 batch**。训练脚本默认导出是固定 batch=1，批处理前需 `export(..., dynamic=True)` 重导出一份动态 batch 的 best.onnx。

---

## 姿态检测优化（补充，已较优，剩余小头）

### ⑤ YOLOX decode+NMS（~3.7 ms）移到 GPU
- **改哪里**：`src/tabletennis/vision/pose/rtmpose_pose.py` → `_yolox_decode_batch`（约 185–212 行）。
- **现状**：动态 batch 导出的 YOLOX 输出原始 `(B,3549,85)`，解码 + 逐类 NMS 全在 numpy CPU 上做，占 YOLOX 段 ~1/3。
- **改法**：把这段 numpy 解码/NMS 用 torch 在 GPU 上做（或把 EfficientNMS 烤回 ONNX）。预期 **3.7 → <0.5 ms**。
- **影响**：无（同样的框，只是算得更快）。

### ⑥ DLT 三角化向量化（6.5 ms/人 → ~1–2 ms/人）
- **改哪里**：`src/tabletennis/reconstruction/triangulate.py` → `triangulate_pose`（约 215–278 行）。
- **现状**：26 个关节逐个 `for j in range(n_joints)` 循环，每个关节点单独做一次小矩阵 SVD + 重投影 + 交会角。
- **改法**：把 26 关节堆成 `(26, 8, 4)` 一次性 `np.linalg.svd`（支持 batch），去掉逐点 Python 循环。**6.5 → 1–2 ms/人**。
- **影响**：无（同一 DLT 公式，只是批量算）。

### ⑦ 姿态灰度→BGR 融合（1.24 ms → ~0.5 ms）
- **改哪里**：`rtmpose_pose.py` → `detect_batch` 里每帧 `cv2.cvtColor(frame.image, COLOR_GRAY2BGR)`（约 452、477 行）。
- **改法**：先对灰度 letterbox 再复制 3 通道（复用优化 ① 的思路），少转一次。影响小，优先级最低。

---

## 组合效果预估（4 相机一轮）

| 方案 | 球检测（4 相机） | 说明 |
|---|---|---|
| 现状 | ~80–92 ms ≈ 12 FPS | 串行，CUDA fp32@1280 |
| ① 融合预处理 | ~56–72 ms | 每相机省 ~6 ms |
| ①+③（TRT@1280） | ~36–44 ms | 推理 6→3.4 ms |
| ①+③+④（TRT+批处理@1280） | **~22–28 ms ≈ 40 FPS** | 批处理省固定开销 |
| ①+②+③+④（TRT+批处理@640） | **~8–12 ms ≈ 90 FPS** | 需先验证 640 的 recall |

姿态重建（1 人）：现状 ~20 ms/轮 ≈ 48 FPS；做完 ⑤⑥ → **~12 ms/轮 ≈ 80 FPS**。

---

## 实施顺序建议

1. **先做 ①**（融合预处理）——纯代码、零风险、立竿见影。
2. **再做 ③+④**（球 TRT + 批处理）——零精度损失，把球从 12 FPS 提到 ~40 FPS。
3. **⑥⑤⑦**（姿态 DLT 向量化 + YOLOX decode 上 GPU + 灰度融合）——姿态再提一档。
4. **最后 ②**（降分辨率）——唯一伤精度的，做完前必须跑验证集确认 recall；先试 960，再考虑 640。
