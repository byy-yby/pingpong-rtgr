# IMU 姿态接入（维特智能 WT9011DCL）

把插在乒乓球拍柄里的维特智能 IMU（WT9011DCL，CH340 USB 转串口）读进来，在
`live_control` 的 3D 窗口里实时显示球拍朝向。

## 用法
- `scripts/live_control.py` 里按 **i** 开关。
- 3D 窗口里球拍 + 坐标架随 IMU 朝向实时旋转（**纯朝向，无绝对位置**，锚在桌面中心上方 0.35m）。
- 再按 **i** 关闭，球拍层隐藏。

## 硬件 / 协议
- 型号 **WT9011DCL**，115200 波特率，默认 10Hz，默认上报 **0x61 组合包**（加速度 + 角速度 + 角度）。
- 帧格式 `0x55 | Flag | DataL DataH ... | Checksum`，校验和 = 从 0x55 起到数据末之和的低 8 位。
- 角度 = int16 / 32768 × 180°（欧拉 Z-Y-X）；四元数 = int16 / 32768。数据一律小端。
- 兼容经典协议 0x53 角度 / 0x59 四元数。

## 模块
- `witmotion.py`：0x55 协议解析 + 角度/四元数 -> 旋转矩阵（纯 numpy，无 I/O）。
- `reader.py`：`ImuReader` 后台串口线程 + 自动波特率探测。

## 坑
- **CH340 有时不自动绑定内核 `ch341` 驱动** → 没有 `/dev/ttyUSB*`。拔插一次，或
  `sudo modprobe usbserial vendor=0x1a86 product=0x7523`。
- 需 pyserial（`pip install pyserial`，已装进 `tt` 环境）。
- **未完成**：IMU 安装角（IMU 轴 vs 拍面朝向的固定偏移）默认按「拍面法线=+Z、手柄=+X」，
  实际安装角需按实物校准后在 `viewer3d._paddle_mesh` / 朝向换算处补一个固定旋转。
