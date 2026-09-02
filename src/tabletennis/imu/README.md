# IMU 姿态接入（维特智能 WT9011DCL-BT5.0）

把插在乒乓球拍柄里的维特智能 IMU（WT9011DCL-BT5.0，**蓝牙 5.0 / BLE**）读进来，在
`live_control` 的 3D 窗口里实时显示球拍朝向。

## 用法
- `scripts/live_control.py` 里按 **i** 开关（自动 BLE 扫描名字含 "WT" 的模块并连接）。
- 3D 窗口里球拍 + 坐标架随 IMU 朝向实时旋转（**纯朝向，无绝对位置**，锚在桌面中心上方 0.35m）。
- 再按 **i** 关闭，球拍层隐藏。
- 扫不到 / 有多个模块时用 `--imu-mac AA:BB:CC:DD:EE:FF` 或 `--imu-name <名字子串>` 直接指定。

## BLE 连接
- GATT 服务 `0000ffe5`、notify 收数据 `0000ffe4`、写命令 `0000ffe9`
  （UUID 来自维特官方 SDK `WitBluetooth_BWT901BLE5_0` 的 `BleUUID.java`）。
- 依赖 `bleak`（`pip install bleak`，已装进 `tt` 环境）。
- 广播名以 **"WT"** 开头（如 WT9011DCL…）。

## 数据协议（BLE 与 UART 同一套帧格式）
- 默认上报 **0x61 组合包**（加速度6B + 角速度6B + 角度6B）@10Hz（可 0.2~200Hz）。
- 帧格式 `0x55 | Flag | DataL DataH ... | Checksum`，校验和 = 从 0x55 起到数据末之和的低 8 位。
- 角度 = int16 / 32768 × 180°（欧拉 Z-Y-X）；四元数 = int16 / 32768。数据一律小端。
- 兼容经典 0x53 角度 / 0x59 四元数。
- **BLE 单包最多 20 字节**，21 字节的 0x61 包会跨 notify 拆分，靠解析器的增量缓冲 + 校验和重组。

## 模块
- `witmotion.py`：0x55 协议解析 + 角度/四元数 -> 旋转矩阵（纯 numpy，无 I/O）。
- `reader.py`：`ImuReader` 后台 BLE 线程（bleak，扫描/连接/notify）。

## 坑
- 模块上电后可能被手机 / 上位机占用（BLE 一般只允许单连接），连不上时先断开其它设备，或用 `--imu-mac` 直接指定。
- 需有可用的蓝牙适配器（`bluetoothctl list` 应看到 hci0）。
- **未完成**：IMU 安装角（IMU 轴 vs 拍面朝向的固定偏移）默认按「拍面法线=+Z、手柄=+X」，
  实际安装角需按实物校准后在 `viewer3d._paddle_mesh` / 朝向换算处补一个固定旋转。
