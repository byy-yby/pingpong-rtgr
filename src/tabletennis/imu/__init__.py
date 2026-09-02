"""IMU 惯性测量单元接入：维特智能 (WitMotion) BLE 模块读取 + 姿态解析。

- :mod:`witmotion`：0x55 协议解析 + 角度/四元数 -> 旋转矩阵（纯 numpy，无 I/O）。
- :mod:`reader`：后台 BLE 线程，线程安全暴露最新姿态。

用法（配合 ``scripts/live_control.py`` 按 ``i``）：
  reader = ImuReader()          # BLE 扫描名字含 "WT" 的维特模块并连接
  reader.start()
  ...
  R = reader.latest_rotation()  # 3x3 旋转矩阵（body -> world），无新数据为 None
"""
from .reader import ImuReader
from .witmotion import WitMotionParser, angle_to_rotmat, quat_to_rotmat

__all__ = [
    "WitMotionParser",
    "angle_to_rotmat",
    "quat_to_rotmat",
    "ImuReader",
]
