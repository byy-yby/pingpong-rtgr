"""IMU 惯性测量单元接入：维特智能 (WitMotion) 模块串口读取 + 姿态解析。

- :mod:`witmotion`：0x55 协议解析 + 角度/四元数 -> 旋转矩阵（纯 numpy，无 I/O）。
- :mod:`reader`：后台串口读取线程，线程安全暴露最新姿态。

用法（配合 ``scripts/live_control.py`` 按 ``i``）：
  reader = ImuReader()          # 自动找 /dev/ttyUSB*，115200 + 自动波特率兜底
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
