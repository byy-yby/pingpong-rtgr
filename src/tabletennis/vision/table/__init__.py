"""球桌识别：用两个大 ArUco 标记定位桌面世界系。

实现 :class:`TableDetector`（已注册到 ``vision/detector.py`` 的工厂），
``scripts/live_control.py`` 按 T 键时做一次性识别，把标准尺寸球桌投影到
各相机视角并生成 Open3D 3D 场景。
"""
from .table_detector import TableDetector

__all__ = ["TableDetector"]
