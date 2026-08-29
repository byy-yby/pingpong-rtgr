"""相机控制模块。

对外只暴露两个稳定入口：

- :class:`CameraManager`：多相机统一启停（对应「相机启动流程」）。
- :class:`Camera`：单台相机的 open/start/close。

底层 SDK 细节（``sdk`` / ``frame`` / ``trigger`` / ``parameter``）不建议直接 import，
统一从本包导入。
"""
from .camera import Camera
from .camera_manager import CameraManager
from .sdk import enumerate_devices, finalize_sdk, get_sdk_version, initialize_sdk

__all__ = [
    "Camera",
    "CameraManager",
    "enumerate_devices",
    "initialize_sdk",
    "finalize_sdk",
    "get_sdk_version",
]
