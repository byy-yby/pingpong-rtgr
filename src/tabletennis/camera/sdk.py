"""海康 MVS SDK 薄封装：SDK 生命周期 + 设备枚举。

这里集中处理两件容易踩坑的事：

1. **`.so` 定位**：MVS 的 Python 绑定在 ``import`` 时通过环境变量
   ``MVCAM_COMMON_RUNENV`` 拼出 ``libMvCameraControl.so`` 路径
   （实际是 ``$MVCAM_COMMON_RUNENV/64/libMvCameraControl.so``）。
   本机该变量已指向 ``/opt/MVS/lib``，这里再 ``setdefault`` 兜底一次，
   保证任何环境下都能加载。

2. **绑定内部用绝对 import**：``MvCameraControl_class.py`` 里写的是
   ``from PixelType_header import *`` 这种**不带点**的 import，因此必须把
   ``mv_import`` 目录加进 ``sys.path`` 才能 import 成功。

所有其它模块（camera / parameter / trigger …）都应从本模块 import SDK 符号，
不要直接碰 ``mv_import``，这样 SDK 的引导逻辑只存在一处。
"""
from __future__ import annotations

import os
import sys
from ctypes import POINTER, cast
from dataclasses import dataclass
from typing import List, Optional

_MV_IMPORT_DIR = os.path.join(os.path.dirname(__file__), "mv_import")

# 必须在 import MvCameraControl_class 之前设置，否则加载 .so 会拼出 None 路径抛 TypeError
os.environ.setdefault("MVCAM_COMMON_RUNENV", "/opt/MVS/lib")

if _MV_IMPORT_DIR not in sys.path:
    sys.path.insert(0, _MV_IMPORT_DIR)

# noqa 抑制 "import not at top" 风格告警 —— 必须先加 sys.path 再 import
from MvCameraControl_class import MvCamera  # noqa: E402
from CameraParams_header import (  # noqa: E402
    MV_CC_DEVICE_INFO,
    MV_CC_DEVICE_INFO_LIST,
    MV_FRAME_OUT,
    MV_FRAME_OUT_INFO_EX,
    MVCC_ENUMVALUE,
    MVCC_FLOATVALUE,
    MVCC_INTVALUE,
    MVCC_STRINGVALUE,
    MVCC_ENUMENTRY,
)
from CameraParams_const import (  # noqa: E402
    MV_ACCESS_Exclusive,
    MV_GENTL_CAMERALINK_DEVICE,
    MV_GENTL_CXP_DEVICE,
    MV_GENTL_GIGE_DEVICE,
    MV_GENTL_XOF_DEVICE,
    MV_GIGE_DEVICE,
    MV_USB_DEVICE,
)
from MvErrorDefine_const import MV_OK  # noqa: E402
from PixelType_header import (  # noqa: E402
    PixelType_Gvsp_Mono8,
    PixelType_Gvsp_Mono10,
    PixelType_Gvsp_Mono10_Packed,
    PixelType_Gvsp_Mono12,
    PixelType_Gvsp_Mono12_Packed,
)

# 枚举设备时覆盖的所有传输层类型
ALL_LAYER_TYPES = (
    MV_GIGE_DEVICE
    | MV_USB_DEVICE
    | MV_GENTL_GIGE_DEVICE
    | MV_GENTL_CAMERALINK_DEVICE
    | MV_GENTL_CXP_DEVICE
    | MV_GENTL_XOF_DEVICE
)

_LAYER_NAMES = {
    MV_GIGE_DEVICE: "GigE",
    MV_USB_DEVICE: "USB3",
    MV_GENTL_GIGE_DEVICE: "GenTL-GigE",
    MV_GENTL_CAMERALINK_DEVICE: "GenTL-CameraLink",
    MV_GENTL_CXP_DEVICE: "GenTL-CXP",
    MV_GENTL_XOF_DEVICE: "GenTL-XoF",
}


@dataclass
class DeviceInfo:
    """一台在线设备的信息，供 :func:`enumerate_devices` 返回。

    Attributes:
        index: 枚举到的序号。
        layer_type: 传输层类型（MV_USB_DEVICE 等）。
        layer_name: 传输层类型的可读名。
        model: 型号名。
        serial: 序列号。
        user_defined_name: 用户自定义名（可为空）。
        raw: 底层 ``MV_CC_DEVICE_INFO`` 结构体，创建句柄时用。
    """

    index: int
    layer_type: int
    layer_name: str
    model: str
    serial: str
    user_defined_name: str
    raw: object


def decode_bytes(char_array) -> str:
    """把 ctypes 字符数组安全解码成字符串（型号 / 序列号等）。"""
    byte_str = memoryview(char_array).tobytes()
    null_index = byte_str.find(b"\x00")
    if null_index != -1:
        byte_str = byte_str[:null_index]
    for encoding in ("gbk", "utf-8", "latin-1"):
        try:
            return byte_str.decode(encoding)
        except UnicodeDecodeError:
            continue
    return byte_str.decode("latin-1", errors="replace")


def initialize_sdk() -> None:
    """初始化 SDK（全局，一次即可）。"""
    ret = MvCamera.MV_CC_Initialize()
    if ret != MV_OK:
        raise RuntimeError(f"MVS SDK 初始化失败: 0x{ret:08x}")


def finalize_sdk() -> None:
    """反初始化 SDK。"""
    MvCamera.MV_CC_Finalize()


def get_sdk_version() -> int:
    """SDK 版本号（形如 0x04080000）。"""
    return MvCamera.MV_CC_GetSDKVersion()


def enumerate_devices() -> List[DeviceInfo]:
    """枚举在线设备，返回 :class:`DeviceInfo` 列表。

    要求先调用 :func:`initialize_sdk`。
    """
    device_list = MV_CC_DEVICE_INFO_LIST()
    ret = MvCamera.MV_CC_EnumDevices(ALL_LAYER_TYPES, device_list)
    if ret != MV_OK:
        raise RuntimeError(f"枚举设备失败: 0x{ret:08x}")

    devices: List[DeviceInfo] = []
    for i in range(device_list.nDeviceNum):
        info = cast(device_list.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
        layer = info.nTLayerType
        special = info.SpecialInfo

        model = serial = user_name = ""
        if layer == MV_USB_DEVICE:
            u = special.stUsb3VInfo
            model = decode_bytes(u.chModelName)
            serial = decode_bytes(u.chSerialNumber)
            user_name = decode_bytes(u.chUserDefinedName)
        elif layer in (MV_GIGE_DEVICE, MV_GENTL_GIGE_DEVICE):
            g = special.stGigEInfo
            model = decode_bytes(g.chModelName)
            serial = decode_bytes(g.chSerialNumber)
            user_name = decode_bytes(g.chUserDefinedName)
        # CML / CXP / XoF 等协议本项目不用，留空即可

        devices.append(
            DeviceInfo(
                index=i,
                layer_type=layer,
                layer_name=_LAYER_NAMES.get(layer, "unknown"),
                model=model,
                serial=serial,
                user_defined_name=user_name,
                raw=info,
            )
        )
    return devices
