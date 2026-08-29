"""SDK 帧缓冲 -> numpy 灰度图，以及 Frame 构造。"""
from __future__ import annotations

import ctypes
from typing import Optional

import numpy as np

from ..core.types import Frame
from .sdk import (
    PixelType_Gvsp_Mono8,
    PixelType_Gvsp_Mono10,
    PixelType_Gvsp_Mono10_Packed,
    PixelType_Gvsp_Mono12,
    PixelType_Gvsp_Mono12_Packed,
)

# 黑白相机的像素格式枚举值（GVSP）。本机固定用 Mono8。
_MONO8 = {PixelType_Gvsp_Mono8, 0x01080001}
_MONO10 = {PixelType_Gvsp_Mono10, 0x01100003}
_MONO12 = {PixelType_Gvsp_Mono12, 0x01100005}
_MONO10_PACKED = {PixelType_Gvsp_Mono10_Packed, 0x010C0004}
_MONO12_PACKED = {PixelType_Gvsp_Mono12_Packed, 0x010C0006}


def buffer_to_image(
    p_buf, width: int, height: int, pixel_format: int, frame_len: int
) -> np.ndarray:
    """把 SDK 帧缓冲拷贝成 numpy 灰度图。

    注意：这里**必须拷贝** —— SDK 在 ``FreeImageBuffer`` 之后会复用同一块缓冲。
    """
    raw = ctypes.string_at(p_buf, frame_len)

    if pixel_format in _MONO8:
        arr = np.frombuffer(raw, dtype=np.uint8)
    elif pixel_format in (_MONO10, _MONO12):
        # 未打包：2 字节 / 像素，低 bit 有效
        arr = np.frombuffer(raw, dtype=np.uint16)
    elif pixel_format in _MONO10_PACKED:
        raise NotImplementedError("Mono10_Packed 解包未实现，请改用 Mono8")
    elif pixel_format in _MONO12_PACKED:
        raise NotImplementedError("Mono12_Packed 解包未实现，请改用 Mono8")
    else:
        raise ValueError(f"不支持的像素格式 0x{pixel_format:08x}，本机黑白相机请用 Mono8")

    return arr.reshape(height, width).copy()


def extract_frame(
    st_frame, camera_id: int, serial: str
) -> Optional[Frame]:
    """从 SDK 的 ``MV_FRAME_OUT`` 结构体构造 :class:`Frame`。

    Args:
        st_frame: ``MV_FRAME_OUT()`` 实例（已由 GetImageBuffer 填充）。
        camera_id: 逻辑相机索引。
        serial: 相机序列号。
    """
    info = st_frame.stFrameInfo
    if not st_frame.pBufAddr:
        return None

    image = buffer_to_image(
        st_frame.pBufAddr, info.nWidth, info.nHeight, info.enPixelType, info.nFrameLen
    )
    device_timestamp = (info.nDevTimeStampHigh << 32) | info.nDevTimeStampLow

    return Frame(
        camera_id=camera_id,
        serial=serial,
        frame_num=info.nFrameNum,
        device_timestamp=device_timestamp,
        host_timestamp=info.nHostTimeStamp,
        image=image,
        pixel_format=info.enPixelType,
        width=info.nWidth,
        height=info.nHeight,
    )
