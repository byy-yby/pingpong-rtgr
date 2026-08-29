"""图像质量控制：曝光、增益、黑电平、伽马、亮度、对比度、帧率。

研究结论（海康 MVS SDK，标准 GenICam / SFNC 节点，经 ``MV_CC_Set/Get*Value`` 读写）：

- 这些参数都是相机的 **GenICam 特征节点**，通过字符串 key 访问：
  ``ExposureTime`` / ``ExposureAuto`` / ``Gain`` / ``GainAuto`` /
  ``BlackLevel`` / ``Gamma`` / ``Brightness`` / ``Contrast`` /
  ``AcquisitionFrameRate`` / ``ResultingFrameRate`` / ``PixelFormat``。
- 本机是**黑白相机**（MV-CS016-10UM），**没有**白平衡（BalanceWhiteAuto）与
  饱和度（Saturation）节点；调节画面亮度靠
  **曝光时间 + 增益 + 黑电平 + 伽马** 组合。
- "Brightness" 是偏显示/软件侧的节点，部分海康机型不存在 —— 本类里做存在性探测，
  节点不存在时 get 返回 ``None``、set 返回非 0 错误码（不抛异常）。
- 设置 ``ExposureTime`` / ``Gain`` 之前，通常要先关掉对应的自动模式
  （``ExposureAuto`` / ``GainAuto`` = Off），否则手动值会被自动算法覆盖。
- 读值接口返回的 ``MVCC_FLOATVALUE`` / ``MVCC_INTVALUE`` 结构体里自带
  ``fMax/fMin``、``nMax/nMin/nInc``，因此可以直接查询每个参数的合法范围。
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

from .sdk import (
    MvCamera,
    MV_OK,
    MVCC_ENUMVALUE,
    MVCC_ENUMENTRY,
    MVCC_FLOATVALUE,
    MVCC_INTVALUE,
    decode_bytes,
)


class ImageControl:
    """封装单台相机所有图像质量相关参数的读写。

    直接持有 ``MvCamera`` 实例（已 open），不负责开关相机。
    """

    def __init__(self, cam: MvCamera):
        self.cam = cam

    # ------------------------------------------------------------------
    # 底层通用读写
    # ------------------------------------------------------------------
    def _get_float(self, node: str) -> Optional[float]:
        val = MVCC_FLOATVALUE()
        if self.cam.MV_CC_GetFloatValue(node, val) != MV_OK:
            return None
        return float(val.fCurValue)

    def _set_float(self, node: str, value: float) -> int:
        return self.cam.MV_CC_SetFloatValue(node, float(value))

    def _get_float_range(self, node: str) -> Optional[Tuple[float, float]]:
        val = MVCC_FLOATVALUE()
        if self.cam.MV_CC_GetFloatValue(node, val) != MV_OK:
            return None
        return float(val.fMin), float(val.fMax)

    def _get_int_range(self, node: str) -> Optional[Tuple[int, int, int]]:
        val = MVCC_INTVALUE()
        if self.cam.MV_CC_GetIntValueEx(node, val) != MV_OK:
            return None
        return int(val.nMin), int(val.nMax), int(val.nInc)

    def _get_enum_str(self, node: str) -> Optional[str]:
        ev = MVCC_ENUMVALUE()
        if self.cam.MV_CC_GetEnumValue(node, ev) != MV_OK:
            return None
        entry = MVCC_ENUMENTRY()
        entry.nValue = ev.nCurValue
        if self.cam.MV_CC_GetEnumEntrySymbolic(node, entry) != MV_OK:
            return None
        return decode_bytes(entry.chSymbolic)

    def _set_enum_str(self, node: str, value: str) -> int:
        return self.cam.MV_CC_SetEnumValueByString(node, value)

    # ------------------------------------------------------------------
    # 曝光
    # ------------------------------------------------------------------
    def set_exposure_time_us(self, t_us: float) -> int:
        """设置曝光时间（微秒）。会先关掉自动曝光。"""
        self.cam.MV_CC_SetEnumValue("ExposureAuto", 0)
        return self._set_float("ExposureTime", t_us)

    def get_exposure_time_us(self) -> Optional[float]:
        return self._get_float("ExposureTime")

    def get_exposure_range_us(self) -> Optional[Tuple[float, float]]:
        """曝光时间合法范围（微秒）。"""
        return self._get_float_range("ExposureTime")

    def set_exposure_auto(self, on: bool) -> int:
        """开关自动曝光（on=True 时手动曝光失效）。"""
        return self.cam.MV_CC_SetEnumValue("ExposureAuto", 1 if on else 0)

    # ------------------------------------------------------------------
    # 增益
    # ------------------------------------------------------------------
    def set_gain_db(self, g_db: float) -> int:
        """设置增益（dB）。会先关掉自动增益。"""
        self.cam.MV_CC_SetEnumValue("GainAuto", 0)
        return self._set_float("Gain", g_db)

    def get_gain_db(self) -> Optional[float]:
        return self._get_float("Gain")

    def get_gain_range_db(self) -> Optional[Tuple[float, float]]:
        return self._get_float_range("Gain")

    def set_gain_auto(self, on: bool) -> int:
        return self.cam.MV_CC_SetEnumValue("GainAuto", 1 if on else 0)

    # ------------------------------------------------------------------
    # 黑电平（"曝光补偿"的暗部偏移，类似亮度下限）
    # ------------------------------------------------------------------
    def set_black_level(self, value: float) -> int:
        return self._set_float("BlackLevel", value)

    def get_black_level(self) -> Optional[float]:
        return self._get_float("BlackLevel")

    def get_black_level_range(self) -> Optional[Tuple[float, float]]:
        return self._get_float_range("BlackLevel")

    # ------------------------------------------------------------------
    # 伽马
    # ------------------------------------------------------------------
    def set_gamma(self, value: float) -> int:
        return self._set_float("Gamma", value)

    def get_gamma(self) -> Optional[float]:
        return self._get_float("Gamma")

    def get_gamma_range(self) -> Optional[Tuple[float, float]]:
        return self._get_float_range("Gamma")

    # ------------------------------------------------------------------
    # 亮度 / 对比度（节点可能不存在，做了存在性探测）
    # ------------------------------------------------------------------
    def set_brightness(self, value: int) -> int:
        """设置亮度（Integer 节点，部分机型不存在）。"""
        return self.cam.MV_CC_SetIntValueEx("Brightness", int(value))

    def get_brightness(self) -> Optional[int]:
        val = MVCC_INTVALUE()
        if self.cam.MV_CC_GetIntValueEx("Brightness", val) != MV_OK:
            return None
        return int(val.nCurValue)

    def get_brightness_range(self) -> Optional[Tuple[int, int, int]]:
        return self._get_int_range("Brightness")

    def set_contrast(self, value: float) -> int:
        return self._set_float("Contrast", value)

    def get_contrast(self) -> Optional[float]:
        return self._get_float("Contrast")

    # ------------------------------------------------------------------
    # 帧率 / 像素格式
    # ------------------------------------------------------------------
    def set_frame_rate(self, hz: float) -> int:
        return self._set_float("AcquisitionFrameRate", hz)

    def get_frame_rate(self) -> Optional[float]:
        return self._get_float("AcquisitionFrameRate")

    def get_resulting_frame_rate(self) -> Optional[float]:
        """当前实际帧率（只读，受曝光时间等约束）。"""
        return self._get_float("ResultingFrameRate")

    def set_pixel_format(self, fmt: str = "Mono8") -> int:
        """设置像素格式（黑白相机用 Mono8）。"""
        return self._set_enum_str("PixelFormat", fmt)

    def get_pixel_format(self) -> Optional[str]:
        return self._get_enum_str("PixelFormat")

    # ------------------------------------------------------------------
    # 汇总 / 探测
    # ------------------------------------------------------------------
    def list_supported(self) -> Dict[str, object]:
        """探测常用图像控制节点，返回「节点名 -> 当前值 / 范围」字典。

        节点不存在时对应项为 ``None``，不会抛异常。可直接用来判断
        某台相机到底支持哪些参数、取值范围是多少。
        """
        return {
            "ExposureTime_us": self.get_exposure_time_us(),
            "ExposureTime_range_us": self.get_exposure_range_us(),
            "Gain_db": self.get_gain_db(),
            "Gain_range_db": self.get_gain_range_db(),
            "BlackLevel": self.get_black_level(),
            "BlackLevel_range": self.get_black_level_range(),
            "Gamma": self.get_gamma(),
            "Gamma_range": self.get_gamma_range(),
            "Brightness": self.get_brightness(),
            "Brightness_range": self.get_brightness_range(),
            "Contrast": self.get_contrast(),
            "PixelFormat": self.get_pixel_format(),
            "AcquisitionFrameRate_hz": self.get_frame_rate(),
            "ResultingFrameRate_hz": self.get_resulting_frame_rate(),
        }
