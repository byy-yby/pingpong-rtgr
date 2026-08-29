"""触发配置：外部触发（信号发生器）与软件触发。

研究结论（海康 MVS SDK，标准 GenICam 节点）：

- **外部触发**：信号发生器输出接相机 6-pin I/O 的 **Line0**（输入线），
  四台相机并联到同一路信号即可实现同步曝光。配置方式：
  ``TriggerMode=On`` + ``TriggerSource=Line0`` + ``TriggerActivation=RisingEdge``，
  再对 Line0 设 ``LineDebouncerTime``（us）防抖动误触发。
- **软件触发**：``TriggerSource=Software``，之后每次调用
  ``MV_CC_SetCommandValue("TriggerSoftware")`` 出一帧。没接信号发生器时用它调试。
- **自由采集**：``TriggerMode=Off``，相机按 ``AcquisitionFrameRate`` 连续出图。

所有函数都返回最后一次 SDK 调用的返回码（``MV_OK=0`` 表示成功），失败抛出或返回码由调用方处理。
"""
from __future__ import annotations

from .sdk import MvCamera, MV_OK


def configure_external_trigger(
    cam: MvCamera,
    source: str = "Line0",
    activation: str = "RisingEdge",
    debouncer_us: int = 50,
    delay_us: float = 0.0,
) -> int:
    """配置为外部硬件触发（信号发生器接 Line0）。

    Args:
        cam: 已 open 的 MvCamera 实例。
        source: 触发源，默认 "Line0"。低延迟场景可用 "Line2"。
        activation: 触发沿，默认 "RisingEdge"（上升沿）。
        debouncer_us: Line0 滤波时间（微秒），误触发时可适当加大。
        delay_us: 触发延迟（微秒）。
    """
    ret = cam.MV_CC_SetEnumValueByString("TriggerMode", "On")
    if ret != MV_OK:
        return ret
    ret = cam.MV_CC_SetEnumValueByString("TriggerSource", source)
    if ret != MV_OK:
        return ret
    ret = cam.MV_CC_SetEnumValueByString("TriggerActivation", activation)
    if ret != MV_OK:
        return ret
    ret = cam.MV_CC_SetFloatValue("TriggerDelay", delay_us)
    if ret != MV_OK:
        return ret
    # 关闭触发缓存：避免把历史触发攒起来一次性吐帧
    ret = cam.MV_CC_SetBoolValue("TriggerCacheEnable", False)
    if ret != MV_OK:
        return ret
    ret = cam.MV_CC_SetEnumValueByString("LineSelector", source)
    if ret != MV_OK:
        return ret
    return cam.MV_CC_SetIntValueEx("LineDebouncerTime", debouncer_us)


def configure_software_trigger(cam: MvCamera) -> int:
    """配置为软件触发（调试用，无需信号发生器）。"""
    ret = cam.MV_CC_SetEnumValueByString("TriggerMode", "On")
    if ret != MV_OK:
        return ret
    return cam.MV_CC_SetEnumValueByString("TriggerSource", "Software")


def configure_continuous(cam: MvCamera) -> int:
    """配置为自由采集（连续出图）。"""
    return cam.MV_CC_SetEnumValueByString("TriggerMode", "Off")


def trigger_software_once(cam: MvCamera) -> int:
    """软件触发一帧（仅在软件触发模式下有效）。"""
    return cam.MV_CC_SetCommandValue("TriggerSoftware")


def get_trigger_mode(cam: MvCamera):
    """读取当前触发源符号名（如 "Line0" / "Software"），失败返回 None。"""
    from .sdk import MVCC_ENUMVALUE, MVCC_ENUMENTRY, decode_bytes

    ev = MVCC_ENUMVALUE()
    ret = cam.MV_CC_GetEnumValue("TriggerSource", ev)
    if ret != MV_OK:
        return None
    entry = MVCC_ENUMENTRY()
    entry.nValue = ev.nCurValue
    ret = cam.MV_CC_GetEnumEntrySymbolic("TriggerSource", entry)
    if ret != MV_OK:
        return None
    return decode_bytes(entry.chSymbolic)
