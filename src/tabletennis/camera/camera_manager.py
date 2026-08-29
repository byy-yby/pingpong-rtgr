"""多相机统一管理：SDK 生命周期 + 按序列号/索引选相机 + 统一启停。

对外的「相机启动流程」入口就是这里：

    CameraManager.__enter__()
        ├─ sdk.initialize_sdk()
        ├─ sdk.enumerate_devices() -> 按 serial / index 匹配
        └─ 对每台 Camera.open()  （建立句柄 + 触发/曝光配置）
    CameraManager.start()
        └─ 对每台 Camera.start()  （StartGrabbing + 采集线程）
    CameraManager.stop() / close()
        └─ 逆序 stop / close

用法：

    with CameraManager(trigger_mode="external") as mgr:
        mgr.start()
        while running:
            bundle = mgr.get_latest_bundle()
            ...
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Union

from .camera import Camera
from .sdk import (
    DeviceInfo,
    enumerate_devices,
    finalize_sdk,
    initialize_sdk,
)
from ..core.types import FrameBundle

logger = logging.getLogger(__name__)


def _kwargs_from_config(config: dict) -> dict:
    """把 ``config/cameras.yaml`` 的 dict 转成 :class:`Camera` 的关键字参数。

    兼容早期约定（``CameraManager(config)``）与当前约定（``CameraManager(**kwargs)``）。
    """
    cfg = config or {}
    trigger = cfg.get("trigger") or {}
    image = cfg.get("image") or {}
    exposure = cfg.get("exposure") or {}
    gain = cfg.get("gain") or {}

    mode = trigger.get("mode", "continuous")
    kwargs = {
        "trigger_mode": mode,
        "trigger_source": trigger.get("source", "Line0"),
        "pixel_format": image.get("pixel_format", "Mono8"),
        "exposure_us": exposure.get("time_us") if not exposure.get("auto", False) else None,
        "gain_db": gain.get("db") if not gain.get("auto", False) else None,
    }
    if mode == "software":
        kwargs["software_rate_hz"] = image.get("frame_rate_hz", 30.0)
    return kwargs


class CameraManager:
    """管理 N 台相机，统一 SDK 生命周期与启停。

    支持两种构造方式（内部统一为 camera kwargs）：

    - ``CameraManager(config_dict)`` —— 早期约定，config 结构见 config/cameras.yaml。
    - ``CameraManager(serials=..., indices=..., **camera_kwargs)`` —— 关键字方式。

    同时提供两套等价的启停方法名（历史约定 + 直观命名）：

    - ``setup()`` == ``open_all()``
    - ``start()`` == ``start_all()``
    - ``stop()``  == ``stop_all()``
    - ``close()`` == ``close_all()``
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        serials: Optional[Sequence[str]] = None,
        indices: Optional[Sequence[int]] = None,
        **camera_kwargs,
    ) -> None:
        # 兼容 CameraManager(config) 形式
        if isinstance(config, dict):
            serials, indices = self._parse_camera_list(config)
            camera_kwargs = {**_kwargs_from_config(config), **camera_kwargs}

        self._serials = list(serials) if serials else None
        self._indices = list(indices) if indices else None
        self._camera_kwargs = camera_kwargs
        self._cameras: Dict[int, Camera] = {}
        self._sdk_initialized = False

    @staticmethod
    def _parse_camera_list(config: dict):
        """从 config 的 ``cameras`` 列表提取序列号/索引选择。"""
        cams = config.get("cameras") or []
        serials = [c["serial"] for c in cams if c.get("serial")]
        indices = [c["index"] for c in cams if not c.get("serial")]
        return (serials or None), (indices or None)

    # ------------------------------------------------------------------
    # 上下文管理
    # ------------------------------------------------------------------
    def __enter__(self) -> "CameraManager":
        self.setup()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def cameras(self) -> List[Camera]:
        return list(self._cameras.values())

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def setup(self) -> None:
        """初始化 SDK、枚举设备并按需打开相机（不启动采集）。"""
        if self._cameras:
            return

        initialize_sdk()
        self._sdk_initialized = True
        devices = enumerate_devices()
        if not devices:
            raise RuntimeError("未枚举到任何相机。请检查 USB 连接，并确认 MVS 客户端已关闭。")

        selected = self._select_devices(devices)

        for logical_id, dev in enumerate(selected):
            cam = Camera(dev, logical_id=logical_id, **self._camera_kwargs)
            cam.open()
            self._cameras[logical_id] = cam

        logger.info("CameraManager 打开 %d/%d 台相机", len(self._cameras), len(devices))

    def start(self) -> None:
        """启动所有相机的采集线程。"""
        if not self._cameras:
            raise RuntimeError("请先调用 setup() / 进入 with 块")
        for cam in self._cameras.values():
            cam.start()

    def stop(self) -> None:
        """停止所有相机采集（不关设备、不销毁句柄）。"""
        for cam in self._cameras.values():
            cam.stop()

    def close(self) -> None:
        """关闭所有相机并反初始化 SDK。"""
        for cam in self._cameras.values():
            try:
                cam.close()
            except Exception as exc:  # noqa: BLE001 —— 释放阶段不能因为单台失败而中断
                logger.warning("关闭相机 %s 失败: %s", cam.logical_id, exc)
        self._cameras.clear()

        if self._sdk_initialized:
            finalize_sdk()
            self._sdk_initialized = False

    # ------------------------------------------------------------------
    # 取帧 / 组帧
    # ------------------------------------------------------------------
    def get_latest_frame(self, logical_id: int, block: bool = True, timeout: float = 1.0):
        cam = self._cameras.get(logical_id)
        return cam.get_latest_frame(block=block, timeout=timeout) if cam else None

    def get_latest_frames(self, block: bool = True, timeout: float = 1.0) -> Dict[int, object]:
        """每台相机各取一帧（逻辑索引 -> Frame）。

        用于实时预览 / 逐相机姿态检测。**不做时间对齐**——那属于四机同步组帧
        （外触发下需按设备时间戳对齐），留待 pipeline 阶段实现。
        """
        return {
            cid: cam.get_latest_frame(block=block, timeout=timeout)
            for cid, cam in self._cameras.items()
        }

    def get_latest_bundle(self, block: bool = True, timeout: float = 1.0) -> FrameBundle:
        """取每台相机最新一帧，包成 :class:`FrameBundle`。

        注：当前是「各机各自最新帧」，未按设备时间戳严格对齐；外部硬触发下
        四机曝光已对齐，对实时预览足够。严格对齐留待 pipeline 阶段。
        """
        bundle = FrameBundle()
        for cid, cam in self._cameras.items():
            frame = cam.get_latest_frame(block=block, timeout=timeout)
            if frame is not None:
                bundle.frames[cid] = frame
        return bundle

    def get_synchronized_bundle(self, block: bool = True, timeout: float = 1.0) -> FrameBundle:
        """严格对齐四机帧：先清空各机队列，再各取下一帧（= 同一触发周期）。

        只有在外触发（信号发生器接 Line0，共享时钟）下才严格对齐：清空后各机
        的下一帧来自同一次触发边沿，四机曝光在同一时钟周期内。连续 / 软触发
        模式下则退化为「各机各自下一帧」（不做时钟对齐，板静止时结果一致）。
        """
        for cam in self._cameras.values():
            cam.drain()
        bundle = FrameBundle()
        for cid, cam in self._cameras.items():
            frame = cam.get_latest_frame(block=block, timeout=timeout)
            if frame is not None:
                bundle.frames[cid] = frame
        return bundle

    # ------------------------------------------------------------------
    # 别名（历史约定）
    # ------------------------------------------------------------------
    def open_all(self) -> None:
        """== setup()"""
        self.setup()

    def start_all(self) -> None:
        """== start()"""
        self.start()

    def stop_all(self) -> None:
        """== stop()"""
        self.stop()

    def close_all(self) -> None:
        """== close()"""
        self.close()

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    def _select_devices(self, devices: List[DeviceInfo]) -> List[DeviceInfo]:
        if self._serials:
            by_serial = {d.serial: d for d in devices}
            selected: List[DeviceInfo] = []
            for s in self._serials:
                if s not in by_serial:
                    raise RuntimeError(f"未找到序列号 {s} 的相机，在线序列号: {list(by_serial)}")
                selected.append(by_serial[s])
            return selected

        if self._indices:
            return [devices[i] for i in self._indices]

        return devices
