"""单台相机封装：open / start / 采集线程 / 有界队列 / close。

这是「相机启动流程」的落点。启动顺序严格遵守已验证过的 SDK 调用链：

    MvCamera() → MV_CC_CreateHandle → MV_CC_OpenDevice(MV_ACCESS_Exclusive)
    → 配置触发模式（trigger.py）→ 配置图像参数（parameter.py，固定短曝光）
    → MV_CC_StartGrabbing → 采集线程 GetImageBuffer → FreeImageBuffer
    → （关闭时）StopGrabbing → CloseDevice → DestroyHandle

采集用一个独立线程 + ``queue.Queue(maxsize)`` 有界队列：消费端（可视化 / 姿态）
处理不过来时**丢最旧帧**，避免内存堆积、延迟越滚越大。

注意：本机是黑白相机（Mono8），``extract_frame`` 产出的 ``Frame.image`` 是
单通道灰度图（H, W）。姿态 / 可视化需要彩色时在各自模块里自行复制成 BGR。
"""
from __future__ import annotations

import ctypes
import logging
import queue
import threading
import time
from typing import Optional

from . import trigger
from .frame import extract_frame
from .parameter import ImageControl
from .sdk import (
    MvCamera,
    MV_FRAME_OUT,
    MV_ACCESS_Exclusive,
    MV_OK,
)

logger = logging.getLogger(__name__)


class Camera:
    """单台相机。

    Args:
        dev_info: ``sdk.enumerate_devices()`` 返回的 :class:`DeviceInfo`。
        logical_id: 逻辑索引（对应 config/cameras.yaml 里的 index）。
        trigger_mode: ``"external"``（信号发生器 Line0）/ ``"software"``（软触发调试）/
            ``"continuous"``（自由采集，无触发信号时用）。
        trigger_source: 外部触发输入线，默认 Line0。
        exposure_us: 固定曝光时间（微秒），None 表示不修改。
        gain_db: 固定增益（dB），None 表示不修改。
        pixel_format: 像素格式，黑白相机用 "Mono8"。
        software_rate_hz: 软触发频率（仅 trigger_mode="software" 时生效）。
        frame_timeout_ms: 抓帧超时。
        max_queue: 有界队列长度，超过丢最旧帧。
    """

    def __init__(
        self,
        dev_info,
        logical_id: int,
        *,
        trigger_mode: str = "continuous",
        trigger_source: str = "Line0",
        exposure_us: Optional[float] = None,
        gain_db: Optional[float] = None,
        pixel_format: str = "Mono8",
        software_rate_hz: float = 30.0,
        frame_timeout_ms: int = 1000,
        max_queue: int = 8,
    ) -> None:
        self.dev_info = dev_info
        self.logical_id = logical_id
        self.serial = dev_info.serial
        self.trigger_mode = trigger_mode
        self.trigger_source = trigger_source
        self.exposure_us = exposure_us
        self.gain_db = gain_db
        self.pixel_format = pixel_format
        self.software_rate_hz = software_rate_hz
        self.frame_timeout_ms = frame_timeout_ms
        self.max_queue = max_queue

        self._cam: Optional[MvCamera] = None
        self._opened = False
        self._grabbing = False
        self._frame_queue: "queue.Queue" = queue.Queue(maxsize=max_queue)
        self._frame_sink = None   # Optional[Callable[[Frame], None]]：每帧旁路回调（录制用）
        self._grab_thread: Optional[threading.Thread] = None
        self._soft_trigger_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._image_control: Optional[ImageControl] = None

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    @property
    def is_open(self) -> bool:
        return self._opened

    @property
    def controls(self) -> ImageControl:
        """图像参数控制器（曝光/增益/黑电平/伽马/亮度等），需 open 之后使用。"""
        if self._image_control is None:
            raise RuntimeError("相机未打开，controls 不可用")
        return self._image_control

    def open(self) -> None:
        """建立句柄并打开设备（独占），应用触发与图像配置。"""
        if self._opened:
            return

        cam = MvCamera()
        ret = cam.MV_CC_CreateHandle(self.dev_info.raw)
        if ret != MV_OK:
            cam.MV_CC_DestroyHandle()
            raise RuntimeError(f"[cam {self.logical_id}] CreateHandle 失败: 0x{ret:08x}")

        ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != MV_OK:
            cam.MV_CC_DestroyHandle()
            raise RuntimeError(
                f"[cam {self.logical_id}] OpenDevice 失败: 0x{ret:08x} "
                f"(若为 0x80000203，说明相机被 MVS 客户端等独占，先关掉它)"
            )

        self._cam = cam
        self._opened = True
        self._image_control = ImageControl(cam)

        # 像素格式先设，避免后续参数设置依赖错误的格式
        if self.pixel_format:
            self._image_control.set_pixel_format(self.pixel_format)

        # 触发模式（外部 / 软件 / 连续）
        if self.trigger_mode == "external":
            ret = trigger.configure_external_trigger(cam, source=self.trigger_source)
            if ret != MV_OK:
                raise RuntimeError(f"[cam {self.logical_id}] 配置外部触发失败: 0x{ret:08x}")
        elif self.trigger_mode == "software":
            ret = trigger.configure_software_trigger(cam)
            if ret != MV_OK:
                raise RuntimeError(f"[cam {self.logical_id}] 配置软件触发失败: 0x{ret:08x}")
        else:  # continuous
            ret = trigger.configure_continuous(cam)
            if ret != MV_OK:
                raise RuntimeError(f"[cam {self.logical_id}] 配置连续采集失败: 0x{ret:08x}")

        # 固定曝光 / 增益（姿态检测要求画面稳定，必须关自动曝光）
        if self.exposure_us is not None:
            ret = self._image_control.set_exposure_time_us(self.exposure_us)
            if ret != MV_OK:
                raise RuntimeError(f"[cam {self.logical_id}] 设置曝光失败: 0x{ret:08x}")
        if self.gain_db is not None:
            ret = self._image_control.set_gain_db(self.gain_db)
            if ret != MV_OK:
                raise RuntimeError(f"[cam {self.logical_id}] 设置增益失败: 0x{ret:08x}")

        logger.info("[cam %s] opened: %s (%s), %s", self.logical_id, self.dev_info.model, self.serial, self.trigger_mode)

    def start(self) -> None:
        """开始抓帧：启动采集线程（软触发模式下再启动触发线程）。"""
        if not self._opened:
            raise RuntimeError("相机未打开")
        if self._grabbing:
            return

        ret = self._cam.MV_CC_StartGrabbing()
        if ret != MV_OK:
            raise RuntimeError(f"[cam {self.logical_id}] StartGrabbing 失败: 0x{ret:08x}")
        self._grabbing = True

        self._stop_event.clear()
        self._grab_thread = threading.Thread(
            target=self._grab_loop, name=f"grab-cam{self.logical_id}", daemon=True
        )
        self._grab_thread.start()

        if self.trigger_mode == "software":
            self._soft_trigger_thread = threading.Thread(
                target=self._software_trigger_loop, name=f"softtrig-cam{self.logical_id}", daemon=True
            )
            self._soft_trigger_thread.start()

        logger.info("[cam %s] grabbing started", self.logical_id)

    def stop(self) -> None:
        """停止抓帧（不关设备）。"""
        self._stop_event.set()
        if self._grab_thread is not None:
            self._grab_thread.join(timeout=2.0)
            self._grab_thread = None
        if self._soft_trigger_thread is not None:
            self._soft_trigger_thread.join(timeout=1.0)
            self._soft_trigger_thread = None

        if self._grabbing and self._cam is not None:
            self._cam.MV_CC_StopGrabbing()
            self._grabbing = False

    def close(self) -> None:
        """逆序释放：停止抓帧 → 关设备 → 销毁句柄。"""
        self.stop()
        if self._cam is not None:
            self._cam.MV_CC_CloseDevice()
            self._cam.MV_CC_DestroyHandle()
            self._cam = None
        self._opened = False
        logger.info("[cam %s] closed", self.logical_id)

    # ------------------------------------------------------------------
    # 取帧
    # ------------------------------------------------------------------
    def get_latest_frame(self, block: bool = True, timeout: float = 1.0):
        """取最新一帧。返回 :class:`Frame` 或 None（超时 / 队列空）。"""
        try:
            return self._frame_queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None

    def set_frame_sink(self, sink) -> None:
        """注册 / 取消每帧旁路回调（四路视频录制用）。

        回调在采集线程里、每抓到一帧时同步调用（与主队列消费互不影响），必须**快速返回**
        ——实现侧应只做入队（满则丢最旧），真正的磁盘编码放自己的后台线程（见
        ``camera/recorder.py::_CameraWriter``）。传 None 取消注册。

        Args:
            sink: ``Callable[[Frame], None]`` 或 None。
        """
        self._frame_sink = sink

    def drain(self) -> None:
        """清空队列（消费端处理不过来时用于快速跳到最新帧）。"""
        try:
            while True:
                self._frame_queue.get_nowait()
        except queue.Empty:
            pass

    # ------------------------------------------------------------------
    # 内部线程
    # ------------------------------------------------------------------
    def _grab_loop(self) -> None:
        st_frame = MV_FRAME_OUT()
        ctypes.memset(ctypes.byref(st_frame), 0, ctypes.sizeof(st_frame))
        while not self._stop_event.is_set():
            ret = self._cam.MV_CC_GetImageBuffer(st_frame, self.frame_timeout_ms)
            if ret != MV_OK:
                # 超时（0x8000000A 之类）在软触发/外部触发下是正常的，继续等
                continue

            frame = extract_frame(st_frame, self.logical_id, self.serial)
            self._cam.MV_CC_FreeImageBuffer(st_frame)

            if frame is None:
                continue

            # 旁路录制：在入主队列之前把帧送给录制 sink（快进快出，录制线程负责编码）
            sink = self._frame_sink
            if sink is not None:
                sink(frame)

            # 有界队列：满了丢最旧帧，保证拿到的是最新画面
            try:
                self._frame_queue.put_nowait(frame)
            except queue.Full:
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._frame_queue.put_nowait(frame)
                except queue.Full:
                    pass

    def _software_trigger_loop(self) -> None:
        period = 1.0 / max(self.software_rate_hz, 1e-3)
        while not self._stop_event.is_set():
            self._cam.MV_CC_SetCommandValue("TriggerSoftware")
            time.sleep(period)
