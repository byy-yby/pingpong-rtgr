"""BLE 读取线程：从维特智能 IMU（WT9011DCL-BT5.0）读姿态。

IMU 走 **蓝牙 5.0 (BLE)**，不是串口。GATT 服务 ``ffe5``、notify 收数据 ``ffe4``、
写命令 ``ffe9``（UUID 见维特源码 ``WitBluetooth_BWT901BLE5_0`` 的 ``BleUUID.java``）。
后台线程用 ``bleak`` 扫描（名字含 "WT"）→ 连接 → 订阅 notify → 逐包喂
:class:`WitMotionParser`，把最新朝向（旋转矩阵 / 欧拉角 / 四元数）存成线程安全
快照，供 ``live_control`` 主循环取用并推给 3D 场景。

- BLE 单包最多 20 字节，21 字节的 0x61 组合包会跨 notify 拆分，靠解析器的增量缓冲
  + 校验和重组（与串口同一套帧格式，解析器复用）。
- 连不上 / 扫描不到不抛异常，只在日志提示，方便按 i 重试。
"""
from __future__ import annotations

import asyncio
import threading
import time
from typing import Optional, Tuple

import numpy as np

from .witmotion import WitMotionParser, angle_to_rotmat, quat_to_rotmat

# BLE 5.0 GATT UUID（维特 WT9011DCL-BT5.0；来自官方 SDK BleUUID.java）
_SERVICE_UUID = "0000ffe5-0000-1000-8000-00805f9a34fb"
_READ_UUID = "0000ffe4-0000-1000-8000-00805f9a34fb"    # notify：收数据
_SEND_UUID = "0000ffe9-0000-1000-8000-00805f9a34fb"    # write：发命令（暂未用）


class ImuReader:
    """后台线程用 BLE 读取维特智能 IMU 姿态（旋转矩阵 / 欧拉角 / 四元数）。

    Args:
        device_name: 广播名子串（大小写不敏感）；None 则匹配名字含 "WT" 的模块。
        mac: 直接指定 MAC 地址连接（跳过扫描），如 ``"AA:BB:CC:DD:EE:FF"``。
        scan_timeout: 扫描超时（秒）。
    """

    def __init__(self, device_name: Optional[str] = None, mac: Optional[str] = None,
                 scan_timeout: float = 6.0) -> None:
        self.device_name = device_name
        self.mac = mac
        self.scan_timeout = scan_timeout
        self.parser = WitMotionParser()

        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.connected = False

        self._last_R: Optional[np.ndarray] = None   # 3x3（body -> world）
        self._last_rpy: Optional[Tuple[float, float, float]] = None
        self._last_quat: Optional[Tuple[float, float, float, float]] = None
        self._last_t: float = 0.0

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(target=self._thread_main, name="imu-ble-reader",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=6.0)
            self._thread = None

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._async_main())
        except Exception as exc:  # noqa: BLE001 —— BLE 失败不影响 2D 主流程
            print(f"[IMU] BLE 运行异常：{exc}")
        finally:
            self.connected = False

    # ------------------------------------------------------------------
    # BLE 扫描 / 连接 / notify（在后台线程的 asyncio loop 里跑）
    # ------------------------------------------------------------------
    async def _async_main(self) -> None:
        from bleak import BleakClient, BleakScanner

        addr = self.mac
        name = self.mac or None
        if addr is None:
            print(f"[IMU] 扫描 BLE 设备（最多 {self.scan_timeout:.0f}s，找名字含 'WT' 的维特模块）…")
            try:
                devices = await BleakScanner.discover(timeout=self.scan_timeout,
                                                      return_adv=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[IMU] 扫描失败（蓝牙适配器没起？）：{exc}")
                return
            if not self._running:
                return
            for d, adv in devices.values():
                n = adv.local_name or d.name or ""
                if self._name_matches(n):
                    addr, name = d.address, n
                    break
            if addr is None:
                found = [(adv.local_name or d.name or d.address)
                         for d, adv in devices.values()]
                print(f"[IMU] 未找到维特 BLE 模块（扫描到 {len(devices)} 台：{found}）——"
                      f"请确认 IMU 已上电、未被手机/上位机占用、距离够近；"
                      f"或用 --imu-mac 直接指定 MAC。")
                return

        print(f"[IMU] 连接 {name or addr} ({addr})…")
        async with BleakClient(addr) as client:
            self.connected = True
            await client.start_notify(_READ_UUID, self._on_notify)
            print(f"[IMU] 已连接 {name or addr}，等待姿态数据…")
            while self._running and client.is_connected:
                await asyncio.sleep(0.1)
            try:
                await client.stop_notify(_READ_UUID)
            except Exception:  # noqa: BLE001
                pass
        self.connected = False

    def _name_matches(self, name: str) -> bool:
        if self.device_name:
            return self.device_name.lower() in (name or "").lower()
        return "WT" in (name or "").upper()

    # ------------------------------------------------------------------
    # notify 回调 -> 解析 -> 更新最新姿态
    # ------------------------------------------------------------------
    def _on_notify(self, _sender, data) -> None:
        for pkt in self.parser.feed(bytes(data)):
            self._handle(pkt)

    def _handle(self, pkt) -> None:
        t = time.time()
        if "quat" in pkt:
            R = quat_to_rotmat(*pkt["quat"])
            with self._lock:
                self._last_R, self._last_quat, self._last_t = R, pkt["quat"], t
                self._last_rpy = None
        elif "angle" in pkt:
            roll, pitch, yaw = pkt["angle"]
            R = angle_to_rotmat(roll, pitch, yaw)
            with self._lock:
                self._last_R, self._last_rpy, self._last_t = R, (roll, pitch, yaw), t
                self._last_quat = None

    # ------------------------------------------------------------------
    # 读取最新姿态（线程安全）
    # ------------------------------------------------------------------
    def latest_rotation(self) -> Optional[np.ndarray]:
        """最新朝向的 3x3 旋转矩阵（body -> world）；无有效数据返回 None。"""
        with self._lock:
            return None if self._last_R is None else self._last_R.copy()

    def latest_rpy(self) -> Optional[Tuple[float, float, float]]:
        """最新欧拉角 ``(roll, pitch, yaw)``（度）；无则 None。"""
        with self._lock:
            return self._last_rpy

    def latest_quat(self) -> Optional[Tuple[float, float, float, float]]:
        """最新四元数 ``(w, x, y, z)``；无则 None。"""
        with self._lock:
            return self._last_quat
