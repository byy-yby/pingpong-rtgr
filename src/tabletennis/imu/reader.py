"""BLE 读取线程：从维特智能 IMU（WT9011DCL-BT5.0）读姿态。

IMU 走 **蓝牙 5.0 (BLE)**，不是串口。GATT 服务 ``ffe5``、notify 收数据 ``ffe4``、
写命令 ``ffe9``（UUID 见维特源码 ``WitBluetooth_BWT901BLE5_0`` 的 ``BleUUID.java``）。
后台线程用 ``bleak`` 扫描（名字含 "WT"）→ 连接 → 订阅 notify → 逐包喂
:class:`WitMotionParser`，把最新朝向（旋转矩阵 / 欧拉角 / 四元数）存成线程安全
快照，供 ``live_control`` 取用并推给 3D 场景。

可靠性设计（解决「卡顿 / 偶尔停更」）：
- **上报率提到期望值**：连接后经 ``ffe9`` 下发官方写命令（NORMAL 协议，**5 字节
  无校验**：``FF AA <reg> <valL> <valH>``），把 RATE 寄存器 0x03 从默认 10Hz 提到
  100Hz（默认）。否则 60fps 窗口里球拍以 10Hz 步进，看起来非常顿挫。
- **自动重连**：连接一旦断开（模块关机 / 离开 / 被其它设备占用）不退出线程，
  指数退避重连（``stop()`` 置 ``_running=False`` 后最多 0.1s 内退出）。
- **数据看门狗**：已连接但 >4s 无有效报文（链路假死 / ``is_connected`` 失效）强制重连。
- **notify 回调容错**：单包解析 / 朝向回调异常不打断订阅，限频打日志。
- **遥测**：每 2s 打印一次实测数据速率（包/秒），与期望值对比，便于定位链路问题。
- **on_orientation 回调**：每个有效姿态包直接回调（默认 None），供主线程把朝向
  推给 3D 场景时**绕开主循环帧率钳制**（主循环只有 ~20FPS，串行推会卡顿）。

- **BLE 流不带校验和（实测坑）**：UART 帧是 ``0x55 | Flag | 18B | 校验`` 共 21B，
  但 WT901BLE5.0 的 BLE 流里 0x61 组合包只有 ``0x55 | 0x61 | 18B`` 共 **20B，没有校验
  字节**，模块还把连续多个包塞进同一条 notify（80B=4 包、40B=2 包）。解析器必须以
  ``checksum=False`` 建（按 20B 解析），否则每个包校验都失败、有效包只剩 1/256 的运气值
  → 实测就是「能连上但姿态几乎不动 / 偶发跳一下」。
- 连不上 / 扫描不到不抛异常，只在日志提示，方便按 i 重试。
"""
from __future__ import annotations

import asyncio
import os
import threading
import time
from typing import Callable, Optional, Tuple

import numpy as np

from .witmotion import WitMotionParser, angle_to_rotmat, quat_to_rotmat

# BLE 5.0 GATT UUID（维特 WT9011DCL-BT5.0；来自官方 SDK BleUUID.java）
_SERVICE_UUID = "0000ffe5-0000-1000-8000-00805f9a34fb"
_READ_UUID = "0000ffe4-0000-1000-8000-00805f9a34fb"    # notify：收数据
_SEND_UUID = "0000ffe9-0000-1000-8000-00805f9a34fb"    # write：发命令

# ---- 官方写命令（NORMAL 协议：FF AA <reg> <valL> <valH>，共 5 字节、无校验）----
# 字节布局见官方 SDK WitStandardProtocol_JY901 的 WitWriteReg()（直发 5 字节），
# 以及 Android 例程 Bwt901cl.unlockReg() / setReturnRate()。
_UNLOCK_CMD = bytes([0xFF, 0xAA, 0x69, 0x88, 0xB5])   # 解锁（KEY=0x69, 数据 0xB588）
_SAVE_CMD = bytes([0xFF, 0xAA, 0x00, 0x00, 0x00])     # 保存（SAVE=0x00, 数据 0x0000）

# RATE 寄存器 0x03 取值 -> Hz（来自官方 SDK WitStandardProtocol_JY901 的 REG.h；
# WT9011DCL 走 NORMAL 协议，默认 0x06=10Hz）
_RRATE_BY_HZ = {
    0.2: 0x01, 0.5: 0x02, 1.0: 0x03, 2.0: 0x04, 5.0: 0x05,
    10.0: 0x06, 20.0: 0x07, 50.0: 0x08, 100.0: 0x09, 200.0: 0x0B,
}
_SUPPORTED_HZ = sorted(_RRATE_BY_HZ)


def _set_rate_cmd(hz: float) -> bytes:
    """把上报率寄存器 0x03 写成 ``hz`` Hz（官方 ``setReturnRate`` 同款命令）。

    ``hz`` 必须是 :data:`_RRATE_BY_HZ` 的键（构造 :class:`ImuReader` 时会就近归整）。
    """
    return bytes([0xFF, 0xAA, 0x03, _RRATE_BY_HZ[hz], 0x00])


def _snap_rate(hz: float) -> float:
    """把请求的 Hz 就近归整到模块支持的取值。"""
    return min(_SUPPORTED_HZ, key=lambda h: abs(h - hz))


class ImuReader:
    """后台线程用 BLE 读取维特智能 IMU 姿态（旋转矩阵 / 欧拉角 / 四元数）。

    Args:
        device_name: 广播名子串（大小写不敏感）；None 则匹配名字含 "WT" 的模块。
        mac: 直接指定 MAC 地址连接（跳过扫描），如 ``"AA:BB:CC:DD:EE:FF"``。
        scan_timeout: 扫描超时（秒）。
        output_rate_hz: 期望上报率（Hz，支持 0.2~200；默认 100）。连接后自动下发
            官方写命令解锁并设置，失败保持模块当前值。传给 3D 显示前会就近归整。
        retry_delay: 断连后首次重连等待（秒），之后指数退避到最多 10s。
    """

    def __init__(self, device_name: Optional[str] = None, mac: Optional[str] = None,
                 scan_timeout: float = 6.0, output_rate_hz: float = 100.0,
                 retry_delay: float = 2.0) -> None:
        self.device_name = device_name
        self.mac = mac
        self.scan_timeout = scan_timeout
        self.output_rate_hz = _snap_rate(float(output_rate_hz))
        self.retry_delay = retry_delay
        self.parser = WitMotionParser(checksum=False)   # BLE 流无校验字节，见模块 docstring

        # 每个有效姿态包回调 ``on_orientation(R)``（R 为 3x3，body->world）。
        # 在 notify 线程调用，须尽快返回（只做跨线程写，别做重活）。
        self.on_orientation: Optional[Callable[[np.ndarray], None]] = None

        # 调试模式（TT_IMU_DEBUG=1）：打印原始 notify（长度 + 十六进制）与解析统计，
        # 用于定位「模块没发够快」还是「模块在发但解析拒了大部分」。
        self.debug = bool(int(os.environ.get("TT_IMU_DEBUG", "0")))
        self._dbg_notifies = 0
        self._dbg_len_counter: dict = {}

        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.connected = False

        self._last_R: Optional[np.ndarray] = None   # 3x3（body -> world）
        self._last_rpy: Optional[Tuple[float, float, float]] = None
        self._last_quat: Optional[Tuple[float, float, float, float]] = None
        self._last_t: float = 0.0      # 最近一次有效姿态时间（仅本线程访问）
        self._connect_t: float = 0.0   # 最近一次连上时间（仅本线程访问）
        self._rate_hz: float = 0.0     # 实测数据速率 EMA（跨线程读，锁保护）

        # 遥测
        self._diag_t0 = time.time()
        self._diag_n0 = 0
        self._diag_n = 0
        self._last_err_t = 0.0

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
            self._set_connected(False)

    # ------------------------------------------------------------------
    # BLE 扫描 / 连接 / notify（在后台线程的 asyncio loop 里跑；断连自动重连）
    # ------------------------------------------------------------------
    async def _async_main(self) -> None:
        from bleak import BleakClient, BleakScanner

        backoff = self.retry_delay
        while self._running:
            addr, name = await self._resolve_device(BleakScanner)
            if not self._running:
                return
            if addr is None:
                print(f"[IMU] 未找到模块，{backoff:.0f}s 后重试扫描…")
                await self._sleep_cancellable(backoff)
                backoff = min(backoff * 1.5, 10.0)
                continue
            backoff = self.retry_delay

            try:
                async with BleakClient(addr, timeout=12.0) as client:
                    self._connect_t = time.time()
                    self._set_connected(True)
                    await client.start_notify(_READ_UUID, self._on_notify)
                    if self.debug:
                        try:
                            print(f"[IMU DBG] MTU={client.mtu_size}（>20 则 0x61 包单条送达）")
                        except Exception:  # noqa: BLE001
                            pass
                    await self._configure_output_rate(client)
                    print(f"[IMU] 已连接 {name or addr}，等待姿态数据…")
                    while self._running:
                        if not client.is_connected:
                            print("[IMU] BLE 连接断开（模块关机 / 离开 / 被其它设备占用？）")
                            break
                        if time.time() - max(self._last_t, self._connect_t) > 4.0:
                            print("[IMU] 已连接但 4s 无数据，强制重连…")
                            break
                        await asyncio.sleep(0.05)
                    try:
                        await client.stop_notify(_READ_UUID)
                    except Exception:  # noqa: BLE001
                        pass
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 —— 断连/超时走自动重连
                print(f"[IMU] 连接/读取异常：{exc}")
            finally:
                self._set_connected(False)

            if not self._running:
                return
            print(f"[IMU] {backoff:.0f}s 后自动重连…")
            await self._sleep_cancellable(backoff)
            backoff = min(backoff * 1.5, 10.0)

    async def _resolve_device(self, scanner) -> Tuple[Optional[str], Optional[str]]:
        """扫描并返回 ``(addr, name)``；指定了 mac 则跳过扫描；找不到返回 (None, None)。"""
        if self.mac:
            return self.mac, self.mac
        print(f"[IMU] 扫描 BLE 设备（最多 {self.scan_timeout:.0f}s，找名字含 'WT' 的模块）…")
        try:
            devices = await scanner.discover(timeout=self.scan_timeout, return_adv=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[IMU] 扫描失败（蓝牙适配器没起？）：{exc}")
            return None, None
        if not self._running:
            return None, None
        for d, adv in devices.values():
            n = adv.local_name or d.name or ""
            if self._name_matches(n):
                return d.address, n
        found = [adv.local_name or d.name or d.address for d, adv in devices.values()]
        print(f"[IMU] 未找到维特 BLE 模块（扫描到 {len(devices)} 台：{found}）")
        return None, None

    async def _configure_output_rate(self, client) -> None:
        """解锁 → 设置上报率 → 保存（官方 NORMAL 协议 5 字节写命令，无校验）。

        失败不影响连接（保持模块当前速率），仅打日志。官方建议每条寄存器写命令
        之间间隔 50~100ms，这里统一 sleep 60ms。
        """
        try:
            await client.write_gatt_char(_SEND_UUID, _UNLOCK_CMD, response=False)
            await asyncio.sleep(0.06)
            await client.write_gatt_char(_SEND_UUID, _set_rate_cmd(self.output_rate_hz),
                                         response=False)
            await asyncio.sleep(0.06)
            await client.write_gatt_char(_SEND_UUID, _SAVE_CMD, response=False)
            print(f"[IMU] 已下发：解锁 + 上报率 {self.output_rate_hz:g}Hz + 保存")
        except Exception as exc:  # noqa: BLE001
            print(f"[IMU] 下发速率命令失败（保持模块当前速率，不影响连接）：{exc}")

    async def _sleep_cancellable(self, seconds: float) -> None:
        """分段 sleep：``stop()`` 置 ``_running=False`` 后最多 0.1s 内返回。"""
        while self._running and seconds > 0:
            step = min(0.1, seconds)
            await asyncio.sleep(step)
            seconds -= step

    def _set_connected(self, v: bool) -> None:
        if self.connected == v:
            return
        self.connected = v
        print(f"[IMU] {'已连接' if v else '已断开'}")

    def _name_matches(self, name: str) -> bool:
        if self.device_name:
            return self.device_name.lower() in (name or "").lower()
        return "WT" in (name or "").upper()

    # ------------------------------------------------------------------
    # notify 回调 -> 解析 -> 更新最新姿态
    # ------------------------------------------------------------------
    def _on_notify(self, _sender, data) -> None:
        try:
            raw = bytes(data)
            if self.debug:
                self._dbg_notifies += 1
                n = len(raw)
                self._dbg_len_counter[n] = self._dbg_len_counter.get(n, 0) + 1
                if self._dbg_notifies <= 30 or self._dbg_notifies % 200 == 0:
                    print(f"[IMU DBG] notify#{self._dbg_notifies} len={n} {raw[:24].hex()}")
            for pkt in self.parser.feed(raw):
                self._handle(pkt)
            self._maybe_report_rate()
        except Exception as exc:  # noqa: BLE001 —— 单包异常不打断订阅
            if time.time() - self._last_err_t > 5.0:
                print(f"[IMU] notify 处理异常（已忽略，继续接收）：{exc}")
                self._last_err_t = time.time()

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
        else:
            return
        self._diag_n += 1
        self._last_t = t
        cb = self.on_orientation
        if cb is not None:
            try:
                cb(R)
            except Exception as exc:  # noqa: BLE001 —— 回调（推 3D）失败不影响数据流
                if time.time() - self._last_err_t > 5.0:
                    print(f"[IMU] on_orientation 回调异常：{exc}")
                    self._last_err_t = time.time()

    def _maybe_report_rate(self) -> None:
        """每 2s 打印一次实测数据速率（包/秒），与期望值对比便于定位链路问题。"""
        now = time.time()
        if now - self._diag_t0 < 2.0:
            return
        dt = now - self._diag_t0
        rate = (self._diag_n - self._diag_n0) / dt
        with self._lock:
            self._rate_hz = rate
        low = " —— 偏低，链路/距离/连接质量问题？" if rate < self.output_rate_hz * 0.6 else ""
        print(f"[IMU] 数据 {rate:.1f} 包/秒（期望 {self.output_rate_hz:g}Hz）{low}")
        if self.debug:
            print(f"[IMU DBG] notify总数={self._dbg_notifies} "
                  f"长度分布={dict(self._dbg_len_counter)} "
                  f"解析统计={dict(self.parser.stats)}")
            self._dbg_len_counter = {}
        self._diag_t0 = now
        self._diag_n0 = self._diag_n

    # ------------------------------------------------------------------
    # 读取最新姿态 / 速率（线程安全）
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

    def data_rate_hz(self) -> float:
        """实测数据速率（包/秒，EMA；连上并出数后才有意义）。"""
        with self._lock:
            return self._rate_hz
