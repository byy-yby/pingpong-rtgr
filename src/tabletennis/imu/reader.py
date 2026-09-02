"""串口读取线程：从维特智能 IMU 读字节流，解析并暴露最新姿态。

IMU 通过 CH340 USB 转串口（``/dev/ttyUSB*``）接入，默认 115200 波特率。
后台线程持续读取、逐包喂 :class:`WitMotionParser`，把最新朝向（旋转矩阵 /
四元数 / 欧拉角）存成线程安全的快照，供 ``live_control`` 主循环取用并推给
3D 场景。

- 串口打开失败 / 没有 /dev/ttyUSB* 时不抛异常，只在日志里提示，方便按 i 重试。
- ``auto_baud=True`` 时按候选波特率依次试，直到解析出角度/四元数包为止。
"""
from __future__ import annotations

import glob
import threading
import time
from typing import Optional, Tuple

import numpy as np

from .witmotion import WitMotionParser, angle_to_rotmat, quat_to_rotmat

# 自动波特率候选（指定波特率放最前，其余按常用度排）
_AUTO_BAUDS = (115200, 9600, 460800, 921600, 230400)


def default_port() -> str:
    """按常见顺序返回第一个存在的串口设备；都没有则返回 ``/dev/ttyUSB0``。"""
    cands = sorted(glob.glob("/dev/ttyUSB*")) + sorted(glob.glob("/dev/ttyACM*"))
    cands += sorted(glob.glob("/dev/serial/by-id/*"))
    return cands[0] if cands else "/dev/ttyUSB0"


class ImuReader:
    """后台线程读取维特智能 IMU 姿态（旋转矩阵 / 欧拉角 / 四元数）。"""

    def __init__(self, port: Optional[str] = None, baud: int = 115200,
                 auto_baud: bool = True) -> None:
        self.port = port or default_port()
        self.baud = baud
        self.auto_baud = auto_baud
        self.parser = WitMotionParser()

        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._ser = None
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
        self._thread = threading.Thread(target=self._run, name="imu-reader", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        self._close_serial()

    def _close_serial(self) -> None:
        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:  # noqa: BLE001
                pass
            self._ser = None

    # ------------------------------------------------------------------
    # 串口打开 + 自动波特率
    # ------------------------------------------------------------------
    def _open(self, baud: int) -> bool:
        import serial
        try:
            ser = serial.Serial(self.port, baud, timeout=0.2)
        except Exception as exc:  # noqa: BLE001 —— 没插好 / 无权限等，交给上层提示
            print(f"[IMU] 打开 {self.port} @ {baud} 失败：{exc}")
            return False
        self._ser = ser
        if not self.auto_baud:
            self.baud = baud
            return True

        # 自动波特率：读一小段时间，能解析出角度/四元数包即为正确波特率
        deadline = time.time() + 0.6
        got_pose = False
        while time.time() < deadline:
            data = ser.read(ser.in_waiting or 1)
            if not data:
                continue
            for pkt in self.parser.feed(data):
                if "angle" in pkt or "quat" in pkt:
                    got_pose = True
            if got_pose:
                break
        if got_pose:
            self.baud = baud
            return True
        ser.close()
        self._ser = None
        return False

    # ------------------------------------------------------------------
    # 读取循环
    # ------------------------------------------------------------------
    def _run(self) -> None:
        bauds = [self.baud] if not self.auto_baud else (
            [self.baud] + [b for b in _AUTO_BAUDS if b != self.baud]
        )
        opened = False
        for b in bauds:
            self.parser = WitMotionParser()  # 换波特率时清掉残流
            if self._open(b):
                opened = True
                break
        if not opened:
            print(f"[IMU] 无法连接 {self.port}（已试 {bauds}）——请确认设备已插好、"
                  f"出现 /dev/ttyUSB*，且波特率正确。")
            return

        self.connected = True
        kind = "自动探测" if self.auto_baud else "手动指定"
        print(f"[IMU] 已连接 {self.port} @ {self.baud}（{kind}）")
        while self._running:
            try:
                data = self._ser.read(self._ser.in_waiting or 1)
            except Exception as exc:  # noqa: BLE001
                print(f"[IMU] 串口读取异常：{exc}")
                break
            if not data:
                continue
            for pkt in self.parser.feed(data):
                self._handle(pkt)
        self._close_serial()
        self.connected = False

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
