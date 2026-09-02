"""维特智能 (WitMotion) IMU 的 0x55 协议解析 + 姿态换算。

WT9011DCL 走「新协议」：默认以组合包 ``0x55 0x61 ...``（加速度 6B + 角速度 6B +
角度 6B）在 115200 波特率输出；经典协议的单字段包（0x51 加速度 / 0x52 角速度 /
0x53 角度 / 0x59 四元数）也一并兼容。数据一律小端、int16。

**帧格式（UART vs BLE 实测不一样，这是本项目踩过的大坑）**：
- UART：``0x55 | Flag | Data | Checksum``，共 ``2+len+1`` 字节，校验和 = 从 0x55
  起到数据末（不含校验字节）全部字节之和的低 8 位。
- **BLE（WT901BLE5.0 实测，MTU 23）：0x61 组合包只有 ``0x55 | 0x61 | 18B 数据``
  共 20 字节，不带校验和字节**；模块把连续多个包塞进同一条 notify（80B=4 包、
  40B=2 包）。若仍按 21B 带校验解析，每个包校验都失败，有效包只剩 1/256 的运气值
  → 表现为「能连上但姿态几乎不动 / 偶发跳一下」。

所以 :class:`WitMotionParser` 带 ``checksum`` 开关：``checksum=True`` 是 UART 21B
带校验（默认，向后兼容）；BLE 读取（:mod:`tabletennis.imu.reader`）用
``checksum=False`` 按 20B 无校验解析。

换算：
- 角度（roll/pitch/yaw，度）= int16 / 32768 * 180；
- 四元数 Q0~Q3 = int16 / 32768（Q0=w 实部，Q1..3=x,y,z）；
- 欧拉角序列为 Z-Y-X（绕 Z 转 yaw、绕 Y 转 pitch、绕 X 转 roll）。

本模块纯 numpy、无 I/O；读取在 :mod:`tabletennis.imu.reader`。
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

# 内部缓冲上限：若字节流长时间不含 0x55（噪声/半包垃圾），feed() 只保留尾部
# 一小段继续等头，防止长时间会话里内存无限增长、find 变慢。
_MAX_BUF = 4096
_KEEP_TAIL = 1024

# 各 flag 对应的数据字节数（不含 0x55 头、flag 字节与校验字节）
_FLAG_LEN = {
    0x50: 11,   # 时间（经典协议；WT9011DCL 默认不上报）
    0x51: 6,    # 加速度 Ax Ay Az
    0x52: 6,    # 角速度 Wx Wy Wz
    0x53: 6,    # 角度 Roll Pitch Yaw（经典协议）
    0x54: 6,    # 磁场 Hx Hy Hz
    0x59: 8,    # 四元数 Q0 Q1 Q2 Q3
    0x61: 18,   # 组合包：加速度 + 角速度 + 角度（WT9011DCL 默认上报）
}

_ACCEL_SCALE = 16.0 / 32768.0     # g
_GYRO_SCALE = 2000.0 / 32768.0    # deg/s
_ANGLE_SCALE = 180.0 / 32768.0    # deg
_QUAT_SCALE = 1.0 / 32768.0


def _i16(buf, off: int) -> int:
    return int.from_bytes(buf[off:off + 2], "little", signed=True)


def _i16s(buf, off: int, n: int, scale: float) -> np.ndarray:
    """从 ``buf`` 读 n 个连续 int16（小端），乘 ``scale``。"""
    return np.asarray([_i16(buf, off + 2 * i) * scale for i in range(n)], dtype=np.float64)


def quat_to_rotmat(w: float, x: float, y: float, z: float) -> np.ndarray:
    """四元数 -> 旋转矩阵（body -> world，Hamilton 约定，w 实部）。

    返回 3x3 正交矩阵 ``R``，``X_world = R @ X_body``。输入不要求已归一化。
    """
    n = float(np.sqrt(w * w + x * x + y * y + z * z))
    if n == 0.0:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def angle_to_rotmat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """欧拉角 -> 旋转矩阵（body -> world，Z-Y-X 序列，角度制入参）。

    WitMotion 的角度定义为：绕 Z 转 yaw、绕 Y 转 pitch、绕 X 转 roll，
    即 ``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``。
    """
    r, p, y = np.radians([roll, pitch, yaw])
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


class WitMotionParser:
    """增量解析 0x55 协议字节流。

    用法：反复 ``feed(chunk)``，每次返回该 chunk 内解析出的完整报文列表；
    跨 chunk 的残包会在内部缓冲，直到凑齐（或校验失败丢弃）。

    Args:
        checksum: True = UART 帧（``2+len+1`` 字节，带校验和，默认）；False =
            BLE 帧（``2+len`` 字节，无校验和，见模块 docstring 的实测结论）。
    """

    def __init__(self, checksum: bool = True) -> None:
        self._buf = bytearray()
        self.checksum = checksum
        # 解析统计（调试用）：bytes=累计输入字节，packets=成功报文，checksum_fail=校验失败，
        # unknown_flag=未知 flag，partial_wait=等更多字节（跨块残包）。
        self.stats = {"bytes": 0, "packets": 0, "checksum_fail": 0,
                      "unknown_flag": 0, "partial_wait": 0}

    def feed(self, chunk: bytes) -> List[Dict]:
        self._buf += chunk
        self.stats["bytes"] += len(chunk)
        out: List[Dict] = []
        pos = 0
        n = len(self._buf)
        while True:
            idx = self._buf.find(b"\x55", pos)
            if idx < 0:
                break
            pos = idx
            if pos + 2 > n:
                break  # 头 + flag 字节没齐，等下一块
            pkt, consumed = self._parse_one(self._buf[pos:])
            if pkt is not None:
                self.stats["packets"] += 1
                out.append(pkt)
                pos += consumed
            elif consumed == 0:
                break  # 数据未到齐，等下一块
            else:
                pos += consumed  # 坏头（校验失败/未知 flag）：跳过这个 0x55 重新找头
        if pos > 0:
            del self._buf[:pos]
        elif len(self._buf) > _MAX_BUF:
            # 本轮没消费任何字节且缓冲超限：说明长时间没有 0x55 头（噪声流），
            # 只保留尾部小窗继续等下一帧，防内存无限增长。
            del self._buf[:len(self._buf) - _KEEP_TAIL]
        return out

    def _parse_one(self, buf) -> Tuple[Optional[Dict], int]:
        """解析 buffer 开头的一个报文。

        Returns:
            ``(pkt, consumed)``：pkt 为 None 表示还没解析出；consumed 为 0 表示
            需要更多字节，为正表示应丢弃 ``consumed`` 字节后重试。
        """
        flag = buf[1]
        ln = _FLAG_LEN.get(flag)
        if ln is None:
            self.stats["unknown_flag"] += 1
            return None, 1  # 未知 flag：丢掉这个 0x55
        total = 2 + ln + (1 if self.checksum else 0)
        if len(buf) < total:
            self.stats["partial_wait"] += 1
            return None, 0  # 数据未到齐
        data = buf[2:2 + ln]
        if self.checksum and (sum(buf[:2 + ln]) & 0xFF) != buf[2 + ln]:
            self.stats["checksum_fail"] += 1
            return None, 1  # 校验和不匹配：丢掉这个 0x55
        return self._decode(flag, data), total

    @staticmethod
    def _decode(flag: int, data: bytes) -> Dict:
        if flag == 0x61:  # 组合包：加速度 + 角速度 + 角度
            ax, ay, az = _i16s(data, 0, 3, _ACCEL_SCALE)
            wx, wy, wz = _i16s(data, 6, 3, _GYRO_SCALE)
            roll, pitch, yaw = _i16s(data, 12, 3, _ANGLE_SCALE)
            return {
                "type": "combined",
                "accel": (float(ax), float(ay), float(az)),
                "gyro": (float(wx), float(wy), float(wz)),
                "angle": (float(roll), float(pitch), float(yaw)),
            }
        if flag == 0x53:  # 角度（经典协议）
            roll, pitch, yaw = _i16s(data, 0, 3, _ANGLE_SCALE)
            return {"type": "angle", "angle": (float(roll), float(pitch), float(yaw))}
        if flag == 0x59:  # 四元数（Q0=w, Q1=x, Q2=y, Q3=z）
            q = _i16s(data, 0, 4, _QUAT_SCALE)
            return {"type": "quaternion", "quat": (float(q[0]), float(q[1]), float(q[2]), float(q[3]))}
        if flag == 0x51:
            ax, ay, az = _i16s(data, 0, 3, _ACCEL_SCALE)
            return {"type": "accel", "accel": (float(ax), float(ay), float(az))}
        if flag == 0x52:
            wx, wy, wz = _i16s(data, 0, 3, _GYRO_SCALE)
            return {"type": "gyro", "gyro": (float(wx), float(wy), float(wz))}
        if flag == 0x54:
            hx, hy, hz = _i16s(data, 0, 3, 1.0)
            return {"type": "mag", "mag": (float(hx), float(hy), float(hz))}
        return {"type": "unknown", "flag": flag}
