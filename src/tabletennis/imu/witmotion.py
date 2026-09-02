"""维特智能 (WitMotion) IMU 的 0x55 协议解析 + 姿态换算。

WT9011DCL 走「新协议」：默认以组合包 ``0x55 0x61 ...``（加速度 6B + 角速度 6B +
角度 6B）在 115200 波特率输出；经典协议的单字段包（0x51 加速度 / 0x52 角速度 /
0x53 角度 / 0x59 四元数）也一并兼容。**BLE 固件（官方 BWT901BLE5.0）的磁力计与
四元数不在周期流里**——要主动发寄存器读命令（``FF AA 27 <reg> 00``），模块以
``0x55 0x71 <regL><regH>`` 开头的 20B 响应帧回传（连续寄存器值，读 0x3A 回
HX/HY/HZ、读 0x51 回四元数 Q0..Q3）。数据一律小端、int16。

**帧格式（UART vs BLE 实测不一样，这是本项目踩过的大坑）**：
- UART：``0x55 | Flag | Data | Checksum``，共 ``2+len+1`` 字节，校验和 = 从 0x55
  起到数据末（不含校验字节）全部字节之和的低 8 位。
- **BLE（WT901BLE5.0 实测，MTU 23）：0x61 组合包只有 ``0x55 | 0x61 | 18B 数据``
  共 20 字节，不带校验和字节**；模块把连续多个包塞进同一条 notify（80B=4 包、
  40B=2 包）。若仍按 21B 带校验解析，每个包校验都失败，有效包只剩 1/256 的运气值
  → 表现为「能连上但姿态几乎不动 / 偶发跳一下」。

所以 :class:`WitMotionParser` 带 ``checksum`` 开关：``checksum=True`` 是 UART 21B
带校验（默认，向后兼容）；BLE 读取（:mod:`tabletennis.imu.reader`）用
``checksum=False`` 按 20B 无校验解析（0x61 与 0x71 都是 20B）。

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
    0x51: 6,    # 加速度 Ax Ay Az（经典协议）
    0x52: 6,    # 角速度 Wx Wy Wz（经典协议）
    0x53: 6,    # 角度 Roll Pitch Yaw（经典协议）
    0x54: 6,    # 磁场 Hx Hy Hz（经典协议）
    0x59: 8,    # 四元数 Q0 Q1 Q2 Q3
    0x61: 18,   # 组合包：加速度 + 角速度 + 角度（WT9011DCL 默认上报）
    0x71: 18,   # BLE 寄存器读响应：reg(2B) + 8×int16 连续寄存器值（官方 BWT901BLE5.0）
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


def so3_project(M: np.ndarray) -> np.ndarray:
    """把任意 3x3 矩阵投影到最近的正交旋转矩阵（SVD + 反射修正）。

    用于对多个朝向样本取均值：样本旋转矩阵逐元素求平均后一般不再是纯旋转
    （含缩放/形变），直接当旋转用会出错；先投影回 SO(3) 再返回。
    """
    U, _, Vt = np.linalg.svd(np.asarray(M, dtype=np.float64))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1.0
        R = U @ Vt
    return R


def imu_to_paddle_world(R_now, R_home, R_ref) -> np.ndarray:
    """IMU 当前朝向 -> 球拍在桌面系(世界系)里的朝向，3D 显示直接套网格（body->world）。

    IMU rigidly 固定在球拍柄内、安装角未知；参考时刻（用户把球拍平放锁定基准）捕获
    ``R_home = R_imu(0)``。设 ``A`` = 固定安装角、``R_ref`` = 参考时刻球拍在世界系中的
    姿态（约定常量，见 ``scripts/live_control._IMU_REF_YZ``），则
    ``R_imu(t) = R_paddle(t) @ A`` 且 ``R_home = R_ref @ A``，消去 ``A`` 得::

        R_paddle(t) = R_now @ R_home.T @ R_ref

    参考时刻输出恰好等于 ``R_ref``，任意运动都贴合真实世界朝向、与安装角无关。
    注意不能图省事用 ``R_home.T @ R_now``（共轭旋转，屏幕朝向会整体偏转错位）。
    """
    Rn = np.asarray(R_now, dtype=np.float64)
    Rh = np.asarray(R_home, dtype=np.float64)
    Rr = np.asarray(R_ref, dtype=np.float64)
    return Rn @ Rh.T @ Rr


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


# ---------------------------------------------------------------------------
# 磁航向锚定（消除陀螺 yaw 漂移）
#
# 背景：WT9011DCL 的片上 Kalman 只有在磁力计被校准/可用时才用磁场锚定 yaw。出厂没做
# 磁场校准、台面附近又有铁磁物时，模块 yaw 退化成纯陀螺积分 → 挥拍几圈后摆回参考姿态，
# 拍面水平对（重力锚定）但手柄航向对不上（陀螺积漂）。上位机/App 的「转 8 字」校准能修，
# 但不方便用上位机的场景需要主机侧等价方案：周期读磁力计寄存器（BLE 0x71 响应，见
# reader），我们由原始 mag 自己算**绝对**航向；模块平放静止时把显示航向钉到磁场
# （漂移清零），快速倾斜挥拍时磁力计不可靠，就冻结上次修正、暂时回落模块自身 yaw
# （单拍 ~0.1-0.5s 内陀螺误差很小）。
#
# 约定（纯函数可单测）：模块欧拉 Z-Y-X（yaw 绕 Z）。``tilt_compensated_mag_heading``
# 用模块自身的 roll/pitch 把原始磁力计读数翻平，取水平投影方位角；模块水平时它==模块
# 的 yaw（模块 +x 指向磁场北时 heading=0，世界 yaw 增则 heading 同步增），因此
# 「heading 相对参考的变化量」≈「真实世界航向相对参考的变化量」，与模块 yaw 漂不漂无关。
# ---------------------------------------------------------------------------

def wrap_pi(x_deg: float) -> float:
    """把角度（度）折到 (-180, 180]。"""
    x = float(x_deg) % 360.0
    if x > 180.0:
        x -= 360.0
    if x <= -180.0:
        x += 360.0
    return x


def tilt_compensated_mag_heading(roll_deg: float, pitch_deg: float, mag) -> float:
    """倾角补偿磁航向（度）：由模块 roll/pitch 确定的倾角把原始 mag 翻平到水平面。

    body->水平 = Ry(pitch) @ Rx(roll)；水平面内取 ``atan2(-My, Mx)`` 得方位角。
    只用当前读数、不含陀螺积分 → 不漂移；但倾角大时对 roll/pitch 误差敏感，调用方
    （:class:`MagYawLock`）应在近水平时才信任它。mag 为 0 向量时返回 0。
    """
    m = np.asarray(mag, dtype=np.float64).reshape(3)
    if np.linalg.norm(m) < 1e-9:
        return 0.0
    r, p = np.radians([roll_deg, pitch_deg])
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    level = Ry @ (Rx @ m)
    return float(np.degrees(np.arctan2(-level[1], level[0])))


def _clamp01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


class MagYawLock:
    """主机侧磁航向锚定：把显示 yaw 钉在磁航向上，消除模块陀螺 yaw 漂移。

    用法：锁参考姿态时（用户把球拍平放于桌面原点静止）调 :meth:`lock` 记录参考航向；
    之后每个姿态包调 :meth:`update`，返回值 = 应叠加在「纯模块 yaw 的显示旋转
    ``R_disp0``」上的**绕世界竖直轴的修正角（度）**：``R_disp = Rz_world(δ) @ R_disp0``
    （Rz 左乘只改世界系里的水平航向、不动拍面倾角）。未锁参考 / 无 mag / 球拍明显倾斜
    或快速转动时返回冻结值或 0（回落到模块自身 yaw）。
    """

    LEVEL_MAX_DEG = 12.0      # 平放门限：|roll|/|pitch| 超过则磁航向对倾角误差敏感
    STILL_MAX_DEG_S = 40.0    # 转动门限：|gyro| 超过（快挥）则磁航向噪声大
    K_ATTACK = 0.6            # 每拍向磁航向逼近的比例（静止平放约 0.1s 收敛）
    W_EMA = 0.5               # 权重平滑系数（防 w 在门限附近抖动）

    def __init__(self) -> None:
        self._home_h: Optional[float] = None   # 参考 heading（度）
        self._home_yaw: float = 0.0            # 参考模块 yaw（度）
        self._delta: float = 0.0               # 当前世界竖直修正（度）
        self._w: float = 0.0                   # 平滑后的「信任磁航向」权重 [0,1]

    @property
    def enabled(self) -> bool:
        return self._home_h is not None

    def lock(self, roll_deg: float, pitch_deg: float, yaw_deg: float, mag) -> None:
        """在参考姿态（模块静止、水平、处于参考世界朝向）记下磁航向与模块 yaw。"""
        self._home_h = wrap_pi(
            tilt_compensated_mag_heading(roll_deg, pitch_deg, mag))
        self._home_yaw = float(yaw_deg)
        self._delta = 0.0
        self._w = 0.0

    def update(self, roll_deg: float, pitch_deg: float, yaw_deg: float,
               gyro_norm_deg_s: float, mag) -> float:
        """逐姿态包调用；返回世界竖直修正角（度），没有有效磁锚定时返回 0/冻结值。"""
        if self._home_h is None or mag is None:
            return 0.0
        w_level = _clamp01(
            (self.LEVEL_MAX_DEG - max(abs(roll_deg), abs(pitch_deg)))
            / self.LEVEL_MAX_DEG)
        w_still = _clamp01((self.STILL_MAX_DEG_S - float(gyro_norm_deg_s))
                           / self.STILL_MAX_DEG_S)
        w = w_level * w_still
        self._w += self.W_EMA * (w - self._w)
        if self._w <= 1e-6:
            return self._delta  # 运动/倾斜中：冻结修正，别把磁噪声带进挥拍
        h = wrap_pi(tilt_compensated_mag_heading(roll_deg, pitch_deg, mag))
        dH = wrap_pi(h - self._home_h)          # 磁锚定的世界航向变化
        dy = wrap_pi(yaw_deg - self._home_yaw)  # 模块自报的航向变化（可能带漂移）
        # 权重合成期望航向变化：w=1（静止平放）取磁场 dH，w=0（运动/倾斜）取模块 dy。
        # 用复数加权避免 ±180 折返处不连续。
        zr = (self._w * np.cos(np.radians(dH))
              + (1.0 - self._w) * np.cos(np.radians(dy)))
        zi = (self._w * np.sin(np.radians(dH))
              + (1.0 - self._w) * np.sin(np.radians(dy)))
        desired = float(np.degrees(np.arctan2(zi, zr)))
        target = wrap_pi(desired - dy)          # 要在 R_disp0 之上补的世界修正
        self._delta = wrap_pi(
            self._delta + self.K_ATTACK * wrap_pi(target - self._delta))
        return self._delta


def handle_bearing(R) -> float:
    """手柄方向在桌面系 XY 平面的方位角（度，atan2，range −180..180]。

    viewer3d 的球拍 mesh 手柄默认沿 +X，故取 ``R @ [1,0,0]`` 在世界系的方向，
    投影到 XY 平面取方位角——平放时 = 手柄世界航向；绕世界 +Z 转 ``deg`` 时方位角
    正好平移 ``deg``（Rz 与 XY 投影可交换），是「世界航向修正」的自然度量。
    """
    d = np.asarray(R, dtype=np.float64).reshape(3, 3) @ np.array([1.0, 0.0, 0.0])
    return float(np.degrees(np.arctan2(d[1], d[0])))


class WorldHeadingHold:
    """静止冻结显示航向：把模块「完全不动」时仍在慢漂的 yaw 整体丢弃。

    模块 yaw 在无磁场基准（本固件不开 0x54）时是纯陀螺积分，静止时零偏也照积
    （手持/桌面几分钟漂 1~3°）。本类在**真静止**（调用方按模块角速度 EMA + 门限判定，
    桌面静置是、手持抖动通常不是）时把显示航向冻在进静止那一刻，模块静止期自己攒的
    漂移不进显示；转动恢复时从冻结值继续累积模块航向的真实变化——**不跳变、静止期的
    漂移被永久丢弃**（不是冻结到转动瞬间再补回来）。它只给「相对保持」、给不了绝对
    航向，「摆回原点应复位」由 live_control 的**原点自动重锁**负责。

    用法：锁参考后逐姿态包 ``delta = update(bearing(R_disp0), still)``，返回绕世界
    +Z 的修正角（度），``R_disp = Rz_world(delta) @ R_disp0`` 使静止时显示航向恒定、
    转动时贴合模块。``still`` 须由调用方对 |gyro| 平滑后与门限比较（见 live_control），
    别把手上 6~12Hz 的抖动判成「转动」。
    """

    def __init__(self) -> None:
        self._disp: Optional[float] = None  # 显示航向（度）：静止冻结、转动累积
        self._prev: Optional[float] = None  # 上一包的模块航向（度）

    def reset(self) -> None:
        """重锁参考姿态后调用：基线归零，下一包起 δ=0（显示航向 = 模块航向）。"""
        self._disp = None
        self._prev = None

    def update(self, heading_deg: float, still: bool) -> float:
        """逐包调用。``heading_deg`` = 纯模块显示的航向（``bearing(R_disp0)``）。"""
        h = float(heading_deg)
        if self._prev is None:
            self._prev = h
            self._disp = h
            return 0.0
        if not still:
            # 转动：把模块航向的真实变化累积进显示航向；静止期攒的漂移不进 _disp
            self._disp = wrap_pi(self._disp + wrap_pi(h - self._prev))
        self._prev = h
        return wrap_pi(self._disp - h)


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
        if flag == 0x71:  # BLE 寄存器读响应（官方 BWT901BLE5.0：0x55 0x71 regL regH + 连续寄存器值）
            # data = reg(2B 小端) + 8×int16；首个值 = reg 寄存器，随后 reg+1, reg+2, …
            reg = data[0] | (data[1] << 8)
            vals = _i16s(data, 2, 4, 1.0)
            return {"type": "reg", "reg": reg,
                    "values": (float(vals[0]), float(vals[1]),
                               float(vals[2]), float(vals[3]))}
        return {"type": "unknown", "flag": flag}
