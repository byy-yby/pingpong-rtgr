"""IMU 协议解析单元测试（合成字节流，无需硬件 / Open3D）。

覆盖：0x61 组合包（角度）、经典 0x53 角度 / 0x59 四元数、校验和校验与重同步、
跨 chunk 分帧，以及角度/四元数 -> 旋转矩阵的正确性与一致性。
"""
import numpy as np

from tabletennis.imu.witmotion import (
    WitMotionParser,
    angle_to_rotmat,
    quat_to_rotmat,
)


def _i16(v: int) -> bytes:
    return int(v).to_bytes(2, "little", signed=True)


def _pkt(flag: int, payload: bytes) -> bytes:
    body = bytes([0x55, flag]) + payload
    return body + bytes([sum(body) & 0xFF])


def _angle_payload(roll: float, pitch: float, yaw: float) -> bytes:
    return _i16(round(roll / 180 * 32768)) + \
           _i16(round(pitch / 180 * 32768)) + \
           _i16(round(yaw / 180 * 32768))


def test_combined_0x61_angle():
    """WT9011DCL 默认组合包：18B = 加速度(6) + 角速度(6) + 角度(6)。"""
    accel = _i16(0) + _i16(0) + _i16(32768 // 16)  # Az = 1g（16g 量程）
    gyro = _i16(0) + _i16(0) + _i16(0)
    pkt = _pkt(0x61, accel + gyro + _angle_payload(10.0, 20.0, 30.0))
    out = WitMotionParser().feed(pkt)
    assert len(out) == 1
    assert out[0]["type"] == "combined"
    roll, pitch, yaw = out[0]["angle"]
    assert abs(roll - 10.0) < 0.01
    assert abs(pitch - 20.0) < 0.01
    assert abs(yaw - 30.0) < 0.01
    assert abs(out[0]["accel"][2] - 1.0) < 0.01


def test_classic_angle_0x53():
    out = WitMotionParser().feed(_pkt(0x53, _angle_payload(0.0, 0.0, 90.0)))
    assert out[0]["type"] == "angle"
    _, _, yaw = out[0]["angle"]
    assert abs(yaw - 90.0) < 0.01


def test_quaternion_0x59_identity():
    # Q=(w,x,y,z)=(1,0,0,0) -> 单位旋转；int16 有符号最大 32767，故 1.0 编码为 32767
    q = _i16(32767) + _i16(0) + _i16(0) + _i16(0)
    out = WitMotionParser().feed(_pkt(0x59, q))
    assert out[0]["type"] == "quaternion"
    assert np.allclose(quat_to_rotmat(*out[0]["quat"]), np.eye(3), atol=1e-6)


def test_bad_checksum_resync():
    """坏校验和包 + 中间垃圾字节后，仍能解析出后面的好包。"""
    good = _pkt(0x53, _angle_payload(0.0, 0.0, 90.0))
    bad = bytearray(good)
    bad[-1] ^= 0xFF  # 破坏校验和
    stream = bytes(bad) + b"\x00\x01\x02" + good
    out = WitMotionParser().feed(stream)
    assert len(out) == 1
    assert out[0]["type"] == "angle"


def test_split_across_chunks():
    """逐字节喂入，跨 chunk 的分帧也能正确解析。"""
    pkt = _pkt(0x61, _i16(0) * 3 + _i16(0) * 3 + _angle_payload(1.0, 2.0, 3.0))
    p = WitMotionParser()
    out = []
    for b in pkt:
        out += p.feed(bytes([b]))
    assert len(out) == 1
    assert out[0]["type"] == "combined"


def test_angle_to_rotmat_yaw90():
    R = angle_to_rotmat(0.0, 0.0, 90.0)
    # 绕 Z 转 90°：+X -> +Y
    assert np.allclose(R @ np.array([1, 0, 0]), np.array([0, 1, 0]), atol=1e-9)
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-9)


def test_quat_to_rotmat_z90():
    # 绕 Z 转 90°：w=cos45°, z=sin45°
    w = z = np.sqrt(0.5)
    R = quat_to_rotmat(w, 0.0, 0.0, z)
    assert np.allclose(R @ np.array([1, 0, 0]), np.array([0, 1, 0]), atol=1e-9)


def test_angle_quat_consistent():
    """同一次绕 Z 转 90°，角度与四元数两种路径结果一致。"""
    R_angle = angle_to_rotmat(0.0, 0.0, 90.0)
    R_quat = quat_to_rotmat(np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5))
    assert np.allclose(R_angle, R_quat, atol=1e-9)
