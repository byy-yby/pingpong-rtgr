"""IMU 协议解析单元测试（合成字节流，无需硬件 / Open3D）。

覆盖：0x61 组合包（角度）、经典 0x53 角度 / 0x59 四元数、校验和校验与重同步、
跨 chunk 分帧、旋转矩阵数学（so3_project），以及 IMU 球拍绑定到右手腕的选人逻辑
（right_wrist_anchor——模块本身不触发 open3d 延迟 import）。
"""
import numpy as np

from tabletennis.core.types import Skeleton3D
from tabletennis.imu.reader import (
    ImuReader,
    _MAG_REG,
    _SAVE_CMD,
    _UNLOCK_CMD,
    _read_reg_cmd,
    _set_rate_cmd,
)
from tabletennis.imu.witmotion import (
    MagYawLock,
    WitMotionParser,
    WorldHeadingHold,
    angle_to_rotmat,
    handle_bearing,
    imu_to_paddle_world,
    quat_to_rotmat,
    so3_project,
    tilt_compensated_mag_heading,
    wrap_pi,
)
from tabletennis.visualization.viewer3d import (
    SceneViewer3D,
    right_wrist_anchor,
)


def _i16(v: int) -> bytes:
    return int(v).to_bytes(2, "little", signed=True)


def _pkt(flag: int, payload: bytes) -> bytes:
    """UART 帧：0x55 | Flag | Data | Checksum（共 2+len+1 字节）。"""
    body = bytes([0x55, flag]) + payload
    return body + bytes([sum(body) & 0xFF])


def _ble_pkt(flag: int, payload: bytes) -> bytes:
    """BLE 帧（WT901BLE5.0 实测）：0x55 | Flag | Data（共 2+len 字节，无校验）。"""
    return bytes([0x55, flag]) + payload


def _reg_pkt(reg: int, vals8) -> bytes:
    """BLE 0x71 寄存器读响应（20B 无校验）：0x55 0x71 regL regH + 8×int16 连续寄存器值。"""
    payload = bytes([reg & 0xFF, reg >> 8]) + b"".join(_i16(int(v)) for v in vals8)
    assert len(vals8) == 8 and len(payload) == 18
    return _ble_pkt(0x71, payload)


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


def test_reader_notify_path():
    """BLE notify（20B 无校验 0x61 包）-> 解析 -> 旋转矩阵。

    ImuReader 用 checksum=False 建解析器；21B 带校验的 UART 包在这里不是合法输入。
    """
    r = ImuReader()
    payload = _i16(0) * 3 + _i16(0) * 3 + _angle_payload(0.0, 0.0, 90.0)
    pkt = _ble_pkt(0x61, payload)
    assert len(pkt) == 20
    r._on_notify(None, pkt)
    R = r.latest_rotation()
    assert R is not None
    assert np.allclose(R @ np.array([1, 0, 0]), np.array([0, 1, 0]), atol=1e-6)


def test_ble_80b_notify_4_packets():
    """实测 notify 帧：80B = 4 个 20B 无校验 0x61 包，一帧解析出 4 个组合包。

    该 hex 来自真实模块（imu_probe.py 阶段1 抓的原始数据）。
    """
    raw = bytes.fromhex(
        "5561fcffefff110800000000000098ff0d00f831"
        "5561fdffecff0e0800000000000098ff0d00f831"
        "5561fdffecff0e0800000000000098ff0d00f831"
        "5561feffe9ff0f0800000000000098ff0d00f831"
    )
    assert len(raw) == 80
    p = WitMotionParser(checksum=False)
    out = p.feed(raw)
    assert len(out) == 4
    assert all(o["type"] == "combined" for o in out)
    assert p.stats["checksum_fail"] == 0
    # 第一包实测读数：Az≈1.008g、yaw≈70.26°（int16/32768×量程）
    _, _, az = out[0]["accel"]
    assert abs(az - 1.008) < 0.01
    _, _, yaw = out[0]["angle"]
    assert abs(yaw - 70.26) < 0.05


def test_ble_40b_notify_2_packets():
    """实测下发 50Hz 命令后的 notify：40B = 2 个 20B 无校验 0x61 包。"""
    raw = bytes.fromhex(
        "5561f3fff3ff0f0800000000000098ff0d00f831"
        "5561f9fff1ff100800000000000098ff0d00f831"
    )
    assert len(raw) == 40
    p = WitMotionParser(checksum=False)
    out = p.feed(raw)
    assert len(out) == 2
    assert p.stats["checksum_fail"] == 0


def test_checksum_true_rejects_ble_stream():
    """回归测试：旧 checksum=True 对 20B BLE 包解析不出——这就是「卡顿」根因。

    20B 按 21B 长度等待 → partial_wait，永远凑不齐 → 有效包只剩 1/256 运气值。
    """
    p = WitMotionParser()   # 默认带校验
    out = p.feed(bytes.fromhex("5561fcffefff110800000000000098ff0d00f831"))
    assert out == []
    assert p.stats["partial_wait"] > 0


def test_write_cmd_bytes_official():
    """写命令字节须与官方 SDK 完全一致（NORMAL 协议 5 字节无校验）。"""
    # Android 例程 Bwt901cl.unlockReg() / setReturnRate()，C SDK WitWriteReg()
    assert _UNLOCK_CMD == bytes([0xFF, 0xAA, 0x69, 0x88, 0xB5])   # 解锁 KEY=0x69
    assert _SAVE_CMD == bytes([0xFF, 0xAA, 0x00, 0x00, 0x00])     # 保存 SAVE=0x00
    assert _set_rate_cmd(10.0) == bytes([0xFF, 0xAA, 0x03, 0x06, 0x00])   # RATE=0x03
    assert _set_rate_cmd(50.0) == bytes([0xFF, 0xAA, 0x03, 0x08, 0x00])
    assert _set_rate_cmd(100.0) == bytes([0xFF, 0xAA, 0x03, 0x09, 0x00])
    assert _set_rate_cmd(200.0) == bytes([0xFF, 0xAA, 0x03, 0x0B, 0x00])
    assert all(len(c) == 5 for c in (_UNLOCK_CMD, _SAVE_CMD, _set_rate_cmd(100.0)))


def test_reader_rate_snap():
    """请求的上报率就近归整到模块支持的取值。"""
    assert ImuReader(output_rate_hz=90).output_rate_hz == 100.0
    assert ImuReader(output_rate_hz=30).output_rate_hz == 20.0
    assert ImuReader(output_rate_hz=1.5).output_rate_hz == 1.0
    assert ImuReader(output_rate_hz=300).output_rate_hz == 200.0


def test_on_orientation_callback():
    """notify 解析出姿态后调用 on_orientation(R)，R 为正确旋转矩阵。"""
    r = ImuReader()
    got = []
    r.on_orientation = got.append
    payload = _i16(0) * 3 + _i16(0) * 3 + _angle_payload(0.0, 0.0, 90.0)
    r._on_notify(None, _ble_pkt(0x61, payload))
    assert len(got) == 1
    assert np.allclose(got[0] @ np.array([1, 0, 0]), np.array([0, 1, 0]), atol=1e-6)


def test_on_orientation_callback_not_called_on_garbage():
    """无有效姿态（非角度/四元数包）不触发回调。"""
    r = ImuReader()
    got = []
    r.on_orientation = got.append
    r._on_notify(None, b"\x00\x01\x02\x03")   # 纯垃圾
    assert got == []


# ----------------------------------------------------------------------
# 旋转矩阵数学（so3_project：多个朝向样本取均值时投影回 SO(3)）
# ----------------------------------------------------------------------
def test_so3_project_identity():
    """单位矩阵投影后仍是单位阵。"""
    assert np.allclose(so3_project(np.eye(3)), np.eye(3), atol=1e-9)


def test_so3_project_reflection_to_rotation():
    """反射矩阵（det=-1）投影后变成纯旋转（det=+1）。"""
    R = so3_project(np.diag([1.0, -1.0, 1.0]))
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
    assert np.isclose(np.linalg.det(R), 1.0)


def test_so3_project_average_two_rotations():
    """两个相差 20° 的旋转逐元素均值投影 ≈ 中间的 10°。"""
    R1 = angle_to_rotmat(0.0, 0.0, 0.0)
    R2 = angle_to_rotmat(0.0, 0.0, 20.0)
    Rm = so3_project((R1 + R2) / 2.0)
    assert np.allclose(Rm, angle_to_rotmat(0.0, 0.0, 10.0), atol=0.05)


# ----------------------------------------------------------------------
# imu_to_paddle_world：IMU 读数 -> 球拍桌面系世界朝向（3D 显示）
# ----------------------------------------------------------------------
_R_REF = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # Rz(-90°)


def test_imu_to_paddle_world_at_reference():
    """参考时刻输出 == R_ref：拍面平放朝上(+Z)、点口端/手柄朝桌面 −Y。"""
    for M in (np.eye(3), angle_to_rotmat(15.0, -20.0, 30.0)):  # 任意安装角
        R_home = _R_REF @ M
        R_disp = imu_to_paddle_world(R_home, R_home, _R_REF)
        assert np.allclose(R_disp, _R_REF, atol=1e-9)
    # mesh 手柄 +X -> 桌面 −Y；拍面法线 +Z 仍朝上（表系 Y=长边）
    assert np.allclose(_R_REF @ [1.0, 0.0, 0.0], [0.0, -1.0, 0.0], atol=1e-9)
    assert np.allclose(_R_REF @ [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], atol=1e-9)


def test_imu_to_paddle_world_tracks_true_world_pose():
    """任意世界旋转 Q：显示朝向 == 真实球拍世界朝向 Q@R_ref，与安装角 M 无关。"""
    for M in (np.eye(3), angle_to_rotmat(10.0, 30.0, -15.0), angle_to_rotmat(120.0, -50.0, 60.0)):
        R_home = _R_REF @ M
        for Q in (np.eye(3), angle_to_rotmat(0.0, 0.0, 40.0),  # 纯桌面内偏航
                  angle_to_rotmat(0.0, 90.0, 0.0),            # 纯前倾
                  angle_to_rotmat(-30.0, 40.0, 25.0),         # 挥拍复合
                  angle_to_rotmat(120.0, 55.0, -10.0)):       # 立起+翻转
            R_imu = Q @ R_home
            R_disp = imu_to_paddle_world(R_imu, R_home, _R_REF)
            assert np.allclose(R_disp, Q @ _R_REF, atol=1e-6)


def test_imu_to_paddle_world_old_conjugate_is_wrong():
    """回归：旧公式 R_home.T @ R 是共轭旋转，一般运动下屏幕朝向会整体错位。

    取 M=I、Q=纯绕 Y 前倾 90°（拍面从 +Z 立到 +X）：
    真值手柄应指向 −Y；旧公式会指向 +X（偏 90°），新公式正确。
    """
    R_home = _R_REF
    Q = angle_to_rotmat(0.0, 90.0, 0.0)
    truth = Q @ _R_REF
    old = R_home.T @ (Q @ R_home)
    new = imu_to_paddle_world(Q @ R_home, R_home, _R_REF)
    v = np.array([1.0, 0.0, 0.0])
    h_true = truth @ v
    h_old = old @ v
    h_new = new @ v
    assert np.allclose(h_true, [0.0, -1.0, 0.0], atol=1e-9)      # 真实手柄 -Y
    assert not np.allclose(h_old, h_true, atol=1e-6)             # 旧公式确实偏
    assert np.allclose(h_new, h_true, atol=1e-9)                 # 新公式正确


# ----------------------------------------------------------------------
# IMU 球拍绑到右手腕：right_wrist_anchor（不触发 open3d 延迟 import）
# ----------------------------------------------------------------------
def _skel26(wrist_pos, wrist_conf=0.9):
    """构造一张 halpe26 骨架：只填右手腕（索引 10），其余全 NaN。"""
    kps = np.full((26, 3), np.nan)
    conf = np.zeros(26)
    if wrist_pos is not None:
        kps[10] = wrist_pos
        conf[10] = wrist_conf
    return Skeleton3D(keypoints=kps, confidence=conf, skeleton="halpe26")


def test_right_wrist_anchor_picks_nearest_origin():
    """多人里选右手腕离桌面原点最近的人（用户初次连接把拍放在原点那一侧）。"""
    me = _skel26([0.1, 0.3, 0.8])        # 原点侧 = 拿拍的人
    other = _skel26([1.0, 2.4, 0.8])     # 球桌远端
    anchor = right_wrist_anchor([me, other])
    assert anchor is not None
    assert np.allclose(anchor, [0.1, 0.3, 0.8], atol=1e-9)


def test_right_wrist_anchor_respects_min_conf():
    """右手腕置信度过低视为无效：跳过近处低置信，选远处合格的人。"""
    low = _skel26([0.1, 0.3, 0.8], wrist_conf=0.1)
    ok = _skel26([1.0, 2.4, 0.8], wrist_conf=0.9)
    assert right_wrist_anchor([low], min_conf=0.3) is None
    anchor = right_wrist_anchor([low, ok], min_conf=0.3)
    assert anchor is not None
    assert np.allclose(anchor, [1.0, 2.4, 0.8], atol=1e-9)


def test_right_wrist_anchor_no_valid_wrist():
    """无骨架 / 右手腕 NaN 时返回 None（调用方保持上一锚点）。"""
    assert right_wrist_anchor([]) is None
    assert right_wrist_anchor([_skel26(None)]) is None


def test_set_imu_anchor_threadsafe_state():
    """set_imu_anchor 线程安全更新锚点；None 忽略（保持上一值）。"""
    v = SceneViewer3D()   # __init__ 不触碰 open3d
    v.set_imu_anchor([0.2, 0.5, 1.0])
    with v._imu_lock:
        assert np.allclose(v._imu_anchor, [0.2, 0.5, 1.0])
    v.set_imu_anchor(None)
    with v._imu_lock:
        assert np.allclose(v._imu_anchor, [0.2, 0.5, 1.0])  # None 被忽略


# ----------------------------------------------------------------------
# 磁航向锚定：tilt_compensated_mag_heading / MagYawLock（消除陀螺 yaw 漂移）
# ----------------------------------------------------------------------
_NORTH = np.array([1.0, 0.0, 0.0])   # 水平磁场指北（任意均匀刻度在 atan2 中抵消）


def _mag_for_rpy(roll, pitch, yaw):
    """模块处于欧拉(roll,pitch,yaw) 且世界水平磁场 = _NORTH 时，body 系磁力计读数。"""
    return angle_to_rotmat(roll, pitch, yaw).T @ _NORTH


def _rotz(deg):
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _ang_between(A, B):
    cos = (np.trace(A.T @ B) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def test_wrap_pi():
    assert abs(wrap_pi(190.0) - (-170.0)) < 1e-9
    assert abs(wrap_pi(-190.0) - 170.0) < 1e-9
    assert abs(wrap_pi(360.0)) < 1e-9
    assert abs(wrap_pi(180.0) - 180.0) < 1e-9
    assert abs(wrap_pi(540.0) - 180.0) < 1e-9
    assert abs(wrap_pi(-540.0) - 180.0) < 1e-9   # ≡ +180°，取区间代表 +180


def test_mag_heading_flat_equals_yaw():
    """水平时 heading == 模块 yaw：+x 指磁场北时 0，yaw 增则 heading 同步增。"""
    for yaw in (-170.0, -90.0, -45.0, 0.0, 45.0, 90.0, 135.0, 170.0, 179.0):
        h = tilt_compensated_mag_heading(0.0, 0.0, _mag_for_rpy(0.0, 0.0, yaw))
        assert abs(wrap_pi(h - yaw)) < 1e-6, f"yaw={yaw} heading={h}"


def test_mag_heading_tilt_invariant():
    """倾角补偿后 heading 只反映世界航向，与 roll/pitch 无关（水平磁场下 == yaw）。"""
    for roll, pitch, yaw in [(25.0, 0.0, 0.0), (0.0, 30.0, 0.0),
                             (15.0, -20.0, 45.0), (-35.0, 20.0, -120.0),
                             (-10.0, 5.0, 178.0)]:
        h = tilt_compensated_mag_heading(roll, pitch,
                                         _mag_for_rpy(roll, pitch, yaw))
        assert abs(wrap_pi(h - yaw)) < 1e-6, f"rpy=({roll},{pitch},{yaw}) h={h}"


def test_mag_heading_zero_mag_returns_zero():
    assert tilt_compensated_mag_heading(0.0, 0.0, (0.0, 0.0, 0.0)) == 0.0


def test_mag_yaw_lock_returns_reference_after_drift():
    """核心场景：模块 yaw 漂 +35°，球拍物理上已回到参考姿态。磁锚定应输出 -35°
    的绕世界竖直修正，使 R_disp0（含漂移的显示旋转）被拉回 R_ref。"""
    lock = MagYawLock()
    lock.lock(0.0, 0.0, 0.0, _mag_for_rpy(0.0, 0.0, 0.0))
    # 模块自报 yaw 从 0 涨到 35（漂移）；磁力计读数同参考（物理姿态相同 → 磁场同）
    delta = 0.0
    for _ in range(60):
        delta = lock.update(0.0, 0.0, 35.0, 0.0, _mag_for_rpy(0.0, 0.0, 0.0))
    assert abs(delta - (-35.0)) < 0.5, f"delta={delta}（期望 ≈ -35）"

    # 落到真实显示路径：R_home = 参考时刻模块 R（M=I 时 = Rz(-90°)）
    R_home = _R_REF
    R_disp0 = imu_to_paddle_world(angle_to_rotmat(0.0, 0.0, -90.0 + 35.0),
                                  R_home, _R_REF)
    R_corr = _rotz(delta) @ R_disp0
    assert _ang_between(R_corr, _R_REF) < 0.5


def test_mag_yaw_lock_no_drift_keeps_zero():
    """无漂移时锚定不加修正：物理转 +90°（模块 yaw=+90、磁场同步），delta 保持 ~0。"""
    lock = MagYawLock()
    lock.lock(0.0, 0.0, 0.0, _mag_for_rpy(0.0, 0.0, 0.0))
    for _ in range(30):
        delta = lock.update(0.0, 0.0, 90.0, 0.0, _mag_for_rpy(0.0, 0.0, 90.0))
    assert abs(delta) < 0.5, f"delta={delta}（无漂移应保持 0）"


def test_mag_yaw_lock_freezes_during_fast_motion():
    """快挥（|gyro| > 门限）时修正冻结：不把磁噪声带进挥拍。"""
    lock = MagYawLock()
    lock.lock(0.0, 0.0, 0.0, _mag_for_rpy(0.0, 0.0, 0.0))
    for _ in range(30):                       # 先在静止下收敛
        v0 = lock.update(0.0, 0.0, 0.0, 0.0, _mag_for_rpy(0.0, 0.0, 0.0))
    for _ in range(5):                        # 快速平面转 +90°，gyro 300°/s
        v = lock.update(0.0, 0.0, 90.0, 300.0, _mag_for_rpy(0.0, 0.0, 90.0))
        assert abs(v - v0) < 1e-6, f"运动中断言 delta 冻结：{v} vs {v0}"


def test_mag_yaw_lock_tilted_does_not_lock():
    """明显倾斜（|roll| > 平放门限）时不锚定：即使模块 yaw 已漂，也冻结上次修正。"""
    lock = MagYawLock()
    lock.lock(0.0, 0.0, 0.0, _mag_for_rpy(0.0, 0.0, 0.0))
    v0 = 0.0
    for _ in range(5):
        v0 = lock.update(40.0, 0.0, 45.0, 0.0,
                         _mag_for_rpy(40.0, 0.0, 45.0))
    assert abs(v0) < 1e-6  # 倾斜即权重 0，修正保持 0（不追 45° 的假锚定）


def test_mag_yaw_lock_disabled_without_mag():
    lock = MagYawLock()
    assert not lock.enabled
    assert lock.update(0.0, 0.0, 35.0, 0.0, (1.0, 0.0, 0.0)) == 0.0
    lock.lock(0.0, 0.0, 0.0, (1.0, 0.0, 0.0))
    assert lock.enabled
    assert lock.update(0.0, 0.0, 0.0, 0.0, None) == 0.0


def test_reader_request_mag_flag_and_snapshot_path():
    """request_mag 透传；0x54 磁场包进快照，on_packet 携带最新 accel/gyro/mag。

    ImuReader._on_notify 是同步可测路径（不走 asyncio/BLE）。
    """
    r = ImuReader(request_mag=True)
    assert r.request_mag is True
    assert ImuReader().request_mag is False
    packets = []
    r.on_packet = packets.append

    # 1) 组合包（无 mag 快照）→ on_packet 触发，mag=None
    comb = (_i16(0) * 3 + _i16(0) * 3 + _angle_payload(0.0, 0.0, 90.0))
    r._on_notify(None, _ble_pkt(0x61, comb))
    assert len(packets) == 1
    assert packets[0]["mag"] is None
    assert abs(packets[0]["yaw"] - 90.0) < 1e-6
    assert packets[0]["accel"] == (0.0, 0.0, 0.0)
    assert r.latest_rotation() is not None

    # 2) 单独 0x54 磁场包：只刷快照，不触发 on_packet
    r._on_notify(None, _ble_pkt(0x54, _i16(100) + _i16(-200) + _i16(300)))
    assert len(packets) == 1          # 没触发第二次
    assert r._n_mag == 1

    # 3) 再来一个组合包 → on_packet 带上上一步的 mag 快照
    r._on_notify(None, _ble_pkt(0x61, comb))
    assert len(packets) == 2
    assert packets[1]["mag"] == (100.0, -200.0, 300.0)

    # 4) 分离包路径：加速度 0x51 / 磁场 0x54 / 角度 0x53 各自到达
    r._on_notify(None, _ble_pkt(0x51, _i16(0) * 3))
    r._on_notify(None, _ble_pkt(0x54, _i16(7) + _i16(8) + _i16(9)))
    r._on_notify(None, _ble_pkt(0x53, _angle_payload(0.0, 0.0, 45.0)))
    assert len(packets) == 3
    assert packets[2]["mag"] == (7.0, 8.0, 9.0)
    assert abs(packets[2]["yaw"] - 45.0) < 1e-6


def test_read_reg_cmd_bytes_official():
    """寄存器读命令 == 官方 BWT901BLE5.0 ReadData：``FF AA 27 <reg> 00``。"""
    assert _read_reg_cmd(0x3A) == bytes([0xFF, 0xAA, 0x27, 0x3A, 0x00])  # 磁力计
    assert _read_reg_cmd(0x51) == bytes([0xFF, 0xAA, 0x27, 0x51, 0x00])  # 四元数
    assert len(_read_reg_cmd(0x3A)) == 5


def test_parse_reg_0x71_response():
    """BLE 0x71 寄存器读响应：``55 71 regL regH`` + 8×int16，解析出 reg 与连续值。

    官方多平台（Android/C#/Python）一致：读 0x3A 时响应首 3 个值 = HX/HY/HZ。
    """
    f = _reg_pkt(0x3A, [100, -200, 300, 0, 0, 0, 0, 0])
    assert len(f) == 20
    out = WitMotionParser(checksum=False).feed(f)
    assert len(out) == 1
    assert out[0]["type"] == "reg"
    assert out[0]["reg"] == 0x3A
    assert out[0]["values"][:3] == (100.0, -200.0, 300.0)


def test_reader_mag_snapshot_from_reg_0x3a():
    """磁力计走官方 BLE 方式：轮询读寄存器 0x3A → 0x71 帧刷新 ``_last_mag``。

    reg 帧只刷快照、不触发 on_packet；随后 0x61 姿态包带上最新 mag（锚定用）。
    """
    r = ImuReader(request_mag=True)
    packets = []
    r.on_packet = packets.append
    comb = _i16(0) * 3 + _i16(0) * 3 + _angle_payload(0.0, 0.0, 0.0)

    r._on_notify(None, _ble_pkt(0x61, comb))
    assert len(packets) == 1 and packets[0]["mag"] is None

    r._on_notify(None, _reg_pkt(_MAG_REG, [111, -22, 333, 0, 0, 0, 0, 0]))
    assert r._n_mag == 1
    assert len(packets) == 1          # reg 帧不触发姿态回调

    r._on_notify(None, _ble_pkt(0x61, comb))
    assert len(packets) == 2
    assert packets[1]["mag"] == (111.0, -22.0, 333.0)

    # 读别的寄存器（0x51 四元数）不影响 mag 快照
    r._on_notify(None, _reg_pkt(0x51, [16384, 0, 0, 0, 0, 0, 0, 0]))
    assert r._n_mag == 1


def test_reader_ready_flag_and_on_ready():
    """配置完成标志：False→True 才触发一次 on_ready；同值不重复触发。

    live_control 靠它把参考锁定挪到速率配置完成后的干净流（修「锁到过渡期低速流」）。
    """
    r = ImuReader()
    calls = []
    r.on_ready = lambda: calls.append(True)
    assert not r.is_ready()
    r._set_ready(False)     # 同值：不触发
    assert not calls
    r._set_ready(True)      # False→True：触发一次
    assert r.is_ready()
    assert len(calls) == 1
    r._set_ready(True)      # 同值：不重复触发
    assert len(calls) == 1
    r._set_ready(False)     # True→False：只在变 True 时回调
    assert not r.is_ready()
    assert len(calls) == 1
    r._set_ready(True)      # 再次 False→True
    assert len(calls) == 2


def test_handle_bearing_flat_equals_yaw():
    """水平（roll=pitch=0）时手柄世界方位角 == 模块 yaw。"""
    for yaw in (-170.0, -90.0, 0.0, 45.0, 135.0, 179.0):
        h = handle_bearing(angle_to_rotmat(0.0, 0.0, yaw))
        assert abs(wrap_pi(h - yaw)) < 1e-6, f"yaw={yaw} bearing={h}"
    # 参考姿态 R_ref=Rz(-90°)：手柄世界方向 = 桌面 −Y
    assert abs(wrap_pi(handle_bearing(_R_REF) - (-90.0))) < 1e-6


def test_rotz_world_rotates_bearing_exactly():
    """绕世界 +Z 转 delta 后手柄方位角正好平移 delta——航向修正的核心恒等式。"""
    R = angle_to_rotmat(15.0, -20.0, 40.0)   # 带倾角的拍
    for delta in (-170.0, -35.0, 0.0, 90.0, 179.0):
        Rz = _rotz(delta)
        b0, b1 = handle_bearing(R), handle_bearing(Rz @ R)
        assert abs(wrap_pi(b1 - b0 - delta)) < 1e-6, f"delta={delta}"


def test_world_heading_hold_discards_rest_creep():
    """核心场景：静止时模块 yaw 零偏积分慢漂 +2°，显示应冻在原处；随后真实转 +30°，
    显示 = 原处 + 30°（静止期漂移被永久丢弃，转动恢复不跳变）。"""
    hh = WorldHeadingHold()
    h = 10.0
    assert abs(hh.update(h, False)) < 1e-9    # 基线
    shown = []
    for _ in range(100):                      # 静止：模块 yaw 每包 creep +0.02°
        h += 0.02
        d = hh.update(h, True)
        shown.append(wrap_pi(h + d))
    assert all(abs(x - 10.0) < 1e-9 for x in shown), "静止漂移进了显示！"
    for _ in range(30):                       # 真实旋转 30°/包（h 从 12 涨到 42）
        h += 1.0
        d = hh.update(h, False)
        shown.append(wrap_pi(h + d))
    assert abs(shown[-1] - 40.0) < 1e-6, f"期望 10+30=40，实际 {shown[-1]}"


def test_world_heading_hold_follows_pure_motion():
    """一直转动 → 修正 0，显示贴模块（真实旋转不被冻结吞掉）。"""
    hh = WorldHeadingHold()
    h = 0.0
    assert abs(hh.update(h, False)) < 1e-9
    for _ in range(25):
        h += 1.0
        d = hh.update(h, False)
        assert abs(d) < 1e-9, f"转动中 δ 应为 0，实际 {d}"
    assert abs(wrap_pi(h + d) - h) < 1e-9


def test_world_heading_hold_reset():
    """重锁参考后 reset()：δ 归零，显示重新从当前航向起算。"""
    hh = WorldHeadingHold()
    h = 5.0
    assert abs(hh.update(h, True)) < 1e-9     # 基线锁在 5°
    for _ in range(10):                       # 静止 + creep 到 6°
        h += 0.1
        d = hh.update(h, True)
    assert abs(wrap_pi(h + d) - 5.0) < 1e-9   # 已冻结在 5°
    hh.reset()
    d = hh.update(9.0, False)                 # 新参考下模块航向已是 9°：δ=0
    assert abs(d) < 1e-9
    assert abs(wrap_pi(9.0 + d) - 9.0) < 1e-9
