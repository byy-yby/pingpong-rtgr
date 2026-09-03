"""球轨迹滤波：numpy 常速度/弹道卡尔曼（6 态 3D）+ 门限外点剔除。

作用：单帧三角化的毫米级误差偏大且含离群点（误检 / 视角不足），对平滑的乒乓轨迹
用卡尔曼做时序平滑，能显著压低误差、剔除跳变。状态 ``[px,py,pz,vx,vy,vz]``（世界系，
米），观测为三角化 3D 球心。

可选把重力 ``g`` 作为已知控制输入加入状态转移（弹道模型，见 ``BallTracker(gravity=…)``），
这样补帧时按 ``p += v·dt + ½g·dt²`` 外推——球被击高后 coast 走真实抛物线，而不是沿速度
方向直线外推上天。纯 numpy 实现（项目环境无 filterpy）。过程噪声用「离散白噪声加速度」
(DWNA) 模型，适应重力以外未建模的加速度（空气阻力 / 马格努斯 / 击球）。
"""
from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = ["BallTracker"]


class BallTracker:
    """3D 球轨迹卡尔曼滤波器（常速度模型，可选已知重力 → 弹道模型，可选桌面反弹）。"""

    def __init__(
        self,
        dt: float = 0.01,
        process_noise: float = 1000.0,
        meas_noise_m: float = 0.002,
        gate_m: float = 0.5,
        min_conf: float = 0.3,
        max_coast: Optional[int] = None,
        gravity: Optional[np.ndarray] = None,
        gate_sigma: Optional[float] = None,
        floor_z: Optional[float] = None,
        restitution: float = 0.9,
    ) -> None:
        """
        Args:
            dt: 默认帧间隔（秒），100fps → 0.01。
            process_noise: 加速度强度 q（≈最大加速度平方，m²/s³）。越大越跟得上急变，
                但平滑越弱。球受重力 ~10、击球 ~100 m/s²，默认取 (30)²。
            meas_noise_m: 单轴观测噪声标准差（米）。毫米级目标取 2mm 起步。
            gate_m: 门限（米）。``gate_sigma`` 为 None 时用欧氏距离门限：观测与预测的
                欧氏距离超过该值判为离群点，只预测不更新。
            min_conf: 观测置信度下限，低于此视为无观测（只预测）。
            max_coast: 连续无观测（缺测 / 门限外点）的最大帧数；超过则判为失联
                （reset + 返回 None）。None 表示永不失联（旧行为，无限外推）。
            gravity: 已知恒定加速度（世界系，m/s²，如 ``[0,0,-9.81]``）作为控制输入
                加入状态转移（弹道模型：``p += v·dt + ½g·dt²``、``v += g·dt``）。
                None = 纯常速度模型（旧行为）。球被击高后 coast 才不致直线外推上天。
            gate_sigma: 马氏距离门限（σ 数）。非 None 时用 ``innovᵀ S⁻¹ innov``（S 为
                新息协方差）判离群点，门限随 coast 期间不确定性增长自动放宽，球重新
                出现时更不易被误拒。None = 用固定欧氏 ``gate_m``。
            floor_z: 球心最低 z（米，世界系；= 桌面 z + 球半径）。非 None 时，预测若
                向下穿过该平面就触发反弹：钳回 floor_z 并竖直速度反向（乘 ``restitution``），
                否则球下落会击穿桌面（穿模）。None = 不约束（旧行为）。
            restitution: 反弹的竖直恢复系数（0~1）。乒乓球在球台上约 0.9。
        """
        self.dt = float(dt)
        self.q = float(process_noise)
        self.R = np.eye(3) * float(meas_noise_m) ** 2
        self.gate_m = float(gate_m)
        self.min_conf = float(min_conf)
        self.max_coast = None if max_coast is None else int(max_coast)
        self.gravity = None if gravity is None else np.asarray(gravity, dtype=np.float64).reshape(3)
        self.gate_sigma = None if gate_sigma is None else float(gate_sigma)
        self.floor_z = None if floor_z is None else float(floor_z)
        self.restitution = float(restitution)

        self.x = np.zeros(6, dtype=np.float64)   # [px,py,pz,vx,vy,vz]
        self.P = np.eye(6, dtype=np.float64) * 1.0
        self.initialized = False
        self._coast = 0   # 自上次有效观测以来的连续缺测帧数

    # ------------------------------------------------------------------
    def _transition(self, dt: float) -> np.ndarray:
        F = np.eye(6, dtype=np.float64)
        F[0:3, 3:6] = np.eye(3) * dt
        return F

    def _process_cov(self, dt: float) -> np.ndarray:
        q = self.q
        dt2, dt3, dt4 = dt ** 2, dt ** 3, dt ** 4
        Q = np.zeros((6, 6), dtype=np.float64)
        Q[0:3, 0:3] = np.eye(3) * (dt4 / 4.0)
        Q[0:3, 3:6] = np.eye(3) * (dt3 / 2.0)
        Q[3:6, 0:3] = np.eye(3) * (dt3 / 2.0)
        Q[3:6, 3:6] = np.eye(3) * dt2
        return q * Q

    def _predict(self, dt: float) -> None:
        F = self._transition(dt)
        self.x = F @ self.x
        if self.gravity is not None:
            # 已知重力作为控制输入（弹道模型）：位置 += ½g·dt²、速度 += g·dt
            self.x[0:3] += self.gravity * (0.5 * dt * dt)
            self.x[3:6] += self.gravity * dt
        if self.floor_z is not None and self.x[2] < self.floor_z and self.x[5] < 0.0:
            # 球向下穿过桌面 → 反弹：钳回桌面平面 + 竖直速度反向（恢复系数）。
            # 否则滤波会在碰撞那几帧继续往下积分，球击穿桌面（穿模）。
            self.x[2] = self.floor_z
            self.x[5] = -self.restitution * self.x[5]
        self.P = F @ self.P @ F.T + self._process_cov(dt)

    def _update(self, z: np.ndarray) -> None:
        H = np.hstack([np.eye(3), np.zeros((3, 3))])
        y = z - H @ self.x
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(6) - K @ H) @ self.P

    # ------------------------------------------------------------------
    def update(
        self,
        X: Optional[np.ndarray],
        conf: float = 1.0,
        dt: Optional[float] = None,
    ) -> Optional[np.ndarray]:
        """喂入一帧观测，返回平滑后 3D 位置（世界系，米）。

        Args:
            X: 三角化 3D 球心 (3,)；None 或 conf 过低表示本帧无观测。
            conf: 观测置信度 0..1。
            dt: 与上一帧的间隔（秒），None 用默认值。

        Returns:
            平滑后位置 (3,)；未初始化且无观测，或连续缺测超过 ``max_coast``
            （判失联，已 reset）时返回 None。
        """
        dt = float(dt) if dt is not None else self.dt

        # 无观测：只预测（coast）。连续缺测超过 max_coast 即失联（球落桌/打飞）。
        if X is None or not np.isfinite(X).all() or conf < self.min_conf:
            if not self.initialized:
                return None
            self._predict(dt)
            self._coast += 1
            if self.max_coast is not None and self._coast > self.max_coast:
                self.reset()
                return None
            return self.x[0:3].copy()

        z = np.asarray(X, dtype=np.float64).reshape(3)
        if not self.initialized:
            self.x[0:3] = z
            self.x[3:6] = 0.0
            self.initialized = True
            self._coast = 0
            return z.copy()

        self._predict(dt)
        H = np.hstack([np.eye(3), np.zeros((3, 3))])
        innov = z - H @ self.x
        S = H @ self.P @ H.T + self.R
        # 门限外点：只预测（coast），不更新；外点同样计缺测（误检 / 球突然飞离会在
        # 累计 max_coast 后失联，避免一直挂在旧位置）。
        if self.gate_sigma is not None:
            # 马氏距离门限（新息协方差归一化）：coast 越久 P 越大 → 门限自动放宽
            d2 = float(innov @ np.linalg.solve(S, innov))
            reject = d2 > self.gate_sigma ** 2
        else:
            reject = float(np.linalg.norm(innov)) > self.gate_m
        if reject:
            self._coast += 1
            if self.max_coast is not None and self._coast > self.max_coast:
                self.reset()
                return None
            return self.x[0:3].copy()

        self._update(z)
        self._coast = 0
        return self.x[0:3].copy()

    @property
    def position(self) -> np.ndarray:
        return self.x[0:3].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.x[3:6].copy()

    @property
    def coast(self) -> int:
        """自上次有效观测以来的连续缺测帧数（含门限外点）。"""
        return self._coast

    def reset(self) -> None:
        self.x = np.zeros(6, dtype=np.float64)
        self.P = np.eye(6, dtype=np.float64) * 1.0
        self.initialized = False
        self._coast = 0
