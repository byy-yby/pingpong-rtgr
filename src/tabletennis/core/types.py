"""跨模块共享的数据类型。

这些类型是各模块之间传递数据的最小契约：相机模块产出 :class:`Frame`，
后续的视觉 / 重建 / 可视化模块消费 Frame 并产出 3D 结果。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class Frame:
    """一帧图像及其元数据。

    Attributes:
        camera_id: 逻辑相机索引（0..3），对应 ``config/cameras.yaml`` 里的 index。
        serial: 相机序列号（物理对应）。
        frame_num: SDK 帧号（每台相机独立递增）。
        device_timestamp: 设备时间戳 ``(nDevTimeStampHigh << 32) | nDevTimeStampLow``。
            外部触发下四机曝光对齐，可用它判断同步。
        host_timestamp: SDK 主机侧时间戳（int64）。
        image: 灰度图，shape=(H, W)，dtype 依像素格式（Mono8 -> uint8）。
        pixel_format: GVSP 像素格式枚举值（Mono8 = 17301505）。
        width / height: 图像尺寸。
    """

    camera_id: int
    serial: str
    frame_num: int
    device_timestamp: int
    host_timestamp: int
    image: np.ndarray
    pixel_format: int
    width: int
    height: int


@dataclass
class FrameBundle:
    """同一触发时刻的一组帧（每台相机一帧）。

    Attributes:
        frames: key = camera_id, value = Frame。
    """

    frames: Dict[int, Frame] = field(default_factory=dict)

    @property
    def cameras(self) -> List[int]:
        return sorted(self.frames)

    def is_complete(self, expected: int) -> bool:
        return len(self.frames) >= expected


# ---- 视觉 / 重建阶段的数据类型 ----


@dataclass
class Pose2D:
    """单帧图像上检测到的一个人的 2D 姿态。

    Attributes:
        camera_id: 逻辑相机索引（-1 表示与相机无关的单图检测）。
        keypoints: 形状 (N, 3) 的数组，每行 ``[x, y, confidence]``（像素坐标）。
            关键点顺序与骨架定义（COCO-17 / Halpe-26 等）一致。
        score: 检测框置信度（有则填，无则 0）。
        bbox: ``[x1, y1, x2, y2]`` 检测框，无则 None。
        skeleton: 关键点骨架名（"coco17" / "halpe26" / "coco133"）。
    """

    camera_id: int
    keypoints: np.ndarray
    score: float = 0.0
    bbox: Optional[np.ndarray] = None
    skeleton: str = "coco17"

    def keypoint(self, idx: int) -> np.ndarray:
        """取第 idx 个关键点的 ``[x, y, conf]``；越界返回全 0。"""
        if 0 <= idx < len(self.keypoints):
            return self.keypoints[idx]
        return np.zeros(3, dtype=np.float32)


@dataclass
class Skeleton3D:
    """多视角三角化得到的 3D 骨架。

    Attributes:
        keypoints: 形状 (N, 3) 的数组，每行 ``[x, y, z]``（世界坐标，米）。
            三角化失败（视角不足 / 置信度过低 / 几何退化）的关键点为 ``[nan, nan, nan]``。
        confidence: 形状 (N,) 的数组，各关键点三角化置信度（0..1，失败为 0）。
        skeleton: 骨架名，须与 2D 姿态一致（"halpe26" / "coco17"）。
        n_views: 形状 (N,) 的 int 数组，各关键点实际参与三角化的视角数（可选，默认 None）。
        reproj_err: 形状 (N,) 的 float 数组，各关键点（加权）重投影误差，像素（可选，默认 None）。
    """

    keypoints: np.ndarray
    confidence: np.ndarray
    skeleton: str = "coco17"
    n_views: Optional[np.ndarray] = None
    reproj_err: Optional[np.ndarray] = None


# ---- 阶段 3（标定 / 重建）占位类型，先定义好供后续模块引用 ----


@dataclass
class CameraIntrinsics:
    """单相机内参（OpenCV 约定）。

    Attributes:
        width / height: 标定时使用的图像分辨率。
        K: 3x3 内参矩阵 ``[[fx,0,cx],[0,fy,cy],[0,0,1]]``。
        dist: 畸变系数 ``(k1,k2,p1,p2,k3,...)``。
    """

    width: int
    height: int
    K: np.ndarray
    dist: np.ndarray


@dataclass
class CameraExtrinsics:
    """单相机外参：世界系 -> 相机系的刚体变换 ``[R | t]``。

    Attributes:
        R: 3x3 旋转矩阵。
        t: 3x1 平移向量。
    """

    R: np.ndarray
    t: np.ndarray

    def project(self, K: np.ndarray) -> np.ndarray:
        """投影矩阵 ``P = K [R | t]``（3x4），供三角化使用。

        Args:
            K: 3x3 内参矩阵（来自 :class:`CameraIntrinsics`）。
        """
        Rt = np.hstack([self.R, self.t.reshape(3, 1)])
        return K @ Rt


# ---- 球 / 球桌检测结果 ----


@dataclass
class Ball2D:
    """单帧图像上检测到的乒乓球（2D 投影）。

    Attributes:
        camera_id: 逻辑相机索引（-1 表示与相机无关）。
        center: 球心像素坐标 ``[x, y]``。
        radius: 球半径（像素）。
        confidence: 置信度 0..1。
    """

    camera_id: int
    center: np.ndarray  # (2,)
    radius: float
    confidence: float = 1.0


@dataclass
class Table2D:
    """单帧图像上检测到的球桌（2D 投影）。

    Attributes:
        camera_id: 逻辑相机索引（-1 表示与相机无关）。
        corners: 桌面 4 个角点，形状 (4, 2)，顺序：左上/右上/右下/左下。
        confidence: 置信度 0..1。
    """

    camera_id: int
    corners: np.ndarray  # (4, 2)
    confidence: float = 1.0


@dataclass
class Table3D:
    """标准尺寸球桌的三维模型（世界系 = 桌面系）。

    桌面系定义见 :func:`~tabletennis.calibration.extrinsics.build_table_frame`：
    原点在原点角标记角点，X 沿短边（宽）、Y 沿长边（长）、Z 竖直向上（桌面 z=0）。
    桌面边框就是桌面 4 角点围成的矩形；整桌（桌面 + 4 条腿 + 球网）按乒乓球桌
    标准尺寸推出，供 2D 投影叠加与 3D 场景建模复用。

    Attributes:
        length: 桌面长边（米），默认 2.74。
        width: 桌面短边（米），默认 1.525。
        height: 桌面离地高度（米），默认 0.76（地面 z=-height）。
        net_height: 球网高（米），默认 0.1525。
    """

    length: float = 2.74
    width: float = 1.525
    height: float = 0.76
    net_height: float = 0.1525

    @property
    def top_corners(self) -> np.ndarray:
        """桌面 4 角点 ``(4,3)``，顺序：原点 → 短边 → 对角 → 长边，z=0。"""
        L, W = self.length, self.width
        return np.array([[0, 0, 0], [W, 0, 0], [W, L, 0], [0, L, 0]], dtype=np.float64)

    @property
    def diagonal(self) -> float:
        """桌面对角线长（米）= 两对角角点距离，用于校验两个大标记的间距。"""
        return float(np.linalg.norm(self.top_corners[2] - self.top_corners[0]))

    def segments(self) -> Dict[str, np.ndarray]:
        """桌面线框，返回 ``{名称: 线段数组 (M,2,3)}``（桌面系，米）。

        keys：``top``（桌面边框）、``legs``（4 条腿）、``floor``（地面框）、
        ``net``（球网）。2D 投影与 3D 场景都消费同一份几何。
        """
        c = self.top_corners
        L, W, H = self.length, self.width, self.height
        top = np.array([[c[i], c[(i + 1) % 4]] for i in range(4)], dtype=np.float64)
        legs = np.array(
            [[c[i], [c[i][0], c[i][1], -H]] for i in range(4)], dtype=np.float64
        )
        floor = np.array(
            [
                [[c[i][0], c[i][1], -H], [c[(i + 1) % 4][0], c[(i + 1) % 4][1], -H]]
                for i in range(4)
            ],
            dtype=np.float64,
        )
        ymid = L / 2.0
        net = np.array(
            [
                [[0, ymid, 0], [W, ymid, 0]],
                [[0, ymid, 0], [0, ymid, self.net_height]],
                [[W, ymid, 0], [W, ymid, self.net_height]],
                [[0, ymid, self.net_height], [W, ymid, self.net_height]],
            ],
            dtype=np.float64,
        )
        return {"top": top, "legs": legs, "floor": floor, "net": net}


@dataclass
class TableDetection:
    """一次球桌识别结果（桌面系 -> 相机系位姿 + 投影到该相机的桌面角点）。

    Attributes:
        camera_id: 逻辑相机索引。
        R / t: 桌面系 -> 相机系的刚体变换 ``X_cam = R @ X_world + t``。
        corners: 桌面 4 角点投影到该相机图像的像素坐标 ``(4,2)``。
        live: True 表示本帧直接检测到大标记（实时识别），False 表示回退到已保存外参。
        confidence: 置信度（固定 1.0，与 :class:`Table2D` 接口兼容）。
    """

    camera_id: int
    R: np.ndarray          # (3, 3)
    t: np.ndarray          # (3, 1)
    corners: np.ndarray    # (4, 2)
    live: bool = True
    confidence: float = 1.0
