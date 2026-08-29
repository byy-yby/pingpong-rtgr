"""视觉检测器抽象接口。

约定的最小契约：

- ``detect(frame)`` 输入一帧 :class:`~tabletennis.core.types.Frame`，
  返回结果列表（:class:`Pose2D` 列表，或 :class:`Ball2D` 列表）。
- 检测器实现放在 ``vision/pose/`` 与 ``vision/ball/`` 下，通过本模块的
  基类统一类型，上层 pipeline / 脚本只依赖抽象，便于替换算法。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from ..core.types import Frame


class Detector(ABC):
    """检测器基类（姿态 / 球共用）。"""

    @abstractmethod
    def detect(self, frame: Frame) -> List[Any]:
        """对一帧做检测，返回结果对象列表。"""

    def close(self) -> None:
        """释放资源（模型等）。默认无操作。"""


class PoseDetector(Detector, ABC):
    """2D 姿态检测器接口：``detect(frame) -> List[Pose2D]``。"""

    @abstractmethod
    def detect(self, frame: Frame) -> List[Any]:
        """对一帧做 2D 姿态检测，返回 :class:`Pose2D` 列表（可能为空）。"""


class BallDetector(Detector, ABC):
    """2D 球检测器接口：``detect(frame) -> List[Ball2D]``。"""

    @abstractmethod
    def detect(self, frame: Frame) -> List[Any]:
        """对一帧做乒乓球检测，返回 :class:`Ball2D` 列表（可能为空）。"""


class TableDetector(Detector, ABC):
    """2D 球桌检测器接口：``detect(frame) -> Optional[Table2D]``。"""

    @abstractmethod
    def detect(self, frame: Frame) -> Any:
        """对一帧做球桌检测，返回 :class:`Table2D` 或 None（未检测到）。"""


# ---- 检测器注册 / 工厂 ----
# 具体算法实现好后（如 RTMPose 姿态、经典 CV 球检测、球桌角点检测）在此注册，
# 上层（scripts/live_control.py 等）通过 create_detector() 获取实例，未实现则返回 None。

_REGISTRY: Dict[str, Any] = {}


def register_detector(kind: str, factory) -> None:
    """注册检测器工厂。例如 ``register_detector("pose", lambda: RTMPoseDetector())``。"""
    _REGISTRY[kind] = factory


def create_detector(kind: str) -> Any:
    """按名称创建检测器；未注册返回 None（调用方据此提示「接口已定义，算法待实现」）。"""
    factory = _REGISTRY.get(kind)
    return factory() if factory else None


def _create_pose_detector() -> Any:
    """创建 RTMPose 姿态检测器（延迟 import 避免循环依赖）。

    默认 GPU + TensorRT（缺 TensorRT/CUDA 时自动回退 CUDA EP / CPU）。
    """
    from .pose.rtmpose_pose import RTMPoseDetector
    return RTMPoseDetector(device="cuda", backend="tensorrt")


def _create_table_detector() -> Any:
    """创建球桌检测器（延迟 import；自动从项目配置加载内参/外参/标记参数）。"""
    from .table.table_detector import TableDetector
    return TableDetector.load_default()


def _create_ball_detector() -> Any:
    """创建乒乓球检测器（经典路线：背景减除 + 帧差 + 尺寸先验 + 亚像素质心）。"""
    from .ball.classical_ball import ClassicalBallDetector
    return ClassicalBallDetector()


def _create_ball_yolo_detector() -> Any:
    """创建 YOLO 球检测器（onnxruntime；模型文件不存在则返回 None，调用方回退经典）。

    模型路径优先取环境变量 ``BALL_ONNX``，否则用训练默认产物
    ``runs/detect/ball/weights/best.onnx``。
    """
    import os

    from ..core.config import project_root

    path = os.environ.get("BALL_ONNX") or os.path.join(
        project_root(), "runs", "detect", "ball", "weights", "best.onnx"
    )
    if not os.path.exists(path):
        return None
    from .ball.yolo_ball import YoloBallDetector
    return YoloBallDetector(path)


# 姿态检测已实现（RTMPose-l-halpe26，26 点，CPU 推理），注册到工厂，
# live_control.py 按 'p' 即可启用。
register_detector("pose", _create_pose_detector)

# 球桌检测已实现（四个大 ArUco 标记，跨相机三角化定桌面世界系 + 标准尺寸投影），
# 注册到工厂，live_control.py 按 't' 识别球桌：四机视角画桌面边框 + 生成 Open3D 3D 场景。
register_detector("table", _create_table_detector)

# 球检测已实现（经典路线：背景减除 + 帧差 + 尺寸先验 + 亚像素质心），注册到工厂，
# live_control.py / reconstruct_ball.py 按名字取。
register_detector("ball", _create_ball_detector)

# YOLO 球检测（onnxruntime，模型由 scripts/train_ball.py 训练导出），注册到工厂；
# 模型不存在时工厂返回 None，调用方据此回退经典路线。
register_detector("ball_yolo", _create_ball_yolo_detector)
