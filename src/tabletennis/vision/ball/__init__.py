"""球检测子模块：2D 乒乓球检测 + 亚像素精修。

- :mod:`refine`：强度加权质心 + 二阶矩的亚像素球心精修（替代霍夫圆）。
- :mod:`classical_ball`：无训练经典检测（背景减除 + 帧差 + 尺寸先验）。
- :mod:`yolo_ball`：YOLO 检测（onnxruntime，训练见 scripts/train_ball.py）。
"""
from .classical_ball import ClassicalBallDetector
from .refine import refine_ball_center
from .yolo_ball import YoloBallDetector

__all__ = ["ClassicalBallDetector", "refine_ball_center", "YoloBallDetector"]
