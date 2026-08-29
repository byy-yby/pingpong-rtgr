"""视觉识别模块：2D 检测（球 / 姿态 / 球桌）。

姿态检测（RTMPose-l，top-down）；球检测（``ball/``：经典无训练路线 + 亚像素精修）。
"""
from .ball import ClassicalBallDetector, YoloBallDetector, refine_ball_center
from .detector import BallDetector, Detector, PoseDetector, TableDetector
from .pose.rtmpose_pose import RTMPoseDetector

__all__ = [
    "Detector",
    "PoseDetector",
    "BallDetector",
    "TableDetector",
    "RTMPoseDetector",
    "ClassicalBallDetector",
    "YoloBallDetector",
    "refine_ball_center",
]
