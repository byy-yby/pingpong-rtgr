"""视觉识别模块：2D 检测（球 / 姿态）。

目前只实现姿态检测（RTMPose-l，top-down）。球检测（``ball/``）留待后续阶段。
"""
from .detector import Detector, PoseDetector
from .pose.rtmpose_pose import RTMPoseDetector

__all__ = ["Detector", "PoseDetector", "RTMPoseDetector"]
