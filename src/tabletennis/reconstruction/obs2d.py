"""2D 观测（球检测 + 姿态关键点）序列化：离线重建存盘，回放查看器读回叠加显示。

``scripts/reconstruct_video.py`` 在检测阶段把每个主时钟帧、每台相机的 2D 球检测
（:class:`Ball2D`）与 2D 姿态关键点（:class:`Pose2D`，**全部检出人，不经过跨相机
匹配**）序列化成两个 JSON，供 ``scripts/visualize_recon.py`` 回放时把检测结果叠回
四路视频画面，方便对比排查「人物动作 / 球检测」的细微问题。

文件格式（键都是字符串，避免 JSON 只能字符串键的坑）：
    pose2d.json = {frame_str: {cid_str: [pose_dict, ...]}}
    ball2d.json = {frame_str: {cid_str: [ball_dict, ...]}}
    pred_boxes.json = {frame_str: {cid_str: [[x1,y1,x2,y2,slot], ...]}}

pose_dict = {"kpts": [[x,y,c]*N], "score": float, "bbox": [x1,y1,x2,y2] | null,
             "skeleton": "halpe26"}
ball_dict = {"x": float, "y": float, "r": float, "c": float}

``pred_boxes.json`` 是**纯显示**的卡尔曼预测框（该相机该帧没检出人时预测根关节的
重投影位置 + 最近 bbox 尺寸），只给回放叠加画灰色虚线框用，**绝不参与重建**——
阶段 A 的预测只用来开 ROI 搜索窗，伪造成检测会引发 3D→框→姿态→3D 自激回路
（见 ``person_track`` 红线①）。``slot`` 是人在该段里的身份序号（0/1）。
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional

import numpy as np

from ..core.types import Ball2D, Pose2D

__all__ = [
    "pose_to_dict", "ball_to_dict", "dict_to_pose", "dict_to_ball",
    "save_pose2d", "save_ball2d", "load_pose2d", "load_ball2d",
    "save_pred_boxes", "load_pred_boxes",
]


# ----------------------------------------------------------------------
# 序列化 / 反序列化（纯函数，可单测）
# ----------------------------------------------------------------------
def pose_to_dict(pose: Pose2D) -> dict:
    """单个 2D 姿态 -> JSON dict（关键点数组转 list，bbox 可空）。"""
    kp = np.asarray(pose.keypoints, dtype=np.float32)
    bbox = pose.bbox
    return {
        "kpts": kp.tolist(),
        "score": float(getattr(pose, "score", 0.0)),
        "bbox": [float(v) for v in bbox] if bbox is not None else None,
        "skeleton": str(getattr(pose, "skeleton", "halpe26")),
    }


def ball_to_dict(ball: Ball2D) -> dict:
    """单个 2D 球检测 -> JSON dict。"""
    return {
        "x": float(ball.center[0]),
        "y": float(ball.center[1]),
        "r": float(ball.radius),
        "c": float(ball.confidence),
    }


def dict_to_pose(d: dict, camera_id: int = -1) -> Pose2D:
    """JSON dict -> 2D 姿态（camera_id 由调用方按帧补上）。"""
    return Pose2D(
        camera_id=camera_id,
        keypoints=np.asarray(d["kpts"], dtype=np.float32),
        score=float(d.get("score", 0.0)),
        bbox=np.asarray(d["bbox"], dtype=np.float32) if d.get("bbox") else None,
        skeleton=str(d.get("skeleton", "halpe26")),
    )


def dict_to_ball(d: dict, camera_id: int = -1) -> Ball2D:
    """JSON dict -> 2D 球检测。"""
    return Ball2D(
        camera_id=camera_id,
        center=np.array([d["x"], d["y"]], dtype=np.float32),
        radius=float(d["r"]),
        confidence=float(d["c"]),
    )


# ----------------------------------------------------------------------
# 存 / 读
# ----------------------------------------------------------------------
def _write_json(path: str, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False)


def _load_json(path: str) -> Dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:  # noqa: BLE001 —— 半截 / 正在写
        return {}


def save_pose2d(out_dir: str, pose2d: Dict) -> None:
    """把 ``{frame_str: {cid_str: [pose_dict]}}`` 写到 ``out_dir/pose2d.json``。"""
    _write_json(os.path.join(out_dir, "pose2d.json"), pose2d)


def save_ball2d(out_dir: str, ball2d: Dict) -> None:
    """把 ``{frame_str: {cid_str: [ball_dict]}}`` 写到 ``out_dir/ball2d.json``。"""
    _write_json(os.path.join(out_dir, "ball2d.json"), ball2d)


def load_pose2d(out_dir: str) -> Dict:
    """读回 pose2d dict；缺文件返回 {}。"""
    return _load_json(os.path.join(out_dir, "pose2d.json"))


def load_ball2d(out_dir: str) -> Dict:
    """读回 ball2d dict；缺文件返回 {}。"""
    return _load_json(os.path.join(out_dir, "ball2d.json"))


def save_pred_boxes(out_dir: str, pred: Dict) -> None:
    """把 ``{frame_str: {cid_str: [[x1,y1,x2,y2,slot], ...]}}`` 写到 ``pred_boxes.json``。

    **只用于回放叠加显示**（灰色虚线预测框），重建流程不读它——见模块 docstring。
    """
    _write_json(os.path.join(out_dir, "pred_boxes.json"), pred)


def load_pred_boxes(out_dir: str) -> Dict:
    """读回 pred_boxes dict；缺文件返回 {}。"""
    return _load_json(os.path.join(out_dir, "pred_boxes.json"))
