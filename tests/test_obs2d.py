"""obs2d 序列化（Pose2D/Ball2D ↔ dict + 存/读 JSON）的纯函数单测。"""
import numpy as np
import pytest

from tabletennis.core.types import Ball2D, Pose2D
from tabletennis.reconstruction.obs2d import (
    ball_to_dict,
    dict_to_ball,
    dict_to_pose,
    load_ball2d,
    load_pose2d,
    load_pred_boxes,
    pose_to_dict,
    save_ball2d,
    save_pose2d,
    save_pred_boxes,
)


def test_pose_roundtrip():
    kp = np.random.RandomState(0).rand(26, 3).astype(np.float32)
    pose = Pose2D(camera_id=2, keypoints=kp, score=0.87,
                  bbox=np.array([10.0, 20.0, 200.0, 300.0], np.float32),
                  skeleton="halpe26")
    d = pose_to_dict(pose)
    assert d["skeleton"] == "halpe26"
    assert np.asarray(d["kpts"]).shape == (26, 3)
    assert d["bbox"] == [10.0, 20.0, 200.0, 300.0]

    back = dict_to_pose(d, camera_id=2)
    np.testing.assert_allclose(back.keypoints, kp)
    assert back.score == 0.87
    assert back.camera_id == 2
    np.testing.assert_allclose(back.bbox, pose.bbox)


def test_pose_roundtrip_no_bbox():
    pose = Pose2D(camera_id=-1, keypoints=np.zeros((26, 3), np.float32),
                  score=0.0, bbox=None, skeleton="halpe26")
    d = pose_to_dict(pose)
    assert d["bbox"] is None
    back = dict_to_pose(d)
    assert back.bbox is None


def test_ball_roundtrip():
    ball = Ball2D(camera_id=1, center=np.array([123.4, 56.7], np.float32),
                  radius=8.5, confidence=0.91)
    d = ball_to_dict(ball)
    assert d["x"] == pytest.approx(123.4)
    assert d["y"] == pytest.approx(56.7)
    assert d["r"] == pytest.approx(8.5)
    assert d["c"] == pytest.approx(0.91)
    back = dict_to_ball(d, camera_id=1)
    np.testing.assert_allclose(back.center, [123.4, 56.7])
    assert back.radius == 8.5 and back.confidence == 0.91 and back.camera_id == 1


def test_save_load_pose_and_ball(tmp_path):
    pose2d = {"5": {"0": [{"kpts": [[0.0, 1.0, 0.9]], "score": 0.5, "bbox": None,
                           "skeleton": "halpe26"}]}}
    ball2d = {"5": {"1": [{"x": 1.0, "y": 2.0, "r": 3.0, "c": 0.8}]}}
    save_pose2d(str(tmp_path), pose2d)
    save_ball2d(str(tmp_path), ball2d)
    assert load_pose2d(str(tmp_path)) == pose2d
    assert load_ball2d(str(tmp_path)) == ball2d


def test_load_missing_returns_empty(tmp_path):
    assert load_pose2d(str(tmp_path)) == {}
    assert load_ball2d(str(tmp_path)) == {}
    assert load_pred_boxes(str(tmp_path)) == {}


def test_save_load_pred_boxes(tmp_path):
    """预测框（纯显示）存读往返：格式 [[x1,y1,x2,y2,slot], ...]。"""
    pred = {"7": {"2": [[10.0, 20.0, 30.0, 60.0, 0], [1.0, 2.0, 3.0, 4.0, 1]]}}
    save_pred_boxes(str(tmp_path), pred)
    assert load_pred_boxes(str(tmp_path)) == pred
