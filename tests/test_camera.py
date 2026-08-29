"""相机模块基础测试。无相机时自动跳过需要硬件的用例。"""
import pytest

from tabletennis.camera.camera import Camera
from tabletennis.camera.sdk import enumerate_devices, finalize_sdk, initialize_sdk


def _enum_once():
    initialize_sdk()
    try:
        return enumerate_devices()
    finally:
        finalize_sdk()


# 模块加载时枚举一次，供 skipif 判断
DEVICES = _enum_once()
HAS_CAMERA = len(DEVICES) > 0


def test_enumerate_no_exception():
    """枚举本身不应抛异常（0 台或 4 台都算正常）。"""
    assert isinstance(DEVICES, list)


@pytest.mark.skipif(not HAS_CAMERA, reason="未检测到相机")
def test_open_close():
    cam = Camera(DEVICES[0], 0)
    cam.open()
    assert cam.controls is not None
    cam.close()


@pytest.mark.skipif(not HAS_CAMERA, reason="未检测到相机")
def test_apply_settings_and_readback():
    cam = Camera(DEVICES[0], 0, trigger_mode="continuous", pixel_format="Mono8")
    cam.open()
    try:
        # 像素格式应能读回 Mono8（黑白相机默认）
        pf = cam.controls.get_pixel_format()
        assert pf is not None
        # list_supported 不应抛异常
        info = cam.controls.list_supported()
        assert isinstance(info, dict)
    finally:
        cam.close()


@pytest.mark.skipif(not HAS_CAMERA, reason="未检测到相机")
def test_soft_trigger_grab():
    """软件触发模式下自动出帧，验证端到端取图。"""
    cam = Camera(DEVICES[0], 0, trigger_mode="software", pixel_format="Mono8")
    cam.open()
    try:
        cam.start()
        frame = cam.get_latest_frame(timeout=3.0)
        assert frame is not None
        assert frame.image.ndim == 2
        assert frame.width > 0 and frame.height > 0
    finally:
        cam.stop()
        cam.close()
