"""简单的 YAML 配置加载与项目路径工具。"""
from __future__ import annotations

import json
import os
from typing import Any, Optional

import yaml

# 项目根目录：src/tabletennis/core/config.py -> 向上三级
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)


def project_root() -> str:
    return _PROJECT_ROOT


def config_dir() -> str:
    return os.path.join(_PROJECT_ROOT, "config")


def load_yaml(path: str) -> Any:
    """读取并解析一个 YAML 文件。"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_yaml_optional(path: str, default: Any = None) -> Any:
    """读取 YAML，文件不存在时返回默认值。"""
    if not os.path.exists(path):
        return default
    return load_yaml(path)


def load_default_config(name: str) -> Any:
    """读取 ``config/<name>.yaml``。"""
    return load_yaml(os.path.join(config_dir(), f"{name}.yaml"))


# ---- 共享相机图像参数（曝光 / 增益 / 伽马）持久化 ----
# 所有打开相机的脚本都从这里读默认值；live_control.py 的「保存」按钮写这里。

CAMERA_SETTINGS_PATH = os.path.join(config_dir(), "camera_settings.json")

_DEFAULT_CAMERA_SETTINGS = {
    "exposure_us": 5000.0,
    "gain_db": 0.0,
    "gamma": 1.0,
}


def load_camera_settings() -> dict:
    """读取共享相机图像参数（config/camera_settings.json）。

    文件不存在或损坏时返回默认值（5000us / 0dB / 1.0）。
    """
    if not os.path.exists(CAMERA_SETTINGS_PATH):
        return dict(_DEFAULT_CAMERA_SETTINGS)
    try:
        with open(CAMERA_SETTINGS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return dict(_DEFAULT_CAMERA_SETTINGS)

    out = dict(_DEFAULT_CAMERA_SETTINGS)
    if isinstance(data, dict):
        for key in out:
            if key in data and data[key] is not None:
                out[key] = data[key]
    return out


def save_camera_settings(exposure_us: float, gain_db: float, gamma: float) -> None:
    """把当前曝光 / 增益 / 伽马写入 config/camera_settings.json（原子写，避免写一半）。"""
    data = {
        "exposure_us": float(exposure_us),
        "gain_db": float(gain_db),
        "gamma": float(gamma),
    }
    os.makedirs(os.path.dirname(CAMERA_SETTINGS_PATH), exist_ok=True)
    tmp = CAMERA_SETTINGS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    os.replace(tmp, CAMERA_SETTINGS_PATH)


def resolve_camera_settings(exposure_us=None, gain_db=None, gamma=None) -> dict:
    """合并相机图像参数。优先级：显式参数 > 保存的设置文件 > 默认值。"""
    cs = load_camera_settings()
    return {
        "exposure_us": cs["exposure_us"] if exposure_us is None else float(exposure_us),
        "gain_db": cs["gain_db"] if gain_db is None else float(gain_db),
        "gamma": cs["gamma"] if gamma is None else float(gamma),
    }
