#!/usr/bin/env python3
"""四机同步采集验证：外部触发下观察各机帧号 / 设备时间戳是否对齐。

用法：
  python scripts/grab_sync.py                 # 读取 config/cameras.yaml
  python scripts/grab_sync.py --config config/cameras.yaml
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tabletennis.camera import CameraManager  # noqa: E402
from tabletennis.core.config import load_yaml, resolve_camera_settings  # noqa: E402


def camera_kwargs(config: dict) -> dict:
    """把 cameras.yaml 的 trigger/image 段转成 kwargs；曝光/增益来自共享设置文件。"""
    trigger = config.get("trigger", {}) or {}
    image = config.get("image", {}) or {}
    cs = resolve_camera_settings()
    return {
        "trigger_mode": trigger.get("mode", "continuous"),
        "trigger_source": trigger.get("source", "Line0"),
        "exposure_us": cs["exposure_us"],
        "gain_db": cs["gain_db"],
        "pixel_format": image.get("pixel_format", "Mono8"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="四机同步采集验证")
    ap.add_argument("--config", default=None, help="cameras.yaml 路径（默认 config/cameras.yaml）")
    args = ap.parse_args()

    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    cfg_path = args.config or os.path.join(root, "config", "cameras.yaml")
    config = load_yaml(cfg_path) if os.path.exists(cfg_path) else {}

    serials = [c.get("serial") for c in config.get("cameras", []) if c.get("serial")] or None

    try:
        with CameraManager(serials=serials, **camera_kwargs(config)) as mgr:
            mgr.start()
            print(f"已启动 {len(mgr.cameras)} 台相机（触发模式见 {cfg_path}），Ctrl+C 退出")
            print("观察各机 frame_num 与 device_timestamp 是否一致：")

            while True:
                frames = {
                    cid: f
                    for cid, f in mgr.get_latest_frames(block=False).items()
                    if f is not None
                }
                if not frames:
                    time.sleep(0.01)
                    continue

                for cid in sorted(frames):
                    f = frames[cid]
                    print(f"  cam{cid}: frame={f.frame_num:>6} ts={f.device_timestamp}")

                ts = [f.device_timestamp for f in frames.values()]
                if len(ts) > 1:
                    spread = max(ts) - min(ts)
                    print(f"  --- 时间戳极差 {spread} tick（越小越同步）---")
                print()
                time.sleep(0.5)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
