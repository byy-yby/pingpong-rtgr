#!/usr/bin/env python3
"""连接一台相机，打印所有图像控制参数（曝光/增益/黑电平/伽马/亮度/…）及其取值范围。

用于研究「亮度、曝光补偿之类的句柄怎么用」—— 直接看这台相机实际支持哪些节点。

用法：
  python scripts/query_parameters.py --camera-id 0
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tabletennis.camera.camera import Camera  # noqa: E402
from tabletennis.camera.sdk import (  # noqa: E402
    enumerate_devices,
    finalize_sdk,
    initialize_sdk,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="查询相机图像控制参数")
    ap.add_argument("--camera-id", type=int, default=0)
    args = ap.parse_args()

    initialize_sdk()
    cam = None
    try:
        devices = enumerate_devices()
        if not devices:
            print("未找到相机")
            return
        dev = devices[min(args.camera_id, len(devices) - 1)]
        print(f"查询相机 [{args.camera_id}] {dev.model} 序列号={dev.serial}\n")

        cam = Camera(dev, args.camera_id)
        cam.open()

        info = cam.controls.list_supported()
        for name, value in info.items():
            if value is None:
                print(f"  {name:<24} = 不支持/读取失败")
            elif isinstance(value, tuple):
                print(f"  {name:<24} = {value}  (范围)")
            else:
                print(f"  {name:<24} = {value}")
    finally:
        if cam is not None:
            cam.close()
        finalize_sdk()


if __name__ == "__main__":
    main()
