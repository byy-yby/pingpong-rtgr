#!/usr/bin/env python3
"""枚举并打印所有在线相机（用于确认连接、拿到序列号）。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tabletennis.camera.sdk import (  # noqa: E402
    enumerate_devices,
    finalize_sdk,
    get_sdk_version,
    initialize_sdk,
)


def main() -> None:
    initialize_sdk()
    try:
        print(f"MVS SDK 版本: 0x{get_sdk_version():08x}")
        devices = enumerate_devices()
        print(f"在线相机数量: {len(devices)}\n")
        if not devices:
            print("未找到相机。请检查：")
            print("  1. USB3 连接是否正常（lsusb 应能看到 2bdf:0001）")
            print("  2. 是否已安装 udev 规则（无权限时枚举为空）")
            return
        for d in devices:
            print(
                f"[{d.index}] {d.layer_name:<8} {d.model:<20} "
                f"序列号={d.serial:<12} 自定义名={d.user_defined_name!r}"
            )
    finally:
        finalize_sdk()


if __name__ == "__main__":
    main()
