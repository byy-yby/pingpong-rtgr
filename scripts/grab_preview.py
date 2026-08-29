#!/usr/bin/env python3
"""单相机实时预览（验证触发、调曝光/增益）。

用法示例：
  python scripts/grab_preview.py                              # 自由采集预览第一台
  python scripts/grab_preview.py --trigger external          # 外部触发（信号发生器接 Line0）
  python scripts/grab_preview.py --trigger software          # 软件触发（无需信号发生器）
  python scripts/grab_preview.py --camera-id 2 --exposure 5000 --gain 5 --gamma 1.0
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tabletennis.camera.camera import Camera  # noqa: E402
from tabletennis.camera.sdk import (  # noqa: E402
    enumerate_devices,
    finalize_sdk,
    initialize_sdk,
)
from tabletennis.core.config import resolve_camera_settings  # noqa: E402

try:
    import cv2  # type: ignore

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def main() -> None:
    ap = argparse.ArgumentParser(description="单相机实时预览")
    ap.add_argument("--camera-id", type=int, default=0, help="相机逻辑索引（0..N-1）")
    ap.add_argument("--trigger", choices=["continuous", "external", "software"], default="continuous")
    ap.add_argument("--exposure", type=float, default=None, help="曝光时间(us)")
    ap.add_argument("--gain", type=float, default=None, help="增益(dB)")
    ap.add_argument("--gamma", type=float, default=None, help="伽马")
    ap.add_argument("--black-level", type=float, default=None, help="黑电平")
    ap.add_argument("--fps", type=float, default=None, help="采集帧率(Hz，自由采集时生效)")
    ap.add_argument("--no-display", action="store_true", help="不弹窗，只打印帧率统计")
    args = ap.parse_args()

    initialize_sdk()
    cam = None
    try:
        devices = enumerate_devices()
        if not devices:
            print("未找到相机")
            return
        if args.camera_id >= len(devices):
            print(f"camera-id {args.camera_id} 越界，只有 {len(devices)} 台")
            return

        dev = devices[args.camera_id]
        print(f"打开相机 [{args.camera_id}] {dev.model} 序列号={dev.serial}")
        cs = resolve_camera_settings(args.exposure, args.gain, args.gamma)
        cam = Camera(
            dev,
            args.camera_id,
            trigger_mode=args.trigger,
            exposure_us=cs["exposure_us"],
            gain_db=cs["gain_db"],
            pixel_format="Mono8",
        )
        cam.open()
        cam.controls.set_gamma(cs["gamma"])

        # 构造函数未覆盖的可选参数
        if args.black_level is not None:
            cam.controls.set_black_level(args.black_level)
        if args.fps is not None:
            cam.controls.set_frame_rate(args.fps)

        cam.start()
        print("按 Ctrl+C 退出" + ("" if HAS_CV2 or args.no_display else "（未安装 opencv，仅打印帧率）"))

        t0, n = time.time(), 0
        while True:
            frame = cam.get_latest_frame(timeout=1.0)
            if frame is None:
                continue
            n += 1
            if n % 30 == 0:
                fps = n / (time.time() - t0)
                print(f"帧率 {fps:6.1f} fps | 帧号 {frame.frame_num} | {frame.width}x{frame.height}")
            if HAS_CV2 and not args.no_display:
                cv2.imshow(f"cam-{args.camera_id}", frame.image)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC 退出
                    break
    except KeyboardInterrupt:
        pass
    finally:
        if cam is not None:
            cam.stop()
            cam.close()
        finalize_sdk()
        if HAS_CV2:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
