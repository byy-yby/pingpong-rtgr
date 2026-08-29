#!/usr/bin/env python3
"""姿态识别实时预览：自动开启相机 → RTMPose-l 检测 → 骨架叠加到相机图像。

这是「姿势识别模块」的启动脚本。运行它时会**自动走相机启动流程**
（枚举 → CreateHandle → OpenDevice → 触发/曝光配置 → StartGrabbing → 采集线程），
然后把 RTMPose-l 检测出的人体骨骼实时渲染到相机画面里。

用法示例：
  # 单相机 + RTMPose-l（top-down：YOLOX 检测人 + RTMPose 关键点），自由采集
  python scripts/pose_preview.py

  # 外部触发（信号发生器接 Line0）
  python scripts/pose_preview.py --trigger external

  # 软件触发（无信号发生器，逐帧软触发）
  python scripts/pose_preview.py --trigger software

  # 多相机平铺（默认 2x2），每路都跑检测
  python scripts/pose_preview.py --all-cameras

  # 只预览画面不跑模型（先验证相机/曝光）
  python scripts/pose_preview.py --no-pose

  # 调模型 / 精度 / 速度
  python scripts/pose_preview.py --model rtmpose-l-halpe26 --device cpu --input-size 192 256 --stride 3

提示：本机 GPU 是 GT 1030 2GB，RTMPose 默认跑 CPU；换好 GPU 后加 --device cuda。
多路实时会卡，建议先单路、按需降分辨率/隔帧。
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

try:
    import cv2
    HAS_CV2 = True
except ImportError:  # pragma: no cover
    HAS_CV2 = False


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="姿态识别实时预览（相机 + RTMPose-l + 骨架叠加）")
    ap.add_argument("--camera-id", type=int, default=0, help="相机逻辑索引（0..N-1，默认第一台）")
    ap.add_argument("--all-cameras", action="store_true", help="打开全部在线相机并平铺显示")
    ap.add_argument("--trigger", choices=["continuous", "external", "software"], default="continuous")
    ap.add_argument("--exposure", type=float, default=None, help="曝光时间(us)")
    ap.add_argument("--gain", type=float, default=None, help="增益(dB)")

    ap.add_argument("--no-pose", action="store_true", help="只预览画面，不加载/运行姿态模型")
    ap.add_argument("--model", default="rtmpose-l-halpe26",
                    help="rtmpose-l-halpe26 / rtmpose-m-halpe26 / rtmpose-s-halpe26 / "
                         "rtmpose-x-halpe26（26点）；不带 -halpe26 为 COCO-17 模型；或本地 onnx 路径")
    ap.add_argument("--device", default="cuda", help="cpu 或 cuda（默认 cuda，缺 CUDA 自动回退）")
    ap.add_argument("--input-size", type=int, nargs=2, default=(192, 256),
                    metavar=("H", "W"), help="姿态模型输入尺寸(H W)，rtmpose-l 用 192 256")
    ap.add_argument("--score-thr", type=float, default=0.5, help="人体检测置信度阈值")
    ap.add_argument("--stride", type=int, default=1, help="隔 N 帧检测一次（复用上次结果，提速）")
    ap.add_argument("--max-side", type=int, default=None, help="检测前把长边缩到该像素（提速）")
    ap.add_argument("--no-display", action="store_true", help="不弹窗，只打印 FPS")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()

    if not HAS_CV2:
        print("未安装 opencv-python，无法显示；请先：conda run -n tt pip install opencv-python")
        sys.exit(1)

    # 延迟 import 相机 / 姿态模块，保证 --no-pose 时不触发 rtmlib 依赖
    from tabletennis.camera.camera import Camera
    from tabletennis.camera.sdk import enumerate_devices, finalize_sdk, initialize_sdk
    from tabletennis.core.config import resolve_camera_settings
    from tabletennis.visualization.overlay2d import annotate_frame

    initialize_sdk()
    cameras = []
    detector = None
    try:
        devices = enumerate_devices()
        if not devices:
            print("未找到相机。检查 USB 连接 / udev 权限 / 是否被 MVS 客户端占用。", flush=True)
            return

        # 决定开哪些相机
        if args.all_cameras:
            sel = list(range(len(devices)))
        else:
            if args.camera_id >= len(devices):
                print(f"camera-id {args.camera_id} 越界，只有 {len(devices)} 台")
                return
            sel = [args.camera_id]

        # 曝光/增益：命令行 > 保存的设置文件 > 默认值
        cs = resolve_camera_settings(args.exposure, args.gain)

        # 按「相机启动流程」逐台打开（open 内部完成触发/曝光配置，但未 StartGrabbing）
        for cid in sel:
            dev = devices[cid]
            print(f"打开相机 [{cid}] {dev.model} 序列号={dev.serial}", flush=True)
            cam = Camera(
                dev,
                cid,
                trigger_mode=args.trigger,
                exposure_us=cs["exposure_us"],
                gain_db=cs["gain_db"],
                pixel_format="Mono8",
            )
            cam.open()
            cameras.append(cam)

        # 加载姿态模型（失败则降级为纯预览，不中断相机）
        if not args.no_pose:
            try:
                from tabletennis.vision.pose.rtmpose_pose import RTMPoseDetector
                detector = RTMPoseDetector(
                    model=args.model,
                    input_size=tuple(args.input_size),
                    device=args.device,
                    score_thr=args.score_thr,
                )
                print(f"RTMPose 模型加载完成（{args.model}, {args.device}）", flush=True)
            except Exception as exc:  # noqa: BLE001
                print(f"[警告] RTMPose 加载失败，降级为纯画面预览：{exc}", flush=True)
                detector = None

        # 启动采集
        for cam in cameras:
            cam.start()
        print("已开始抓帧，按 ESC 或 Ctrl+C 退出", flush=True)

        last_poses = {cid: [] for cid in sel}
        frame_idx = 0
        t0, n = time.time(), 0
        while True:
            if args.no_display:
                # 无 GUI 时没有 waitKey 限速，这里统一限速，避免空转烧 CPU
                time.sleep(0.033)
            for cam in cameras:
                # Camera 的有界队列满时会丢最旧帧，直接取最新即可，无需手动 drain
                frame = cam.get_latest_frame(timeout=0.1)
                if frame is None:
                    continue

                poses = []
                if detector is not None and frame_idx % max(args.stride, 1) == 0:
                    from dataclasses import replace
                    scale = 1.0
                    det_frame = frame
                    if args.max_side and max(frame.height, frame.width) > args.max_side:
                        scale = args.max_side / max(frame.height, frame.width)
                        small = cv2.resize(
                            frame.image, None, fx=scale, fy=scale,
                            interpolation=cv2.INTER_AREA,
                        )
                        # 检测用缩小图，坐标再映射回原图
                        det_frame = replace(frame, image=small)
                    poses = detector.detect(det_frame)
                    if scale != 1.0:
                        for p in poses:
                            p.keypoints[:, :2] /= scale
                    last_poses[cam.logical_id] = poses
                else:
                    poses = last_poses[cam.logical_id]

                annotated = annotate_frame(
                    frame.image,
                    poses,
                    title=f"cam{cam.logical_id} ({frame.width}x{frame.height})",
                )

                if args.no_display:
                    continue
                cv2.imshow(f"pose-cam{cam.logical_id}", annotated)

            frame_idx += 1
            n += 1
            if n % 30 == 0:
                fps = n / (time.time() - t0)
                print(f"处理帧率 {fps:6.1f} fps | 相机数 {len(cameras)} | 模型 {'RTMPose' if detector else '关闭'}", flush=True)

            if not args.no_display and (cv2.waitKey(1) & 0xFF == 27):
                break
    except KeyboardInterrupt:
        pass
    finally:
        for cam in cameras:
            cam.stop()
            cam.close()
        finalize_sdk()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
