#!/usr/bin/env python3
"""把某次离线重建（``scripts/reconstruct_video.py`` 的输出目录）做成 3D 回放。

在 Open3D 新 Filament 渲染器里搭出 球桌 + 地面 + 相机视锥 场景，逐帧回放 SMPL
拟合结果——人物网格按太阳光照着色并在桌面/地面投射**真实接触阴影**（不是旧
``Visualizer`` 的 headlight 平面白）。播放语义与重建一致：no_person/失败帧清空
人体；被 ``--stride`` 跳过的帧保持上一姿态。

用法
  python scripts/visualize_recon.py <out_dir>        # 回放某次重建结果
  python scripts/visualize_recon.py --watch <out_dir>  # 重建进行中实时观看（追帧）
  python scripts/visualize_recon.py --render 42 out.png <out_dir>  # 离线出第 42 帧 PNG

不传 <out_dir> 时默认找最新的 ``data/video/*/recon``。

交互
  鼠标：左拖=旋转，右键拖=平移，滚轮=缩放
  键盘：Space=播放/暂停，←/→=步进一帧，Home/End=首/尾，R=复位视角，
        -=/+ 半速/倍速，Esc=退出

要求重建输出里带 ``recon_faces.npy``（网格拓扑，SMPL 网格必需；新版
reconstruct_video.py 会自动写）。旧输出缺它时可加 ``--easymocap-root`` 让本脚本用
SMPL 模型补写一次。
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS)
sys.path.insert(0, os.path.join(_ROOT, "src"))


def _latest_recon() -> str:
    """最新的 data/video/<session>/recon 输出目录。"""
    hits = sorted(glob.glob(os.path.join(_ROOT, "data", "video", "*", "recon")))
    if not hits:
        raise SystemExit("✗ 找不到任何 data/video/*/recon —— 请传 <out_dir>")
    return hits[-1]


def build_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir", nargs="?", help="reconstruct_video.py 的输出目录（默认最新）")
    ap.add_argument("--watch", action="store_true", help="重建进行中实时追帧（recon 还在跑时用）")
    ap.add_argument("--fps", type=float, default=0.0, help="回放速度（帧/秒；默认=录像真实出帧率）")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--render", nargs=2, metavar=("T", "OUT_PNG"),
                    help="离线渲染主时钟第 T 帧到 PNG（无窗口，EGL headless，验证/导出用）")
    ap.add_argument("--root", default="", help="项目根（含 data/calibration、data/extrinsics；"
                                               "默认项目根）")
    ap.add_argument("--easymocap-root", default="/home/yby/projects/EasyMocap",
                    help="faces 缺失时用 SMPL 模型补写（默认路径）")
    return ap.parse_args()


def main() -> None:
    args = build_args()
    from tabletennis.visualization.recon_player import (
        ReconTimeline,
        load_faces,
        play_gui,
        render_still,
    )

    out_dir = args.out_dir or _latest_recon()
    if not os.path.isdir(out_dir):
        print(f"✗ 找不到输出目录：{out_dir}")
        sys.exit(2)

    root = args.root or None
    if args.render:
        t, png = int(args.render[0]), args.render[1]
        tl = ReconTimeline(out_dir)
        arr = render_still(tl, t, args.width, args.height, out_png=png, root=root)
        shown = tl.person_path_at(t)
        print(f"✓ 已渲染 t={t} -> {png}（{arr.shape[1]}×{arr.shape[0]}，"
              f"{'人物' if shown else '无人'}）")
        return

    play_gui(out_dir, easymocap_root=args.easymocap_root,
             width=args.width, height=args.height, fps=args.fps,
             watch=args.watch, root=root)


if __name__ == "__main__":
    main()
