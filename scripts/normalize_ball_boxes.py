#!/usr/bin/env python3
"""统一手绘标注框：refine 精修球心+半径 → 以球心为中心的正方形框。

手绘框常见问题：中心偏移、长宽不方正（球是圆的）、尺寸偏大。
本脚本对每张**已有框**的图调 ``refine_ball_center``（强度加权质心+二阶矩），
得到亚像素球心与半径，重写为「中心=精修球心、边长≈2.4×半径」的正方形框；
refine 失败（conf<0.3）时回退为「原框中心 + 正方形（边长=原 max 边）」。
空标签（无球帧）保持不变。

用法：
  python scripts/normalize_ball_boxes.py                 # 处理 data/ball_dataset 前 N 张
  python scripts/normalize_ball_boxes.py --root data/ball_dataset --limit 6000
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import cv2

from tabletennis.vision.ball.refine import refine_ball_center

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def norm_to_px(cx, cy, w, h, W, H) -> tuple:
    return int(cx * W), int(cy * H), w * W, h * H


def write_yolo(path: str, box: Optional[tuple], W: int, H: int) -> None:
    """box=(x,y,r_px) 精修结果或回退；空则写空文件。"""
    if box is None:
        open(path, "w").close()
        return
    x, y, r = box
    side = 2.4 * r
    side = min(max(side, 6.0), 64.0)           # 防止极端值
    cx = min(max(x / W, 0.0), 1.0)
    cy = min(max(y / H, 0.0), 1.0)
    w = min(max(side / W, 0.0), 1.0)
    h = min(max(side / H, 0.0), 1.0)
    with open(path, "w") as f:
        f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="统一手绘标注框（精修球心+正方形）")
    ap.add_argument("--root", default=None, help="数据集根目录（默认 data/ball_dataset）")
    ap.add_argument("--limit", type=int, default=6000, help="只处理排序前 N 张（默认 6000）")
    ap.add_argument("--dry-run", action="store_true", help="只统计不改写")
    args = ap.parse_args()

    root = args.root
    if root is None:
        from tabletennis.core.config import project_root
        root = os.path.join(project_root(), "data", "ball_dataset")
    img_dir = os.path.join(root, "images")
    lbl_dir = os.path.join(root, "labels")
    if not os.path.isdir(img_dir):
        sys.exit(f"[错误] 找不到 {img_dir}")

    names = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(IMG_EXTS))[:args.limit]
    if not names:
        sys.exit("[错误] 没有图片")
    print(f"处理前 {len(names)} 张图（{root}）...")

    n_ok = n_fallback = n_empty = n_skip = 0
    for name in names:
        lbl_path = os.path.join(lbl_dir, os.path.splitext(name)[0] + ".txt")
        if not os.path.exists(lbl_path):
            n_skip += 1
            continue
        line = open(lbl_path).read().strip()
        if not line:
            n_empty += 1
            continue
        parts = line.split()
        if len(parts) < 5:
            n_skip += 1
            continue

        img = cv2.imread(os.path.join(img_dir, name), cv2.IMREAD_GRAYSCALE)
        if img is None:
            n_skip += 1
            continue
        H, W = img.shape[:2]

        cx, cy, w, h = (float(p) for p in parts[1:5])
        px, py, pw, ph = norm_to_px(cx, cy, w, h, W, H)
        r_hint = max(pw, ph) / 2.0

        x, y, r, conf = refine_ball_center(img, px, py, radius_hint=r_hint)
        if conf > 0.3 and 2.0 <= r <= 32.0:
            new_box = (x, y, r)
            n_ok += 1
        else:
            # 回退：原框中心 + 正方形（边长=原 max 边）
            new_box = (px, py, max(pw, ph) / 2.0)
            n_fallback += 1

        if not args.dry_run:
            write_yolo(lbl_path, new_box, W, H)

    print(f"✓ 精修正方形框: {n_ok} | 回退(原框转正方形): {n_fallback} | "
          f"空标签(无球帧保留): {n_empty} | 跳过: {n_skip}")

    if args.dry_run:
        print("（--dry-run 未改写任何文件）")
    else:
        print("原标签已备份（如需要回滚可手动恢复）。")


if __name__ == "__main__":
    main()
