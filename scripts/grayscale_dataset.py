#!/usr/bin/env python3
"""把 YOLO 格式数据集的图片转成灰度（对齐黑白相机），用于域适应。

公开数据集是 RGB，而你的相机是 Mono8 灰度。训练前把图转灰度，让模型学到的特征
与推理时的输入一致（否则 RGB 上训的模型在灰度图上有明显 domain gap）。

用法：
  conda activate tt
  python scripts/grayscale_dataset.py --in data/public_ball_dataset --out data/ball_dataset
  python scripts/grayscale_dataset.py --in data/public_ball_dataset --inplace   # 原地覆盖

``--in`` 目录需含 ``images/`` 与 ``labels/``（YOLO 格式，标签原样复制）。
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

import cv2

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def main() -> None:
    ap = argparse.ArgumentParser(description="YOLO 数据集灰度化（对齐黑白相机）")
    ap.add_argument("--in", dest="in_dir", required=True, help="输入数据集根目录（含 images/ 与 labels/）")
    ap.add_argument("--out", default=None, help="输出目录；省略则 --inplace")
    args = ap.parse_args()

    in_img = os.path.join(args.in_dir, "images")
    in_lbl = os.path.join(args.in_dir, "labels")
    if not os.path.isdir(in_img):
        sys.exit(f"[错误] 找不到 {in_img}")

    out_dir = args.out or args.in_dir
    out_img = os.path.join(out_dir, "images")
    out_lbl = os.path.join(out_dir, "labels")
    os.makedirs(out_img, exist_ok=True)
    if os.path.isdir(in_lbl):
        os.makedirs(out_lbl, exist_ok=True)

    names = sorted(f for f in os.listdir(in_img) if f.lower().endswith(IMG_EXTS))
    n = 0
    for name in names:
        img = cv2.imread(os.path.join(in_img, name), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        cv2.imwrite(os.path.join(out_img, name), img)
        lbl_src = os.path.join(in_lbl, os.path.splitext(name)[0] + ".txt")
        if os.path.exists(lbl_src):
            shutil.copy(lbl_src, os.path.join(out_lbl, os.path.splitext(name)[0] + ".txt"))
        n += 1

    print(f"✓ 已灰度化 {n} 张图片 → {out_dir}")
    print("  下一步：python scripts/train_ball.py --out " + out_dir)


if __name__ == "__main__":
    main()
