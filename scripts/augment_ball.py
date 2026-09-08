#!/usr/bin/env python3
"""乒乓球数据增强：把 ``--limit`` 张原始图扩充到 ``(limit × (1+per))`` 张。

对每张原始图生成 ``--per``（默认 3）个变体，每个变体随机组合：
- **几何**（同步变换 bbox）：水平/垂直翻转、平移、缩放、小角度旋转
- **光度**（只改像素）：亮度/对比度/伽马/高斯噪声/高斯模糊

变体数量 = 原图 + 3×增强 = 4×，即 3000 → 12000。输出到独立目录，
不覆盖原始数据集。空标签（无球帧）只做光度增强（作为背景负样本）。

用法：
  python scripts/augment_ball.py --root data/ball_dataset --out data/ball_dataset_aug
  python scripts/augment_ball.py --per 3 --seed 0

输出：``--out/images/*.png`` + ``--out/labels/*.txt``（YOLO 格式），可直接喂 train_ball.py。
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import cv2
import numpy as np

from tabletennis.core.config import project_root

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def read_box(path: str) -> Optional[Tuple[float, float, float, float]]:
    """读 YOLO 标签第一行 → (cx, cy, w, h) 归一化；空返回 None。"""
    if not os.path.exists(path):
        return None
    line = open(path).read().strip()
    if not line:
        return None
    p = line.split()
    if len(p) < 5:
        return None
    return tuple(float(v) for v in p[1:5])


def in_frame(box: Optional[Tuple[float, float, float, float]]) -> bool:
    """几何变换后 bbox 是否仍在图内（完全或大部分）。"""
    if box is None:
        return True
    cx, cy, w, h = box
    return 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0


def geometric(img: np.ndarray, box: Optional[Tuple[float, float, float, float]],
              rng: np.random.Generator
              ) -> Tuple[np.ndarray, Optional[Tuple[float, float, float, float]]]:
    """随机几何变换（同步变换 bbox）。框被旋出画面则返回原图不变换。"""
    H, W = img.shape[:2]
    op = int(rng.integers(0, 5))
    b = box

    if op == 0:  # 水平翻转
        img = cv2.flip(img, 1)
        if b: b = (1 - b[0], b[1], b[2], b[3])
    elif op == 1:  # 垂直翻转
        img = cv2.flip(img, 0)
        if b: b = (b[0], 1 - b[1], b[2], b[3])
    elif op == 2:  # 平移 ±20%
        tx = float(rng.uniform(-0.20, 0.20) * W)
        ty = float(rng.uniform(-0.20, 0.20) * H)
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        img = cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REPLICATE)
        if b:
            b = (b[0] + tx / W, b[1] + ty / H, b[2], b[3])
    elif op == 3:  # 缩放 0.7~1.4（绕图像中心）
        s = float(rng.uniform(0.7, 1.4))
        M = cv2.getRotationMatrix2D((W / 2, H / 2), 0, s)
        img = cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REPLICATE)
        if b:
            cx, cy, w, h = b
            cx = (cx - 0.5) * s + 0.5
            cy = (cy - 0.5) * s + 0.5
            b = (cx, cy, w * s, h * s)
    else:  # 旋转 ±30°
        ang = float(rng.uniform(-30, 30))
        M = cv2.getRotationMatrix2D((W / 2, H / 2), ang, 1.0)
        img = cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REPLICATE)
        if b:
            cx, cy, w, h = b
            cx_p = (cx - 0.5) * W
            cy_p = (cy - 0.5) * H
            a = np.deg2rad(ang)
            # 注意：cv2.getRotationMatrix2D 的矩阵是 [alpha beta; -beta alpha]，
            # 必须用与 warpAffine 一致的变换方向（否则旋转后球心镜像，框错位）。
            alpha, beta = np.cos(a), np.sin(a)
            cx2 = alpha * cx_p + beta * cy_p
            cy2 = -beta * cx_p + alpha * cy_p
            # 球是圆的，旋转后 bbox 仍是以球心为中心的正方形，边长不变
            b = (cx2 / W + 0.5, cy2 / H + 0.5, w, h)
            if not (0.05 <= b[0] <= 0.95 and 0.05 <= b[1] <= 0.95):
                return img, None  # 球被旋出主体区域，丢弃该变体

    if not in_frame(b):
        return img, None
    return img, b


def photometric(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """随机光度变换（不改 bbox）。幅度已加大（亮度±45 / 对比度0.6~1.45 /
    伽马0.6~1.5 / 噪声σ6~14 / 模糊σ0.5~1.2）以加强域适应；球只有 12~24px，
    极端模糊/伽马/噪声仍可能把小球抹掉产生坏样本，训练后若 mAP 掉需回退模糊/伽马上限。"""
    op = int(rng.integers(0, 5))
    img = img.astype(np.float32)

    if op == 0:  # 亮度 ±45（整体平移，球保持对比）
        img += float(rng.uniform(-45, 45))
    elif op == 1:  # 对比度 0.6~1.45（剧烈）
        img = (img - 128.0) * float(rng.uniform(0.6, 1.45)) + 128.0
    elif op == 2:  # 伽马 0.6~1.5（剧烈，暗帧更强）
        g = float(rng.uniform(0.6, 1.5))
        img = np.clip(img / 255.0, 0, 1) ** g * 255.0
    elif op == 3:  # 高斯噪声 σ=6~14
        img += rng.normal(0, float(rng.uniform(6, 14)), img.shape)
    else:  # 高斯模糊 σ=0.5~1.2（球径 12px 以上才扛得住，σ≤1.2）
        img = cv2.GaussianBlur(img, (0, 0), float(rng.uniform(0.5, 1.2)))

    return np.clip(img, 0, 255).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser(description="乒乓球数据增强（几何+光度，同步变换 bbox）")
    ap.add_argument("--root", default=None, help="原始数据集（默认 data/ball_dataset）")
    ap.add_argument("--out", default=None, help="输出数据集（默认 data/ball_dataset_aug）")
    ap.add_argument("--limit", type=int, default=3000, help="只用排序前 N 张原始图（默认 3000）")
    ap.add_argument("--per", type=int, default=3, help="每张生成几个变体（默认 3 → 总 4×）")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = args.root
    if root is None:
        root = os.path.join(project_root(), "data", "ball_dataset")
    out = args.out
    if out is None:
        out = os.path.join(project_root(), "data", "ball_dataset_aug")

    img_in = os.path.join(root, "images")
    lbl_in = os.path.join(root, "labels")
    img_out = os.path.join(out, "images")
    lbl_out = os.path.join(out, "labels")
    os.makedirs(img_out, exist_ok=True)
    os.makedirs(lbl_out, exist_ok=True)

    names = sorted(f for f in os.listdir(img_in) if f.lower().endswith(IMG_EXTS))[:args.limit]
    if not names:
        sys.exit(f"[错误] {img_in} 里没有图片")
    rng = np.random.default_rng(args.seed)

    total = len(names) * (1 + args.per)
    n_written = 0
    n_box = n_empty = 0

    for i, name in enumerate(names):
        img = cv2.imread(os.path.join(img_in, name), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        stem = os.path.splitext(name)[0]
        box = read_box(os.path.join(lbl_in, stem + ".txt"))
        if box is not None:
            n_box += 1
        else:
            n_empty += 1

        # 原图原样复制（含标签）
        cv2.imwrite(os.path.join(img_out, name), img)
        with open(os.path.join(lbl_out, stem + ".txt"), "w") as f:
            if box:
                f.write(f"0 {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n")
        n_written += 1

        # 生成 per 个变体
        for v in range(args.per):
            # 几何变换（空标签帧跳过几何，只做光度）
            aug = img
            aug_box = box
            if box is not None:
                aug, aug_box = geometric(img, box, rng)
                if aug_box is None:
                    # 几何变换把球旋出画面 → 用原图重采样一次
                    aug, aug_box = geometric(img, box, rng)
                    if aug_box is None:
                        continue
            aug = photometric(aug, rng)

            vname = f"{stem}_v{v+1}.png"
            cv2.imwrite(os.path.join(img_out, vname), aug)
            vstem = os.path.splitext(vname)[0]
            with open(os.path.join(lbl_out, vstem + ".txt"), "w") as f:
                if aug_box:
                    f.write(f"0 {aug_box[0]:.6f} {aug_box[1]:.6f} {aug_box[2]:.6f} {aug_box[3]:.6f}\n")
            n_written += 1

        if (i + 1) % 500 == 0:
            print(f"  已处理 {i+1}/{len(names)}")

    print(f"✓ 输出 {n_written} 张 → {out}（含 {n_box} 有球原图 / {n_empty} 空标签原图，各 ×{1+args.per}）")
    print(f"  目标 {total} 张；几何变换出界丢弃了 {total - n_written} 张")
    print("  下一步：python scripts/train_ball.py --out " + out)


if __name__ == "__main__":
    main()
