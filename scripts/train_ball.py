#!/usr/bin/env python3
"""训练单类「ball」检测器（ultralytics YOLO）+ 导出 ONNX。

前置（一次性）：
  conda activate tt
  pip install torch --index-url https://download.pytorch.org/whl/cu128   # RTX5080=Blackwell，需 CUDA12.8
  pip install ultralytics

流程：读 ``data/ball_dataset`` 的图片+标签 → 按 9:1 分 train/val（写 train.txt/val.txt
与 ball.yaml）→ 训练 YOLOv8n（单类，输入 ≥1280 以检测 12~24px 小球）→ 导出 ONNX。

用法：
  python scripts/train_ball.py                          # 默认 yolov8n, 100 epochs, imgsz 1280, GPU0
  python scripts/train_ball.py --epochs 200 --imgsz 1280 --batch 16
  python scripts/train_ball.py --model yolov8s.pt --device 0
  python scripts/train_ball.py --no-export              # 只训练，不导出 ONNX

输出：``runs/detect/train*/weights/best.pt``（训练权重）与 ``best.onnx``（推理用）。
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np

from tabletennis.core.config import project_root


def build_dataset(root: str, val_ratio: float = 0.1, seed: int = 0) -> str:
    """把平铺的 images/labels 分成 train/val，写 train.txt/val.txt + ball.yaml，返回 yaml 路径。"""
    img_dir = os.path.join(root, "images")
    lbl_dir = os.path.join(root, "labels")
    os.makedirs(lbl_dir, exist_ok=True)

    names = sorted(f for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg", ".bmp")))
    if not names:
        raise SystemExit(f"[错误] {img_dir} 里没有图片，先跑 capture_ball.py 采集。")

    # 缺标签的图片补空标签（当背景负样本）
    for name in names:
        lbl = os.path.join(lbl_dir, os.path.splitext(name)[0] + ".txt")
        if not os.path.exists(lbl):
            open(lbl, "w").close()

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(names))
    n_val = max(1, int(len(names) * val_ratio))
    val_idx = set(order[:n_val].tolist())

    train_lines, val_lines = [], []
    for i, name in enumerate(names):
        line = os.path.join(img_dir, name)
        (val_lines if i in val_idx else train_lines).append(line)

    with open(os.path.join(root, "train.txt"), "w") as f:
        f.write("\n".join(train_lines) + "\n")
    with open(os.path.join(root, "val.txt"), "w") as f:
        f.write("\n".join(val_lines) + "\n")

    yaml_path = os.path.join(root, "ball.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {root}\n")
        f.write("train: train.txt\n")
        f.write("val: val.txt\n")
        f.write("names:\n  0: ball\n")

    n_pos = sum(
        1 for name in names
        if os.path.getsize(os.path.join(lbl_dir, os.path.splitext(name)[0] + ".txt")) > 0
    )
    print(f"数据集：{len(names)} 张（{n_pos} 有球 / {len(names) - n_pos} 背景负样本），"
          f"train {len(train_lines)} / val {len(val_lines)}")
    return yaml_path


def main() -> None:
    ap = argparse.ArgumentParser(description="训练单类 ball 检测器 + 导出 ONNX")
    ap.add_argument("--out", default=None, help="数据集根目录（默认 data/ball_dataset）")
    ap.add_argument("--model", default="yolov8n.pt", help="预训练权重（yolov8n/s/m...）")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=1280, help="输入分辨率（小球需 ≥1280）")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--patience", type=int, default=None,
                    help="val mAP 早停耐心（None=ultralytics 默认不早停）")
    ap.add_argument("--device", default="0", help="0/1/2...=GPU，cpu=CPU")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--no-export", action="store_true", help="只训练不导出 ONNX")
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit(
            "[错误] 未安装 ultralytics/torch。先执行：\n"
            "  conda activate tt\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cu128\n"
            "  pip install ultralytics"
        )

    root = args.out or os.path.join(project_root(), "data", "ball_dataset")
    yaml_path = build_dataset(root, val_ratio=args.val_ratio)

    model = YOLO(args.model)
    model.train(
        data=yaml_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        single_cls=True,
        patience=args.patience,
        project=os.path.join(project_root(), "runs", "detect"),
        name="ball",
    )

    if not args.no_export:
        best_pt = os.path.join(project_root(), "runs", "detect", "ball", "weights", "best.pt")
        model = YOLO(best_pt)
        onnx_path = model.export(format="onnx", imgsz=args.imgsz)
        print(f"\n✓ 训练完成。ONNX 模型：{onnx_path}")
        print("  推理时把该路径传给 reconstruct_ball.py --ball-model 或 live_control。")


if __name__ == "__main__":
    main()
