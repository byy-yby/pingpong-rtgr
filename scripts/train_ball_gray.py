#!/usr/bin/env python3
"""微调 yolo11n-grayscale（1 通道灰度）单类 ball + 导出动态 batch ONNX。

与 ``train_ball.py`` 的区别：用官方灰度预训练 ``yolo11n-grayscale.pt``（第一层
in_channels=1），数据集 yaml 加 ``channels: 1``，训练出的模型直接吃单通道灰度图，
推理时无需再复制成 3 通道（省预处理 + 输入 tensor 减到 1/3）。

用法：
  conda activate tt
  python scripts/train_ball_gray.py                       # 默认 100 epochs @1280
  python scripts/train_ball_gray.py --epochs 200 --imgsz 1440
  python scripts/train_ball_gray.py --no-export           # 只训练不导出

前置：模型权重首次自动从 ultralytics 下载（~5.3MB），若网络慢可手动放到
``data/weights/gray/yolo11n-grayscale.pt``。

输出：``runs/detect/ball_gray/weights/best.pt`` 与 ``best.onnx``（动态 batch，1 通道）。
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tabletennis.core.config import project_root

# 灰度预训练权重（优先本地缓存，缺失时 ultralytics 自动下载）
_MODEL = os.path.join(project_root(), "data", "weights", "gray", "yolo11n-grayscale.pt")


def _write_gray_yaml(root: str) -> str:
    """在数据集目录写 ball_gray.yaml（channels: 1 让 dataloader 按灰度加载）。"""
    yaml_path = os.path.join(root, "ball_gray.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {root}\n")
        f.write("train: train.txt\n")
        f.write("val: val.txt\n")
        f.write("channels: 1\n")
        f.write("names:\n  0: ball\n")
    return yaml_path


def _build_split(root: str, val_ratio: float = 0.1, seed: int = 0) -> None:
    """生成 train.txt / val.txt（9:1 切分，与 train_ball.py::build_dataset 一致）。

    augment_ball.py 只产出 images/ + labels/，不写切分文件；训练前必须先生成，
    否则 ultralytics check_det_dataset 报 "images not found, missing path .../val.txt"。
    """
    import numpy as np

    img_dir = os.path.join(root, "images")
    lbl_dir = os.path.join(root, "labels")
    os.makedirs(lbl_dir, exist_ok=True)

    names = sorted(f for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg", ".bmp")))
    if not names:
        raise SystemExit(f"[错误] {img_dir} 里没有图片，先跑 augment_ball.py 生成增强集。")

    # 缺标签的图片补空标签（当背景负样本）
    for name in names:
        lbl = os.path.join(lbl_dir, os.path.splitext(name)[0] + ".txt")
        if not os.path.exists(lbl):
            open(lbl, "w").close()

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(names))
    n_val = max(1, int(len(names) * val_ratio))
    val_idx = set(order[:n_val].tolist())

    with open(os.path.join(root, "train.txt"), "w") as f:
        f.write("\n".join(os.path.join(img_dir, n) for i, n in enumerate(names) if i not in val_idx) + "\n")
    with open(os.path.join(root, "val.txt"), "w") as f:
        f.write("\n".join(os.path.join(img_dir, n) for i, n in enumerate(names) if i in val_idx) + "\n")

    n_pos = sum(
        1 for name in names
        if os.path.getsize(os.path.join(lbl_dir, os.path.splitext(name)[0] + ".txt")) > 0
    )
    print(f"数据集：{len(names)} 张（{n_pos} 有球 / {len(names) - n_pos} 背景负样本），"
          f"train {len(names) - n_val} / val {n_val}")


def main() -> None:
    ap = argparse.ArgumentParser(description="微调 yolo11n-grayscale 单类 ball")
    ap.add_argument("--out", default=None, help="数据集根目录（默认 data/ball_dataset_aug）")
    ap.add_argument("--model", default=_MODEL, help="灰度预训练权重路径")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=1280, help="输入分辨率（训练仅支持正方形整数）")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--device", default="0")
    ap.add_argument("--name", default="ball_gray")
    ap.add_argument("--no-export", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.model):
        raise SystemExit(f"[错误] 未找到灰度预训练权重 {args.model}。\n"
                         "  请手动下载 yolo11n-grayscale.pt 到 data/weights/gray/，或让 ultralytics 自动下载。")

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit("[错误] 未安装 ultralytics/torch。")

    root = args.out or os.path.join(project_root(), "data", "ball_dataset_aug")
    _build_split(root)
    yaml_path = _write_gray_yaml(root)

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
        name=args.name,
        exist_ok=True,
    )

    if not args.no_export:
        best_pt = os.path.join(project_root(), "runs", "detect", args.name, "weights", "best.pt")
        m = YOLO(best_pt)
        onnx_path = m.export(format="onnx", imgsz=args.imgsz, dynamic=True, simplify=True)
        print(f"\n✓ 训练完成。ONNX 模型：{onnx_path}")


if __name__ == "__main__":
    main()
