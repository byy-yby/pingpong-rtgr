#!/usr/bin/env python3
"""乒乓球检测数据标注工具（OpenCV 拖框，YOLO 格式）。

配合 ``capture_ball.py`` 使用：逐张显示图片，鼠标左键拖框标球，存成 YOLO 标签
（``0 cx cy w h`` 归一化）。已有标签（如经典检测器预标注的）会自动载入，方便校正。

用法：
  conda activate tt
  python scripts/label_ball.py                          # 标注 data/ball_dataset/images
  python scripts/label_ball.py --out data/ball_dataset  # 指定数据集根目录

操作：
  左键拖拽 = 画/改框      空格 = 保存并下一张
  d = 删除当前框           n/p = 上一张/下一张（不保存）
  q / ESC = 退出
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import cv2

from tabletennis.core.config import project_root
from tabletennis.visualization.overlay2d import gray_to_bgr

WIN = "label-ball"


def box_to_yolo(x0, y0, x1, y1, W: int, H: int) -> str:
    cx = (x0 + x1) / 2.0 / W
    cy = (y0 + y1) / 2.0 / H
    w = abs(x1 - x0) / W
    h = abs(y1 - y0) / H
    return f"0 {min(max(cx,0),1):.6f} {min(max(cy,0),1):.6f} {min(max(w,0),1):.6f} {min(max(h,0),1):.6f}"


def load_label(path: str) -> Optional[tuple]:
    """读 YOLO 标签返回 ``(x0,y0,x1,y1)`` 像素框；无框返回 None。"""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        line = f.readline().strip()
    if not line:
        return None
    parts = line.split()
    if len(parts) < 5:
        return None
    return tuple(float(p) for p in parts[1:5])  # cx, cy, w, h（归一化）


def main() -> None:
    ap = argparse.ArgumentParser(description="乒乓球检测数据标注")
    ap.add_argument("--out", default=None, help="数据集根目录（默认 data/ball_dataset）")
    args = ap.parse_args()

    root = args.out or os.path.join(project_root(), "data", "ball_dataset")
    img_dir = os.path.join(root, "images")
    lbl_dir = os.path.join(root, "labels")
    os.makedirs(lbl_dir, exist_ok=True)

    names = sorted(f for f in os.listdir(img_dir) if f.lower().endswith((".png", ".jpg", ".bmp")))
    if not names:
        print(f"[错误] {img_dir} 里没有图片，先跑 capture_ball.py 采集。")
        sys.exit(1)

    idx = 0
    box: Optional[tuple] = None   # 归一化 (cx, cy, w, h)
    mode = None                   # 当前鼠标动作: "draw"(画新框) / "move"(拖动已有框)
    drag_anchor = None            # draw 模式: 按压起始点 (x0, y0) 像素
    grab_off = None               # move 模式: 点击点相对框左上角的偏移 (ox, oy)
    img = None
    W = H = 0

    def load(idx: int) -> None:
        nonlocal box, img, W, H
        name = names[idx]
        img = cv2.imread(os.path.join(img_dir, name), cv2.IMREAD_GRAYSCALE)
        H, W = img.shape[:2]
        norm = load_label(os.path.join(lbl_dir, os.path.splitext(name)[0] + ".txt"))
        box = norm  # (cx, cy, w, h) 归一化

    def norm_to_px(norm) -> tuple:
        cx, cy, w, h = norm
        return (int((cx - w / 2) * W), int((cy - h / 2) * H),
                int((cx + w / 2) * W), int((cy + h / 2) * H))

    def on_mouse(event, x, y, flags, param) -> None:
        nonlocal box, mode, drag_anchor, grab_off
        if event == cv2.EVENT_LBUTTONDOWN:
            # 点在已有框内 → 进入 move；否则 → 进入 draw
            inside = False
            if box is not None:
                bx0, by0, bx1, by1 = norm_to_px(box)
                inside = bx0 <= x <= bx1 and by0 <= y <= by1
            if inside:
                mode = "move"
                grab_off = (x - bx0, y - by0)
            else:
                mode = "draw"
                drag_anchor = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            if mode == "draw" and drag_anchor is not None:
                x0, y0 = drag_anchor
                x1, y1 = x, y
                tlx, tly = min(x0, x1), min(y0, y1)
                brx, bry = max(x0, x1), max(y0, y1)
                cx = (tlx + brx) / 2.0 / W
                cy = (tly + bry) / 2.0 / H
                w = (brx - tlx) / W
                h = (bry - tly) / H
                box = (cx, cy, w, h)
            elif mode == "move" and grab_off is not None:
                ox, oy = grab_off
                cx = (x - ox + box[2] * W / 2.0) / W
                cy = (y - oy + box[3] * H / 2.0) / H
                box = (cx, cy, box[2], box[3])
        elif event == cv2.EVENT_LBUTTONUP:
            mode = None
            drag_anchor = None
            grab_off = None

    def save() -> None:
        nonlocal box
        name = names[idx]
        path = os.path.join(lbl_dir, os.path.splitext(name)[0] + ".txt")
        line = box_to_yolo(*norm_to_px(box), W, H) if box is not None else ""
        with open(path, "w") as f:
            f.write(line + ("\n" if line else ""))
        print(f"  保存 {name}: {'有球' if line else '无球'}")

    load(0)
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WIN, on_mouse)

    while True:
        disp = gray_to_bgr(img).copy()
        if box is not None:
            x0, y0, x1, y1 = norm_to_px(box)
            cv2.rectangle(disp, (x0, y0), (x1, y1), (0, 0, 255), 2)
        cv2.putText(disp, f"{idx+1}/{len(names)}  {names[idx]}", (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(disp, "拖框标球 / 点住框内拖动可移动 | 空格=保存+下一张 | d=删框 | n/p=翻页 | q=退出",
                    (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.imshow(WIN, disp)

        key = cv2.waitKey(10) & 0xFF  # 轮询而非阻塞：拖动时每 10ms 重绘一次，实时看到框
        if key in (27, ord("q")):
            break
        elif key == ord(" "):
            save()
            idx = min(idx + 1, len(names) - 1)
            load(idx)
        elif key == ord("d"):
            box = None
        elif key == ord("n"):
            idx = min(idx + 1, len(names) - 1)
            load(idx)
        elif key == ord("p"):
            idx = max(idx - 1, 0)
            load(idx)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
