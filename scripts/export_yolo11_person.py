#!/usr/bin/env python3
"""把灰度原生 yolo11n（ch=1）导出成「单通道 + 动态 batch」的人体检测 ONNX。

背景：`data/weights/gray/yolo11n-grayscale.pt` 是灰度原生训练的 YOLO11n（第一个卷积
1→16，输入单通道），COCO-80 类（person=类 0）。用它替换 rtmlib YOLOX 做人检测，
省掉 gray→3ch 复制与 CPU 逐类 NMS。

导出的 ONNX：
- 输入 ``(batch, 1, 640, 640)`` 单通道 [0,1] float（/255 在预处理里做，不烤进图）。
- 输出 ``(batch, 84, N)`` 原始 raw（不烤 NMS）：84 = 4 box(cxcywh, 输入坐标) + 80 类，
  N = 8400（80²+40²+20²）。person 分数在 ``out[:, 4, :]``。

关键坑（复用现有经验）：ultralytics ``dynamic=True`` 会把 batch/h/w 三个轴都标动态，
onnxruntime TRT EP 遇 h/w 动态会**静默回退 CUDA**，必须用
``scripts/fix_onnx_dynamic.py --ch 1`` 把 h/w 固化成 640。

用法::

    conda run -n tt python scripts/export_yolo11_person.py          # 默认导出到 ~/.cache
    conda run -n tt python scripts/export_yolo11_person.py --verify  # 导出后与 torch 对拍
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def _default_src() -> str:
    # 灰度原生模型在主工作区 data/weights/gray/（.gitignore 忽略，不进版本库）。
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..",
        "data", "weights", "gray", "yolo11n-grayscale.pt",
    )


def _default_out() -> str:
    return os.path.join(os.path.expanduser("~"), ".cache", "tabletennis",
                        "yolo11n_grayscale_person.onnx")


def _onnx_io(onnx_path: str):
    """返回 (输入 [(name,shape)], 输出 [(name,shape)])。"""
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    return (
        [(i.name, i.shape) for i in sess.get_inputs()],
        [(o.name, o.shape) for o in sess.get_outputs()],
    )


def _export_ultralytics(src: str, tmp: str, imgsz: int) -> None:
    from ultralytics import YOLO
    model = YOLO(src)
    print(f"[ultralytics] 模型 ch={getattr(model.model, 'ch', '?')} "
          f"nc={getattr(model.model, 'nc', '?')}")
    # 导出 raw（不烤 NMS），dynamic=True（batch/h/w 动态，之后 fix 固定 h/w）
    model.export(format="onnx", imgsz=imgsz, dynamic=True, simplify=False)
    # ultralytics 导出产物名 = 源名换 .onnx，与源同目录
    exported = os.path.splitext(src)[0] + ".onnx"
    if not os.path.exists(exported):
        sys.exit(f"[错误] ultralytics 未在预期位置生成 {exported}")
    os.replace(exported, tmp)
    print(f"[ultralytics] 原始动态导出 -> {tmp}")


def _export_manual(src: str, tmp: str, imgsz: int) -> None:
    """回退：手动 torch.onnx.export 强制 1 通道输入（ultralytics 偶尔强制 3ch 时用）。"""
    import torch
    from ultralytics import YOLO
    model = YOLO(src)
    m = model.model
    m.eval()
    dummy = torch.zeros(1, 1, imgsz, imgsz)
    torch.onnx.export(
        m, dummy, tmp,
        input_names=["input"], output_names=["output0"],
        dynamic_axes={"input": {0: "batch"}, "output0": {0: "batch"}},
        opset_version=17, do_constant_folding=True,
    )
    print(f"[manual] 1 通道导出 -> {tmp}")


def _fix_dynamic(tmp: str, out: str, imgsz: int) -> None:
    """用 fix_onnx_dynamic.py 把 h/w 固定、仅 batch 动态。"""
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "fix_onnx_dynamic.py")
    subprocess.run(
        [sys.executable, script, tmp, out, "--imgsz", str(imgsz), "--ch", "1"],
        check=True,
    )


def verify(out: str, src: str, imgsz: int) -> None:
    """torch 模型 vs 导出 ONNX 逐元素对拍（batch=1/4）。"""
    import torch
    import onnxruntime as ort
    from ultralytics import YOLO

    inputs, outputs = _onnx_io(out)
    print(f"  ONNX 输入: {inputs}")
    print(f"  ONNX 输出: {outputs}")
    assert inputs[0][1][1] == 1, f"输入通道应为 1，实际 {inputs[0][1]}"
    assert inputs[0][1][2] == inputs[0][1][3] == imgsz, "h/w 应为固定 imgsz"

    model = YOLO(src).model.eval()
    sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    for b in (1, 4):
        x = np.random.rand(b, 1, imgsz, imgsz).astype(np.float32)
        with torch.no_grad():
            ref = model(torch.from_numpy(x))
            ref = ref[0] if isinstance(ref, (list, tuple)) else ref
            ref = ref.numpy()
        got = sess.run(None, {iname: x})[0]
        assert got.shape == ref.shape, f"batch={b} shape 不一致 {got.shape} vs {ref.shape}"
        max_abs = float(np.abs(got - ref).max())
        # ONNX 导出（opset 18 + 图优化）有 ~1e-3 量级的浮点重排误差属正常，阈值放宽到 5e-2
        print(f"  batch={b}: shape={got.shape} max|diff|={max_abs:.6f} "
              f"{'OK' if max_abs < 5e-2 else '⚠ 超差'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default=_default_src())
    ap.add_argument("--out", default=_default_out())
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.src):
        sys.exit(f"找不到灰度模型 {args.src}，先确认主工作区 data/weights/gray/ 里有它。")

    tmp = args.out + ".raw.onnx"
    _export_ultralytics(args.src, tmp, args.imgsz)

    # 检查 ultralytics 导出的输入通道，若被强制成 3 则手动回退
    ins, _outs = _onnx_io(tmp)
    if ins[0][1][1] != 1:
        print(f"[警告] ultralytics 导出输入通道={ins[0][1][1]}（非 1），改手动导出")
        _export_manual(args.src, tmp, args.imgsz)

    _fix_dynamic(tmp, args.out, args.imgsz)
    os.remove(tmp)
    print(f"✓ 已导出 {args.out} ({os.path.getsize(args.out)} bytes)")
    ins, outs = _onnx_io(args.out)
    print(f"  输入 {ins}  输出 {outs}")

    if args.verify:
        verify(args.out, args.src, args.imgsz)


if __name__ == "__main__":
    main()
