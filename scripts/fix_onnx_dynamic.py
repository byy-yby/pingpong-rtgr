#!/usr/bin/env python3
"""把 ultralytics ``dynamic=True`` 导出的 ONNX 改成「仅 batch 动态、h/w 固定」，
让 onnxruntime TensorrtExecutionProvider 能建引擎。

为什么必须做：ultralytics ``dynamic=True`` 会把 batch、height、width **三个轴都**标成
动态。onnxruntime TRT EP 遇到含 h/w 的动态输入会**静默回退 CUDA**（不建引擎、不报错），
TRT 就白配了。h/w 固定成 ``--imgsz`` 后 TRT EP 才按遇到的 batch 尺寸（1、4 等）各建一个
引擎并缓存到 ``~/.cache/tabletennis/trt_engines``。

用法::

    python scripts/fix_onnx_dynamic.py best_raw.onnx best.onnx --imgsz 1280
"""
import argparse

import onnx


def _shape_str(node) -> str:
    dims = []
    for d in node.type.tensor_type.shape.dim:
        dims.append(d.dim_param if d.dim_param else str(d.dim_value))
    return "[" + ", ".join(dims) + "]"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", help="ultralytics dynamic=True 导出的 onnx")
    ap.add_argument("dst", help="输出 onnx（仅 batch 动态）")
    ap.add_argument("--imgsz", type=int, default=1280, help="固定输入 H/W")
    args = ap.parse_args()

    m = onnx.load(args.src)
    print("原始输入:", _shape_str(m.graph.input[0]))

    inp = m.graph.input[0]
    tt = inp.type.tensor_type
    fixed = {1: 3, 2: args.imgsz, 3: args.imgsz}
    for i, dim in enumerate(tt.shape.dim):
        if i == 0:
            dim.ClearField("dim_value")
            dim.dim_param = "batch"
        elif i in fixed:
            dim.ClearField("dim_param")
            dim.dim_value = fixed[i]

    for o in m.graph.output:  # 输出 batch 动态，其余保持导出的固定值
        tt = o.type.tensor_type
        for i, dim in enumerate(tt.shape.dim):
            if i == 0:
                dim.ClearField("dim_value")
                dim.dim_param = "batch"

    print("修改后输入:", _shape_str(m.graph.input[0]))
    for o in m.graph.output:
        print("输出:", _shape_str(o))
    onnx.save(m, args.dst)
    print(f"✓ 已保存 {args.dst}（TRT EP 现在可以建引擎了）")


if __name__ == "__main__":
    main()
