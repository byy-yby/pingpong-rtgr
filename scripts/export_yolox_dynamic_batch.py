#!/usr/bin/env python3
"""把固定 batch=1 的 YOLOX ONNX 转成动态 batch，实现多相机一次 forward。

⚠️ 状态：**不完整 / 不可行**。mmpose SDK 导出的 YOLOX（已烤入 EfficientNMS）在整张图里
有**大量** batch=1 硬编码——不止 head 里 12 个 Reshape，还有 Squeeze/Unsqueeze/Gather/
NonMaxSuppression 等，修一个冒一个（batch=1 输出正确、batch=2 在 Squeeze_556 崩）。
图手术不是可行路径；正确做法是**用 mmdet/mmpose 从 PyTorch 重新导出动态 batch**。
本脚本保留作为那次尝试的记录与起点。

用法：
    conda run -n tt python scripts/export_yolox_dynamic_batch.py \
        <src.onnx> <dst.onnx> [--verify]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def make_dynamic_batch(model: onnx.ModelProto) -> onnx.ModelProto:
    g = model.graph
    inits = {i.name: i for i in g.initializer}

    # 1) 输入 batch 维 -> 动态
    for inp in g.input:
        d0 = inp.type.tensor_type.shape.dim[0]
        if d0.dim_value == 1:
            d0.ClearField("dim_value")
            d0.dim_param = "batch"

    # 2) 输出 batch 维 -> 动态
    for out in g.output:
        d0 = out.type.tensor_type.shape.dim[0]
        if d0.dim_value == 1:
            d0.ClearField("dim_value")
            d0.dim_param = "batch"

    # 3) 收集硬编码 [1, ...] 的 Reshape/Expand/Tile 节点
    targets = []
    for n in g.node:
        if n.op_type not in ("Reshape", "Expand", "Tile"):
            continue
        shape_in = n.input[1] if len(n.input) > 1 else None
        if shape_in and shape_in in inits:
            arr = numpy_helper.to_array(inits[shape_in])
            flat = arr.reshape(-1)
            # 只处理「batch=1, ...」的多维 reshape；[1] 这种标量 reshape 不是 batch 维，跳过
            if flat.size > 1 and flat[0] == 1:
                targets.append((n, shape_in, flat.astype(np.int64)))

    # 4) 对每个节点插入 Shape/Gather/Concat，把 [1,...] 换成 [batch,...]
    insert_before: dict = {}  # reshape name -> [新节点...]
    for n, shape_in, flat in targets:
        rest = flat[1:]  # 去掉 batch=1，其余 dims 不变
        s_name = f"{n.name}_shape"
        g_name = f"{n.name}_batch"
        c_name = f"{n.name}_dynshape"
        idx_name = f"{n.name}_idx0"
        rest_name = f"{n.name}_rest"

        nodes = [helper.make_node("Shape", [n.input[0]], [s_name])]
        g.initializer.append(
            numpy_helper.from_array(np.array([0], dtype=np.int64), name=idx_name)
        )
        nodes.append(helper.make_node("Gather", [s_name, idx_name], [g_name], axis=0))
        if rest.size:
            g.initializer.append(numpy_helper.from_array(rest, name=rest_name))
            nodes.append(
                helper.make_node("Concat", [g_name, rest_name], [c_name], axis=0)
            )
        else:  # 无 rest，shape 就是 [batch]
            c_name = g_name

        # 改原节点的 shape 输入指向动态 shape
        for nn in g.node:
            if nn.name == n.name:
                nn.input[1] = c_name
                break
        insert_before[n.name] = nodes

    # 5) 把 Shape/Gather/Concat 插到各自 Reshape 之前，保证拓扑序
    original_nodes = list(g.node)
    out_nodes = []
    for node in original_nodes:
        if node.name in insert_before:
            out_nodes.extend(insert_before[node.name])
        out_nodes.append(node)
    del g.node[:]
    g.node.extend(out_nodes)

    return model


def main() -> None:
    ap = argparse.ArgumentParser(description="YOLOX ONNX 固定 batch=1 -> 动态 batch")
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--verify", action="store_true", help="batch=1 时对比转换前后输出")
    args = ap.parse_args()

    model = onnx.load(args.src)
    model = make_dynamic_batch(model)
    # 注意：不做 onnx.shape_inference —— 它会按 batch=1 误标 Shape 节点的输出，
    # 导致 onnxruntime 把 Shape->Gather->Concat 链常量折叠成 [1,...]，batch 动态化失效。
    onnx.save(model, args.dst)
    print(f"已导出动态 batch 模型 -> {args.dst} ({os.path.getsize(args.dst)} bytes)")

    if args.verify:
        _verify(args.src, args.dst)


def _verify(src: str, dst: str) -> None:
    import onnxruntime as ort

    x = np.random.rand(1, 3, 640, 640).astype(np.float32)
    so = ort.SessionOptions()
    s0 = ort.InferenceSession(src, sess_options=so, providers=["CPUExecutionProvider"])
    s1 = ort.InferenceSession(dst, sess_options=so, providers=["CPUExecutionProvider"])
    o0 = s0.run(None, {s0.get_inputs()[0].name: x})
    o1 = s1.run(None, {s1.get_inputs()[0].name: x})
    for a, b in zip(o0, o1):
        if a.shape == b.shape:
            print(f"  batch=1 输出一致: shape={a.shape}, max|diff|={np.abs(a - b).max():.6f}")
        else:
            print(f"  ⚠ 输出 shape 不一致: {a.shape} vs {b.shape}")
    # 再试 batch=2
    x2 = np.random.rand(2, 3, 640, 640).astype(np.float32)
    try:
        o2 = s1.run(None, {s1.get_inputs()[0].name: x2})
        print(f"  batch=2 可运行: {[o.shape for o in o2]}")
    except Exception as e:  # noqa: BLE001
        print(f"  batch=2 运行失败: {type(e).__name__}: {str(e)[:200]}")


if __name__ == "__main__":
    main()
