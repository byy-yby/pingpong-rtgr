#!/usr/bin/env python3
"""把项目内纯 numpy 的 ``SMPL_NEUTRAL.npz`` 转回官方 EasyMocap ``load_model`` 要的 ``.pkl``。

背景：EasyMocap 官方 ``easymocap.smplmodel.load_model`` 用
``SMPLlayer(join(model_path, 'smpl'), ...)``，而 ``load_bodydata`` 只认
``pickle.load``（.pkl），不认 .npz。官方 SMPL 下载的 .pkl 是 chumpy 序列化，
numpy 2.x 下没法直接加载；本项目已有的 ``SMPL_NEUTRAL.npz`` 是从官方 pkl 转出的
纯 numpy，所有字段（f/J_regressor/v_template/weights/posedirs/shapedirs/kintree_table）
与官方结构一致。本脚本把它写回一个**纯 numpy 的 .pkl**（无 chumpy），让官方
``SMPLlayer`` 能直接 ``pickle.load``。

用法：
  python scripts/npz_to_smpl_pkl.py                       # 默认 -> data/bodymodels/smpl/SMPL_NEUTRAL.pkl
  python scripts/npz_to_smpl_pkl.py -i xxx.npz -o out.pkl

同时把官方 ``J_regressor_body25.npy`` 拷到 ``model_path`` 根下
（``load_model(skel_type='body25')`` 需要 ``<model_path>/J_regressor_body25.npy``）。
官方文件在 ``$EASYMOCAP_ROOT/data/smplx/J_regressor_body25.npy``，用
``EASYMOCAP_ROOT`` 环境变量可覆盖。
"""
from __future__ import annotations

import argparse
import os
import pickle
import shutil
import sys

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-i", "--input", default="data/bodymodels/SMPL_NEUTRAL.npz")
    ap.add_argument("-o", "--output", default="data/bodymodels/smpl/SMPL_NEUTRAL.pkl")
    ap.add_argument("-m", "--model-path", default="data/bodymodels",
                    help="load_model 的 model_path（= pkl 上级的上级，放 J_regressor_body25.npy）")
    args = ap.parse_args()

    d = np.load(args.input)
    keys = ["f", "J_regressor", "v_template", "weights", "posedirs",
            "shapedirs", "kintree_table"]
    out = {k: np.ascontiguousarray(d[k]) for k in keys}
    for k, v in out.items():
        print(f"  {k}: shape={v.shape} dtype={v.dtype}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(out, f, protocol=4)
    print(f"[npz->pkl] 写入 {args.output} ({os.path.getsize(args.output)} 字节)")

    # J_regressor_body25.npy（body25 顶点->关节回归器，load_model 依赖）
    em_root = os.environ.get("EASYMOCAP_ROOT", "/home/yby/projects/EasyMocap")
    src_reg = os.path.join(em_root, "data", "smplx", "J_regressor_body25.npy")
    dst_reg = os.path.join(args.model_path, "J_regressor_body25.npy")
    if os.path.exists(src_reg):
        os.makedirs(args.model_path, exist_ok=True)
        shutil.copyfile(src_reg, dst_reg)
        print(f"[npz->pkl] 拷贝 {src_reg} -> {dst_reg}")
    else:
        print(f"[npz->pkl] 警告：找不到 {src_reg}（J_regressor_body25.npy），"
              f"body25 关节将不可用", file=sys.stderr)


if __name__ == "__main__":
    main()
