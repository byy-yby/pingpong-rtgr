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

实现复用 ``tabletennis.reconstruction.easymocap.convert_npz_to_pkl``（运行时
自动生成与 CLI 走同一份代码，不会漂移）。
"""
from __future__ import annotations

import argparse
import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_THIS), "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from tabletennis.reconstruction.easymocap import (  # noqa: E402
    DEFAULT_EASYMOCAP_ROOT,
    convert_npz_to_pkl,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-i", "--input", default="data/bodymodels/SMPL_NEUTRAL.npz")
    ap.add_argument("-o", "--output", default="data/bodymodels/smpl/SMPL_NEUTRAL.pkl")
    ap.add_argument("-m", "--model-path", default="data/bodymodels",
                    help="load_model 的 model_path（= pkl 上级的上级，放 J_regressor_body25.npy）")
    args = ap.parse_args()

    em_root = os.environ.get("EASYMOCAP_ROOT", DEFAULT_EASYMOCAP_ROOT)
    convert_npz_to_pkl(args.input, args.model_path, em_root)
    print(f"[npz->pkl] 写入 {args.output} "
          f"({os.path.getsize(args.output)} 字节) + J_regressor_body25.npy")


if __name__ == "__main__":
    main()
