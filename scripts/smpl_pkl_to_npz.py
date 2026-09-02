#!/usr/bin/env python3
"""把官方 SMPL .pkl（chumpy 序列化）转成 .npz（纯 numpy，免 chumpy）。

SMPL 官方下载（smpl.is.tue.mpg.de → SMPL for Python → SMPL_python_v.1.1.0.zip）
给的是 chumpy 格式的 ``.pkl``，在 numpy 2.x 下加载需要已停维护的 chumpy。这里用
最小 shim 绕过 chumpy，直接把模型各字段抽成 numpy 数组存成 ``.npz``，供
``reconstruction/easymocap.py``（以及 smplx 等）直接使用。

用法：
  python scripts/smpl_pkl_to_npz.py basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl \
      -o data/bodymodels/SMPL_NEUTRAL.npz
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import types

import numpy as np


class _Ch:
    """chumpy.Ch 的最小替身：只负责把 pickle 反序列化的 state 还原成数组。"""

    def __setstate__(self, d: dict) -> None:
        self.__dict__ = d

    def __array__(self, dtype=None):
        return np.asarray(self._array(), dtype=dtype)

    def _array(self):
        d = self.__dict__
        # r 是计算缓存（序列化时通常被丢弃），x 是输入项；SMPL 里二者都是纯 numpy
        for key in ("r", "x"):
            if key in d:
                return d[key]
        raise KeyError(f"chumpy.Ch state 缺少 r/x：{list(d.keys())}")


def _inject_chumpy_shim() -> None:
    """往 sys.modules 注入假的 chumpy / chumpy.ch，让 pickle 能反序列化 Ch 对象。"""
    chumpy = types.ModuleType("chumpy")
    chumpy.Ch = _Ch
    chumpy_ch = types.ModuleType("chumpy.ch")
    chumpy_ch.Ch = _Ch
    sys.modules["chumpy"] = chumpy
    sys.modules["chumpy.ch"] = chumpy_ch


def _to_np(v):
    if isinstance(v, _Ch):
        return _to_np(v._array())
    if isinstance(v, (str, bytes, bool, int, float)):
        return v
    import scipy.sparse as sp
    if sp.issparse(v):
        return np.asarray(v.toarray())
    return np.asarray(v)


def convert(pkl_path: str, out_path: str) -> None:
    _inject_chumpy_shim()
    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    print("原始 .pkl 顶层字段：", list(data.keys()))
    out = {}
    for k, v in data.items():
        arr = _to_np(v)
        if isinstance(arr, np.ndarray):
            print(f"  {k}: shape={arr.shape} dtype={arr.dtype}")
            out[k] = arr
        else:
            print(f"  {k}: {arr!r} (跳过非数组字段)")

    required = ["f", "J_regressor", "kintree_table", "shapedirs", "weights",
                "posedirs", "v_template"]
    missing = [k for k in required if k not in out]
    if missing:
        raise SystemExit(f"缺少必要字段 {missing}，pkl 内容异常")

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.savez(out_path, **out)
    print(f"✓ 已保存 {out_path}（{os.path.getsize(out_path) / 1e6:.1f} MB）")


def main() -> None:
    ap = argparse.ArgumentParser(description="SMPL .pkl -> .npz（免 chumpy）")
    ap.add_argument("pkl_path")
    ap.add_argument("-o", "--out", default=None, help="输出 .npz 路径（默认同目录同名）")
    args = ap.parse_args()
    out = args.out or os.path.splitext(args.pkl_path)[0] + ".npz"
    convert(args.pkl_path, out)


if __name__ == "__main__":
    main()
