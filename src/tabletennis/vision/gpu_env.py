"""GPU 推理环境准备：加载 pip 安装的 nvidia CUDA / cuDNN 运行库。

本机没有系统级 CUDA toolkit（无 ``/usr/local/cuda``、无 ``nvcc``），而
onnxruntime-gpu / TensorRT 靠 ``dlopen`` 按 soname（如 ``libcublasLt.so.13``）查找
CUDA 运行库。这些库由 pip 的 ``nvidia-*-cuXX`` 包装在
``site-packages/nvidia/<pkg>/lib/`` 下（带版本后缀，如 ``libcudart.so.13``），
默认不在动态链接器搜索路径里。

本模块在 ``import onnxruntime`` 之前用 ``ctypes`` 按绝对路径预加载这些 ``.so``
（RTLD_GLOBAL），使后续 onnxruntime 的 ``dlopen("libcublasLt.so.13")`` 命中已
加载的库，从而**无需手工设置 LD_LIBRARY_PATH**。仅在启用 GPU 时显式调用
:func:`preload_nvidia_libs`。
"""
from __future__ import annotations

import ctypes
import glob
import os
import site
from typing import List


def nvidia_lib_dirs() -> List[str]:
    """返回所有 ``site-packages/nvidia/<pkg>/lib`` 目录（pip 装的 CUDA/cuDNN）。"""
    roots = list(site.getsitepackages())
    try:
        roots.append(site.getusersitepackages())
    except Exception:  # noqa: BLE001
        pass
    dirs: List[str] = []
    for r in roots:
        if not r or not os.path.isdir(r):
            continue
        for d in sorted(glob.glob(os.path.join(r, "nvidia", "*", "lib"))):
            dirs.append(d)
    return dirs


def preload_nvidia_libs() -> List[str]:
    """预加载 nvidia wheel 里的 CUDA / cuDNN 运行库，返回加载成功的 .so 文件名列表。

    nvidia wheel 里的库是带版本后缀的真实文件（``libcudart.so.13`` 等），没有
    无后缀基名符号链接，所以匹配 ``*.so`` 与 ``*.so.*`` 两种形式。
    """
    loaded: List[str] = []
    for libdir in nvidia_lib_dirs():
        for pat in (os.path.join(libdir, "*.so"), os.path.join(libdir, "*.so.*")):
            for so in sorted(glob.glob(pat)):
                try:
                    ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
                    loaded.append(os.path.basename(so))
                except OSError:
                    pass
    # 兜底：把 nvidia lib 目录加进 LD_LIBRARY_PATH（对按 soname 的 dlopen 友好）
    dirs = nvidia_lib_dirs()
    if dirs:
        extra = os.pathsep.join(dirs)
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = (
            extra + (os.pathsep + existing if existing else "")
        )
    return loaded
