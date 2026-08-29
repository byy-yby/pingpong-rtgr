"""骨架定义：关键点名称、骨骼连线。

目前内置 COCO-17（RTMPose 默认输出）。后续要上 Halpe-26 / WholeBody-133
时在此处新增字典即可，检测器与可视化都从这里取定义，保证连线/命名一致。
"""
from __future__ import annotations

from typing import Dict, List, Tuple

# COCO-17 关键点索引 -> 名称
COCO17_NAMES: List[str] = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

# COCO-17 骨骼连线：(起, 止)，画在图像上
COCO17_EDGES: List[Tuple[int, int]] = [
    (0, 1), (0, 2), (1, 3), (2, 4),            # 头
    (5, 6),                                    # 肩
    (5, 7), (7, 9), (6, 8), (8, 10),          # 手臂
    (5, 11), (6, 12), (11, 12),                # 躯干 + 髋
    (11, 13), (13, 15), (12, 14), (14, 16),    # 腿
]

# Halpe-26 关键点（AlphaPose / mmpose 顺序）：COCO-17 + 头顶/颈/骨盆 + 6 个脚点
HALPE26_NAMES: List[str] = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
    "head", "neck", "hip",
    "left_big_toe", "right_big_toe", "left_small_toe", "right_small_toe",
    "left_heel", "right_heel",
]

# Halpe-26 骨骼连线（与 mmpose/rtmlib 的 halpe26 定义一致）
HALPE26_EDGES: List[Tuple[int, int]] = [
    (15, 13), (13, 11), (11, 19),              # 左腿 → 髋(骨盆)
    (16, 14), (14, 12), (12, 19),              # 右腿 → 髋(骨盆)
    (17, 18), (18, 19),                        # 头顶 → 颈 → 骨盆（脊柱）
    (18, 5), (5, 7), (7, 9),                   # 左臂
    (18, 6), (6, 8), (8, 10),                  # 右臂
    (1, 2), (0, 1), (0, 2), (1, 3), (2, 4),    # 头
    (3, 5), (4, 6),                            # 耳 → 肩
    (15, 20), (15, 22), (15, 24),              # 左脚趾/脚跟
    (16, 21), (16, 23), (16, 25),              # 右脚趾/脚跟
]

SKELETONS: Dict[str, Dict] = {
    "coco17": {
        "names": COCO17_NAMES,
        "edges": COCO17_EDGES,
    },
    "halpe26": {
        "names": HALPE26_NAMES,
        "edges": HALPE26_EDGES,
    },
}


def get_skeleton(name: str) -> Dict:
    """按名字取骨架定义；未知名字回退到 coco17。"""
    return SKELETONS.get(name, SKELETONS["coco17"])
