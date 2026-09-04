"""ViTPose 检测器：coco_25→halpe26 重排映射的一致性纯函数单测。

核心保证：ViTPosePoseDetector 把 easy_ViTPose ``coco_25``（25 点，顺序见
vitpose_pose 模块 docstring）重排成下游 halpe26 布局。只要
``coco25 → halpe26 → body25`` 的复合映射与 ``COCO25_TO_BODY25`` 锚点表一致，
下游三角化 / SMPL 拟合 / pose2d.json 就无需任何改动。
"""
import numpy as np

from tabletennis.reconstruction.easymocap import HALPE26_TO_BODY25
from tabletennis.vision.pose.vitpose_pose import (
    COCO25_TO_BODY25,
    COCO25_TO_HALPE26,
    coco25_to_halpe26,
)

# coco_25 恰好缺 halpe26 的 17（头顶，body25 无对应关节，本就该留空）
_HEAD_TOP_HALPE = 17


def test_reorder_covers_all_non_head_slots_exactly_once():
    """coco_25 的 25 个关节恰好映射到 halpe26 除 17 外的全部 25 个槽（无重无漏）。"""
    src = sorted(c for c, _ in COCO25_TO_HALPE26)
    dst = sorted(h for _, h in COCO25_TO_HALPE26)
    assert src == list(range(25)), f"coco 源索引应恰为 0..24，实际缺口：{src}"
    expected_dst = [h for h in range(26) if h != _HEAD_TOP_HALPE]
    assert dst == expected_dst, f"halpe 目标应缺 17，实际：{dst}"


def test_anchors_match_downstream_body25():
    """复合 coco25→halpe26→body25 与 COCO25_TO_BODY25 逐点一致。"""
    halpe_to_body25 = dict(HALPE26_TO_BODY25)
    for c, h in COCO25_TO_HALPE26:
        assert h in halpe_to_body25, f"halpe{h} 下游未消费（应只漏头顶 17）"
    for c, h in COCO25_TO_HALPE26:
        assert COCO25_TO_BODY25[c] == halpe_to_body25[h], (
            f"coco{c}->halpe{h}->body25{halpe_to_body25[h]} "
            f"≠ 锚点 body25{COCO25_TO_BODY25[c]}")


def test_coco25_to_halpe26_copy_and_head_zero():
    """数值拷贝正确、头顶槽(h17) 恒为 0、halpe 其它行不受污染。"""
    rng = np.random.RandomState(0)
    k25 = rng.rand(25, 3).astype(np.float32)
    k25[:, 0] *= 1440
    k25[:, 1] *= 1080
    out = coco25_to_halpe26(k25)
    assert out.shape == (26, 3)
    assert np.all(out[_HEAD_TOP_HALPE] == 0.0), "头顶槽应为 (0,0,0)"
    for c, h in COCO25_TO_HALPE26:
        np.testing.assert_allclose(out[h], k25[c], err_msg=f"coco{c}->halpe{h}")
