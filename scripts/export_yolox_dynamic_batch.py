#!/usr/bin/env python3
"""用纯 PyTorch 重建 YOLOX-tiny 并从 mmdet 权重重新导出**动态 batch** ONNX。

背景：mmpose SDK 的 YOLOX ONNX 已烤入 EfficientNMS 且 batch=1 硬编码（12 个 Reshape +
Squeeze/Unsqueeze + NMS），图手术不可行（本文件旧版本试过）。正确路径是从 PyTorch 权重
**重新导出**：按 mmdet 的模块命名重建 YOLOX-tiny（CSPDarknet + YOLOXPAFPN + YOLOXHead），
加载 humanart 的 pth，导出 batch 维动态、不烤 NMS 的模型——输出 (B, 3549, 85)，每行
[reg_xy(2), reg_wh(2), obj.sigmoid(1), cls.sigmoid(80)]（3549 = 52²+26²+13²，stride 8/16/32
按序展平，y-major）。NMS 留给上层 numpy 逐类做（batch 内各图独立）。

导出细节（与 mmdet 严格一致，保证权重加载正确）：
- Focus 切块顺序 top_left/bot_left/top_right/bot_right；CSPLayer cat [main, short]。
- BN: eps=1e-3, momentum=0.03（训练时的 eps 必须一致）；激活 Swish=SiLU。
- **不烤 /255 归一化**：humanart 配置的 DetDataPreprocessor 未配 mean/std，训练时直接吃
  0-255 原始输入（rtmlib 也喂 0-255）。加了 /255 反而会让权重跑出垃圾（实测 0 检出）。
- 固定 416×416 空间尺寸（rtmlib 预处理固定 resize 到该尺寸），仅 batch 维动态，
  便于 TensorRT 建引擎。

用法：
    conda run -n tt python scripts/export_yolox_dynamic_batch.py \
        [--pth PATH] [--out PATH] [--verify]
默认从 ~/.cache/tabletennis/yolox_tiny_humanart.pth 读取、导出到
~/.cache/tabletennis/yolox_tiny_dynamic_416.onnx。
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# mmdet 等价模块重建（命名与 state_dict 完全一致，strict 加载）
# ---------------------------------------------------------------------------


class ConvModule(nn.Module):
    """mmcv ConvModule：conv(bias=False) + BN(eps=1e-3, mom=0.03) + SiLU。"""

    def __init__(self, in_c: int, out_c: int, k: int, s: int = 1, p: int = None):
        super().__init__()
        if p is None:
            p = (k - 1) // 2
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_c, momentum=0.03, eps=0.001)
        self.act = nn.SiLU(inplace=False)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Focus(nn.Module):
    """YOLOX stem：4 个降采样切片 concat 后过一个 3×3 conv。"""

    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv = ConvModule(in_c * 4, out_c, 3, 1, 1)

    def forward(self, x):
        tl = x[..., ::2, ::2]
        tr = x[..., ::2, 1::2]
        bl = x[..., 1::2, ::2]
        br = x[..., 1::2, 1::2]
        x = torch.cat((tl, bl, tr, br), dim=1)  # 与 mmdet 顺序一致
        return self.conv(x)


class DarknetBottleneck(nn.Module):
    def __init__(self, in_c: int, out_c: int, expansion: float = 0.5,
                 add_identity: bool = True):
        super().__init__()
        hidden = int(out_c * expansion)
        self.conv1 = ConvModule(in_c, hidden, 1)
        self.conv2 = ConvModule(hidden, out_c, 3, 1, 1)
        self.add_identity = add_identity and in_c == out_c

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        return out + x if self.add_identity else out


class CSPLayer(nn.Module):
    def __init__(self, in_c: int, out_c: int, num_blocks: int = 1,
                 add_identity: bool = True):
        super().__init__()
        mid = out_c // 2
        self.main_conv = ConvModule(in_c, mid, 1)
        self.short_conv = ConvModule(in_c, mid, 1)
        self.final_conv = ConvModule(2 * mid, out_c, 1)
        self.blocks = nn.Sequential(*[
            DarknetBottleneck(mid, mid, 1.0, add_identity)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        x_short = self.short_conv(x)
        x_main = self.main_conv(x)
        x_main = self.blocks(x_main)
        return self.final_conv(torch.cat((x_main, x_short), dim=1))  # main 在前


class SPPBottleneck(nn.Module):
    def __init__(self, in_c: int, out_c: int, kernel_sizes: Tuple = (5, 9, 13)):
        super().__init__()
        mid = in_c // 2
        self.conv1 = ConvModule(in_c, mid, 1)
        self.poolings = nn.ModuleList([
            nn.MaxPool2d(ks, stride=1, padding=ks // 2) for ks in kernel_sizes
        ])
        self.conv2 = ConvModule(mid * (len(kernel_sizes) + 1), out_c, 1)

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(torch.cat([x] + [p(x) for p in self.poolings], dim=1))


class CSPDarknet(nn.Module):
    """P5 架构：widen_factor 0.375 / deepen_factor 0.33（tiny）。"""

    # (in, out, num_blocks, add_identity, use_spp)，与 mmdet P5 一致
    _ARCH = (
        (64, 128, 3, True, False),
        (128, 256, 9, True, False),
        (256, 512, 9, True, False),
        (512, 1024, 3, False, True),
    )

    def __init__(self, widen: float = 0.375, deepen: float = 0.33):
        super().__init__()
        self.stem = Focus(3, int(64 * widen))
        for i, (in_c, out_c, nb, add_ident, use_spp) in enumerate(self._ARCH):
            in_c, out_c = int(in_c * widen), int(out_c * widen)
            nb = max(round(nb * deepen), 1)
            stage = [ConvModule(in_c, out_c, 3, 2, 1)]
            if use_spp:
                stage.append(SPPBottleneck(out_c, out_c, (5, 9, 13)))
            stage.append(CSPLayer(out_c, out_c, nb, add_ident))
            self.add_module(f"stage{i + 1}", nn.Sequential(*stage))

    def forward(self, x):
        x = self.stem(x)
        outs = []
        for i, name in enumerate(("stage1", "stage2", "stage3", "stage4")):
            x = getattr(self, name)(x)
            if i + 1 in (2, 3, 4):  # out_indices=(2,3,4)
                outs.append(x)
        return tuple(outs)


class YOLOXPAFPN(nn.Module):
    def __init__(self, in_channels: Tuple = (96, 192, 384), out_channels: int = 96,
                 num_csp_blocks: int = 1):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        n = len(in_channels)
        self.reduce_layers = nn.ModuleList()
        self.top_down_blocks = nn.ModuleList()
        for idx in range(n - 1, 0, -1):
            self.reduce_layers.append(
                ConvModule(in_channels[idx], in_channels[idx - 1], 1))
            self.top_down_blocks.append(
                CSPLayer(in_channels[idx - 1] * 2, in_channels[idx - 1],
                         num_csp_blocks, add_identity=False))
        self.downsamples = nn.ModuleList()
        self.bottom_up_blocks = nn.ModuleList()
        for idx in range(n - 1):
            self.downsamples.append(
                ConvModule(in_channels[idx], in_channels[idx], 3, 2, 1))
            self.bottom_up_blocks.append(
                CSPLayer(in_channels[idx] * 2, in_channels[idx + 1],
                         num_csp_blocks, add_identity=False))
        self.out_convs = nn.ModuleList([
            ConvModule(in_channels[i], out_channels, 1) for i in range(n)
        ])

    def forward(self, inputs):
        n = len(inputs)
        inner_outs = [inputs[-1]]
        for idx in range(n - 1, 0, -1):
            feat_high = inner_outs[0]
            feat_low = inputs[idx - 1]
            feat_high = self.reduce_layers[n - 1 - idx](feat_high)
            inner_outs[0] = feat_high
            upsample_feat = self.upsample(feat_high)
            inner_outs.insert(
                0, self.top_down_blocks[n - 1 - idx](
                    torch.cat([upsample_feat, feat_low], 1)))
        outs = [inner_outs[0]]
        for idx in range(n - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            out = self.bottom_up_blocks[idx](
                torch.cat([self.downsamples[idx](feat_low), feat_height], 1))
            outs.append(out)
        for i, conv in enumerate(self.out_convs):
            outs[i] = conv(outs[i])
        return tuple(outs)


class YOLOXHead(nn.Module):
    def __init__(self, num_classes: int = 80, in_channels: int = 96,
                 feat_channels: int = 96, stacked_convs: int = 2,
                 strides: Tuple = (8, 16, 32)):
        super().__init__()
        self.strides = strides
        self.multi_level_cls_convs = nn.ModuleList()
        self.multi_level_reg_convs = nn.ModuleList()
        self.multi_level_conv_cls = nn.ModuleList()
        self.multi_level_conv_reg = nn.ModuleList()
        self.multi_level_conv_obj = nn.ModuleList()
        for _ in strides:
            self.multi_level_cls_convs.append(nn.Sequential(*[
                ConvModule(in_channels if i == 0 else feat_channels,
                           feat_channels, 3, 1, 1) for i in range(stacked_convs)]))
            self.multi_level_reg_convs.append(nn.Sequential(*[
                ConvModule(in_channels if i == 0 else feat_channels,
                           feat_channels, 3, 1, 1) for i in range(stacked_convs)]))
            self.multi_level_conv_cls.append(nn.Conv2d(feat_channels, num_classes, 1))
            self.multi_level_conv_reg.append(nn.Conv2d(feat_channels, 4, 1))
            self.multi_level_conv_obj.append(nn.Conv2d(feat_channels, 1, 1))

    def forward(self, x):
        outs = []
        for i in range(len(self.strides)):
            cls_score = self.multi_level_conv_cls[i](
                self.multi_level_cls_convs[i](x[i]))
            reg_feat = self.multi_level_reg_convs[i](x[i])
            bbox_pred = self.multi_level_conv_reg[i](reg_feat)
            objectness = self.multi_level_conv_obj[i](reg_feat)
            # (B, 4+1+80, H, W) -> (B, HW, 85)，y-major 展平
            out = torch.cat(
                [bbox_pred, objectness.sigmoid(), cls_score.sigmoid()], dim=1)
            out = out.permute(0, 2, 3, 1).reshape(out.shape[0], -1, 85)
            outs.append(out)
        return torch.cat(outs, dim=1)  # (B, 3549, 85)，stride 8/16/32 顺序


class YOLOXDynamic(nn.Module):
    """带 /255 归一化的完整 YOLOX-tiny，输出 (B, 3549, 85)。"""

    def __init__(self):
        super().__init__()
        self.backbone = CSPDarknet()
        self.neck = YOLOXPAFPN()
        self.bbox_head = YOLOXHead()

    def forward(self, x):
        # 注意：humanart 配置的 DetDataPreprocessor 未配 mean/std → 训练时**不做归一化**，
        # 模型直接吃 0-255 原始输入（mmpose 导出的 ONNX 同样不烤 /255，rtmlib 也直接喂 0-255）。
        feats = self.backbone(x)
        feats = self.neck(feats)
        return self.bbox_head(feats)


def _default_pth() -> str:
    return os.path.join(os.path.expanduser("~"), ".cache", "tabletennis",
                        "yolox_tiny_humanart.pth")


def _default_out() -> str:
    return os.path.join(os.path.expanduser("~"), ".cache", "tabletennis",
                        "yolox_tiny_dynamic_416.onnx")


def build_model(pth: str) -> nn.Module:
    ckpt = torch.load(pth, map_location="cpu")
    sd = ckpt["state_dict"]
    model = YOLOXDynamic()
    model.eval()
    missing, unexpected = model.load_state_dict(sd, strict=True)
    assert not missing and not unexpected, (
        f"权重 key 不匹配: missing={missing}, unexpected={unexpected}")
    return model


def export(pth: str, out: str, opset: int = 17) -> None:
    model = build_model(pth)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    dummy = torch.randn(1, 3, 416, 416)
    torch.onnx.export(
        model,
        dummy,
        out,
        input_names=["input"],
        output_names=["dets_raw"],
        dynamic_axes={"input": {0: "batch"}, "dets_raw": {0: "batch"}},
        opset_version=opset,
        do_constant_folding=True,
    )
    print(f"已导出动态 batch YOLOX -> {out} ({os.path.getsize(out)} bytes)")


def verify(pth: str, out: str, batches=(1, 4)) -> None:
    """torch 模型输出 vs 导出的 ONNX 输出逐元素比对（batch=1/4）。"""
    import onnxruntime as ort

    model = build_model(pth)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    sess = ort.InferenceSession(out, sess_options=so,
                                providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    print(f"  ONNX 输入: {inp.name} {inp.shape}  输出: "
          f"{[(o.name, o.shape) for o in sess.get_outputs()]}")
    for b in batches:
        x = (np.random.rand(b, 3, 416, 416) * 255).astype(np.float32)
        with torch.no_grad():
            ref = model(torch.from_numpy(x)).numpy()
        got = sess.run(None, {inp.name: x})[0]
        assert got.shape == ref.shape, f"batch={b} shape 不一致 {got.shape} vs {ref.shape}"
        max_abs = float(np.abs(got - ref).max())
        print(f"  batch={b}: shape={got.shape} max|diff|={max_abs:.6f} "
              f"{'OK' if max_abs < 1e-3 else '⚠ 超差'}")


def main() -> None:
    ap = argparse.ArgumentParser(description="从 mmdet 权重重导出动态 batch YOLOX ONNX")
    ap.add_argument("--pth", default=_default_pth())
    ap.add_argument("--out", default=_default_out())
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.pth):
        sys.exit(f"找不到权重 {args.pth}，先用 wget 下载。")
    export(args.pth, args.out, args.opset)
    if args.verify:
        verify(args.pth, args.out)


if __name__ == "__main__":
    main()
