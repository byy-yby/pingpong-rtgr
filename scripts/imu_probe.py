#!/usr/bin/env python3
"""IMU BLE 原始数据探针：直接观察 WT9011DCL-BT5.0 到底发了什么。

三阶段各观察 6s：
  1. 原始默认：只订阅 notify，不发送任何命令 → 模块「自然」上报速率；
  2. 下发官方 50Hz 命令（解锁 + RATE + 保存）后 → 看是否变快；
  3. 下发官方 100Hz 命令后 → 看是否变快（WT901BLE5.0 文档标称上限 50Hz，
     100Hz 可能被固件拒绝，正好借此验证）。

输出：MTU、notify 事件数/长度分布、前 12 条 notify 十六进制、解析统计
（成功报文 / 校验失败 / 未知 flag）、有效包速率。用于定位「模块没发够快」
还是「模块在发但解析拒了大部分」。

用法：
  python scripts/imu_probe.py                              # 扫描自动找 WT 模块
  python scripts/imu_probe.py --mac AA:BB:CC:DD:EE:FF      # 直接指定 MAC
  python scripts/imu_probe.py --phase2 20 --phase3 0       # 只试 20Hz、跳过阶段3
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from collections import Counter
from typing import Callable, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from tabletennis.imu.reader import (  # noqa: E402
    WitMotionParser,
    _READ_UUID,
    _SAVE_CMD,
    _SEND_UUID,
    _UNLOCK_CMD,
    _set_rate_cmd,
)


def _split_segments(raw: bytes):
    """把 notify 按 ``55 61`` 头切成连续段，报告每段长度并验证校验和。

    段长 21B = 带校验和（55 61 + 18B 数据 + 校验）；段长 20B = 无校验和
    （55 61 + 18B 数据）。用于验证 BLE 流里 0x61 包到底带不带校验字节。
    """
    starts = []
    i = 0
    while True:
        j = raw.find(b"\x55\x61", i)
        if j < 0:
            break
        starts.append(j)
        i = j + 2
    out = []
    for k, s in enumerate(starts):
        e = starts[k + 1] if k + 1 < len(starts) else len(raw)
        seg = raw[s:e]
        if len(seg) == 21:
            ck = seg[20]
            ok = (sum(seg[:20]) & 0xFF) == ck
            out.append((seg, f"21B 带校验 校验字节={ck:02x} {'通过' if ok else '失败'}", ok))
        elif len(seg) == 20:
            out.append((seg, "20B 无校验（55 61+18B 数据）", True))
        else:
            out.append((seg, f"{len(seg)}B 异常段", False))
    return out


async def _observe(client, label: str, seconds: float) -> Tuple[int, float]:
    """订阅 notify 观察 ``seconds`` 秒，打印长度分布 / hex / 解析统计。"""
    print(f"\n===== {label}（{seconds:.0f}s）=====")
    # BLE 流 0x61 包是 20B 无校验（实测结论），解析器必须 checksum=False，
    # 否则包按 21B 等待全 partial_wait、有效速率恒 0。
    parser = WitMotionParser(checksum=False)
    lens: Counter = Counter()
    samples = []
    packets = 0

    def on_notify(_sender, data) -> None:
        nonlocal packets
        raw = bytes(data)
        lens[len(raw)] += 1
        if len(samples) < 12:
            samples.append(raw)
        for pkt in parser.feed(raw):
            packets += 1

    await client.start_notify(_READ_UUID, on_notify)
    t0 = time.time()
    await asyncio.sleep(seconds)
    dt = time.time() - t0
    await client.stop_notify(_READ_UUID)

    n_notify = sum(lens.values())
    print(f"notify 事件: {n_notify} 个（{n_notify / dt:.1f}/s）")
    print(f"notify 长度分布: {dict(lens)}")
    print(f"有效报文: {packets} 个（{packets / dt:.1f}/s）")
    print(f"解析统计: {dict(parser.stats)}")
    for i, raw in enumerate(samples[:3]):
        print(f"notify#{i} 完整 hex ({len(raw)}B): {raw.hex()}")
        # 按「0x55 0x61 + 18B 数据」切段，验证段末是否有校验和字节
        segs = _split_segments(raw)
        for j, (seg, info, ok) in enumerate(segs):
            print(f"  段{j} {info} → 数据={seg[2:].hex()}")
    return packets, dt


async def main(mac: Optional[str], phase2_hz: float, phase3_hz: Optional[float]) -> None:
    from bleak import BleakClient, BleakScanner

    addr, name = mac, (mac or "?")
    if not addr:
        print("[probe] 扫描 BLE 设备（找名字含 'WT'）…")
        devices = await BleakScanner.discover(timeout=8.0, return_adv=True)
        for d, adv in devices.values():
            n = adv.local_name or d.name or ""
            if "WT" in n.upper():
                addr, name = d.address, n
                break
        if not addr:
            print("[probe] 没找到名字含 'WT' 的模块——上电了吗？被手机占用了吗？")
            return

    print(f"[probe] 连接 {name} ({addr})…")
    async with BleakClient(addr, timeout=12.0) as client:
        print(f"[probe] MTU = {client.mtu_size}（>20 则 21B 的 0x61 包单条送达，不拆包）")
        p1, _ = await _observe(client, "阶段1 原始默认（不发任何命令）", 6.0)

        print(f"\n[probe] 下发命令：解锁 + 上报率 {phase2_hz:g}Hz + 保存")
        for cmd in (_UNLOCK_CMD, _set_rate_cmd(phase2_hz), _SAVE_CMD):
            await client.write_gatt_char(_SEND_UUID, cmd, response=False)
            await asyncio.sleep(0.06)
        p2, _ = await _observe(client, f"阶段2 命令后（{phase2_hz:g}Hz）", 6.0)

        if phase3_hz:
            print(f"\n[probe] 再下发：解锁 + 上报率 {phase3_hz:g}Hz + 保存")
            for cmd in (_UNLOCK_CMD, _set_rate_cmd(phase3_hz), _SAVE_CMD):
                await client.write_gatt_char(_SEND_UUID, cmd, response=False)
                await asyncio.sleep(0.06)
            p3, _ = await _observe(client, f"阶段3 命令后（{phase3_hz:g}Hz）", 6.0)
        else:
            p3 = None

        print("\n===== 结论 =====")
        print(f"  原始默认: {p1}/6s")
        print(f"  {phase2_hz:g}Hz 命令后: {p2}/6s")
        if p3 is not None:
            print(f"  {phase3_hz:g}Hz 命令后: {p3}/6s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="IMU BLE 原始数据探针")
    ap.add_argument("--mac", default=None, help="模块 MAC（默认扫描找 WT）")
    ap.add_argument("--phase2", type=float, default=50.0, help="阶段2 试的 Hz")
    ap.add_argument("--phase3", type=float, default=100.0,
                    help="阶段3 试的 Hz；传 0 跳过阶段3")
    args = ap.parse_args()
    asyncio.run(main(args.mac, args.phase2, args.phase3 if args.phase3 else None))
