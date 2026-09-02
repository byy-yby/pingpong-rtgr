"""离线录像 / 重建输入的测试：时间戳对齐（纯函数）+ 录制往返（无相机硬件）。

- ``align_maps`` / ``_pulse_ids``：单测跨相机对齐在「各相机丢不同帧」下仍逐拍正确。
- ``SessionVideoRecorder`` → ``VideoSource``：用假相机（只实现 recorder 需要的
  ``logical_id / serial / set_frame_sink``）录一段 mp4 往返，验证编码线程、ts 副产物、
  解码端重新对齐到同一触发脉冲（用每帧灰度值编码脉冲号来对账）。
"""
import os
import time

import numpy as np
import pytest

from tabletennis.core.types import Frame
from tabletennis.reconstruction.video_source import VideoSource, _pulse_ids, align_maps

_TICK = 1_000_000  # ~100MHz tick @ 100 脉冲/s


# ----------------------------------------------------------------------
# 纯函数：_pulse_ids / align_maps
# ----------------------------------------------------------------------
def _mk_ts(n: int, drop_after_kept: int, ndrop: int, start_pulse: int = 5) -> np.ndarray:
    """生成某相机的设备时间戳序列：写 ``n`` 帧，``drop_after_kept`` 帧后连丢 ndrop 拍。"""
    ts = []
    pulse = start_pulse
    kept = 0
    while kept < n:
        ts.append(pulse * _TICK + (kept % 7))     # 加一点抖动（<1 周期）
        pulse += 1
        kept += 1
        if kept == drop_after_kept:
            pulse += ndrop
    return np.asarray(ts, dtype=np.uint64)


def test_pulse_ids_strictly_increasing_with_drops():
    """脉冲号必须严格递增，且丢帧处跳过对应的拍号。"""
    ts0 = _mk_ts(60, 10 ** 9, 0)      # 无丢帧：id == 帧号
    ts1 = _mk_ts(40, 20, 2)           # 第 20 帧后丢 2 拍
    ids0 = _pulse_ids(ts0)
    ids1 = _pulse_ids(ts1)
    assert np.array_equal(ids0, np.arange(60))
    assert bool(np.all(np.diff(ids1) > 0))
    # 丢 2 拍 → 最后写帧的脉冲号比帧号大 2
    assert ids1[-1] == len(ts1) - 1 + 2
    # 前 20 帧仍与帧号一致（丢帧发生在之后）
    assert np.array_equal(ids1[:20], np.arange(20))


def test_align_maps_matches_same_physical_pulse():
    """四台相机起点同拍、各自丢不同帧：对齐后每个映射都指向同一脉冲的帧。"""
    t0 = _mk_ts(60, 10 ** 9, 0)       # ref：无丢
    t1 = _mk_ts(60, 20, 1)            # 丢 1 拍
    t2 = _mk_ts(60, 15, 3)            # 连丢 3 拍
    t3 = _mk_ts(60, 10 ** 9, 0)       # 无丢
    ts_by = {0: t0, 1: t1, 2: t2, 3: t3}
    nf = {c: len(t) for c, t in ts_by.items()}

    maps = align_maps(ts_by, nf, ref_cam=0)
    assert len(maps) == len(t0)

    ids = {c: _pulse_ids(t) for c, t in ts_by.items()}
    miss = {1: 0, 2: 0}
    for k, d in enumerate(maps):
        assert d[0] == k                       # 主时钟恒逐帧
        for c in (1, 2, 3):
            if c in d:
                # 目标帧脉冲号 == 主时钟帧脉冲号 ⇒ 同一物理触发
                assert ids[c][d[c]] == ids[0][k], (c, k)
            else:
                miss[c] = miss.get(c, 0) + 1
    assert miss[1] == 1 and miss[2] == 3       # 只有真丢帧的拍缺帧
    for k, d in enumerate(maps):               # 无丢帧的相机与主时钟 1:1
        assert d[3] == k


def test_align_maps_no_ts_falls_back_to_index():
    """完全没有 ts 时退化为逐帧号对齐。"""
    maps = align_maps({}, {0: 10, 1: 8, 2: 10}, ref_cam=0)
    assert len(maps) == 10
    assert all(d.get(0) == k for k, d in enumerate(maps))
    assert [1 in d and d[1] == k for k, d in enumerate(maps)] == [True] * 8 + [False] * 2
    assert all(d.get(2) == k for k, d in enumerate(maps))


# ----------------------------------------------------------------------
# 往返：SessionVideoRecorder（假相机）→ 文件 → VideoSource 重新对齐
# ----------------------------------------------------------------------
class _FakeCam:
    """只实现 recorder / 主流程用到的相机接口。"""

    def __init__(self, logical_id: int) -> None:
        self.logical_id = logical_id
        self.serial = f"SN{logical_id:06d}"
        self._sink = None

    def set_frame_sink(self, sink) -> None:
        self._sink = sink

    def feed(self, frame: Frame) -> None:
        """模拟采集线程把帧交给 sink（旁路录制回调）。"""
        if self._sink is not None:
            self._sink(frame)


def _flat_frame(cid: int, pulse: int) -> Frame:
    """一帧「白带在第 pulse 行」的位置编码帧——码流做 YUV/有损变换后位置仍可靠。"""
    img = np.zeros((120, 160), np.uint8)
    img[pulse:pulse + 3, :] = 255
    return Frame(
        camera_id=cid, serial=str(cid), frame_num=pulse,
        device_timestamp=pulse * _TICK, host_timestamp=0,
        image=img, pixel_format=0, width=160, height=120,
    )


def _band_row(img: np.ndarray) -> int:
    """读回帧里白带**起始行**（YUV 码流有轻微行间涂抹，取第一个亮行最稳）。"""
    rows = img.mean(axis=1)
    bright = np.argmax(rows > 100.0)
    return int(bright)


def _feed_session(cams, n_pulses: int, drop_map: dict) -> None:
    """给每台相机喂 n_pulses 帧；drop_map[cid]=被丢掉的脉冲号集合。"""
    for cid in drop_map:
        for pulse in range(n_pulses):
            if pulse in drop_map[cid]:
                continue
            cams[cid].feed(_flat_frame(cid, pulse))


def test_recorder_roundtrip_and_videosource(tmp_path):
    """录 3 路 mp4（cam2 丢 2 拍）→ ts 副产物 → VideoSource 按脉冲重新对齐。"""
    cv2 = pytest.importorskip("cv2")

    session = str(tmp_path / "sess")
    n_cam, n_pulses = 3, 30
    cams = {cid: _FakeCam(cid) for cid in range(n_cam)}
    drop_map = {0: set(), 1: set(), 2: {15, 16}}   # cam2 丢 2 个连拍
    feed_frames = {c: n_pulses - len(d) for c, d in drop_map.items()}

    from tabletennis.camera.recorder import SessionVideoRecorder

    rec = SessionVideoRecorder(list(cams.values()), session_dir=session, fps=100.0)
    rec.start()
    assert rec.is_recording
    _feed_session(cams, n_pulses, drop_map)
    time.sleep(0.5)              # 给编码线程排空（160×120 mp4v 很快）
    meta = rec.stop()
    assert not rec.is_recording

    # 产物齐全
    for cid in range(n_cam):
        assert os.path.isfile(os.path.join(session, f"cam{cid}.mp4"))
        ts = np.load(os.path.join(session, f"cam{cid}_ts.npy"))
        assert ts.dtype == np.uint64 and len(ts) == feed_frames[cid]
    assert meta["frames_per_cam"] == {str(c): feed_frames[c] for c in range(n_cam)}

    # VideoSource：主时钟 cam0，按 ts 重对齐到同一脉冲
    src = VideoSource(session, ref_cam=0)
    assert src.n_ref == feed_frames[0]
    assert src.ref_cam == 0

    aligned_ok = missing = 0
    for k in range(src.n_ref):
        fs = src.frames_for_ref(k)
        pulse0 = k                       # cam0 无丢帧 → 脉冲号 == 帧号
        for cid in range(n_cam):
            if cid not in fs:
                assert cid in (1, 2) and k in {15, 16}  # cam2 缺 15/16
                missing += 1
                continue
            img = fs[cid].image
            # 对齐正确 ⇒ 读到的是与主时钟同一脉冲的帧（白带行 = 脉冲号）
            assert _band_row(img) == pulse0, (cid, k, _band_row(img))
            assert fs[cid].device_timestamp // _TICK == pulse0
            aligned_ok += 1
    assert missing == 2
    assert aligned_ok == src.n_ref * 3 - 2
    src.close()
