"""离线重建的输入：把 ``data/video/<session>/`` 的四路 mp4 + ts 副产物读成多视角帧流。

录制端 ``camera/recorder.py`` 每台相机独立后台编码，编码跟不上时会丢帧——所以同一次
录像里四段视频的**帧号并不一一对应同一触发脉冲**。本模块用每段视频旁的设备时间戳
（``cam{cid}_ts.npy``，uint64、~100MHz tick）做跨相机对齐，把四路重新绑回同一时刻。

对齐策略（沿 CLAUDE.md 时间戳节的结论）
--------------------------------------
- **绝不在相机之间直接比设备时间戳的绝对大小**：各机时钟基准偏移秒级、不可比。
  改为每台相机各自算「脉冲号」——设备时间戳只在**同一台相机内部**相邻帧相减，
  该帧与上一帧之间丢了几拍（编码跟不上）就加几号。正常 1 帧/拍，中位周期差即 1 拍。
- 录制端把 sink 挂在已经以同一外部触发跑着的抓帧流上 → 四台从**同一拍**开始写，
  所以「脉冲号相同 ⇒ 同一物理触发」成立。对主时钟第 k 帧（脉冲 P），目标相机里
  找脉冲号 == P 的那一帧；目标相机这一拍被编码丢掉了就没有帧可给（该视角缺帧）。
- 完全没有 ts 副产物时退化为「按帧号对齐」（等长假设，丢帧则错位——仅作回退）。

对外 API
--------
- :func:`discover_session` —— 列出 session 文件夹里的相机视频。
- :func:`align_maps` —— 纯函数：返回主时钟每帧 -> 各相机应读帧号（缺帧的相机不在里）。
- :class:`VideoSource` —— 上述 + 顺序解码，:meth:`frames_for_ref` 给出某一主时钟帧的
  各相机 :class:`Frame`（灰度），供姿态检测 / 重建直接使用。
"""
from __future__ import annotations

import os
import re
from typing import Dict, List, Optional

import cv2
import numpy as np

from ..core.types import Frame

_CAM_RE = re.compile(r"^cam(\d+)\.(mp4|avi)$")

# 假设的设备时间戳频率（录制端 CLAUDE.md：~100MHz → 1 tick = 1e-8 s）
_TICK_HZ = 100e6


def discover_session(session_dir: str) -> Dict[int, str]:
    """扫描 ``session_dir``，返回 ``{cid: 视频文件路径}``（按 ``cam{cid}.mp4`` 命名）。"""
    out: Dict[int, str] = {}
    for name in sorted(os.listdir(session_dir)):
        m = _CAM_RE.match(name)
        if m:
            out[int(m.group(1))] = os.path.join(session_dir, name)
    return out


def _load_ts(session_dir: str, cid: int) -> Optional[np.ndarray]:
    p = os.path.join(session_dir, f"cam{cid}_ts.npy")
    if os.path.isfile(p):
        return np.load(p)
    return None


def _period_ticks(ts: np.ndarray) -> float:
    """脉冲周期（ticks）：连续时间戳差的中位数（robust）。"""
    if ts is None or len(ts) < 3:
        return 1.0
    d = np.diff(np.asarray(ts, dtype=np.float64))
    d = d[d > 0]
    if d.size == 0:
        return 1.0
    return float(np.median(d))


def _pulse_ids(ts: np.ndarray) -> np.ndarray:
    """单台相机内部：把设备时间戳转成「脉冲号」（丢帧的拍号被跳过）。

    ts 是同一相机的时间戳序列（~100MHz tick）。相邻差 ≈ 1 个脉冲周期；若某处差了
    ~2 周期说明中间丢了一拍没写进文件 → 从那一拍起脉冲号 +1。正常帧差就 +1。
    返回与 ts 等长的 ``int64``：``pulse_id[i]`` = 第 i 个写进文件的帧对应的触发拍号。
    """
    ts = np.asarray(ts, dtype=np.float64)
    n = len(ts)
    if n == 0:
        return np.empty(0, dtype=np.int64)
    if n == 1:
        return np.zeros(1, dtype=np.int64)
    d = np.diff(ts)
    ok = d > 0
    if not ok.any():
        return np.arange(n, dtype=np.int64)
    period = float(np.median(d[ok]))
    if period <= 0:
        return np.arange(n, dtype=np.int64)
    dropped = np.clip(np.round(d / period).astype(np.int64) - 1, 0, None)
    cum = np.concatenate(([0], np.cumsum(dropped)))
    return np.arange(n, dtype=np.int64) + cum


def align_maps(
    ts_by_cam: Dict[int, np.ndarray],
    n_frames_by_cam: Dict[int, int],
    ref_cam: int,
) -> List[Dict[int, int]]:
    """跨相机对齐（纯函数，便于单测）。

    Args:
        ts_by_cam: ``{cid: uint64 设备时间戳}``；缺失 = 无 ts。
        n_frames_by_cam: ``{cid: 视频帧数}``（无 ts 时的帧号对齐回退）。
        ref_cam: 主时钟相机（其脉冲序列当主时间轴）。

    Returns:
        ``maps[k] = {cid: 该相机文件帧号}``；长度 = 主时钟写帧数；
        该相机在某拍缺帧时不出现在那个 dict 里。
    """
    ref_ts = ts_by_cam.get(ref_cam)
    if ref_ts is None or len(ref_ts) == 0:
        # 没有 ts 可对齐 → 帧号回退（仅当全部/主相机缺 ts 时走到）
        n_ref = n_frames_by_cam.get(ref_cam, 0)
        maps: List[Dict[int, int]] = [dict() for _ in range(n_ref)]
        for cid in n_frames_by_cam:
            n_c = n_frames_by_cam.get(cid, 0)
            for k in range(min(n_ref, n_c)):
                maps[k][cid] = k
        return maps

    ref_ids = _pulse_ids(ref_ts)
    n_ref = len(ref_ids)
    maps: List[Dict[int, int]] = [dict() for _ in range(n_ref)]

    for cid, n_c in n_frames_by_cam.items():
        if n_c == 0:
            continue
        cam_ts = ts_by_cam.get(cid)
        if cam_ts is None or len(cam_ts) == 0 or cid == ref_cam:
            # 主相机逐帧对应；其它无 ts 的相机按帧号 1:1 对齐（丢帧则错位，仅回退）
            for k in range(min(n_ref, n_c)):
                maps[k][cid] = k
            continue
        cam_ids = _pulse_ids(cam_ts)
        for k in range(n_ref):
            p = int(np.searchsorted(cam_ids, int(ref_ids[k]), side="left"))
            # ref/cam 脉冲号都单调不减 → 命中位置天然单调前移，无需额外约束。
            if p < n_c and cam_ids[p] == ref_ids[k]:
                maps[k][cid] = p
            # p>=n_c 或脉冲号不等 → 该拍在 cam 里被丢掉 / 已到结尾 → 缺帧
    return maps


class VideoSource:
    """一个录制 session 的四路视频源（时间戳对齐 + 顺序解码）。

    ``frames_for_ref(k)`` 返回主时钟第 k 帧对应的各相机 :class:`Frame`
    （灰度 ``image``），缺失视角不出现在结果里。
    """

    def __init__(self, session_dir: str, ref_cam: Optional[int] = None) -> None:
        files = discover_session(session_dir)
        if not files:
            raise FileNotFoundError(
                f"{session_dir} 里没有 cam{ref_cam if ref_cam is not None else ''}N.mp4 视频"
            )
        self.session_dir = session_dir
        self.files = files
        self.cids: List[int] = sorted(files)
        self.ts_by_cam: Dict[int, np.ndarray] = {
            c: t for c in self.cids if (t := _load_ts(session_dir, c)) is not None
        }
        self.n_frames_by_cam: Dict[int, int] = {}
        for cid, path in files.items():
            cap = cv2.VideoCapture(path)
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            self.n_frames_by_cam[cid] = max(0, n)
        self.ref_cam = ref_cam if ref_cam in self.cids else self.cids[0]
        self.maps = align_maps(
            self.ts_by_cam, self.n_frames_by_cam, self.ref_cam
        )
        self._caps: Dict[int, cv2.VideoCapture] = {}
        self._pos: Dict[int, int] = {}
        self._cache: Dict[int, np.ndarray] = {}

    # ------------------------------------------------------------------
    def period_sec(self, cid: Optional[int] = None) -> float:
        cid = cid if cid is not None else self.ref_cam
        ts = self.ts_by_cam.get(cid)
        return _period_ticks(ts) / _TICK_HZ if ts is not None else 0.0

    @property
    def n_ref(self) -> int:
        return len(self.maps)

    def summary(self) -> dict:
        return {
            "ref_cam": self.ref_cam,
            "cams": {c: {"frames": self.n_frames_by_cam[c],
                         "period_s": self.period_sec(c)} for c in self.cids},
            "aligned_ref_frames": self.n_ref,
            "has_ts": {c: c in self.ts_by_cam for c in self.cids},
        }

    # ------------------------------------------------------------------
    def _read(self, cid: int, idx: int) -> Optional[np.ndarray]:
        """读某相机第 idx 帧灰度图；顺序推进，目标帧已读过则用缓存。"""
        if idx < 0:
            return None
        if self._cache.get(cid) is not None and self._pos.get(cid, -1) == idx:
            return self._cache[cid]
        cap = self._caps.get(cid)
        if cap is None:
            cap = cv2.VideoCapture(self.files[cid])
            if not cap.isOpened():
                return None
            self._caps[cid] = cap
            self._pos[cid] = -1
        pos = self._pos.get(cid, -1)
        if idx <= pos:                     # 倒退/重复：重新开文件（正常不会倒退）
            cap.release()
            cap = cv2.VideoCapture(self.files[cid])
            self._caps[cid] = cap
            pos = -1
        img: Optional[np.ndarray] = None
        while pos < idx:
            ok, bgr = cap.read()
            if not ok:
                return None
            pos += 1
            if pos == idx:
                img = bgr
        self._pos[cid] = pos
        if img is None:
            return None
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self._cache[cid] = gray
        return gray

    def frames_for_ref(self, k: int) -> Dict[int, Frame]:
        """主时钟第 k 帧的各相机对齐帧（灰度 Frame）。跳过缺帧/读不出的视角。"""
        out: Dict[int, Frame] = {}
        if k >= len(self.maps):
            return out
        for cid, idx in self.maps[k].items():
            gray = self._read(cid, idx)
            if gray is None:
                continue
            ts = self.ts_by_cam.get(cid)
            dev_ts = int(ts[idx]) if ts is not None and idx < len(ts) else 0
            h, w = gray.shape[:2]
            out[cid] = Frame(
                camera_id=cid, serial=str(cid), frame_num=int(idx),
                device_timestamp=dev_ts, host_timestamp=0,
                image=gray, pixel_format=0, width=w, height=h,
            )
        return out

    def seek(self, ref: int) -> None:
        """把各相机解码位置跳到主时钟 ``ref`` 帧附近（2D 叠加随机查看用）。

        重建管线走顺序解码（``frames_for_ref`` 单调递增），不调用它；这里给
        「拖进度条 / 首次打开 2D 叠加窗」这类大跳一个快路径：``CAP_PROP_POS_FRAMES``
        seek 到目标帧最近的关键帧，之后 ``_read`` 再顺序解到精确 ``idx``。h264 无
        精确随机访问，落点误差几帧内，诊断叠加可接受。
        """
        if not (0 <= ref < len(self.maps)):
            return
        for cid, idx in self.maps[ref].items():
            cap = self._caps.get(cid)
            if cap is None:
                cap = cv2.VideoCapture(self.files[cid])
                if not cap.isOpened():
                    continue
                self._caps[cid] = cap
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            self._pos[cid] = int(idx) - 1
            self._cache.pop(cid, None)

    def close(self) -> None:
        for cap in self._caps.values():
            cap.release()
        self._caps.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
