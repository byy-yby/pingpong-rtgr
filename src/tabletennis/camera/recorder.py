"""四路相机同步录像到 ``data/video/<session>/``（供离线 EasyMocap 重建）。

为什么这样设计
--------------
- live_control 的主循环 / 球 / EasyMocap 线程都在消费相机帧队列；录像若挂在主循环上，
  会随显示 / 重建节奏丢帧。这里每台相机独立一个后台编码线程，帧在**采集线程**里通过
  ``Camera.set_frame_sink`` 直达录制队列——不受主循环 / 重建线程影响，抓多少录多少。
- 若 CPU 编码跟不上出帧率，录制线程会**丢最旧未写帧**（不阻塞抓帧、不阻塞主线程）。
  写成文件的每一帧都记下设备时间戳（``cam{cid}_ts.npy``），离线脚本
  ``scripts/reconstruct_video.py`` 靠时间戳把 4 路视频重新对齐到同一触发脉冲，
  所以个别丢帧不影响重建正确性。

产物布局（一个 session 一个文件夹）
-----------------------------------
``data/video/<YYYYmmdd_HHMMSS>/``
    cam0.mp4 .. cam3.mp4      四段灰度视频（mp4v 编码，逻辑相机号 = 标定里的 cid）
    cam0_ts.npy ..            ``uint64`` 设备时间戳数组，与 mp4 帧一一对应（长度 = 写入帧数）
    meta.json                 相机→序列号、fps、起止墙钟、各相机帧数（对账用）
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
from typing import Dict, List, Optional

import cv2
import numpy as np

from ..core.config import project_root
from ..core.types import Frame

# 本机实测：1440×1080 Mono8 下 mp4v ~8ms/帧（≈100fps 预算内）；MJPG 26ms 太慢、
# avc1(h264) 无 v4l2 设备直接打不开。故统一写 mp4v → .mp4。
_FOURCC = "mp4v"
_MAX_QUEUE = 128


def default_session_dir(root: Optional[str] = None) -> str:
    """``data/video/<YYYYmmdd_HHMMSS>``；同秒重名自动加 ``_1`` 后缀。"""
    base = os.path.join(root or project_root(), "data", "video")
    name = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(base, name)
    n = 1
    while os.path.exists(path):
        path = os.path.join(base, f"{name}_{n}")
        n += 1
    return path


class _CameraWriter(threading.Thread):
    """一台相机一个编码线程：消费有界队列里的灰度帧 → BGR mp4v + 攒设备时间戳。"""

    def __init__(self, cam, path: str, fps: float, max_queue: int = _MAX_QUEUE) -> None:
        super().__init__(name=f"rec-cam{cam.logical_id}", daemon=True)
        self.cam = cam
        self.path = path
        self.fps = float(fps)
        self._q: "queue.Queue" = queue.Queue(maxsize=max_queue)
        self._ts: List[int] = []
        self.count = 0
        self.n_fed = 0       # 采集侧送进来多少帧（含被丢掉的）
        self.n_dropped = 0   # 队列满被丢掉多少帧（≈ 编码跟不上时的丢帧）
        self._first_ts: Optional[int] = None
        self._last_ts: Optional[int] = None

    # -- 采集线程侧（快进快出）------------------------------------------------
    def feed(self, frame: Frame) -> None:
        """把一帧交给录制队列；满则丢最旧（保持近实时，丢帧由 ts 副产物记录）。"""
        self.n_fed += 1
        try:
            self._q.put_nowait(frame)
        except queue.Full:
            try:
                self._q.get_nowait()      # 丢最旧，给新帧腾位
                self.n_dropped += 1
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(frame)
            except queue.Full:            # 仍满：本次帧也丢（极端）
                self.n_dropped += 1

    def stop(self) -> None:
        """请求停止（置哨兵），随后由调用方 join()。

        编码线程若已因错误退出则没人消费队列——避免 put 永久阻塞，此时直接放弃
        （join 会立即返回，调用方照常收尾）。
        """
        while self.is_alive():
            try:
                self._q.put_nowait(None)
                return
            except queue.Full:
                time.sleep(0.01)      # 等编码线程腾出空位（最坏排空 ~128 帧 ≈1s）

    # -- 编码线程侧 -----------------------------------------------------------
    def run(self) -> None:
        vw: Optional[cv2.VideoWriter] = None
        try:
            while True:
                frame = self._q.get()
                if frame is None:
                    break
                gray = frame.image
                if vw is None:
                    h, w = gray.shape[:2]
                    vw = cv2.VideoWriter(
                        self.path, cv2.VideoWriter_fourcc(*_FOURCC), self.fps, (w, h))
                    if not vw.isOpened():
                        print(f"[录像] ✗ 打不开编码器，文件作废：{self.path}")
                        break
                    self._first_ts = int(frame.device_timestamp)
                bgr = gray if gray.ndim == 3 else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                vw.write(bgr)
                self._ts.append(int(frame.device_timestamp))
                self.count += 1
                self._last_ts = int(frame.device_timestamp)
        finally:
            if vw is not None:
                vw.release()
            self._first_ts = self._first_ts if self._first_ts is not None else 0
            self._last_ts = self._last_ts if self._last_ts is not None else 0

    @property
    def ts_array(self) -> np.ndarray:
        return np.asarray(self._ts, dtype=np.uint64)


class SessionVideoRecorder:
    """把一组相机的画面录成一个 session 文件夹。

    Args:
        cameras: ``CameraManager.cameras``（须含 ``logical_id`` / ``serial`` /
            ``set_frame_sink``）。录制按 ``logical_id`` 命名 ``cam{cid}.mp4``，
            与标定文件里的相机号一致。
        session_dir: 目标文件夹；None = ``data/video/<时间戳>``。
        fps: mp4 头标称帧率。默认按触发模式给 100 / 30，只影响播放速度，
            离线重建读的是帧序号 + ts 副产物，不受影响。

    用法::

        rec = SessionVideoRecorder(mgr.cameras)
        rec.start()
        ...（录制中，无状态需要维护）...
        meta = rec.stop()     # meta["dir"] / meta["frames_per_cam"] / ...
    """

    def __init__(self, cameras, session_dir: Optional[str] = None,
                 fps: Optional[float] = None) -> None:
        self.cameras = list(cameras)
        self.fps = fps
        self.session_dir = session_dir or default_session_dir()
        self._writers: Dict[int, _CameraWriter] = {}
        self._started = False
        self._t0 = 0.0

    @property
    def is_recording(self) -> bool:
        return self._started

    def start(self) -> None:
        """开录：建目录、给每台相机开编码线程并挂上帧 sink。"""
        if self._started:
            return
        os.makedirs(self.session_dir, exist_ok=True)
        # mp4 头标称帧率（只影响播放速度）。构造时由调用方按触发模式给 100/30。
        fps = self.fps if self.fps is not None else 100.0
        for cam in self.cameras:
            path = os.path.join(self.session_dir, f"cam{cam.logical_id}.mp4")
            w = _CameraWriter(cam, path, fps)
            w.start()
            self._writers[cam.logical_id] = w
            cam.set_frame_sink(w.feed)
        self._started = True
        self._t0 = time.time()
        print(f"[录像] ● 开始：{self.session_dir}（{len(self.cameras)} 路，{fps:.0f}fps）")

    def stop(self) -> dict:
        """停录：摘 sink → 各线程收尾 → 写 ts 副产物与 meta.json。返回 meta。"""
        if not self._started:
            return {}
        self._started = False
        for cam in self.cameras:
            cam.set_frame_sink(None)
        frames_per_cam: Dict[str, int] = {}
        periods: Dict[str, float] = {}
        fed_per_cam: Dict[str, int] = {}
        dropped_per_cam: Dict[str, int] = {}
        for cid, w in self._writers.items():
            w.stop()
            w.join(timeout=10.0)
            ts = w.ts_array
            frames_per_cam[str(cid)] = int(len(ts))
            fed_per_cam[str(cid)] = int(w.n_fed)
            dropped_per_cam[str(cid)] = int(w.n_dropped)
            # 实测帧率（ticks→s：设备时间戳 ~100MHz → 1 tick=1e-8s）
            if len(ts) > 2:
                med = float(np.median(np.diff(np.asarray(ts, dtype=np.float64))))
                periods[str(cid)] = round(med * 1e-8, 6)   # 秒
            np.save(os.path.join(self.session_dir, f"cam{cid}_ts.npy"), ts)
        self._writers = {}

        wall_s = time.time() - self._t0
        meta = {
            "session_dir": self.session_dir,
            "started_wall": self._t0,
            "duration_s": round(wall_s, 3),
            "fps": (self.fps if self.fps is not None else 100.0),
            "camera_serials": {str(c.logical_id): c.serial for c in self.cameras},
            "frames_per_cam": frames_per_cam,
            # 诊断：喂进来 vs 写进文件 vs 编码丢帧——差多少一眼看出瓶颈在哪侧
            "fed_per_cam": fed_per_cam,
            "encoder_dropped_per_cam": dropped_per_cam,
            "measured_period_s": periods,
        }
        with open(os.path.join(self.session_dir, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)
        print(f"[录像] ■ 停止：本次 {wall_s:.1f}s")
        for cid in sorted(frames_per_cam):
            written = frames_per_cam[cid]
            fed = fed_per_cam[cid]
            dropped = dropped_per_cam[cid]
            fps_est = written / wall_s if wall_s > 0 else 0.0
            print(f"  cam{cid}: 喂 {fed} → 写 {written} 帧（编码丢 {dropped}）"
                  f"· 实测 {fps_est:5.1f} fps / 100 目标")
        print(f"        → {self.session_dir}")
        return meta
