"""四路相机同步录像到 ``data/video/<session>/``（供离线 EasyMocap 重建）。

为什么这样设计
--------------
- live_control 的主循环 / 球 / EasyMocap 线程都在消费相机帧队列；录像若挂在主循环上，
  会随显示 / 重建节奏丢帧。这里每台相机独立一个后台编码线程，帧在**采集线程**里通过
  ``Camera.set_frame_sink`` 直达录制队列——不受主循环 / 重建线程影响，抓多少录多少。
- 编码在 **ffmpeg 子进程**里做（主路径 GPU ``h264_nvenc``，见 ``_resolve_encoder``），
  编码线程只把灰度帧转 yuv420p 喂管道；若编码跟不上出帧率，录制线程会**丢最旧未写帧**
  （不阻塞抓帧、不阻塞主线程）。写成文件的每一帧都记下设备时间戳
  （``cam{cid}_ts.npy``），离线脚本 ``scripts/reconstruct_video.py`` 靠时间戳把 4 路视频
  重新对齐到同一触发脉冲，所以个别丢帧不影响重建正确性。

产物布局（一个 session 一个文件夹）
-----------------------------------
``data/video/<YYYYmmdd_HHMMSS>/``
    cam0.mp4 .. cam3.mp4      四段灰度视频（H.264，逻辑相机号 = 标定里的 cid）
    cam0_ts.npy ..            ``uint64`` 设备时间戳数组，与 mp4 帧一一对应（长度 = 写入帧数）
    meta.json                 相机→序列号、fps、起止墙钟、各相机帧数 + 供帧停顿诊断
                              ``feed_diag_per_cam``（入队墙钟空档 + GetImageBuffer 超时，
                              用于判定「进程冻结」还是「相机/总线停供」，见 stop 打印）
"""
from __future__ import annotations

import functools
import json
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..core.config import project_root
from ..core.types import Frame

# 编码器：默认走 ffmpeg 子进程的 GPU h264_nvenc（本机实测 1440×1080 灰度 4 路并发
# ~168fps/路，远超 100fps 目标）；h264_nvenc 不可用 → libx264；再不可用才退回 cv2 mp4v
# （本机 mp4v 在噪声内容上只有 ~44fps/路，跟不上 100fps 会丢帧——纯兜底）。
# 详见 _resolve_encoder / _FfmpegWriter。
_FOURCC = "mp4v"
_MAX_QUEUE = 128

# 本机自编的 NVENC ffmpeg（源码+头文件见 tools/；也可用 TT_FFMPEG 环境变量覆盖，
# 或把任意带 h264_nvenc 的 ffmpeg 放进 PATH）。
_TOOLS_FFMPEG = "/home/yby/tools/ffmpeg-nvenc/bin/ffmpeg"


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


def _feed_gap_report(wall: List[float], t0: float, t1: float,
                     thresh_s: float = 0.06, cap: int = 50) -> dict:
    """把一路相机的入队墙钟序列转成「供帧停顿」诊断。

    Args:
        wall: ``_CameraWriter._feed_wall``（``time.perf_counter`` 单调秒）。
        t0: 录像开始墙钟（单调秒，recorder.start 记的）。
        t1: 停止墙钟（单调秒，recorder.stop 一进来就记的）。
        thresh_s: 相邻入队间隔超过它才算一次停顿（正常应 ~10ms）。
        cap: gaps 列表最多记多少条。

    设备时间戳只能说明相机侧节奏；要回答「主机在录到一半停供了多久、当时抓帧线程
    是卡死还是在超时轮询」，必须有墙钟侧的这一份记录。
    """
    off = [x - t0 for x in wall if x >= t0]     # 相对录像开始的偏移（秒）
    diag = {"n_fed": len(off)}
    if len(off) < 2:
        return diag
    diffs = [off[i + 1] - off[i] for i in range(len(off) - 1)]
    gaps = [(round(off[i], 3), round(off[i + 1], 3), round(g, 3))
            for i, g in enumerate(diffs) if g > thresh_s]
    t1_off = t1 - t0
    diag.update({
        "first_feed_s": round(off[0], 3),
        "last_feed_s": round(off[-1], 3),
        "span_s": round(off[-1] - off[0], 3),
        "silence_after_last_s": round(max(t1_off - off[-1], 0.0), 3),
        "gaps": gaps[:cap],
        "n_gaps": len(gaps),
        "max_gap_s": round(max(diffs), 3) if diffs else 0.0,
    })
    return diag


# ---------------------------------------------------------------------------
# 编码器解析：ffmpeg(h264_nvenc → libx264) → cv2 mp4v 兜底
# ---------------------------------------------------------------------------
# 编码器候选（codec, 附加参数）。顺序即优先级；nvenc 与 libx264 的差异参数在
# _ffmpeg_encode_ok 里用「真实小编码」验证（存在 ≠ 能在本驱动上打开）。
_CODEC_CANDIDATES: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("h264_nvenc", ("-preset", "p4", "-cq", "20")),
    ("libx264", ("-preset", "veryfast", "-crf", "18")),
)


def _find_ffmpeg_bin() -> Optional[str]:
    """找可用的 ffmpeg：TT_FFMPEG 环境变量 > 本机自编 NVENC 版 > PATH。"""
    forced = os.environ.get("TT_FFMPEG", "")
    if forced and os.path.isfile(forced):
        return forced
    if os.path.isfile(_TOOLS_FFMPEG):
        return _TOOLS_FFMPEG
    return shutil.which("ffmpeg")


def _ffmpeg_has_encoder(ffmpeg: str, codec: str) -> bool:
    """ffmpeg -encoders 里是否注册了该编码器（不保证能在本驱动上打开）。"""
    try:
        out = subprocess.run([ffmpeg, "-hide_banner", "-encoders"],
                             capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return False
    return f" {codec} " in out


@functools.lru_cache(maxsize=None)
def _ffmpeg_encode_ok(ffmpeg: str, codec: str, args: Tuple[str, ...]) -> bool:
    """跑一次真实小编码：编码器能打开 + mp4 能落盘才算数。

    BtbN「latest」的 h264_nvenc 就是反例：-encoders 里在，但驱动 580 只支持
    nvenc API 13.0、它要 13.1 → 一打开就报错。所以必须实测，不能只看清单。
    """
    # yuv420p 3 帧小图（内容无关，只验证能开）；直接 feed 文件头里 160x120。
    frame = bytearray(160 * 120)          # Y = 灰渐变（任意值即可）
    for i in range(160 * 120):
        frame[i] = i & 0xFF
    frame += bytes([128]) * (160 * 120 // 2)   # U/V = 128
    frame = bytes(frame) * 3
    with tempfile.TemporaryDirectory(prefix="tt_enc_probe_") as td:
        cmd = [ffmpeg, "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "yuv420p", "-video_size", "160x120",
               "-r", "30", "-i", "pipe:0", "-c:v", codec, *args,
               os.path.join(td, "probe.mp4")]
        try:
            p = subprocess.run(cmd, input=frame, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=30)
            return p.returncode == 0
        except Exception:
            return False


@functools.lru_cache(maxsize=1)
def _resolve_encoder() -> Optional[Tuple[str, str, Tuple[str, ...]]]:
    """一次性解析用哪个编码器 → (ffmpeg, codec, args)；None = 走 cv2 mp4v 兜底。

    可用 ``TT_RECORDER_CODEC=auto|nvenc|x264|cv2`` 强制指定（默认 auto：
    有 h264_nvenc 且能打开就用它，否则 libx264，再否则 cv2）。
    """
    force = os.environ.get("TT_RECORDER_CODEC", "auto").strip().lower()
    if force == "cv2":
        return None
    ffmpeg = _find_ffmpeg_bin()
    if not ffmpeg:
        return None
    only = {"nvenc": "h264_nvenc", "x264": "libx264"}.get(force)  # None = auto
    for codec, args in _CODEC_CANDIDATES:
        if only is not None and codec != only:
            continue
        if not _ffmpeg_has_encoder(ffmpeg, codec):
            continue
        if _ffmpeg_encode_ok(ffmpeg, codec, args):
            return (ffmpeg, codec, args)
    return None


class _FfmpegWriter:
    """ffmpeg 子进程编码（nvenc/x264）。灰度帧在 Python 侧转成 yuv420p 直接喂。

    为什么喂 yuv420p 而不是 gray：gray 会让 ffmpeg 每帧做一次软件上采样
    （swscale gray→yuv420p）——本机实测正是这堵墙把 4 路压到 ~93fps/路；
    改喂 yuv420p（灰度图没有颜色，U/V 恒 128=中性灰，逐帧只变 Y 平面）后
    4 路飙到 ~168fps/路。Python 侧代价只是一次 Y 拷贝 + 两个 os.write。
    """
    ok = False

    def __init__(self, ffmpeg: str, codec: str, args: Tuple[str, ...],
                 path: str, fps: float, w: int, h: int) -> None:
        # w/h 必须偶数（yuv420p / nvenc 要求）；调用方保证偶数才进 ffmpeg 分支
        self._fd = None
        self.path = path
        self._chroma = bytes([128]) * (w * h // 2)   # U+V 两平面恒 128，只建一次
        self._proc = subprocess.Popen(
            [ffmpeg, "-y", "-loglevel", "error",
             "-f", "rawvideo", "-pix_fmt", "yuv420p",
             "-video_size", f"{w}x{h}", "-r", str(int(fps)),
             "-i", "pipe:0",
             "-c:v", codec, *args, path],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        self._fd = self._proc.stdin.fileno()
        self.ok = self._proc.poll() is None

    def write(self, gray: np.ndarray) -> bool:
        """写一帧灰度图。返回 False = 编码进程已死（丢帧/中断）。"""
        try:
            os.write(self._fd, gray.tobytes())       # Y 平面（1.55MB @1080p）
            os.write(self._fd, self._chroma)         # U/V 平面（0.78MB @1080p）
            return True
        except (BrokenPipeError, OSError):
            return False

    def close(self) -> None:
        """关 stdin → 等 ffmpeg 收尾写 moov → 报退出码。"""
        try:
            if self._proc.stdin is not None:
                self._proc.stdin.close()
        except Exception:      # 进程已死时可能 ValueError/BrokenPipe，忽略
            pass
        try:
            rc = self._proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()
            rc = self._proc.returncode
        if rc != 0:
            try:
                err = (self._proc.stderr.read() or b"").decode("utf-8", "replace")
            except Exception:
                err = ""
            print(f"[录像] ⚠ {os.path.basename(self.path)} 编码进程退出码 {rc}"
                  f"：{err[-300:] if err else '无 stderr'}")


class _Cv2Writer:
    """兜底：OpenCV mp4v（本机 1440×1080 噪声内容 ~44fps/路，跟不上 100fps，
    正常情况不该走到这里——只有机器上没有可用 ffmpeg 时才用）。"""

    def __init__(self, path: str, fps: float, w: int, h: int) -> None:
        self.path = path
        self._vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*_FOURCC), fps, (w, h))
        self._ok = self._vw.isOpened()

    def write(self, gray: np.ndarray) -> bool:
        if not self._ok:
            return False
        bgr = gray if gray.ndim == 3 else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        self._vw.write(bgr)
        return True

    def close(self) -> None:
        if self._ok:
            self._vw.release()


def _open_writer(path: str, fps: float, w: int, h: int, gray2d: bool = True):
    """按机器能力开一个 writer；全失败返回 None（文件作废）。

    ffmpeg 分支只吃 2D 灰度（yuv420p 直喂按 Y 平面拷）；彩色帧走 cv2 兜底。
    """
    plan = _resolve_encoder()
    if plan is not None and gray2d and w % 2 == 0 and h % 2 == 0:
        try:
            wr = _FfmpegWriter(*plan, path, fps, w, h)
            if wr.ok:
                return wr
            wr.close()
        except Exception as e:
            print(f"[录像] ✗ ffmpeg 启动失败，回退 cv2：{e}")
    try:
        wr = _Cv2Writer(path, fps, w, h)
        if wr._ok:
            return wr
        wr.close()
    except Exception:
        pass
    return None


class _CameraWriter(threading.Thread):
    """一台相机一个编码线程：消费有界队列里的灰度帧 → 编码落盘 + 攒设备时间戳。"""

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
        self._feed_wall: List[float] = []  # 每帧入队墙钟（单调秒，诊断静默窗口用）
        self._first_ts: Optional[int] = None
        self._last_ts: Optional[int] = None

    # -- 采集线程侧（快进快出）------------------------------------------------
    def feed(self, frame: Frame) -> None:
        """把一帧交给录制队列；满则丢最旧（保持近实时，丢帧由 ts 副产物记录）。"""
        self.n_fed += 1
        # 入队墙钟：一帧一次 append，开销 ~µs 级。设备 ts 只反映相机侧节奏，
        # 要定位「主机停供」发生在哪段墙钟必须在这里打点。
        self._feed_wall.append(time.perf_counter())
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
        enc = None  # _FfmpegWriter / _Cv2Writer，首个有图帧才开（需要 w/h）
        try:
            while True:
                frame = self._q.get()
                if frame is None:
                    break
                gray = frame.image
                if enc is None:
                    h, w = gray.shape[:2]
                    enc = _open_writer(self.path, self.fps, w, h, gray.ndim == 2)
                    if enc is None:
                        print(f"[录像] ✗ 打不开编码器，文件作废：{self.path}")
                        break
                    self._first_ts = int(frame.device_timestamp)
                if not enc.write(gray):
                    print(f"[录像] ✗ 编码线程写入失败，中断：{self.path}")
                    break
                self._ts.append(int(frame.device_timestamp))
                self.count += 1
                self._last_ts = int(frame.device_timestamp)
        finally:
            if enc is not None:
                enc.close()
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
        self._t0_perf = 0.0   # 开始/停止的单调墙钟（供帧停顿诊断的基准）
        self._t1_perf = 0.0

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
        self._t0_perf = time.perf_counter()
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
        self._t1_perf = time.perf_counter()   # 用户按停的瞬间（供帧停顿诊断的截止点）
        self._started = False
        for cam in self.cameras:
            cam.set_frame_sink(None)
        frames_per_cam: Dict[str, int] = {}
        periods: Dict[str, float] = {}
        fed_per_cam: Dict[str, int] = {}
        dropped_per_cam: Dict[str, int] = {}
        feed_diag: Dict[str, dict] = {}
        cam_by_cid = {c.logical_id: c for c in self.cameras}
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
            # 供帧停顿诊断：入队墙钟的空档 + 该窗口里 GetImageBuffer 有没有在超时轮询
            diag = _feed_gap_report(w._feed_wall, self._t0_perf, self._t1_perf)
            cam = cam_by_cid.get(cid)
            if cam is not None:
                # 诊断属性在真相机上由 camera.py 提供；假相机/别的实现缺了就置空
                tw_all = getattr(cam, "grab_timeout_wall", [])
                t1_off = self._t1_perf - self._t0_perf
                tw = [round(x - self._t0_perf, 3) for x in tw_all
                      if 0.0 <= x - self._t0_perf <= t1_off]
                diag["getimg_timeouts"] = int(getattr(cam, "grab_timeouts", 0))
                diag["timeout_at_s"] = tw[-50:]   # 停录前的超时时刻（最多 50 条）
            feed_diag[str(cid)] = diag
            np.save(os.path.join(self.session_dir, f"cam{cid}_ts.npy"), ts)
        self._writers = {}

        # 时长口径：duration_s 是从 start() 到 stop() 返回（含 4 路编码线程排空 backlog +
        # mp4 落盘 + meta 写入，实测 ~1-2s）——不是真实录制长度！真实长度看 capture_s
        # （用户按停瞬间 − 开始），否则会把收尾耗时误读成「录到一半停供了」。
        wall_s = time.time() - self._t0                       # 含收尾，向后兼容保留
        capture_s = self._t1_perf - self._t0_perf             # 真实录制长度（按停前）
        meta = {
            "session_dir": self.session_dir,
            "started_wall": self._t0,
            "duration_s": round(wall_s, 3),
            "capture_s": round(max(capture_s, 0.0), 3),       # 真实录制长度（对账用）
            "stop_overhead_s": round(max(wall_s - capture_s, 0.0), 3),
            "fps": (self.fps if self.fps is not None else 100.0),
            "camera_serials": {str(c.logical_id): c.serial for c in self.cameras},
            "frames_per_cam": frames_per_cam,
            # 诊断 1：喂进来 vs 写进文件 vs 编码丢帧——差多少一眼看出瓶颈在哪侧
            "fed_per_cam": fed_per_cam,
            "encoder_dropped_per_cam": dropped_per_cam,
            "measured_period_s": periods,
            # 诊断 2：墙钟侧供帧停顿（区分「进程冻结」vs「相机/总线停供」，见下方打印）
            "feed_diag_per_cam": feed_diag,
        }
        with open(os.path.join(self.session_dir, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, ensure_ascii=False, indent=2)
        print(f"[录像] ■ 停止：录制 {capture_s:.1f}s"
              f"（收尾 {max(wall_s - capture_s, 0.0):.1f}s）")
        for cid in sorted(frames_per_cam):
            written = frames_per_cam[cid]
            fed = fed_per_cam[cid]
            dropped = dropped_per_cam[cid]
            fps_est = written / capture_s if capture_s > 0 else 0.0
            print(f"  cam{cid}: 喂 {fed} → 写 {written} 帧（编码丢 {dropped}）"
                  f"· 实测 {fps_est:5.1f} fps / 100 目标")
            d = feed_diag.get(str(cid), {})
            silent = d.get("silence_after_last_s", 0.0)
            # 中段停顿（n_gaps>0）很敏感；尾部静默要 >0.5s 才算异常（用户按停本身
            # 有零点几秒的正常延迟，20260903 那种 1.35s 不会漏）
            if d.get("n_gaps", 0) or silent > 0.5:
                # 静默区 = 各停顿间隙 [s,e] + 末帧后的尾部 [last_feed, t1]
                regions = [(s, e) for s, e, _ in d.get("gaps", [])]
                last = d.get("last_feed_s", 0.0)
                if silent > 0.5:
                    regions.append((last, round(self._t1_perf - self._t0_perf, 3)))
                n_tmo_in_gap = sum(
                    1 for t in d.get("timeout_at_s", [])
                    if any(s - 0.02 <= t <= e + 0.02 for s, e in regions))
                side = ("相机/总线停供（抓帧线程在超时轮询，但相机没把帧送上来）"
                        if n_tmo_in_gap > 0
                        else "进程冻结（静默区里抓帧线程一次都没来取帧，疑似 GIL/阻塞调用）")
                print(f"      ⚠ 供帧 {d.get('span_s', 0):.2f}s 后静默 {silent:.2f}s"
                      f"（{d.get('n_gaps', 0)} 处停顿，最大 {d.get('max_gap_s', 0):.2f}s）"
                      f"· 静默区内 GetImageBuffer 超时 {n_tmo_in_gap} 次 → {side}")
        print(f"        → {self.session_dir}")
        return meta
