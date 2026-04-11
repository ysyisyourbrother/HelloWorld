# -*- coding: utf-8 -*-
"""
在线流视频输入：GStreamer 拉流 + 非 I 帧平均包大小相对历史抬升触发，
在触发窗口内按包大小累积曲线采样若干帧解码，将 RGB 帧封装为 FrameData 入队（供 Memory 注入线程消费）。

可选录制（RTSP）：独立 GStreamer 管线 rtspsrc → depay → parse → splitmuxsink，按时间分段写 MP4，
与 test_rtsp/rtsp_segment_record.py 相同：解封装复用、不重编码、不经 YUV，无需 ffmpeg/OpenCV。

stream_record_max_seconds 在主解码管线上用 GLib 计时，到时结束会话；退出时段内 segment_*.mp4 已全部落盘。

依赖：GStreamer（含 gst-plugins-bad 的 splitmuxsink）、PyGObject（gi）。
若当前解释器无 gi（如 conda 环境），解码会自动改用子进程 tools/stream_gst_child.py，
解释器路径同 GOP 扫描：config video_input.gop_scan_python 或环境变量 GST_GOP_SCAN_PYTHON，默认 /usr/bin/python3。

支持：
- 本地文件路径或 file:// URI（与 gst_frame_info 类似的 demux+parse+解码链路）
- RTSP（H.264 或 H.265，由 config stream_rtsp_depay 指定）
"""

from __future__ import absolute_import, division, print_function

import glob
import json
import logging
import multiprocessing as mp
import os
import queue
import struct
import subprocess
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from src.config import Config
from src.video_input.video_input import FrameData
from src.video_input.stream_gst_runner import (
    FRAME_MAGIC,
    normalize_stream_location,
    run_stream_pipeline,
)
from src.video_utils.gst_frame_info import (
    _project_root,
    resolve_gop_scan_python_exe,
)
from src.video_utils.stream_window_policy import select_frames_for_window


def _can_import_gi():
    try:
        import gi
        gi.require_version("Gst", "1.0")
        return True
    except Exception:
        return False


class StreamVideoInput(object):
    """
    在线流输入：与 VideoInputOnline 类似提供 frame_queue / start / stop / get_frame_queue，
    内部用 GStreamer 线程解码；仅将策略选中的帧放入队列（Memory 侧建议关闭 FrameVectorizer 的二次间隔筛选）。
    """

    def __init__(self, config=None):
        if config is None:
            config = Config()
        self._config = config
        self.log_file = getattr(config, "stream_log_file", "logs/stream_input.log")
        self.stream_uri = getattr(config, "stream_uri", "") or getattr(
            config, "video_file_path", ""
        )
        self.window_duration_sec = float(
            getattr(config, "stream_window_duration_sec", 1.0)
        )
        self.target_decode_fps = float(
            getattr(config, "stream_target_decode_fps", 2.0)
        )
        self.trigger_ratio = float(getattr(config, "stream_trigger_ratio", 1.35))
        self.baseline_ewma_alpha = float(
            getattr(config, "stream_baseline_ewma_alpha", 0.08)
        )
        self.warmup_windows = int(getattr(config, "stream_warmup_windows", 3))
        qmax = int(getattr(config, "stream_queue_maxsize", 100))
        self.frame_queue = mp.Queue(maxsize=qmax)
        self.video_ipc_send_frame = getattr(config, "video_ipc_send_frame", True)

        self.stream_rtsp_depay = str(
            getattr(config, "stream_rtsp_depay", "auto")
        ).lower()

        self.logger = None
        self._set_logger()

        self.running_event = threading.Event()
        self._gst_thread = None
        self._pipeline = None
        self._main_loop = None
        self._gst_error = None

        self._baseline_non_i = None  # type: Optional[float]
        self._windows_seen = 0

        self._window_frames = []  # type: List[Dict[str, Any]]
        self._window_t0 = None  # type: Optional[float]

        # --- 分段录制（磁盘，低内存）---
        self._record_enable = bool(
            getattr(config, "stream_record_enable", False)
        )
        self._record_segment_sec = max(
            1.0,
            float(getattr(config, "stream_record_segment_minutes", 1.0)) * 60.0,
        )
        self._record_base_dir = getattr(
            config, "stream_record_dir", "logs/stream_recordings"
        )
        self._record_max_sec = float(
            getattr(config, "stream_record_max_seconds", 0.0) or 0.0
        )
        self._record_latency_ms = int(
            getattr(config, "stream_record_latency_ms", 200)
        )
        self._record_rtsp_tcp = bool(
            getattr(config, "stream_record_rtsp_tcp", False)
        )

        self._rec_session_id = None  # type: Optional[str]
        self._rec_dir = None  # type: Optional[str]
        self._splitmux_thread = None  # type: Optional[threading.Thread]
        self._record_pipeline = None  # type: Any
        self._record_loop = None  # type: Any
        self._rec_finalized = False
        self._exit_reason = None  # type: Optional[str]
        self._on_session_end = None  # type: Optional[Callable[[], None]]
        self._gst_subproc = None  # type: Optional[subprocess.Popen]
        self._gst_python_exe = resolve_gop_scan_python_exe(
            getattr(config, "video_gop_scan_python", None)
        )

    def set_on_session_end(self, callback):
        """达到 stream_record_max_seconds 时由主解码管线触发（在 Gst 线程内调用回调，回调内勿阻塞）。"""
        self._on_session_end = callback

    def get_exit_reason(self):
        return self._exit_reason

    def get_recording_dir(self):
        """RTSP 分段录制目录（内含 segment_00000.mp4 …）；未启用或未初始化则为 None。"""
        return self._rec_dir

    def get_recording_merged_path(self):
        """已废弃：splitmux 仅生成分段文件，不再做 ffmpeg 合并。请使用 get_recording_dir()。"""
        return None

    def _set_logger(self):
        log_file = self.log_file
        pattern = log_file.replace(".log", "*")
        for f in glob.glob(pattern):
            try:
                os.remove(f)
            except Exception:
                pass

        self.logger = logging.getLogger(name=self.__class__.__name__)
        self.logger.handlers.clear()
        self.logger.setLevel(logging.DEBUG)
        ch = logging.StreamHandler()
        ch.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        ch.setLevel(logging.INFO)
        self.logger.addHandler(ch)
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        from logging.handlers import RotatingFileHandler

        fh = RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5, mode="w"
        )
        fh.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        fh.setLevel(logging.DEBUG)
        self.logger.addHandler(fh)
        self.logger.propagate = False

    def _put_frame_safely(self, frame_data, timeout=None):
        try:
            if timeout is None:
                self.frame_queue.put(frame_data)
            else:
                self.frame_queue.put(frame_data, timeout=timeout)
        except Exception as e:
            self.logger.error("入队失败: %s", e)

    def _recording_init_session(self):
        if not self._record_enable:
            return
        self._rec_session_id = time.strftime("%Y%m%d_%H%M%S")
        self._rec_dir = os.path.join(
            os.path.abspath(self._record_base_dir), self._rec_session_id
        )
        os.makedirs(self._rec_dir, exist_ok=True)
        self._rec_finalized = False
        self.logger.info(
            "RTSP splitmux 录制目录: %s（按约 %.1f 分钟分段）",
            self._rec_dir,
            self._record_segment_sec / 60.0,
        )

    def _stop_splitmux_recording_safe(self):
        """从任意线程停止 splitmux 录制管线。"""
        loop = self._record_loop
        pipe = self._record_pipeline
        if loop is not None:
            try:
                import gi

                gi.require_version("Gst", "1.0")
                from gi.repository import GLib

                GLib.idle_add(loop.quit)
            except Exception:
                try:
                    loop.quit()
                except Exception:
                    pass
        if pipe is not None:
            try:
                import gi

                gi.require_version("Gst", "1.0")
                from gi.repository import Gst

                pipe.set_state(Gst.State.NULL)
            except Exception:
                pass

    def _run_splitmux_recording(self, uri):
        """
        与 test_rtsp/rtsp_segment_record.py 相同：解封装 + splitmuxsink，不重编码。
        在独立线程中运行，与主解码管线并行。
        """
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import GLib, Gst

        Gst.init(None)

        dep = self.stream_rtsp_depay
        if dep == "auto":
            dep = "h264"
        if dep == "h264":
            depay = "rtph264depay ! h264parse"
        elif dep == "h265":
            depay = "rtph265depay ! h265parse"
        else:
            self.logger.error("splitmux 录制不支持 stream_rtsp_depay=%s", dep)
            return

        latency = max(0, self._record_latency_ms)
        if self._record_rtsp_tcp:
            head = "rtspsrc name=recsrc latency=%d protocols=tcp ! " % latency
        else:
            head = "rtspsrc name=recsrc latency=%d ! " % latency
        pl_str = "%s%s ! splitmuxsink name=recmux muxer-factory=mp4mux" % (head, depay)

        try:
            pipeline = Gst.parse_launch(pl_str)
        except Exception as e:
            self.logger.error("splitmux 录制管线创建失败: %s", e)
            return

        self._record_pipeline = pipeline
        src = pipeline.get_by_name("recsrc")
        src.set_property("location", uri)
        mux = pipeline.get_by_name("recmux")
        ns = int(self._record_segment_sec * Gst.SECOND)
        mux.set_property("max_size_time", ns)
        pattern = os.path.join(os.path.abspath(self._rec_dir), "segment_%05d.mp4")
        mux.set_property("location", pattern)

        loop = GLib.MainLoop()
        self._record_loop = loop

        def on_bus(_bus, msg):
            if msg.type == Gst.MessageType.ERROR:
                err, dbg = msg.parse_error()
                self.logger.error("splitmux 录制 ERROR: %s %s", err, dbg or "")
                loop.quit()
            elif msg.type == Gst.MessageType.EOS:
                self.logger.info("splitmux 录制 EOS")
                loop.quit()
            return True

        bus = pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", on_bus)

        ret = pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            self.logger.error("splitmux 录制无法进入 PLAYING")
            pipeline.set_state(Gst.State.NULL)
            self._record_pipeline = None
            self._record_loop = None
            return

        self.logger.info("splitmux 录制已启动: %s", pattern)
        try:
            loop.run()
        finally:
            pipeline.set_state(Gst.State.NULL)
            self._record_pipeline = None
            self._record_loop = None
            self.logger.info("splitmux 录制线程已结束")

    def _finalize_recording_session(self):
        if self._rec_finalized:
            return
        if self._record_enable and self._splitmux_thread is not None:
            self._stop_splitmux_recording_safe()
            t = self._splitmux_thread
            if t is not None and t.is_alive():
                t.join(timeout=12.0)
            self._splitmux_thread = None
        self._rec_finalized = True

    def _finalize_window(self):
        rows = self._window_frames
        self._window_frames = []
        self._window_t0 = None
        if not rows:
            return

        self._windows_seen += 1
        selected, self._baseline_non_i, triggered = select_frames_for_window(
            rows,
            self.target_decode_fps,
            self.window_duration_sec,
            self._baseline_non_i,
            self.trigger_ratio,
            self.warmup_windows,
            self._windows_seen,
            self.baseline_ewma_alpha,
        )

        if triggered:
            self.logger.info(
                "窗口触发编码: 帧数=%d, 选中=%s, baseline_non_i=%.1f",
                len(rows),
                selected,
                self._baseline_non_i or -1.0,
            )
        else:
            self.logger.debug(
                "窗口未触发: 帧数=%d, windows_seen=%d",
                len(rows),
                self._windows_seen,
            )

        for idx in selected:
            if idx < 0 or idx >= len(rows):
                continue
            r = rows[idx]
            frame = r.get("rgb")
            if frame is None:
                continue
            ts = time.time()
            fd = FrameData(
                frame=frame if self.video_ipc_send_frame else None,
                timestamp=ts,
                frame_id=int(r["frame_idx"]),
                source_path=self.stream_uri,
                total_frames=None,
                video_fps=None,
                duration=None,
                trace_ts={
                    "video_extracted_at": ts,
                    "stream_window_triggered": 1.0 if triggered else 0.0,
                },
            )
            self._put_frame_safely(fd)

    def _append_decoded_frame(self, frame_idx, pkt_size, is_keyframe, rgb):
        now = time.time()
        if self._window_t0 is None:
            self._window_t0 = now
        self._window_frames.append(
            {
                "frame_idx": frame_idx,
                "pkt_size": pkt_size,
                "is_keyframe": is_keyframe,
                "rgb": rgb,
                "timestamp": now,
            }
        )
        if now - self._window_t0 >= self.window_duration_sec and self._window_frames:
            self._finalize_window()

    def _finalize_stream_window_on_eos(self):
        if self._window_frames:
            self._finalize_window()

    def _run_gst(self):
        loc, kind = normalize_stream_location(self.stream_uri)
        if self._record_enable:
            if kind == "rtsp":
                self._recording_init_session()
                self._splitmux_thread = threading.Thread(
                    target=self._run_splitmux_recording,
                    args=(loc,),
                    name="StreamSplitmuxRecord",
                    daemon=True,
                )
                self._splitmux_thread.start()
            else:
                self.logger.warning(
                    "stream_record_enable 仅对 RTSP 生效；当前为本地文件源，已跳过分段录制（splitmuxsink）"
                )

        cfg = {
            "stream_uri": self.stream_uri,
            "stream_rtsp_depay": self.stream_rtsp_depay,
            "record_max_seconds": self._record_max_sec,
        }

        def on_rgb_frame(frame_idx, pkt_size, is_key, rgb_bytes, w, h):
            arr = np.frombuffer(rgb_bytes, dtype=np.uint8)
            try:
                rgb = arr.reshape((h, w, 3))
            except ValueError:
                self.logger.warning(
                    "reshape 失败 w=%d h=%d len=%d", w, h, len(arr)
                )
                return
            self._append_decoded_frame(frame_idx, pkt_size, is_key, rgb.copy())

        try:
            run_stream_pipeline(cfg, on_rgb_frame, self.logger, self)
        finally:
            self._finalize_recording_session()
            if self._exit_reason == "max_time" and self._on_session_end is not None:
                try:
                    self._on_session_end()
                except Exception:
                    self.logger.exception("on_session_end 回调异常")

    _STREAM_FRAME_HDR = struct.Struct("<IIIQIB3xI")

    @staticmethod
    def _read_exact(stream, n):
        # type: (Any, int) -> Optional[bytes]
        buf = b""
        while len(buf) < n:
            chunk = stream.read(n - len(buf))
            if not chunk:
                return None
            buf += chunk
        return buf

    def _run_gst_subprocess(self):
        if self._record_enable:
            self.logger.warning(
                "当前 Python 无 PyGObject（gi），已用子进程解码；"
                "RTSP 分段录制（stream_record_enable）在此模式下不可用，已跳过"
            )
        root = _project_root()
        child = os.path.join(root, "tools", "stream_gst_child.py")
        if not os.path.isfile(child):
            raise FileNotFoundError("流解码子进程脚本不存在: %s" % child)

        env = os.environ.copy()
        sep = os.pathsep
        pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = root + (sep + pp if pp else "")

        cfg = {
            "stream_uri": self.stream_uri,
            "stream_rtsp_depay": self.stream_rtsp_depay,
            "record_max_seconds": self._record_max_sec,
        }
        cmd = [self._gst_python_exe, child]
        self.logger.info(
            "GStreamer 在子进程中运行（%s），主进程无需 gi",
            self._gst_python_exe,
        )
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            env=env,
        )
        self._gst_subproc = proc
        try:
            proc.stdin.write(
                (json.dumps(cfg, separators=(",", ":")) + "\n").encode("utf-8")
            )
            proc.stdin.flush()
            proc.stdin.close()
        except Exception as e:
            self._gst_error = str(e)
            self.logger.error("写入子进程配置失败: %s", e)
            try:
                proc.terminate()
            except Exception:
                pass
            return

        hdr = self._STREAM_FRAME_HDR
        hdr_size = hdr.size
        stderr_lines = []  # type: List[str]

        def _drain_stderr():
            try:
                for line in iter(proc.stderr.readline, b""):
                    if not line:
                        break
                    t = line.decode("utf-8", errors="replace").rstrip()
                    stderr_lines.append(t)
                    if t:
                        self.logger.info("[gst-child] %s", t)
            except Exception:
                pass

        t_err = threading.Thread(target=_drain_stderr, name="StreamGst-Stderr", daemon=True)
        t_err.start()

        try:
            while self.running_event.is_set():
                raw = self._read_exact(proc.stdout, hdr_size)
                if raw is None or len(raw) < hdr_size:
                    break
                magic, w, h, frame_idx, pkt_size, is_key, rgb_len = hdr.unpack(raw)
                if magic != FRAME_MAGIC:
                    self.logger.error(
                        "子进程帧头 magic 无效: 0x%x (期望 0x%x)",
                        magic,
                        FRAME_MAGIC,
                    )
                    break
                payload = self._read_exact(proc.stdout, rgb_len)
                if payload is None or len(payload) < rgb_len:
                    break
                arr = np.frombuffer(payload, dtype=np.uint8)
                try:
                    rgb = arr.reshape((h, w, 3))
                except ValueError:
                    self.logger.warning(
                        "子进程 reshape 失败 w=%d h=%d len=%d", w, h, len(arr)
                    )
                    continue
                self._append_decoded_frame(
                    int(frame_idx), int(pkt_size), bool(is_key), rgb.copy()
                )
        finally:
            self._gst_subproc = None
            if proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=5.0)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
            else:
                try:
                    proc.wait(timeout=2.0)
                except Exception:
                    pass
            t_err.join(timeout=2.0)

        if self._window_frames:
            self._finalize_window()

        _prev_exit = self._exit_reason
        if _prev_exit != "user_stop":
            for ln in reversed(stderr_lines):
                if ln.startswith("STREAM_GST_EXIT_REASON "):
                    r = ln.split(" ", 1)[1].strip()
                    if r:
                        self._exit_reason = r
                    break
            if (
                self._exit_reason == "max_time"
                and self._on_session_end is not None
            ):
                try:
                    self._on_session_end()
                except Exception:
                    self.logger.exception("on_session_end 回调异常")

        code = proc.poll()
        if code not in (0, None) and code != -15 and code != -9:
            tail = "; ".join(stderr_lines[-8:]) if stderr_lines else ""
            self._gst_error = "stream_gst_child 退出码 %s" % code
            if tail:
                self._gst_error += ": " + tail
            self.logger.error(self._gst_error)

    def _gst_thread_main(self):
        try:
            if _can_import_gi():
                self._run_gst()
            else:
                self._run_gst_subprocess()
        except Exception as e:
            self._gst_error = str(e)
            self.logger.exception("GStreamer 线程异常: %s", e)

    def start(self):
        if not self.stream_uri:
            raise RuntimeError("未配置 stream_uri（或 video_file_path）")
        self.running_event.set()
        self._gst_thread = threading.Thread(
            target=self._gst_thread_main, name="StreamVideoInput-Gst", daemon=True
        )
        self._gst_thread.start()

    def stop(self):
        if self._exit_reason is None:
            self._exit_reason = "user_stop"
        self.running_event.clear()
        if self._gst_subproc is not None:
            try:
                self._gst_subproc.terminate()
            except Exception:
                pass
        if self._main_loop is not None:
            try:
                self._main_loop.quit()
            except Exception:
                pass
        if self._gst_thread is not None:
            self._gst_thread.join(timeout=8.0)
            self._gst_thread = None
        self._finalize_recording_session()

    def get_frame(self):
        try:
            return self.frame_queue.get(timeout=1.0)
        except queue.Empty:
            return None

    def get_frame_queue(self):
        return self.frame_queue
