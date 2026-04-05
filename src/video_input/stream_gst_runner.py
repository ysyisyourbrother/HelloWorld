# -*- coding: utf-8 -*-
"""
GStreamer 流解码管线（需 PyGObject）。供 StreamVideoInput 进程内调用，或由 tools/stream_gst_child.py 在系统 Python 子进程中运行。

on_rgb_frame(frame_idx, pkt_size, is_key, rgb_bytes, width, height)：每解码一帧调用一次；rgb_bytes 长度为 width*height*3。
"""

from __future__ import absolute_import, division, print_function

import os
import threading
import time
from collections import deque
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.video_utils.gst_frame_info import (
    _demuxer_name_for_path,
    _make_video_parser,
    _resolve_pad_caps,
)

# 子进程 stdout 二进制帧头（与 stream_input._FRAME_MAGIC 一致）
FRAME_MAGIC = 0x53544D31  # b"STM1"


def normalize_stream_location(uri):
    # type: (str) -> Tuple[str, str]
    if not uri:
        raise ValueError("stream_uri / video_file_path 未配置")
    if uri.startswith("rtsp://"):
        return uri, "rtsp"
    if uri.startswith("file://"):
        path = uri.replace("file://", "")
        if os.name == "nt" and path.startswith("/"):
            path = path[1:]
        path = os.path.abspath(path)
        return path, "file"
    path = os.path.abspath(uri)
    return path, "file"


def decoder_for_caps(caps, Gst):
    st = caps.get_structure(0)
    name = st.get_name()
    if name in ("video/x-h264", "video/x-avc"):
        return Gst.ElementFactory.make("avdec_h264", None)
    if name in ("video/x-h265", "video/hevc"):
        return Gst.ElementFactory.make("avdec_h265", None)
    return None


def build_file_pipeline(path, Gst, GstApp, probe_cb):
    # type: (str, Any, Any, str, Any, Any) -> Tuple[Any, Any, Dict[str, Any]]
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError("视频文件不存在: %s" % path)

    pipeline = Gst.Pipeline.new("stream-file")
    src = Gst.ElementFactory.make("filesrc", "src")
    demux = Gst.ElementFactory.make(_demuxer_name_for_path(path), "demux")
    q_el = Gst.ElementFactory.make("queue", "q")
    sink = Gst.ElementFactory.make("appsink", "sink")
    if not all((pipeline, src, demux, q_el, sink)):
        raise RuntimeError("无法创建 GStreamer 基础元素")

    src.set_property("location", path)
    sink.set_property("emit-signals", True)
    sink.set_property("sync", False)
    sink.set_property("max-buffers", 2)
    try:
        sink.set_property("drop", False)
    except Exception:
        pass
    caps_rgb = Gst.Caps.from_string("video/x-raw,format=RGB")
    sink.set_property("caps", caps_rgb)

    pipeline.add(src)
    pipeline.add(demux)
    pipeline.add(q_el)
    pipeline.add(sink)

    if not src.link(demux):
        raise RuntimeError("filesrc -> demux 链接失败")

    state = {"linked": False, "error": None, "drop_idx": 0, "parser": None}

    def _add_drop_sink(pad):
        fake = Gst.ElementFactory.make("fakesink", "drop_%d" % state["drop_idx"])
        state["drop_idx"] += 1
        if fake is None:
            return False
        fake.set_property("sync", False)
        try:
            fake.set_property("async", False)
        except Exception:
            pass
        pipeline.add(fake)
        fake.sync_state_with_parent()
        sp = fake.get_static_pad("sink")
        if pad.link(sp) != Gst.PadLinkReturn.OK:
            return False
        return True

    def _try_link_video_pad(pad):
        caps = _resolve_pad_caps(pad)
        if caps is None:
            return
        st = caps.get_structure(0)
        name = st.get_name()
        if state["linked"]:
            if name.startswith("video/"):
                _add_drop_sink(pad)
            return
        if not name.startswith("video/"):
            _add_drop_sink(pad)
            return
        parser = _make_video_parser(caps)
        if parser is None:
            state["error"] = "不支持的视频编码: %s" % name
            return
        dec = decoder_for_caps(caps, Gst)
        if dec is None:
            state["error"] = "无法创建解码器: %s" % name
            return
        videoconvert = Gst.ElementFactory.make("videoconvert", None)
        if videoconvert is None:
            state["error"] = "无法创建 videoconvert"
            return

        pipeline.add(parser)
        pipeline.add(dec)
        pipeline.add(videoconvert)

        psink = q_el.get_static_pad("sink")
        if pad.link(psink) != Gst.PadLinkReturn.OK:
            state["error"] = "demux -> queue 链接失败"
            return
        if not q_el.link(parser):
            state["error"] = "queue -> parser 失败"
            return
        if not parser.link(dec):
            state["error"] = "parser -> decoder 失败"
            return
        if not dec.link(videoconvert):
            state["error"] = "decoder -> videoconvert 失败"
            return
        if not videoconvert.link(sink):
            state["error"] = "videoconvert -> appsink 失败"
            return

        psrc = parser.get_static_pad("src")
        if psrc is not None:
            psrc.add_probe(Gst.PadProbeType.BUFFER, probe_cb, None)

        parser.sync_state_with_parent()
        dec.sync_state_with_parent()
        videoconvert.sync_state_with_parent()
        q_el.sync_state_with_parent()
        sink.sync_state_with_parent()
        state["linked"] = True

    def on_notify_caps(pad, _pspec):
        if pad.is_linked():
            return
        _try_link_video_pad(pad)

    def on_pad_added(_element, pad):
        if pad.is_linked():
            return
        caps = _resolve_pad_caps(pad)
        if caps is None or caps.is_empty():
            pad.connect("notify::caps", on_notify_caps)
            return
        _try_link_video_pad(pad)

    demux.connect("pad-added", on_pad_added)

    return pipeline, sink, state


def build_rtsp_pipeline(uri, Gst, GstApp, stream_rtsp_depay):
    dep = stream_rtsp_depay
    if dep == "auto":
        dep = "h264"
    if dep == "h264":
        pl_str = (
            "rtspsrc name=src latency=80 ! rtph264depay ! h264parse name=p ! "
            "avdec_h264 ! videoconvert ! appsink name=rgb"
        )
    elif dep == "h265":
        pl_str = (
            "rtspsrc name=src latency=80 ! rtph265depay ! h265parse name=p ! "
            "avdec_h265 ! videoconvert ! appsink name=rgb"
        )
    else:
        raise ValueError("stream_rtsp_depay 须为 auto|h264|h265，当前: %s" % stream_rtsp_depay)

    pipeline = Gst.parse_launch(pl_str)
    src = pipeline.get_by_name("src")
    src.set_property("location", uri)
    p = pipeline.get_by_name("p")
    sink = pipeline.get_by_name("rgb")
    sink.set_property("emit-signals", True)
    sink.set_property("sync", False)
    sink.set_property("max-buffers", 2)
    try:
        sink.set_property("drop", True)
    except Exception:
        pass
    caps_rgb = Gst.Caps.from_string("video/x-raw,format=RGB")
    sink.set_property("caps", caps_rgb)

    return pipeline, sink, p


def run_stream_pipeline(
    cfg,
    on_rgb_frame,
    log,
    owner,
):
    # type: (Dict[str, Any], Callable[..., None], Any, Any) -> None
    """
    cfg: stream_uri, stream_rtsp_depay, record_max_seconds（秒，0 表示不限制）
    log: 具备 info/error/warning/debug 方法的对象（如 logging.Logger）
    owner: StreamVideoInput 实例；设置 owner._main_loop / owner._pipeline 供 stop() 使用
    """
    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GstApp", "1.0")
    from gi.repository import GLib, Gst, GstApp

    stream_uri = cfg["stream_uri"]
    stream_rtsp_depay = str(cfg.get("stream_rtsp_depay", "auto")).lower()
    record_max_seconds = float(cfg.get("record_max_seconds", 0.0) or 0.0)

    Gst.init(None)
    main_loop = GLib.MainLoop()
    loc, kind = normalize_stream_location(stream_uri)

    meta_deque = deque()  # type: deque
    meta_lock = threading.Lock()
    global_frame_idx = [0]

    def probe_cb(pad, info, _user_data):
        buf = info.get_buffer()
        if buf is None:
            return Gst.PadProbeReturn.OK
        size = buf.get_size()
        is_delta = buf.has_flags(Gst.BufferFlags.DELTA_UNIT)
        is_key = not is_delta
        with meta_lock:
            meta_deque.append((size, is_key))
        return Gst.PadProbeReturn.OK

    def on_new_sample(sink):
        sample = None
        ps = getattr(sink, "pull_sample", None)
        if callable(ps):
            sample = ps()
        else:
            cast = getattr(GstApp.AppSink, "cast", None)
            if callable(cast):
                sample = cast(sink).pull_sample()
        if sample is None:
            return Gst.FlowReturn.OK
        buf = sample.get_buffer()
        caps = sample.get_caps()
        if buf is None or caps is None:
            return Gst.FlowReturn.OK
        st = caps.get_structure(0)
        w = st.get_int("width")[1]
        h = st.get_int("height")[1]
        size = buf.get_size()
        expected = w * h * 3
        if size < expected:
            log.warning("缓冲区小于预期 RGB: got %d expect>=%d", size, expected)
        data = buf.extract_dup(0, min(size, expected))

        meta = None
        with meta_lock:
            if meta_deque:
                meta = meta_deque.popleft()
        if meta is None:
            log.warning("元数据队列为空，跳过一帧解码")
            return Gst.FlowReturn.OK
        pkt_size, is_key = meta
        idx = global_frame_idx[0]
        global_frame_idx[0] += 1
        on_rgb_frame(idx, pkt_size, is_key, data, w, h)
        return Gst.FlowReturn.OK

    if kind == "rtsp":
        pipeline, appsink_elem, parse_elem = build_rtsp_pipeline(
            loc, Gst, GstApp, stream_rtsp_depay
        )
        state = None
    else:
        pipeline, appsink_elem, state = build_file_pipeline(
            loc, Gst, GstApp, probe_cb
        )
        parse_elem = None

    owner._pipeline = pipeline
    owner._main_loop = main_loop
    appsink_elem.connect("new-sample", lambda s: on_new_sample(s))

    bus = pipeline.get_bus()
    bus.add_signal_watch()

    def on_bus(_bus, msg):
        if msg.type == Gst.MessageType.ERROR:
            err, dbg = msg.parse_error()
            owner._gst_error = str(err)
            if dbg:
                owner._gst_error += " (%s)" % dbg
            log.error("GStreamer ERROR: %s", owner._gst_error)
            owner._exit_reason = "error"
            main_loop.quit()
        elif msg.type == Gst.MessageType.EOS:
            log.info("GStreamer EOS")
            owner._exit_reason = "eos"
            _eos_finalize = getattr(owner, "_finalize_stream_window_on_eos", None)
            if callable(_eos_finalize):
                try:
                    _eos_finalize()
                except Exception:
                    pass
            main_loop.quit()
        elif msg.type == Gst.MessageType.WARNING:
            pass
        return True

    bus.connect("message", on_bus)

    max_sec = int(record_max_seconds) if record_max_seconds > 0 else 0
    if max_sec > 0:

        def _on_decode_max_time():
            owner._exit_reason = "max_time"
            log.info("已达 stream_record_max_seconds=%d，结束解码管线", max_sec)
            main_loop.quit()
            return False

        GLib.timeout_add_seconds(max_sec, _on_decode_max_time)

    if kind == "rtsp":
        ret = pipeline.set_state(Gst.State.PAUSED)
        if ret == Gst.StateChangeReturn.FAILURE:
            pipeline.set_state(Gst.State.NULL)
            owner._gst_error = "RTSP 管线无法进入 PAUSED"
            log.error(owner._gst_error)
            owner._main_loop = None
            owner._pipeline = None
            return
        pipeline.get_state(5 * Gst.SECOND)
        pe = parse_elem
        psrc = pe.get_static_pad("src") if pe is not None else None
        if psrc is None:
            pipeline.set_state(Gst.State.NULL)
            owner._gst_error = "无法取得 RTSP parser src pad"
            log.error(owner._gst_error)
            owner._main_loop = None
            owner._pipeline = None
            return
        psrc.add_probe(Gst.PadProbeType.BUFFER, probe_cb, None)

    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        pipeline.set_state(Gst.State.NULL)
        owner._gst_error = "管线无法进入 PLAYING"
        log.error(owner._gst_error)
        owner._main_loop = None
        owner._pipeline = None
        return

    if kind != "rtsp":
        for _ in range(400):
            if state.get("linked") or state.get("error"):
                break
            time.sleep(0.05)
        if state.get("error"):
            pipeline.set_state(Gst.State.NULL)
            log.error(state["error"])
            owner._main_loop = None
            owner._pipeline = None
            return
        if not state.get("linked"):
            pipeline.set_state(Gst.State.NULL)
            owner._gst_error = "超时未链接视频轨"
            log.error(owner._gst_error)
            owner._main_loop = None
            owner._pipeline = None
            return

    try:
        main_loop.run()
    finally:
        pipeline.set_state(Gst.State.NULL)
        if getattr(owner, "_exit_reason", None) is None:
            owner._exit_reason = "gst_done"
        owner._main_loop = None
        owner._pipeline = None
