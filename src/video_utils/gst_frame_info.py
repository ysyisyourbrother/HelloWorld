# -*- coding: utf-8 -*-
"""
使用 GStreamer（不解码视频）从本地文件扫描 H.264/H.265 访问单元，得到 GOP 切分所需信息：
I 帧索引、每帧粗略类型（关键帧记为 I，其余记为 P）、每帧压缩大小。

不依赖 ffprobe/ffmpeg 可执行文件；需系统已安装 GStreamer 与 PyGObject（gi），Jetson 上通常已具备。

说明：非关键帧统一标记为 P，不区分 P/B；与 ffprobe 的 pict_type 在 B 帧场景下可能不一致，但满足 GOP 按 I 帧切分的需求。
"""

from __future__ import absolute_import, division, print_function

import json
import os
import subprocess
from typing import Any, Dict, List, Optional, Tuple


def _demuxer_name_for_path(video_path: str) -> str:
    ext = os.path.splitext(video_path)[1].lower()
    if ext in (".mkv", ".webm"):
        return "matroskademux"
    if ext in (".avi"):
        return "avidemux"
    return "qtdemux"


def _resolve_pad_caps(pad):
    caps = pad.get_current_caps()
    if caps is None or caps.is_empty():
        caps = pad.query_caps(None)
    if caps is None or caps.is_empty():
        return None
    return caps


def _make_video_parser(caps):
    """根据 demux 输出的 video caps 创建 h264parse 或 h265parse；不支持则返回 None。"""
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    st = caps.get_structure(0)
    name = st.get_name()
    if name in ("video/x-h264", "video/x-avc"):
        p = Gst.ElementFactory.make("h264parse", None)
        if p is not None:
            try:
                p.set_property("alignment", "au")
            except Exception:
                pass
        return p
    if name in ("video/x-h265", "video/hevc"):
        p = Gst.ElementFactory.make("h265parse", None)
        if p is not None:
            try:
                p.set_property("alignment", "au")
            except Exception:
                pass
        return p
    return None


def get_frame_info_for_stream_gst(video_path: str) -> Tuple[List[int], List[str], List[int]]:
    """
    与 ffprobe_utils.get_frame_info_for_stream 返回形状一致：
    (i_frame_indices, frame_types, pkt_sizes)

    i_frame_indices：关键帧（非 DELTA_UNIT）的 0-based 索引。
    frame_types：关键帧为 \"I\"，其余为 \"P\"（不区分 P/B）。
    pkt_sizes：每个访问单元对应 buffer 的字节数。
    """
    import gi

    gi.require_version("Gst", "1.0")
    gi.require_version("GstApp", "1.0")
    from gi.repository import GLib, Gst, GstApp

    def _appsink_pull_sample(sink):
        """兼容不同 PyGObject：有的元素可直接 pull_sample，有的需 AppSink.cast，否则 emit。"""
        ps = getattr(sink, "pull_sample", None)
        if callable(ps):
            return ps()
        cast = getattr(GstApp.AppSink, "cast", None)
        if callable(cast):
            return cast(sink).pull_sample()
        return sink.emit("pull-sample")

    path = os.path.abspath(video_path)
    if not os.path.isfile(path):
        raise FileNotFoundError("视频文件不存在: %s" % video_path)

    Gst.init(None)

    i_frame_indices = []  # type: List[int]
    frame_types = []  # type: List[str]
    pkt_sizes = []  # type: List[int]
    frame_idx = 0

    state = {
        "linked": False,
        "error": None,  # type: Optional[str]
        "drop_idx": 0,
    }  # type: Dict[str, Any]

    pipeline = Gst.Pipeline.new("gst-gop-scan")
    src = Gst.ElementFactory.make("filesrc", "src")
    demux = Gst.ElementFactory.make(_demuxer_name_for_path(path), "demux")
    queue = Gst.ElementFactory.make("queue", "q")
    sink = Gst.ElementFactory.make("appsink", "sink")
    if not all((pipeline, src, demux, queue, sink)):
        raise RuntimeError("无法创建 GStreamer 基础元素（请检查 GStreamer 安装）")

    src.set_property("location", path)
    sink.set_property("emit-signals", True)
    sink.set_property("sync", False)
    sink.set_property("max-buffers", 0)
    try:
        sink.set_property("drop", False)
    except Exception:
        pass

    pipeline.add(src)
    pipeline.add(demux)
    pipeline.add(queue)
    pipeline.add(sink)
    if not src.link(demux):
        raise RuntimeError("filesrc -> demux 链接失败")

    loop = GLib.MainLoop()

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
            # 已接入第一条视频轨；其余视频轨必须消费，否则部分 demux 会阻塞
            if name.startswith("video/"):
                _add_drop_sink(pad)
            return
        if not name.startswith("video/"):
            _add_drop_sink(pad)
            return
        parser = _make_video_parser(caps)
        if parser is None:
            state["error"] = "不支持的视频编码（需 H.264 或 H.265）: %s" % name
            loop.quit()
            return
        pipeline.add(parser)
        psink = queue.get_static_pad("sink")
        if pad.link(psink) != Gst.PadLinkReturn.OK:
            state["error"] = "demux -> queue 链接失败"
            loop.quit()
            return
        if not queue.link(parser):
            state["error"] = "queue -> parser 链接失败"
            loop.quit()
            return
        if not parser.link(sink):
            state["error"] = "parser -> appsink 链接失败"
            loop.quit()
            return
        parser.sync_state_with_parent()
        queue.sync_state_with_parent()
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

    def on_new_sample(appsink):
        nonlocal frame_idx
        sample = _appsink_pull_sample(appsink)
        if sample is None:
            return Gst.FlowReturn.OK
        buf = sample.get_buffer()
        if buf is None:
            return Gst.FlowReturn.OK
        size = buf.get_size()
        is_delta = buf.has_flags(Gst.BufferFlags.DELTA_UNIT)
        is_key = not is_delta
        pkt_sizes.append(size)
        frame_types.append("I" if is_key else "P")
        if is_key:
            i_frame_indices.append(frame_idx)
        frame_idx += 1
        return Gst.FlowReturn.OK

    sink.connect("new-sample", on_new_sample)

    def on_bus(_bus, msg):
        if msg.type == Gst.MessageType.ERROR:
            err, dbg = msg.parse_error()
            state["error"] = str(err)
            if dbg:
                state["error"] = "%s (%s)" % (state["error"], dbg)
            loop.quit()
        elif msg.type == Gst.MessageType.EOS:
            loop.quit()
        elif msg.type == Gst.MessageType.WARNING:
            pass
        return True

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_bus)

    ret = pipeline.set_state(Gst.State.PLAYING)
    if ret == Gst.StateChangeReturn.FAILURE:
        pipeline.set_state(Gst.State.NULL)
        raise RuntimeError("GStreamer 管线无法进入 PLAYING（文件格式或插件是否匹配？）")

    try:
        loop.run()
    finally:
        pipeline.set_state(Gst.State.NULL)

    if state["error"]:
        raise RuntimeError(state["error"])

    if not state["linked"]:
        raise RuntimeError("未找到可解析的视频轨（请确认容器与编码为 H.264/H.265）")

    return i_frame_indices, frame_types, pkt_sizes


def _project_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def resolve_gop_scan_python_exe(video_gop_scan_python=None):
    """
    解析用于 GOP 子进程的 Python 可解释器路径。
    优先级：video_gop_scan_python 非空 > 环境变量 GST_GOP_SCAN_PYTHON > /usr/bin/python3

    Args:
        video_gop_scan_python: 与 Config.video_gop_scan_python 相同，来自 video_input.gop_scan_python
    """
    if video_gop_scan_python:
        return str(video_gop_scan_python)
    return os.environ.get("GST_GOP_SCAN_PYTHON", "/usr/bin/python3")


def get_frame_info_for_stream_gst_subprocess(
    video_path,
    python_exe=None,
    project_root=None,
):
    """
    在独立子进程中执行 get_frame_info_for_stream_gst，主进程无需 import gi。

    子进程应使用已安装 PyGObject 与 GStreamer 的解释器（Jetson 上常为 /usr/bin/python3）。

    Args:
        video_path: 视频路径
        python_exe: 解释器路径；为 None 时用 resolve_gop_scan_python_exe()（仅环境变量与默认）
        project_root: 项目根目录；为 None 时根据本文件推断

    Returns:
        与 get_frame_info_for_stream_gst 相同：(i_frame_indices, frame_types, pkt_sizes)
    """
    path = os.path.abspath(video_path)
    if not os.path.isfile(path):
        raise FileNotFoundError("视频文件不存在: %s" % video_path)

    if project_root is None:
        project_root = _project_root()
    child = os.path.join(project_root, "tools", "gop_scan_gst_child.py")
    if not os.path.isfile(child):
        raise FileNotFoundError("GOP 扫描子进程脚本不存在: %s" % child)

    if python_exe is None:
        python_exe = resolve_gop_scan_python_exe()

    env = os.environ.copy()
    sep = os.pathsep
    pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = project_root + (sep + pp if pp else "")

    cmd = [python_exe, child, path]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "GOP 子进程失败 (exit=%s, python=%s):\nstderr:\n%s\nstdout:\n%s"
            % (proc.returncode, python_exe, proc.stderr.strip(), proc.stdout.strip())
        )
    out = proc.stdout.strip()
    if not out:
        raise RuntimeError("GOP 子进程无标准输出。stderr:\n%s" % proc.stderr.strip())
    try:
        data = json.loads(out)
    except ValueError as e:
        raise RuntimeError(
            "GOP 子进程输出非 JSON（前 500 字符）: %r\nstderr: %s"
            % (out[:500], proc.stderr)
        ) from e
    if not isinstance(data, dict):
        raise RuntimeError("GOP 子进程返回非对象 JSON: %r" % data)
    for _k in ("i_frame_indices", "frame_types", "pkt_sizes"):
        if _k not in data:
            raise RuntimeError("GOP 子进程 JSON 缺少字段 %s: %r" % (_k, data))
    return (
        data["i_frame_indices"],
        data["frame_types"],
        data["pkt_sizes"],
    )
