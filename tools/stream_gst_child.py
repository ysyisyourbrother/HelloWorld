#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
流解码 GStreamer 子进程：由主进程以带 gi 的 Python（如 /usr/bin/python3）拉起，从 stdin 读一行 JSON 配置，
将解码后的 RGB 帧以二进制格式写入 stdout（见 stream_input 与 stream_gst_runner.FRAME_MAGIC）。

主进程（如 conda）无需安装 PyGObject。
"""

from __future__ import absolute_import, division, print_function

import argparse
import json
import os
import struct
import sys


class _ChildOwner(object):
    """满足 run_stream_pipeline 对 owner 的最小约定。"""

    def __init__(self):
        self._main_loop = None
        self._pipeline = None
        self._gst_error = None
        self._exit_reason = None

    def _finalize_stream_window_on_eos(self):
        pass


def parse_args():
    p = argparse.ArgumentParser(description="GStreamer 流解码（子进程专用）")
    p.add_argument(
        "--config-json",
        help="若省略则从 stdin 读一行 JSON（与 --config-json 内容相同）",
    )
    return p.parse_args()


def _frame_header_struct():
    # magic, w, h, frame_idx (Q), pkt_size, is_key, pad(3), rgb_len -> 32 bytes
    return struct.Struct("<IIIQIB3xI")


if __name__ == "__main__":
    _args = parse_args()
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _root not in sys.path:
        sys.path.insert(0, _root)

    from src.video_input.stream_gst_runner import FRAME_MAGIC, run_stream_pipeline

    if _args.config_json:
        _cfg = json.loads(_args.config_json)
    else:
        _line = sys.stdin.readline()
        if not _line:
            sys.stderr.write("stream_gst_child: 无配置输入\n")
            sys.exit(2)
        _cfg = json.loads(_line)

    _hdr = _frame_header_struct()
    _out = sys.stdout.buffer
    _owner = _ChildOwner()

    def _fmt(msg, a):
        if not a:
            return msg
        try:
            return msg % a
        except Exception:
            return "%s %s" % (msg, a)

    class _Log(object):
        def info(self, msg, *a):
            sys.stderr.write(_fmt(msg, a) + "\n")
            sys.stderr.flush()

        def error(self, msg, *a):
            sys.stderr.write(_fmt(msg, a) + "\n")
            sys.stderr.flush()

        def warning(self, msg, *a):
            sys.stderr.write(_fmt(msg, a) + "\n")
            sys.stderr.flush()

        def debug(self, msg, *a):
            pass

    def _on_rgb_frame(frame_idx, pkt_size, is_key, rgb_bytes, w, h):
        n = len(rgb_bytes)
        _out.write(
            _hdr.pack(
                FRAME_MAGIC,
                int(w),
                int(h),
                int(frame_idx),
                int(pkt_size) & 0xFFFFFFFF,
                1 if is_key else 0,
                n,
            )
        )
        _out.write(rgb_bytes)
        _out.flush()

    try:
        run_stream_pipeline(_cfg, _on_rgb_frame, _Log(), _owner)
    except Exception as _e:
        sys.stderr.write("stream_gst_child: %s\n" % _e)
        sys.stderr.flush()
        sys.exit(1)

    if _owner._gst_error:
        sys.stderr.write("stream_gst_child: %s\n" % _owner._gst_error)
        sys.stderr.flush()
        sys.exit(1)

    _reason = getattr(_owner, "_exit_reason", None) or "gst_done"
    sys.stderr.write("STREAM_GST_EXIT_REASON %s\n" % _reason)
    sys.stderr.flush()
