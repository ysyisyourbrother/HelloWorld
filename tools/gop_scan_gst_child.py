#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GStreamer GOP 扫描子进程入口：由主进程以「带 gi 的 Python」拉起，标准输出仅一行 JSON。

主进程（如 conda）无需安装 PyGObject；本脚本需系统 Python（如 /usr/bin/python3）且已安装 gi + GStreamer。
"""

from __future__ import absolute_import, division, print_function

import argparse
import json
import os
import sys


def parse_args():
    p = argparse.ArgumentParser(description="GStreamer GOP 扫描（子进程专用）")
    p.add_argument("video_path", help="视频文件绝对或相对路径")
    return p.parse_args()


if __name__ == "__main__":
    _args = parse_args()
    _root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _root not in sys.path:
        sys.path.insert(0, _root)

    try:
        from src.video_utils.gst_frame_info import get_frame_info_for_stream_gst

        _i, _t, _p = get_frame_info_for_stream_gst(_args.video_path)
        _payload = {
            "i_frame_indices": _i,
            "frame_types": _t,
            "pkt_sizes": _p,
        }
        sys.stdout.write(json.dumps(_payload, separators=(",", ":")))
        sys.stdout.flush()
    except Exception as _e:
        sys.stderr.write("gop_scan_gst_child: %s\n" % _e)
        sys.stderr.flush()
        sys.exit(1)
