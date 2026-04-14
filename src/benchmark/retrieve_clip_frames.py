# -*- coding: utf-8 -*-
"""检索导出的 GOP mp4：按序读入全部帧，供 Benchmark / Motivation 与 Reasoner 对齐。"""

import logging
import os
from typing import Any, Dict, List, Optional


def count_existing_clip_mp4s(clip_info):
    # type: (Optional[Dict[str, Any]]) -> int
    """paths 中 path 存在且为文件的条目数（与解码顺序一致）。"""
    if not clip_info or not isinstance(clip_info, dict):
        return 0
    return len(_sorted_clip_path_entries(clip_info))


def _sorted_clip_path_entries(clip_info):
    # type: (Dict[str, Any]) -> List[Dict[str, Any]]
    """paths 项含 gop_start、gop_end、path；按 GOP 起点排序，且文件存在。"""
    paths = clip_info.get("paths") or []
    items = []
    for p in paths:
        if not isinstance(p, dict):
            continue
        path = p.get("path")
        if not path or not os.path.isfile(path):
            continue
        items.append(p)
    items.sort(
        key=lambda x: (int(x.get("gop_start", 0)), int(x.get("gop_end", 0)))
    )
    return items


def decode_all_frames_bgr_from_mp4(video_path):
    # type: (str) -> List[Any]
    """顺序读取 mp4 中每一帧，BGR uint8 ndarray。"""
    import cv2

    out = []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return out
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            out.append(frame)
    finally:
        cap.release()
    return out


def decode_clip_info_all_frames_bgr(clip_info, logger=None):
    # type: (Optional[Dict[str, Any]], Optional[logging.Logger]) -> List[Any]
    """
    按 GOP 顺序拼接多个检索 mp4 的全部帧（BGR）。
    clip_info 形如 MemoryManager 返回的 {"paths": [{"gop_start", "gop_end", "path"}, ...]}。

    若传入 ``logger``，在交给大模型解码路径上对每个 clip 打一条 INFO：
    文件名 / GOP 区间 / 该 clip 帧数。
    """
    if not clip_info or not isinstance(clip_info, dict):
        return []
    out = []
    entries = _sorted_clip_path_entries(clip_info)
    if logger is not None and entries:
        logger.info("检索 clip 共 %d 个，开始按文件解码帧数", len(entries))
    for ent in entries:
        path = ent.get("path")
        if not path:
            continue
        part = decode_all_frames_bgr_from_mp4(str(path))
        if logger is not None:
            gs = ent.get("gop_start")
            ge = ent.get("gop_end")
            logger.info(
                "检索 clip: file=%s gop=[%s,%s) frames=%d",
                os.path.basename(str(path)),
                gs,
                ge,
                len(part),
            )
        out.extend(part)
    if logger is not None and entries:
        logger.info("检索 clip 解码合计帧数=%d（将送 Reasoner）", len(out))
    return out


def count_reported_frames_in_clip_info(clip_info):
    # type: (Optional[Dict[str, Any]]) -> int
    """
    不逐帧解码，用 OpenCV 的帧数统计各 clip 长度并求和（供 Motivation 等轻量统计）。
    """
    if not clip_info or not isinstance(clip_info, dict):
        return 0
    import cv2

    total = 0
    for ent in _sorted_clip_path_entries(clip_info):
        path = ent.get("path")
        if not path:
            continue
        cap = cv2.VideoCapture(str(path))
        if cap.isOpened():
            try:
                total += int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            finally:
                cap.release()
    return max(0, total)
