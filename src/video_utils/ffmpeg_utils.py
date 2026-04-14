# -*- coding: utf-8 -*-
"""ffmpeg 封装：按 GOP（I 帧边界）无重编码导出临时 mp4。"""

import bisect
import hashlib
import os
import shutil
import subprocess
import tempfile
from typing import Iterable, List, Optional, Sequence, Tuple

# Python 3.6 兼容


def gop_range_containing_frame(
    i_frames,
    frame_id,
    total_frames,
):
    # type: (Sequence[int], int, Optional[int]) -> Optional[Tuple[int, int]]
    """
    与 SymVideoInputByGOP.iter_gop_ranges 一致：第 k 段为 [i_frames[k], i_frames[k+1])，
    最后一段右端为 total_frames。

    若 frame_id 落在某段内则返回 (gop_start, gop_end)；否则返回 None
    （例如 frame_id 早于第一个 I 帧，且第一个 I 非 0）。
    """
    if not i_frames or total_frames is None:
        return None
    try:
        tf = int(total_frames)
        fid = int(frame_id)
    except (TypeError, ValueError):
        return None
    if tf <= 0:
        return None
    ixs = sorted(int(x) for x in i_frames)
    if not ixs:
        return None
    k = bisect.bisect_right(ixs, fid) - 1
    if k < 0:
        return None
    start = ixs[k]
    end = ixs[k + 1] if k + 1 < len(ixs) else tf
    if fid < start or fid >= end:
        return None
    if start >= end:
        return None
    return start, end


def unique_gop_ranges_for_frame_ids(i_frames, frame_ids, total_frames):
    # type: (Sequence[int], Iterable[int], Optional[int]) -> List[Tuple[int, int]]
    """对多个 frame_id 求所在 GOP，按 (start,end) 去重后按 start 排序。"""
    seen = set()
    out = []
    for fid in frame_ids:
        r = gop_range_containing_frame(i_frames, fid, total_frames)
        if r is None:
            continue
        if r not in seen:
            seen.add(r)
            out.append(r)
    out.sort(key=lambda x: (x[0], x[1]))
    return out


def export_gop_range_copy_mp4(
    video_path,
    gop_start,
    gop_end,
    video_fps,
    output_path,
):
    # type: (str, int, int, float, str) -> str
    """
    使用 ffmpeg 流复制截取 [gop_start, gop_end) 对应时间区间到 output_path（无视频重编码）。

    时间由帧索引 / fps 换算，与项目中按帧索引建库的方式一致。
    """
    if not video_path or not os.path.isfile(video_path):
        raise FileNotFoundError("视频文件不存在: %s" % (video_path,))
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise OSError("未找到 ffmpeg 可执行文件，请安装 ffmpeg 并加入 PATH")

    fps = float(video_fps) if video_fps else 0.0
    if fps <= 0:
        raise ValueError("video_fps 必须为正数，当前为 %s" % (video_fps,))

    start = int(gop_start)
    end = max(start + 1, int(gop_end) - 1)
    if start < 0 or end <= start:
        raise ValueError("无效的 GOP 区间: [%s, %s)" % (start, end))

    ss = start / fps
    dur = (end - start) / fps
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # cmd = [
    #     ffmpeg,
    #     "-hide_banner",
    #     "-loglevel", "error",
    #     "-y",
    #     "-ss", "%.6f" % ss,
    #     "-i", video_path,
    #     "-t", "%.6f" % dur,
    #     "-c", "copy",
    #     "-avoid_negative_ts", "make_zero",
    #     output_path,
    # ]

    # cmd = [
    #     ffmpeg,
    #     "-i", video_path,          # -ss 放在 -i 之后，或者不用 -ss
    #     "-vf", f"trim=start={ss}:duration={dur},setpts=PTS-STARTPTS", # 使用滤镜裁剪
    #     "-c:v", "libx264",         # 必须重编码，因为切断了 GOP 依赖
    #     "-c:a", "aac",             # 音频也要处理
    #     "-strict", "experimental",
    #     output_path
    # ]

    cmd = [
        ffmpeg,
        "-i", video_path,          # -ss 放在 -i 之后，或者不用 -ss
        "-vf", f"select='between(n,{start},{end})',setpts=N/({fps}*TB)", # 使用滤镜裁剪
        "-af", f"atrim=start={start/fps}:end={end/fps},asetpts=PTS-STARTPTS",
        "-shortest",
        "-c:v", "libx264",         # 必须重编码，因为切断了 GOP 依赖
        "-c:a", "aac",             # 音频也要处理
        output_path
    ]
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    if p.returncode != 0:
        err = (p.stderr or "").strip()
        raise RuntimeError(
            "ffmpeg 导出 GOP 失败 (code=%s): %s" % (p.returncode, err or p.stdout)
        )
    return output_path


def export_unique_gop_mp4s_for_source(
    video_path,
    i_frames,
    frame_ids,
    video_fps,
    total_frames,
    output_dir=None,
    prefix="gop",
):
    # type: (str, Sequence[int], Sequence[int], float, Optional[int], Optional[str], str) -> List[Tuple[Tuple[int, int], str]]
    """
    根据若干 frame_id 去重得到 GOP，将每个 GOP 导出为独立临时 mp4。

    Returns:
        [((gop_start, gop_end), mp4_path), ...]
    """
    if not i_frames:
        return []
    ranges = unique_gop_ranges_for_frame_ids(i_frames, frame_ids, total_frames)
    if not ranges:
        return []

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="helloworld_gop_")
    else:
        os.makedirs(output_dir, exist_ok=True)

    results = []
    for i, (gs, ge) in enumerate(ranges):
        name = "%s_%d_%d_%d.mp4" % (prefix, i, gs, ge)
        out_path = os.path.join(output_dir, name)
        export_gop_range_copy_mp4(video_path, gs, ge, video_fps, out_path)
        results.append(((gs, ge), out_path))
    return results


def export_unique_gop_mp4s_from_memory_records(records, output_dir=None, prefix="gop"):
    # type: (Sequence[dict], Optional[str], str) -> List[Tuple[Tuple[int, int], str]]
    """
    根据检索元数据列表（每项含 source_path, frame_id, video_fps, total_frames, i_frames）
    按路径分组，对每组去重 GOP 并导出。

    兼容：缺少 i_frames 或为空、total_frames 缺失的记录会被跳过。
    """
    from collections import defaultdict

    by_path = defaultdict(list)
    meta = {}
    for rec in records:
        if not isinstance(rec, dict):
            continue
        path = rec.get("source_path") or ""
        if not path or not os.path.isfile(path):
            continue
        ixs = rec.get("i_frames")
        if not ixs:
            continue
        fid = rec.get("frame_id")
        if fid is None:
            continue
        fps = rec.get("video_fps") or 0.0
        tf = rec.get("total_frames")
        by_path[path].append(int(fid))
        meta[path] = (ixs, float(fps), tf)

    out_all = []
    for path, fids in by_path.items():
        ixs, fps, tf = meta[path]
        tag = hashlib.md5(path.encode("utf-8")).hexdigest()[:8]
        base = os.path.splitext(os.path.basename(path))[0]
        safe_base = "".join(c if c.isalnum() else "_" for c in base)[:24]
        pfx = "%s_%s_%s" % (prefix, tag, safe_base)
        part = export_unique_gop_mp4s_for_source(
            path, ixs, fids, fps, tf, output_dir=output_dir, prefix=pfx
        )
        out_all.extend(part)
    return out_all
