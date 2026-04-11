"""ffprobe 相关工具函数"""

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _parse_ffprobe_time_seconds(raw: Any) -> Optional[float]:
    if raw is None or raw == "N/A" or raw == "":
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _frame_media_time_seconds(f: Dict[str, Any]) -> Optional[float]:
    for key in ("pkt_pts_time", "best_effort_timestamp_time", "pts_time"):
        t = _parse_ffprobe_time_seconds(f.get(key))
        if t is not None:
            return t
    return None


def get_i_frame_indices(video_path: str) -> List[int]:
    """
    使用 ffprobe 获取视频中 pict_type 为 I 的帧编号（0-based 索引），以列表形式返回。

    Args:
        video_path: 视频文件路径

    Returns:
        所有 I 帧的帧编号列表（按时间顺序）

    Raises:
        FileNotFoundError: 视频文件不存在
        subprocess.CalledProcessError: ffprobe 执行失败
        ValueError: 无法解析 ffprobe 输出
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-select_streams", "v:0",
        "-show_entries", "frame=pict_type",
        str(video_path),
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        check=True,
    )

    data = json.loads(result.stdout)
    frames = data.get("frames", [])

    return [i for i, f in enumerate(frames) if f.get("pict_type") == "I"]


def get_frame_types(video_path: str) -> List[str]:
    """
    使用 ffprobe 获取视频每一帧的 pict_type（I/P/B），按显示顺序返回。

    Args:
        video_path: 视频文件路径

    Returns:
        每帧的 pict_type 列表，如 ["I", "P", "B", "P", ...]

    Raises:
        FileNotFoundError: 视频文件不存在
        subprocess.CalledProcessError: ffprobe 执行失败
        ValueError: 无法解析 ffprobe 输出
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-select_streams", "v:0",
        "-show_entries", "frame=pict_type",
        str(video_path),
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        check=True,
    )

    data = json.loads(result.stdout)
    frames = data.get("frames", [])

    return [f.get("pict_type", "?") for f in frames]

def get_pkt_size(video_path: str) -> List[int]:
    """
    使用 ffprobe 获取视频中每个 packet 的 size（字节），按顺序返回。

    Args:
        video_path: 视频文件路径

    Returns:
        每个 packet 的 size 列表（字节数）

    Raises:
        FileNotFoundError: 视频文件不存在
        subprocess.CalledProcessError: ffprobe 执行失败
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "packet=pts,time,size",
        "-of", "csv=p=0",
        str(video_path),
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        check=True,
    )

    lines = result.stdout.strip().split("\n")
    return [int(line.split(",")[2]) for line in lines if line.strip()]


def get_frame_info_with_pkt_size(video_path: str) -> List[Dict[str, Any]]:
    """
    使用 ffprobe 获取视频每一帧的 pict_type、pkt_pts_time、pkt_size，按显示顺序返回。

    Args:
        video_path: 视频文件路径

    Returns:
        每帧信息列表，每项含 pict_type、pkt_pts_time、pkt_size，以及可选的 media_time（秒，来自
        pkt_pts_time / best_effort_timestamp_time / pts_time 中首个有效字段；无则为 None）。

    Raises:
        FileNotFoundError: 视频文件不存在
        subprocess.CalledProcessError: ffprobe 执行失败
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-select_streams", "v:0",
        "-show_entries", "frame=pict_type,pkt_pts_time,pkt_size",
        str(video_path),
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        check=True,
    )

    data = json.loads(result.stdout)
    frames = data.get("frames", [])

    rows = []
    for f in frames:
        mt = _frame_media_time_seconds(f)
        pkt_pts = f.get("pkt_pts_time")
        try:
            pkt_pts_f = float(pkt_pts) if pkt_pts not in (None, "N/A", "") else 0.0
        except (TypeError, ValueError):
            pkt_pts_f = 0.0
        rows.append(
            {
                "pict_type": f.get("pict_type", "?"),
                "pkt_pts_time": pkt_pts_f,
                "pkt_size": int(f.get("pkt_size", 0)),
                "media_time": mt,
            }
        )
    return rows


def get_frame_info_for_stream_ffprobe(video_path: str) -> Tuple[List[int], List[str], List[int]]:
    """
    一次 ffprobe 调用返回 I 帧索引、帧类型、每帧 pkt_size。
    等价于 (get_i_frame_indices, get_frame_types, get_pkt_size) 的合并结果。

    Args:
        video_path: 视频文件路径

    Returns:
        (i_frame_indices, frame_types, pkt_sizes) 元组
    """
    frames_info = get_frame_info_with_pkt_size(video_path)
    i_frame_indices = [i for i, f in enumerate(frames_info) if f.get("pict_type") == "I"]
    frame_types = [f.get("pict_type", "?") for f in frames_info]
    pkt_sizes = [f.get("pkt_size", 0) for f in frames_info]
    return i_frame_indices, frame_types, pkt_sizes


def get_frame_info_for_stream_ffprobe_with_media_time(
    video_path: str,
) -> Tuple[List[int], List[str], List[int], List[float]]:
    """
    一次 ffprobe 调用返回 I 帧索引、帧类型、每帧 pkt_size、每帧媒体时间（秒）。

    media_time 优先 pkt_pts_time，其次 best_effort_timestamp_time、pts_time；若某帧仍无有效时间，
    则填 0.0，由调用方在已知 fps 时用索引/fps 回填（见 SymVideoInputByStreamWindow）。

    Args:
        video_path: 视频文件路径

    Returns:
        (i_frame_indices, frame_types, pkt_sizes, media_times)
    """
    frames_info = get_frame_info_with_pkt_size(video_path)
    i_frame_indices = [i for i, f in enumerate(frames_info) if f.get("pict_type") == "I"]
    frame_types = [f.get("pict_type", "?") for f in frames_info]
    pkt_sizes = [int(f.get("pkt_size", 0)) for f in frames_info]
    media_times = []
    for f in frames_info:
        mt = f.get("media_time")
        if mt is None:
            media_times.append(0.0)
        else:
            media_times.append(float(mt))
    return i_frame_indices, frame_types, pkt_sizes, media_times
