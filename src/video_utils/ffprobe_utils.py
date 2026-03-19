"""ffprobe 相关工具函数"""

import json
import subprocess
from pathlib import Path


def get_i_frame_indices(video_path: str) -> list[int]:
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
        text=True,
        check=True,
    )

    data = json.loads(result.stdout)
    frames = data.get("frames", [])

    return [i for i, f in enumerate(frames) if f.get("pict_type") == "I"]


def get_frame_types(video_path: str) -> list[str]:
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
        text=True,
        check=True,
    )

    data = json.loads(result.stdout)
    frames = data.get("frames", [])

    return [f.get("pict_type", "?") for f in frames]

def get_pkt_size(video_path: str) -> list[int]:
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
        text=True,
        check=True,
    )

    lines = result.stdout.strip().split("\n")
    return [int(line.split(",")[2]) for line in lines if line.strip()]


def get_frame_info_with_pkt_size(video_path: str) -> list[dict]:
    """
    使用 ffprobe 获取视频每一帧的 pict_type、pkt_pts_time、pkt_size，按显示顺序返回。

    Args:
        video_path: 视频文件路径

    Returns:
        每帧信息列表，每项为 {"pict_type": str, "pkt_pts_time": float, "pkt_size": int}

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
        text=True,
        check=True,
    )

    data = json.loads(result.stdout)
    frames = data.get("frames", [])

    return [
        {
            "pict_type": f.get("pict_type", "?"),
            "pkt_pts_time": float(f.get("pkt_pts_time", 0)),
            "pkt_size": int(f.get("pkt_size", 0)),
        }
        for f in frames
    ]


def get_frame_info_for_stream(video_path: str) -> tuple[list[int], list[str], list[int]]:
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
