"""从视频中提取单帧图片的工具函数"""

from pathlib import Path

import cv2
from decord import VideoReader, cpu


def extract_frame_by_index(
    video_path: str,
    output_path: str,
    frame_index: int,
) -> bool:
    """
    根据帧索引从视频中提取单帧并保存为图片。

    Args:
        video_path: 视频文件路径
        output_path: 输出单帧图片的保存路径
        frame_index: 要提取的帧索引（从 0 开始）

    Returns:
        成功返回 True，失败返回 False
    """
    video_path = Path(video_path)
    output_path = Path(output_path)

    if not video_path.exists():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    vr = VideoReader(str(video_path), ctx=cpu(0))
    total_frames = len(vr)

    if frame_index < 0 or frame_index >= total_frames:
        raise ValueError(
            f"帧索引 {frame_index} 超出范围 [0, {total_frames - 1}]"
        )

    frame = vr[frame_index].asnumpy()  # RGB, [H, W, C], uint8
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return cv2.imwrite(str(output_path), frame_bgr)


def extract_frame_by_time(
    video_path: str,
    output_path: str,
    time_seconds: float,
) -> bool:
    """
    根据时间（秒）从视频中提取单帧并保存为图片。

    Args:
        video_path: 视频文件路径
        output_path: 输出单帧图片的保存路径
        time_seconds: 要提取的时间点（秒），支持小数

    Returns:
        成功返回 True，失败返回 False
    """
    video_path = Path(video_path)
    output_path = Path(output_path)

    if not video_path.exists():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    if time_seconds < 0:
        raise ValueError(f"时间不能为负数: {time_seconds}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    vr = VideoReader(str(video_path), ctx=cpu(0))
    fps = vr.get_avg_fps()
    total_frames = len(vr)

    frame_index = int(time_seconds * fps)
    if frame_index >= total_frames:
        frame_index = total_frames - 1
    elif frame_index < 0:
        frame_index = 0

    frame = vr[frame_index].asnumpy()  # RGB, [H, W, C], uint8
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return cv2.imwrite(str(output_path), frame_bgr)
