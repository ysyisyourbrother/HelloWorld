"""从视频中提取单帧图片的工具函数"""

from pathlib import Path
from typing import List
import numpy as np


def extract_save_frame_by_index(
    video_path: str,
    output_path: str,
    frame_index: int,
    backend: str = "decord",
) -> bool:
    """
    根据帧索引从视频中提取单帧并保存为图片。

    Args:
        video_path: 视频文件路径
        output_path: 输出单帧图片的保存路径
        frame_index: 要提取的帧索引（从 0 开始）
        backend: 选择后端库，可选 "cv2" 或 "decord"

    Returns:
        成功返回 True，失败返回 False
    """
    video_path = Path(video_path)
    output_path = Path(output_path)

    if not video_path.exists():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if backend not in {"cv2", "decord"}:
        raise ValueError(f"不支持的 backend: {backend}，请使用 'cv2' 或 'decord'")

    if backend == "decord":
        from decord import VideoReader, cpu
        from PIL import Image

        vr = VideoReader(str(video_path), ctx=cpu(0))
        total_frames = len(vr)

        if frame_index < 0 or frame_index >= total_frames:
            raise ValueError(
                f"帧索引 {frame_index} 超出范围 [0, {total_frames - 1}]"
            )

        frame = vr[frame_index].asnumpy()  # RGB, [H, W, C], uint8
        Image.fromarray(frame).save(str(output_path))
        return True

    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频文件: {video_path}")
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames > 0 and (frame_index < 0 or frame_index >= total_frames):
            raise ValueError(
                f"帧索引 {frame_index} 超出范围 [0, {total_frames - 1}]"
            )
        if frame_index < 0:
            raise ValueError("帧索引不能为负数")

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame_bgr = cap.read()  # BGR
        if not ok or frame_bgr is None:
            raise RuntimeError(f"读取第 {frame_index} 帧失败: {video_path}")
        return bool(cv2.imwrite(str(output_path), frame_bgr))
    finally:
        cap.release()


def extract_frame_by_index(
    video_path: str,
    frame_index: int,
    backend: str = "cv2",
) -> "np.ndarray":
    """
    根据帧索引从视频中提取单帧并返回（不保存文件）。

    Args:
        video_path: 视频文件路径
        frame_index: 要提取的帧索引（从 0 开始）
        backend: 选择后端库，可选 "cv2" 或 "decord"

    Returns:
        提取到的单帧，格式为 BGR `np.ndarray`
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")
    if backend not in {"cv2", "decord"}:
        raise ValueError(f"不支持的 backend: {backend}，请使用 'cv2' 或 'decord'")
    if frame_index < 0:
        raise ValueError("帧索引不能为负数")

    if backend == "decord":
        from decord import VideoReader, cpu
        import cv2

        vr = VideoReader(str(video_path), ctx=cpu(0))
        total_frames = len(vr)
        if frame_index >= total_frames:
            raise ValueError(f"帧索引 {frame_index} 超出范围 [0, {total_frames - 1}]")
        frame_rgb = vr[frame_index].asnumpy()  # RGB
        return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频文件: {video_path}")
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total_frames > 0 and frame_index >= total_frames:
            raise ValueError(f"帧索引 {frame_index} 超出范围 [0, {total_frames - 1}]")
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            raise RuntimeError(f"读取第 {frame_index} 帧失败: {video_path}")
        return frame_bgr
    finally:
        cap.release()


def extract_save_frame_by_time(
    video_path: str,
    output_path: str,
    time_seconds: float,
    backend: str = "decord",
) -> bool:
    """
    根据时间（秒）从视频中提取单帧并保存为图片。

    Args:
        video_path: 视频文件路径
        output_path: 输出单帧图片的保存路径
        time_seconds: 要提取的时间点（秒），支持小数
        backend: 选择后端库，可选 "cv2" 或 "decord"

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

    if backend not in {"cv2", "decord"}:
        raise ValueError(f"不支持的 backend: {backend}，请使用 'cv2' 或 'decord'")

    if backend == "decord":
        from decord import VideoReader, cpu
        from PIL import Image

        vr = VideoReader(str(video_path), ctx=cpu(0))
        fps = vr.get_avg_fps()
        total_frames = len(vr)

        frame_index = int(time_seconds * fps)
        if frame_index >= total_frames:
            frame_index = total_frames - 1
        elif frame_index < 0:
            frame_index = 0

        frame = vr[frame_index].asnumpy()  # RGB, [H, W, C], uint8
        Image.fromarray(frame).save(str(output_path))
        return True

    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频文件: {video_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            raise RuntimeError(f"无法获取 FPS: {video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        frame_index = int(time_seconds * fps)
        if total_frames > 0:
            if frame_index >= total_frames:
                frame_index = total_frames - 1
            elif frame_index < 0:
                frame_index = 0
        else:
            if frame_index < 0:
                frame_index = 0

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame_bgr = cap.read()  # BGR
        if not ok or frame_bgr is None:
            raise RuntimeError(f"读取第 {frame_index} 帧失败: {video_path}")
        return bool(cv2.imwrite(str(output_path), frame_bgr))
    finally:
        cap.release()
