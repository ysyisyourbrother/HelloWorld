"""
本地视频文件读取适配层：统一 decord 与 OpenCV，便于 Jetson 等无 decord 环境使用 cv2。

约定与 decord.VideoReader 对齐：
- get_avg_fps()
- len(reader)
- reader[i].asnumpy() -> RGB uint8 [H, W, C]
"""

import os
from typing import Any, Union

import numpy as np


class _RgbAsNumpy:
    """与 decord 帧对象一致，提供 asnumpy() -> RGB ndarray。"""

    __slots__ = ("_arr",)

    def __init__(self, arr: np.ndarray):
        self._arr = arr

    def asnumpy(self) -> np.ndarray:
        return self._arr


def _decord_importable() -> bool:
    try:
        import decord  # noqa: F401
        return True
    except ImportError:
        return False


class DecordFileVideoReader:
    """decord 后端（x86 等已安装 decord 的环境）。"""

    __slots__ = ("_vr",)

    def __init__(self, path: str):
        from decord import VideoReader, cpu

        self._vr = VideoReader(path, ctx=cpu(0))

    def get_avg_fps(self) -> float:
        return float(self._vr.get_avg_fps())

    def __len__(self) -> int:
        return len(self._vr)

    def __getitem__(self, idx: int) -> Any:
        n = len(self)
        if idx < 0:
            idx += n
        if idx < 0 or idx >= n:
            raise IndexError(f"frame index {idx} out of range [0, {n})")
        return self._vr[idx]

    def close(self) -> None:
        self._vr = None


class Cv2FileVideoReader:
    """OpenCV 后端（Jetson Orin 等推荐）。"""

    __slots__ = ("_path", "_cap", "_fps", "_n")

    def __init__(self, path: str):
        import cv2

        self._path = path
        self._cap = cv2.VideoCapture(path)
        if not self._cap.isOpened():
            raise RuntimeError(f"无法打开视频文件: {path}")
        fps = float(self._cap.get(cv2.CAP_PROP_FPS) or 0.0)
        self._fps = fps if fps > 0 else 30.0
        n = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self._n = max(0, n)

    def get_avg_fps(self) -> float:
        return self._fps

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> _RgbAsNumpy:
        import cv2

        n = len(self)
        if idx < 0:
            idx += n
        if idx < 0 or idx >= n:
            raise IndexError(f"frame index {idx} out of range [0, {n})")
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, bgr = self._cap.read()
        if not ok or bgr is None:
            raise RuntimeError(f"读取第 {idx} 帧失败: {self._path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return _RgbAsNumpy(rgb)

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


FileVideoReader = Union[DecordFileVideoReader, Cv2FileVideoReader]


def open_file_video_reader(path: str, backend: str = "auto") -> FileVideoReader:
    """
    打开本地视频文件读取器。

    Args:
        path: 视频路径
        backend: "auto" | "decord" | "cv2"
            - auto: 环境变量 VIDEO_READER_BACKEND 为 decord/cv2 时优先采用；
              否则能 import decord 则用 decord，否则用 cv2。

    Returns:
        与 decord.VideoReader 索引语义一致的读取器
    """
    b = (backend or "auto").strip().lower()
    env_b = os.environ.get("VIDEO_READER_BACKEND", "").strip().lower()
    if b == "auto" and env_b in ("cv2", "decord"):
        b = env_b

    if b == "auto":
        if _decord_importable():
            return DecordFileVideoReader(path)
        return Cv2FileVideoReader(path)
    if b == "decord":
        if not _decord_importable():
            raise ImportError("backend=decord 但未安装 decord，请安装 decord 或改用 backend=cv2/auto")
        return DecordFileVideoReader(path)
    if b == "cv2":
        return Cv2FileVideoReader(path)
    raise ValueError(f"不支持的 reader backend: {backend!r}，请使用 auto、decord 或 cv2")
