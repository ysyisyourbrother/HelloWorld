# -*- coding: utf-8 -*-
"""
本地媒体音频抽取与封装：为 Memory / ASR 提供 16 kHz 单声道 float32 波形。

- 整段模式：一次性返回整条音轨（对应视频时长内的音频）。
- 分块模式：参考 test_whisper_streaming/whisper_online.py 的 simultaneous 循环，
  按墙钟（可计算感知）或固定步长（sim_comp_unaware）切片，便于流式 ASR。
"""

from __future__ import annotations

import glob
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from typing import Iterator, Optional, Tuple

import numpy as np

from src.config import Config

SAMPLING_RATE = 16000


@dataclass
class AudioData:
    """
    供 ASR 消费的音频数据结构。

    Attributes:
        audio: 一维 float32，16 kHz 单声道。
        audio_type: True 表示流式 chunk；False 表示整段音轨。
        timestamp: 系统时间 time.time()，用于测时。
        start_time: 仅 chunk 模式有效，该 chunk 在媒体时间轴上的起始时间（秒）。
        end_time: chunk 时为媒体时间轴结束时间（秒）；整段时为整段音频时长（秒）。
    """

    audio: np.ndarray
    audio_type: bool
    timestamp: float
    start_time: Optional[float]
    end_time: float


def load_audio_mono_16k_f32(media_path: str) -> np.ndarray:
    """
    从音频或视频文件加载 16 kHz 单声道 float32 波形。

    优先使用 librosa（与 whisper_online 一致）；不可用时回退到 ffmpeg 管道解码。
    """
    if not os.path.isfile(media_path):
        raise FileNotFoundError("媒体文件不存在: %s" % media_path)

    try:
        import librosa

        audio, _ = librosa.load(media_path, sr=SAMPLING_RATE, mono=True, dtype=np.float32)
        return np.ascontiguousarray(audio)
    except Exception:
        pass

    cmd = [
        "ffmpeg",
        "-nostdin",
        "-nostats",
        "-loglevel",
        "error",
        "-i",
        media_path,
        "-vn",
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "1",
        "-ar",
        str(SAMPLING_RATE),
        "-",
    ]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace")[:4000]
        raise RuntimeError("ffmpeg 解码音频失败 (%s): %s" % (media_path, err))

    raw = proc.stdout
    if not raw:
        return np.array([], dtype=np.float32)
    arr = np.frombuffer(raw, dtype=np.dtype("<f4"))
    return np.ascontiguousarray(arr.copy())


class AudioInputBase:
    """
    音频输入基类：从配置初始化，按 chunk 或整段提供 numpy 波形并封装为 AudioData。

    与 VideoInputBase 类似，通过 Config 中的 audio_input 段驱动；源文件默认同 video_file_path。
    """

    def __init__(self, config: Optional[Config] = None):
        if config is None:
            config = Config()
        self._config = config

        self.log_file = config.audio_log_file
        path_opt = config.audio_source_path
        self.source_path = (path_opt or "").strip() or config.video_file_path

        self.stream_as_chunks = config.audio_stream_as_chunks
        self.chunk_duration_sec = config.audio_chunk_duration_sec
        self.sim_comp_unaware = config.audio_sim_comp_unaware

        self._full_audio: Optional[np.ndarray] = None
        self._duration_sec: float = 0.0
        self._sim_beg: float = 0.0
        self._wall_start: float = 0.0
        self._stream_started: bool = False
        self._fixed_beg: float = 0.0

        self.logger: Optional[logging.Logger] = None
        self._set_logger()

    def _set_logger(self):
        log_file = self.log_file
        pattern = log_file.replace(".log", "*")
        for f in glob.glob(pattern):
            try:
                os.remove(f)
            except OSError:
                pass

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.handlers.clear()
        self.logger.setLevel(logging.DEBUG)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        console_handler.setLevel(logging.INFO)
        self.logger.addHandler(console_handler)

        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file, maxBytes=10 * 1024 * 1024, backupCount=5, mode="w"
        )
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        file_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        self.logger.propagate = False

    def init_source(self, media_path: Optional[str] = None) -> None:
        """指定媒体路径并预加载整段波形（后续 chunk/整段均基于此缓存）。"""
        if media_path:
            self.source_path = media_path
        if not self.source_path:
            raise ValueError("未配置音频源路径（audio_input.source_path 或 video_file_path）")
        self._full_audio = load_audio_mono_16k_f32(self.source_path)
        n = int(self._full_audio.shape[0])
        self._duration_sec = float(n) / float(SAMPLING_RATE)
        self.reset_stream_cursor()
        if self.logger:
            self.logger.info(
                "音频已加载: %s, 采样点=%d, 时长=%.3fs"
                % (self.source_path, n, self._duration_sec)
            )

    def _ensure_loaded(self) -> None:
        if self._full_audio is None:
            self.init_source()

    def reset_stream_cursor(self) -> None:
        """重置分块游标（同一文件上重复模拟流式时可调用）。"""
        self._sim_beg = 0.0
        self._wall_start = 0.0
        self._stream_started = False
        self._fixed_beg = 0.0

    def get_full_audio_numpy(self) -> Tuple[np.ndarray, float]:
        """
        整段模式：返回整条波形与时长（秒）。

        Returns:
            (audio, duration_sec)，audio 为 float32 一维数组。
        """
        self._ensure_loaded()
        assert self._full_audio is not None
        return self._full_audio, self._duration_sec

    def next_chunk_numpy(self) -> Optional[Tuple[np.ndarray, float, float]]:
        """
        分块模式：取下一帧切片（视频/媒体时间轴上的 [start_time, end_time)，单位秒）。

        - sim_comp_unaware=False：与 whisper_online simultaneous 类似，用墙钟 + sleep，
          若单次调用间隔较长，则本切片可能长于 chunk_duration_sec（计算可感知）。
        - sim_comp_unaware=True：每步固定前进 chunk_duration_sec（计算无意识仿真）。

        已到达媒体末尾时返回 None。
        """
        self._ensure_loaded()
        assert self._full_audio is not None

        if self.sim_comp_unaware:
            return self._next_chunk_fixed_step()
        return self._next_chunk_wallclock()

    def _next_chunk_fixed_step(self) -> Optional[Tuple[np.ndarray, float, float]]:
        """固定步长分块（与 whisper_online --comp_unaware 切片方式一致）。"""
        assert self._full_audio is not None
        beg = self._fixed_beg
        if beg >= self._duration_sec:
            return None
        end = min(beg + self.chunk_duration_sec, self._duration_sec)
        i0 = int(beg * SAMPLING_RATE)
        i1 = int(end * SAMPLING_RATE)
        chunk = self._full_audio[i0:i1].copy()
        self._fixed_beg = end
        return chunk, beg, end

    def _next_chunk_wallclock(self) -> Optional[Tuple[np.ndarray, float, float]]:
        """墙钟 + sleep（与 whisper_online simultaneous 默认可计算感知模式一致）。"""
        assert self._full_audio is not None
        if self._sim_beg >= self._duration_sec:
            return None

        if not self._stream_started:
            self._wall_start = time.time()
            self._stream_started = True

        min_chunk = self.chunk_duration_sec
        while True:
            now = time.time() - self._wall_start
            if now >= self._sim_beg + min_chunk:
                break
            sleep_sec = self._sim_beg + min_chunk - now
            if sleep_sec > 0:
                time.sleep(sleep_sec)

        end = min(time.time() - self._wall_start, self._duration_sec)
        beg = self._sim_beg
        i0 = int(beg * SAMPLING_RATE)
        i1 = int(end * SAMPLING_RATE)
        chunk = self._full_audio[i0:i1].copy()
        self._sim_beg = end
        return chunk, beg, end

    def pack_audio_data_chunk(
        self, audio: np.ndarray, start_time: float, end_time: float
    ) -> AudioData:
        """将一分段波形封装为 chunk 型 AudioData。"""
        return AudioData(
            audio=np.ascontiguousarray(audio.astype(np.float32, copy=False)),
            audio_type=True,
            timestamp=time.time(),
            start_time=float(start_time),
            end_time=float(end_time),
        )

    def pack_audio_data_full(self, audio: np.ndarray, duration_sec: float) -> AudioData:
        """将整段波形封装为 AudioData（整段无 start_time）。"""
        return AudioData(
            audio=np.ascontiguousarray(audio.astype(np.float32, copy=False)),
            audio_type=False,
            timestamp=time.time(),
            start_time=None,
            end_time=float(duration_sec),
        )

    def iter_audio_data(self) -> Iterator[AudioData]:
        """
        按当前 stream_as_chunks 设置产出 AudioData。

        Yields:
            分块或单条整段；分块结束时自然结束迭代。
        """
        if not self.stream_as_chunks:
            wav, dur = self.get_full_audio_numpy()
            yield self.pack_audio_data_full(wav, dur)
            return

        while True:
            nxt = self.next_chunk_numpy()
            if nxt is None:
                break
            wav, s, e = nxt
            if wav.size == 0:
                continue
            yield self.pack_audio_data_chunk(wav, s, e)
