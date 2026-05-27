from __future__ import annotations

import glob
import logging
import os
import re
import time
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from src.config import Config
from src.video_input.audio_input import AudioData

SUPPORTED_WHISPER_MODELS = {
    "tiny.en",
    "tiny",
    "base.en",
    "base",
    "small.en",
    "small",
    "medium.en",
    "medium",
    "large-v1",
    "large-v2",
    "large-v3",
    "large",
    "large-v3-turbo",
}


@dataclass
class ASRSegment:
    text: str
    start_time: float
    end_time: float
    timestamp: float


class SymASRBase:
    def __init__(self, config: Optional[Config] = None):
        if config is None:
            config = Config()
        self._config = config

        self.log_file = config.asr_log_file
        self.language = config.asr_language
        self.model_size = config.asr_model_size
        self.model_path = config.asr_model_path
        self.audio_stream_as_chunks = bool(config.audio_stream_as_chunks)
        self.backend = config.asr_backend
        self.device = config.asr_device
        self.compute_type = config.asr_compute_type
        self.beam_size = config.asr_beam_size
        self.buffer_trimming = config.asr_buffer_trimming
        self.buffer_trimming_sec = config.asr_buffer_trimming_sec

        self.logger: Optional[logging.Logger] = None
        self._set_logger()

    def _set_logger(self) -> None:
        pattern = self.log_file.replace(".log", "*")
        for file_path in glob.glob(pattern):
            try:
                os.remove(file_path)
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

        log_dir = os.path.dirname(self.log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = RotatingFileHandler(
            self.log_file, maxBytes=10 * 1024 * 1024, backupCount=5, mode="w"
        )
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        file_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        self.logger.propagate = False


def _infer_model_size_from_path(model_path: str) -> Optional[str]:
    base_name = Path(model_path).name.lower()
    match = re.search(
        r"whisper-(tiny(?:\.en)?|base(?:\.en)?|small(?:\.en)?|medium(?:\.en)?|large(?:-v[123]|-v3-turbo)?)",
        base_name,
    )
    if match is None:
        return None
    model_name = match.group(1)
    if model_name in {"large-v1", "large-v2", "large-v3", "large-v3-turbo"}:
        return model_name
    return model_name


def _validate_model_size(model_size: str) -> None:
    if model_size not in SUPPORTED_WHISPER_MODELS:
        raise ValueError(
            "不支持的 whisper 模型版本: %s, 支持: %s"
            % (model_size, ", ".join(sorted(SUPPORTED_WHISPER_MODELS)))
        )


def _normalize_language(language: str) -> Optional[str]:
    if language == "auto":
        return None
    return language


class SymASR(SymASRBase):
    def __init__(self, config: Optional[Config] = None):
        super().__init__(config=config)
        self.asr_pipe = None
        self._offset_seconds = 0.0
        self._prev_raw_start: Optional[float] = None
        self._prev_global_end = 0.0
        self._load_model()

    def _validate_model_dir(self) -> None:
        model_dir = Path(self.model_path)
        if not model_dir.is_dir():
            raise FileNotFoundError("ASR 模型目录不存在: %s" % self.model_path)
        _validate_model_size(self.model_size)
        inferred = _infer_model_size_from_path(self.model_path)
        if inferred is not None and inferred != self.model_size:
            raise ValueError(
                "模型路径与配置版本不一致: path=%s, inferred=%s, config=%s"
                % (self.model_path, inferred, self.model_size)
            )

    def _resolve_device_and_dtype(self) -> Tuple[int, "torch.dtype"]:
        import torch

        if self.device == "cpu":
            return -1, torch.float32
        if self.device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("配置要求 CUDA，但当前环境不可用")
            return 0, torch.float16
        if torch.cuda.is_available():
            return 0, torch.float16
        return -1, torch.float32

    def _load_model(self) -> None:
        self._validate_model_dir()
        if self.backend != "transformers":
            self.logger.info("SymASR 将 backend=%s 回退到 transformers", self.backend)
        from transformers import pipeline

        device, dtype = self._resolve_device_and_dtype()
        self.asr_pipe = pipeline(
            task="automatic-speech-recognition",
            model=self.model_path,
            tokenizer=self.model_path,
            feature_extractor=self.model_path,
            torch_dtype=dtype,
            device=device,
        )
        lang = _normalize_language(self.language)
        if lang is None:
            self.logger.info("SymASR 模型加载完成: %s，语言=auto", self.model_path)
        else:
            self.logger.info(
                "SymASR 模型加载完成: %s，语言=%s", self.model_path, lang
            )

    def _build_generate_kwargs(self) -> dict:
        lang = _normalize_language(self.language)
        if lang is None:
            return {}
        return {"language": lang, "task": "transcribe"}

    def _stitch_global_ts(self, start: float, end: float) -> Tuple[float, float]:
        if self._prev_raw_start is not None and start + 0.2 < self._prev_raw_start:
            self._offset_seconds = self._prev_global_end
        global_start = start + self._offset_seconds
        global_end = end + self._offset_seconds
        if global_end < global_start:
            global_end = global_start
        self._prev_raw_start = start
        self._prev_global_end = global_end
        return global_start, global_end

    def transcribe_audio_data(self, audio_data: AudioData) -> List[ASRSegment]:
        if audio_data.audio_type:
            raise ValueError("SymASR 只接受整段音频(audio_type=False)")
        if self.asr_pipe is None:
            raise RuntimeError("ASR 模型未初始化")

        ret = self.asr_pipe(
            audio_data.audio,
            return_timestamps=True,
            generate_kwargs=self._build_generate_kwargs(),
        )
        chunks = ret.get("chunks", [])
        output: List[ASRSegment] = []

        for chunk in chunks:
            text = (chunk.get("text", "") or "").strip()
            if not text:
                continue
            start_time, end_time = chunk.get("timestamp", (None, None))
            if start_time is None:
                start_time = 0.0
            if end_time is None:
                end_time = start_time
            global_start, global_end = self._stitch_global_ts(
                float(start_time), float(end_time)
            )
            output.append(
                ASRSegment(
                    text=text,
                    start_time=global_start,
                    end_time=global_end,
                    timestamp=time.time(),
                )
            )
        return output


class _HypothesisBuffer:
    def __init__(self):
        self.commited_in_buffer: List[Tuple[float, float, str]] = []
        self.buffer: List[Tuple[float, float, str]] = []
        self.new: List[Tuple[float, float, str]] = []
        self.last_commited_time = 0.0

    def insert(self, new_words: Sequence[Tuple[float, float, str]], offset: float) -> None:
        shifted = [(a + offset, b + offset, t) for a, b, t in new_words]
        self.new = [(a, b, t) for a, b, t in shifted if a > self.last_commited_time - 0.1]
        if len(self.new) < 1 or not self.commited_in_buffer:
            return

        first_start, _, _ = self.new[0]
        if abs(first_start - self.last_commited_time) >= 1.0:
            return

        c_len = len(self.commited_in_buffer)
        n_len = len(self.new)
        max_n = min(min(c_len, n_len), 5)
        for i in range(1, max_n + 1):
            commited_tail = " ".join(
                [self.commited_in_buffer[-j][2] for j in range(1, i + 1)][::-1]
            )
            new_head = " ".join([self.new[j - 1][2] for j in range(1, i + 1)])
            if commited_tail == new_head:
                for _ in range(i):
                    self.new.pop(0)
                break

    def flush(self) -> List[Tuple[float, float, str]]:
        commit: List[Tuple[float, float, str]] = []
        while self.new and self.buffer:
            na, nb, nt = self.new[0]
            if nt != self.buffer[0][2]:
                break
            commit.append((na, nb, nt))
            self.last_commited_time = nb
            self.buffer.pop(0)
            self.new.pop(0)
        self.buffer = self.new
        self.new = []
        self.commited_in_buffer.extend(commit)
        return commit

    def complete(self) -> List[Tuple[float, float, str]]:
        return self.buffer

    def pop_commited(self, trim_time: float) -> None:
        while self.commited_in_buffer and self.commited_in_buffer[0][1] <= trim_time:
            self.commited_in_buffer.pop(0)


class SymStreamASR(SymASRBase):
    SAMPLING_RATE = 16000

    def __init__(self, config: Optional[Config] = None):
        super().__init__(config=config)
        self.chunk_duration_sec = float(self._config.audio_chunk_duration_sec)

        self.model = None
        self.audio_buffer = np.array([], dtype=np.float32)
        self.transcript_buffer = _HypothesisBuffer()
        self.buffer_time_offset = 0.0
        self.commited: List[Tuple[float, float, str]] = []
        self._load_model()

    def _validate_stream_model_dir(self) -> str:
        model_dir = Path(self.model_path)
        if not model_dir.is_dir():
            raise FileNotFoundError("ASR 模型目录不存在: %s" % self.model_path)
        _validate_model_size(self.model_size)
        inferred = _infer_model_size_from_path(self.model_path)
        if inferred is not None and inferred != self.model_size:
            raise ValueError(
                "模型路径与配置版本不一致: path=%s, inferred=%s, config=%s"
                % (self.model_path, inferred, self.model_size)
            )

        name = model_dir.name
        if name.endswith("-ct2"):
            return str(model_dir)

        sibling_ct2 = model_dir.parent / (name + "-ct2")
        if sibling_ct2.is_dir():
            self.logger.info("检测到同名 -ct2 模型目录，自动切换: %s", str(sibling_ct2))
            return str(sibling_ct2)
        raise ValueError(
            "SymStreamASR 需要 CTranslate2 模型目录（-ct2 结尾）: %s" % self.model_path
        )

    def _resolve_fw_device(self) -> str:
        if self.device in {"cpu", "cuda"}:
            return self.device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _load_model(self) -> None:
        if self.backend != "faster-whisper":
            self.logger.info("SymStreamASR 强制使用 faster-whisper，忽略 backend=%s", self.backend)
        from faster_whisper import WhisperModel

        ct2_path = self._validate_stream_model_dir()
        device = self._resolve_fw_device()
        self.model = WhisperModel(
            ct2_path,
            device=device,
            compute_type=self.compute_type,
        )
        self.logger.info("SymStreamASR 模型加载完成: %s", ct2_path)

    def _ts_words(self, segments) -> List[Tuple[float, float, str]]:
        words: List[Tuple[float, float, str]] = []
        for segment in segments:
            if segment.no_speech_prob > 0.9:
                continue
            for word in segment.words:
                words.append((float(word.start), float(word.end), word.word))
        return words

    def _to_flush(self, words: Sequence[Tuple[float, float, str]]) -> Tuple[Optional[float], Optional[float], str]:
        text = "".join([w[2] for w in words]).strip()
        if len(words) == 0:
            return None, None, ""
        return float(words[0][0]), float(words[-1][1]), text

    def _chunk_completed_segment(self, segments) -> None:
        if len(self.commited) == 0:
            return
        ends = [float(seg.end) for seg in segments]
        if len(ends) <= 1:
            return
        t_end = self.commited[-1][1]
        candidate = ends[-2] + self.buffer_time_offset
        while len(ends) > 2 and candidate > t_end:
            ends.pop(-1)
            candidate = ends[-2] + self.buffer_time_offset
        if candidate <= t_end:
            self._chunk_at(candidate)

    def _chunk_at(self, ts: float) -> None:
        self.transcript_buffer.pop_commited(ts)
        cut_sec = ts - self.buffer_time_offset
        self.audio_buffer = self.audio_buffer[int(cut_sec * self.SAMPLING_RATE) :]
        self.buffer_time_offset = ts

    def _build_prompt(self) -> str:
        if len(self.commited) == 0:
            return ""
        k = max(0, len(self.commited) - 1)
        while k > 0 and self.commited[k - 1][1] > self.buffer_time_offset:
            k -= 1
        prompt_words = [x[2] for x in self.commited[:k]]
        out: List[str] = []
        total_chars = 0
        while prompt_words and total_chars < 200:
            token = prompt_words.pop(-1)
            total_chars += len(token) + 1
            out.append(token)
        out.reverse()
        return " ".join(out)

    def transcribe_audio_data(self, audio_data: AudioData) -> List[ASRSegment]:
        if not audio_data.audio_type:
            raise ValueError("SymStreamASR 只接受分块音频(audio_type=True)")
        if self.model is None:
            raise RuntimeError("ASR 模型未初始化")
        if audio_data.audio.size == 0:
            return []

        self.audio_buffer = np.append(self.audio_buffer, audio_data.audio.astype(np.float32))
        prompt = self._build_prompt()
        segments, _ = self.model.transcribe(
            self.audio_buffer,
            language=_normalize_language(self.language),
            initial_prompt=prompt,
            beam_size=self.beam_size,
            word_timestamps=True,
            condition_on_previous_text=True,
        )
        segments = list(segments)

        words = self._ts_words(segments)
        self.transcript_buffer.insert(words, self.buffer_time_offset)
        newly_commited = self.transcript_buffer.flush()
        self.commited.extend(newly_commited)
        start_time, end_time, text = self._to_flush(newly_commited)

        buffered_sec = len(self.audio_buffer) / float(self.SAMPLING_RATE)
        if self.buffer_trimming == "segment" and buffered_sec > self.buffer_trimming_sec:
            self._chunk_completed_segment(segments)

        if text == "" or start_time is None or end_time is None:
            return []
        return [
            ASRSegment(
                text=text,
                start_time=start_time,
                end_time=end_time,
                timestamp=time.time(),
            )
        ]

    def finish(self) -> List[ASRSegment]:
        pending = self.transcript_buffer.complete()
        start_time, end_time, text = self._to_flush(pending)
        if text == "" or start_time is None or end_time is None:
            return []
        self.buffer_time_offset += len(self.audio_buffer) / float(self.SAMPLING_RATE)
        self.audio_buffer = np.array([], dtype=np.float32)
        return [
            ASRSegment(
                text=text,
                start_time=start_time,
                end_time=end_time,
                timestamp=time.time(),
            )
        ]
