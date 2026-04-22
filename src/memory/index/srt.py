import bisect
import os
import re
import threading
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


class ThreadSafeSRT:
    """线程安全的 SRT 读写与检索类。"""

    _TIME_PATTERN = re.compile(
        r"^\s*(\d{2}):(\d{2}):(\d{2}),(\d{1,3})\s*-->\s*(\d{2}):(\d{2}):(\d{2}),(\d{1,3})\s*$"
    )
    _SENTENCE_ENDINGS = (".", "!", "?", "。", "！", "？")

    def __init__(self):
        self.start_time: List[float] = []
        self.end_time: List[float] = []
        self.texts: List[str] = []
        self.whole_texts: str = ""
        self.latest_idx: int = 0
        self._lock = threading.RLock()

    @contextmanager
    def acquire(self):
        try:
            self._lock.acquire()
            yield self
        finally:
            self._lock.release()

    @staticmethod
    def _time_to_seconds(time_str: str) -> float:
        match = ThreadSafeSRT._TIME_PATTERN.match(time_str)
        if match is None:
            raise ValueError(f"无效时间行: {time_str}")
        hh, mm, ss, ms, hh2, mm2, ss2, ms2 = (int(x) for x in match.groups())
        start = hh * 3600 + mm * 60 + ss + ms / 1000.0
        end = hh2 * 3600 + mm2 * 60 + ss2 + ms2 / 1000.0
        return start, end

    @staticmethod
    def _seconds_to_time(seconds: float) -> str:
        total_ms = int(round(max(0.0, float(seconds)) * 1000))
        hh = total_ms // 3600000
        rem = total_ms % 3600000
        mm = rem // 60000
        rem = rem % 60000
        ss = rem // 1000
        ms = rem % 1000
        return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"

    def _refresh_whole_texts(self):
        self.whole_texts = " ".join(self.texts).strip()

    @staticmethod
    def _normalize_item(item: Dict[str, Any]) -> Tuple[float, float, str]:
        if not isinstance(item, dict):
            raise TypeError("字幕条目必须是字典")
        if "start_time" not in item or "end_time" not in item or "text" not in item:
            raise ValueError("字幕条目必须包含 start_time/end_time/text")
        st = float(item["start_time"])
        et = float(item["end_time"])
        txt = str(item["text"]).strip()
        if et < st:
            et = st
        return st, et, txt

    def load_local(self, path: str) -> bool:
        if not os.path.isfile(path):
            return False

        with open(path, "r", encoding="utf-8") as f:
            lines = [line.rstrip("\n") for line in f]

        parsed: List[Tuple[float, float, str]] = []
        i = 0
        n = len(lines)
        while i < n:
            line = lines[i].strip()
            if not line:
                i += 1
                continue
            if line.isdigit():
                i += 1
                if i >= n:
                    break
                line = lines[i].strip()
            if "-->" not in line:
                i += 1
                continue
            st, et = self._time_to_seconds(line)
            i += 1
            text_lines: List[str] = []
            while i < n and lines[i].strip():
                text_lines.append(lines[i].strip())
                i += 1
            txt = " ".join(text_lines).strip()
            parsed.append((st, et, txt))
            i += 1

        with self.acquire():
            self.start_time = [x[0] for x in parsed]
            self.end_time = [x[1] for x in parsed]
            self.texts = [x[2] for x in parsed]
            self._refresh_whole_texts()
            self.latest_idx = len(self.start_time)
        return True

    def save_local(self, path: str):
        with self.acquire():
            items = list(
                zip(
                    self.start_time[self.latest_idx :],
                    self.end_time[self.latest_idx :],
                    self.texts[self.latest_idx :],
                )
            )
            start_index = self.latest_idx
            total_len = len(self.start_time)

        if not items:
            return

        dir_path = os.path.dirname(path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

        mode = "a" if os.path.isfile(path) and os.path.getsize(path) > 0 else "w"
        with open(path, mode, encoding="utf-8") as f:
            if mode == "a":
                f.write("\n")
            for offset, (st, et, txt) in enumerate(items, start=1):
                idx = start_index + offset
                f.write(f"{idx}\n")
                f.write(
                    f"{self._seconds_to_time(st)} --> {self._seconds_to_time(et)}\n"
                )
                f.write(f"{txt}\n\n")

        with self.acquire():
            self.latest_idx = total_len

    def append_subtitle(
        self, item_or_items: Union[Dict[str, Any], Sequence[Dict[str, Any]]]
    ):
        if isinstance(item_or_items, dict):
            items = [item_or_items]
        else:
            items = list(item_or_items)
        if not items:
            return

        normalized = [self._normalize_item(item) for item in items]
        with self.acquire():
            for st, et, txt in normalized:
                self.start_time.append(st)
                self.end_time.append(et)
                self.texts.append(txt)
            self._refresh_whole_texts()

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        with self.acquire():
            if idx < 0:
                idx += len(self.start_time)
            if idx < 0 or idx >= len(self.start_time):
                raise IndexError("srt index out of range")
            return {
                "idx": idx,
                "start_time": self.start_time[idx],
                "end_time": self.end_time[idx],
                "text": self.texts[idx],
            }

    def __len__(self) -> int:
        with self.acquire():
            return len(self.start_time)

    def _locate_idx_by_time(self, t: float) -> Optional[int]:
        if not self.start_time:
            return None
        pos = bisect.bisect_right(self.start_time, t) - 1
        if pos < 0:
            return None
        if t > self.end_time[pos]:
            return None
        return pos

    def get_text_by_time(self, t: float) -> Optional[Dict[str, Any]]:
        with self.acquire():
            idx = self._locate_idx_by_time(float(t))
            if idx is None:
                return None
            return {
                "idx": idx,
                "start_time": self.start_time[idx],
                "end_time": self.end_time[idx],
                "text": self.texts[idx],
            }

    def _sentence_range_for_idx(self, idx: int) -> Tuple[int, int]:
        left = idx
        while left > 0:
            prev_text = self.texts[left - 1].strip()
            if prev_text.endswith(self._SENTENCE_ENDINGS):
                break
            left -= 1
        right = idx
        n = len(self.texts)
        while right < n - 1:
            cur = self.texts[right].strip()
            if cur.endswith(self._SENTENCE_ENDINGS):
                break
            right += 1
        return left, right

    def _build_sentence(self, left: int, right: int) -> Dict[str, Any]:
        return {
            "start_idx": left,
            "end_idx": right,
            "start_time": self.start_time[left],
            "end_time": self.end_time[right],
            "text": " ".join(self.texts[left : right + 1]).strip(),
            "items": [
                {
                    "idx": i,
                    "start_time": self.start_time[i],
                    "end_time": self.end_time[i],
                    "text": self.texts[i],
                }
                for i in range(left, right + 1)
            ],
        }

    def get_sentence_by_time(self, t: float) -> Optional[Dict[str, Any]]:
        with self.acquire():
            idx = self._locate_idx_by_time(float(t))
            if idx is None:
                return None
            left, right = self._sentence_range_for_idx(idx)
            return self._build_sentence(left, right)

    def get_text_by_period(
        self, start_t: Optional[float] = None, end_t: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        with self.acquire():
            if not self.start_time:
                return []
            st = float(start_t) if start_t is not None else float("-inf")
            et = float(end_t) if end_t is not None else float("inf")
            ret = []
            for i, (seg_st, seg_et, text) in enumerate(
                zip(self.start_time, self.end_time, self.texts)
            ):
                if seg_et < st:
                    continue
                if seg_st > et:
                    break
                ret.append(
                    {
                        "idx": i,
                        "start_time": seg_st,
                        "end_time": seg_et,
                        "text": text,
                    }
                )
            return ret

    def get_sentence_by_period(
        self, start_t: Optional[float] = None, end_t: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        with self.acquire():
            items = self.get_text_by_period(start_t, end_t)
            if not items:
                return []
            ranges = []
            for item in items:
                left, right = self._sentence_range_for_idx(item["idx"])
                ranges.append((left, right))
            ranges = sorted(set(ranges))
            return [self._build_sentence(left, right) for left, right in ranges]

    @staticmethod
    def _compile_word_pattern(
        word: str, whole_word: bool = True, case_sensitive: bool = False
    ) -> re.Pattern:
        escaped = re.escape(word)
        pattern = rf"\b{escaped}\b" if whole_word else escaped
        flags = 0 if case_sensitive else re.IGNORECASE
        return re.compile(pattern, flags)

    def search_word_idx(
        self, word: str, whole_word: bool = True, case_sensitive: bool = False
    ) -> List[int]:
        if not word:
            return []
        matcher = self._compile_word_pattern(word, whole_word, case_sensitive)
        with self.acquire():
            ret: List[int] = []
            for i, txt in enumerate(self.texts):
                matches = list(matcher.finditer(txt))
                ret.extend([i] * len(matches))
            return ret

    def search_word_time(
        self, word: str, whole_word: bool = True, case_sensitive: bool = False
    ) -> List[float]:
        if not word:
            return []
        matcher = self._compile_word_pattern(word, whole_word, case_sensitive)
        with self.acquire():
            ret: List[float] = []
            for i, txt in enumerate(self.texts):
                seg_st = self.start_time[i]
                seg_et = self.end_time[i]
                duration = max(0.0, seg_et - seg_st)
                text_len = max(1, len(txt))
                for mt in matcher.finditer(txt):
                    center = (mt.start() + mt.end()) / 2.0
                    ratio = center / text_len
                    t = seg_st + ratio * duration
                    ret.append(t)
            return ret
