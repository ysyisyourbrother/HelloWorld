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
        '''
        初始化字幕数据容器与线程锁。
        维护时间轴、文本列表和增量保存位置。
        '''
        self.start_time: List[float] = []
        self.end_time: List[float] = []
        self.texts: List[str] = []
        self.whole_texts: str = ""
        self.latest_idx: int = 0
        self._lock = threading.RLock()

    @contextmanager
    def acquire(self):
        '''
        提供线程安全上下文，统一管理加锁与解锁。
        用于保护字幕读写操作的并发一致性。
        '''
        try:
            self._lock.acquire()
            yield self
        finally:
            self._lock.release()

    @staticmethod
    def _time_to_seconds(time_str: str) -> float:
        '''
        将 SRT 时间行解析为起止秒数。
        输入格式需为 "HH:MM:SS,ms --> HH:MM:SS,ms"。
        '''
        match = ThreadSafeSRT._TIME_PATTERN.match(time_str)
        if match is None:
            raise ValueError(f"无效时间行: {time_str}")
        hh, mm, ss, ms, hh2, mm2, ss2, ms2 = (int(x) for x in match.groups())
        start = hh * 3600 + mm * 60 + ss + ms / 1000.0
        end = hh2 * 3600 + mm2 * 60 + ss2 + ms2 / 1000.0
        return start, end

    @staticmethod
    def _seconds_to_time(seconds: float) -> str:
        '''
        将秒数格式化为 SRT 时间字符串。
        输出格式固定为 "HH:MM:SS,ms"。
        '''
        total_ms = int(round(max(0.0, float(seconds)) * 1000))
        hh = total_ms // 3600000
        rem = total_ms % 3600000
        mm = rem // 60000
        rem = rem % 60000
        ss = rem // 1000
        ms = rem % 1000
        return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"

    def _refresh_whole_texts(self):
        '''
        基于当前字幕片段重建整体文本缓存。
        便于进行全局文本读取或后续扩展检索。
        '''
        self.whole_texts = " ".join(self.texts).strip()

    @staticmethod
    def _normalize_item(item: Dict[str, Any]) -> Tuple[float, float, str]:
        '''
        校验并规范化单条字幕输入。
        返回统一的 (start_time, end_time, text) 三元组。
        '''
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
        '''
        从本地 SRT 文件加载字幕到内存索引。
        文件不存在时返回 False，成功解析并写入返回 True。
        '''
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
        '''
        将新增字幕以增量方式写入本地 SRT 文件。
        仅保存 latest_idx 之后的条目并更新保存游标。
        '''
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
        '''
        追加一条或多条字幕到内存索引。
        自动做输入标准化并刷新整体文本缓存。
        '''
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
        '''
        按索引读取单条字幕信息。
        支持负索引并返回包含时间与文本的字典。
        '''
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
        '''
        返回当前字幕条目总数。
        用于快速获知内存索引规模。
        '''
        with self.acquire():
            return len(self.start_time)

    def _locate_idx_by_time(self, t: float) -> Optional[int]:
        '''
        根据时间戳定位所属字幕片段索引。
        若时间不落在任何片段内则返回 None。
        '''
        if not self.start_time:
            return None
        pos = bisect.bisect_right(self.start_time, t) - 1
        if pos < 0:
            return None
        if t > self.end_time[pos]:
            return None
        return pos

    def get_text_by_time(self, t: float) -> Optional[Dict[str, Any]]:
        '''
        查询指定时间点对应的字幕片段。
        命中时返回单条字幕信息，未命中返回 None。
        '''
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
        '''
        以给定字幕索引为中心，向两侧扩展句子边界。
        依据句末标点确定完整句子的起止索引。
        '''
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
        '''
        按句子索引范围组装句子结构化结果。
        包含句子文本、时间范围及组成片段明细。
        '''
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
        '''
        查询指定时间点所在的完整句子。
        先定位片段，再按标点规则扩展并返回句子结果。
        '''
        with self.acquire():
            idx = self._locate_idx_by_time(float(t))
            if idx is None:
                return None
            left, right = self._sentence_range_for_idx(idx)
            return self._build_sentence(left, right)

    def get_text_by_period(
        self, start_t: Optional[float] = None, end_t: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        '''
        查询时间区间内相交的字幕片段列表。
        支持开区间输入，结果按时间顺序返回。
        '''
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
        '''
        查询时间区间内涉及到的完整句子列表。
        自动去重句子范围，避免同一句重复返回。
        '''
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
        '''
        按检索参数构建关键词正则表达式。
        支持整词匹配与大小写敏感配置。
        '''
        escaped = re.escape(word)
        pattern = rf"\b{escaped}\b" if whole_word else escaped
        flags = 0 if case_sensitive else re.IGNORECASE
        return re.compile(pattern, flags)

    def search_word_idx(
        self, word: str, whole_word: bool = True, case_sensitive: bool = False
    ) -> List[int]:
        '''
        检索关键词出现位置对应的字幕索引。
        同一条字幕多次命中会重复记录该索引。
        '''
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
        '''
        估算关键词在视频中的出现时间点。
        基于命中字符中心在字幕片段时长中的线性映射。
        '''
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
