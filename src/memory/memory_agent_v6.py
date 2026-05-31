#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MemoryAgentV6: disjoint faiss subsets, scoped tools, retrieval context formatting."""

import json
import math
import os
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.agent.prompts_for_symphony_v6 import (
    prompt_answer_now,
    prompt_answer_or_replan,
    prompt_generate_plan_with_faiss_and_srt_of_long_video,
    prompt_generate_plan_with_faiss_of_long_video,
    prompt_generate_plan_with_faiss_of_short_video,
    prompt_scope_funccall_1,
    prompt_scope_funccall_2,
    prompt_search_funccall_1,
    prompt_search_funccall_2,
)
from src.config import Config
from src.llm.reasoner import AgenticMultimodalRetrieverAPI
from src.memory.local_plan_parser import LocalPlanParser
from src.memory.memory_manager import MemoryManagerBase
from src.memory.query.query_vectorizer import QueryVectorizer
from src.video_utils.about_frame import extract_save_frame_by_index

LONG_VIDEO_THRE = 600.0
EX_SUBTITLE_THRE = 100


def local_tool_use(func):
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


class FaissSubset:
    """Disjoint union of contiguous faiss index ranges [start_id, start_id + count)."""

    __slots__ = ("_segments",)

    def __init__(self, segments: Optional[Sequence[Tuple[int, int]]] = None):
        self._segments = self._normalize_segments(segments or [])

    @property
    def segments(self) -> List[Tuple[int, int]]:
        return list(self._segments)

    def is_empty(self) -> bool:
        return not self._segments

    def total_count(self) -> int:
        return sum(count for _, count in self._segments)

    @classmethod
    def empty(cls) -> "FaissSubset":
        return cls([])

    @staticmethod
    def _normalize_segments(segments: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
        cleaned: List[Tuple[int, int]] = []
        for item in segments:
            if not item or len(item) != 2:
                continue
            start_id = int(item[0])
            count = int(item[1])
            if start_id < 0 or count <= 0:
                continue
            cleaned.append((start_id, count))
        if not cleaned:
            return []
        cleaned.sort(key=lambda x: x[0])
        merged: List[Tuple[int, int]] = [cleaned[0]]
        for start_id, count in cleaned[1:]:
            prev_start, prev_count = merged[-1]
            prev_end = prev_start + prev_count
            if start_id <= prev_end:
                new_end = max(prev_end, start_id + count)
                merged[-1] = (prev_start, new_end - prev_start)
            else:
                merged.append((start_id, count))
        return merged

    def union_segment(self, start_id: int, count: int) -> "FaissSubset":
        if start_id < 0 or count <= 0:
            return self
        return FaissSubset(self._segments + [(int(start_id), int(count))])

    def union_with(self, other: Optional["FaissSubset"]) -> "FaissSubset":
        if other is None or other.is_empty():
            return self
        if self.is_empty():
            return FaissSubset(other._segments)
        return FaissSubset(self._segments + other._segments)

    def union_period_segment(self, start_id: int, subset_vectors: np.ndarray) -> "FaissSubset":
        if start_id < 0 or subset_vectors is None or subset_vectors.size == 0:
            return self
        count = int(subset_vectors.shape[0])
        return self.union_segment(int(start_id), count)

    def fetch_vectors(self, index) -> Tuple[np.ndarray, List[int]]:
        if self.is_empty() or index is None:
            return np.empty((0, 0), dtype=np.float32), []
        parts: List[np.ndarray] = []
        global_ids: List[int] = []
        for start_id, count in self._segments:
            block = index.subset(int(start_id), int(count))
            if block.size == 0:
                continue
            if block.ndim == 1:
                block = block.reshape(1, -1)
            parts.append(block.astype(np.float32))
            global_ids.extend(range(int(start_id), int(start_id) + int(block.shape[0])))
        if not parts:
            return np.empty((0, 0), dtype=np.float32), []
        return np.vstack(parts), global_ids

    def subset_to_time_range(
        self, databasemap, vector_count: int
    ) -> Tuple[Optional[float], Optional[float]]:
        if self.is_empty() or vector_count <= 0:
            return None, None
        first_id = self._segments[0][0]
        last_seg_start, last_seg_count = self._segments[-1]
        last_id = last_seg_start + last_seg_count - 1
        if first_id >= vector_count or last_id >= vector_count:
            return None, None
        first_rec = databasemap[int(first_id)]
        last_rec = databasemap[int(last_id)]
        start_t = float(first_rec["frame_id"]) / float(first_rec.get("video_fps") or 1.0)
        end_t = float(last_rec["frame_id"]) / float(last_rec.get("video_fps") or 1.0)
        if start_t > end_t:
            return end_t, start_t
        return start_t, end_t


def merge_time_ranges(
    ranges: Sequence[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    cleaned: List[Tuple[float, float]] = []
    for st, et in ranges:
        start_t = float(st)
        end_t = float(et)
        if end_t < start_t:
            start_t, end_t = end_t, start_t
        cleaned.append((start_t, end_t))
    if not cleaned:
        return []
    cleaned.sort(key=lambda x: x[0])
    merged: List[Tuple[float, float]] = [cleaned[0]]
    for start_t, end_t in cleaned[1:]:
        prev_st, prev_et = merged[-1]
        if start_t <= prev_et:
            merged[-1] = (prev_st, max(prev_et, end_t))
        else:
            merged.append((start_t, end_t))
    return merged


class MemoryAgentV6(MemoryManagerBase):
    """Scoped retrieval tools with disjoint faiss subset unions."""

    def __init__(self, config: Config = None):
        super().__init__(config=config)
        self.replan_budget = None
        self.agentic_retriever = AgenticMultimodalRetrieverAPI(config=self._config)
        self._initialize_tools_list()

    def _initialize_tools_list(self) -> None:
        period_tool = {
            "type": "function",
            "function": {
                "name": "_get_subset_by_period",
                "description": "Obtain the subset of semantic vector library according to the time interval of video",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "start_time": {
                            "type": "number",
                            "description": "The start time (second) of the period",
                        },
                        "end_time": {
                            "type": "number",
                            "description": "The end time (second) of the period",
                        },
                    },
                    "required": ["start_time", "end_time"],
                },
            },
        }
        event_tool = {
            "type": "function",
            "function": {
                "name": "_get_subset_by_event_frame",
                "description": "Obtain a temporal subset by anchoring at the most relevant frame for an event entity",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "event": {
                            "type": "string",
                            "description": "Event entity description used to locate the anchor frame. Event MUST be a noun phrase",
                        },
                        "scope": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "Two integers [left_seconds, right_seconds] around the anchor time, e.g., [-45, 15], [0, 60]. You can scale it according to the duration of the video.",
                        },
                    },
                    "required": ["event", "scope"],
                },
            },
        }
        keyword_scope_tool = {
            "type": "function",
            "function": {
                "name": "_get_subset_by_keyword",
                "description": "Expand scope using 1-4 subtitle keywords; union sentence time ranges with the current subset",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "keywords": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "1-4 keywords to locate subtitle sentences. If there are less than four words, you can add some synonyms or words of different forms appropriately.",
                        },
                    },
                    "required": ["keywords"],
                },
            },
        }
        frames_entities_tool = {
            "type": "function",
            "function": {
                "name": "_search_frames_with_multiple_entities",
                "description": "Search frames by multiple entity texts (similarities aggregated) and return top_k frame metadata",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "entities": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "A list of entity strings to search for.",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of top frames to return. Depending on the duration of the video, it can take 5 to 10",
                        },
                    },
                    "required": ["entities", "top_k"],
                },
            },
        }
        frames_sample_tool = {
            "type": "function",
            "function": {
                "name": "_search_frames_just_by_scope",
                "description": "Uniformly sample a budget of frames within the current scope and return their metadata",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "budget": {
                            "type": "integer",
                            "description": "Number of frames to sample within the scope",
                        }
                    },
                    "required": ["budget"],
                },
            },
        }
        subtitles_keywords_tool = {
            "type": "function",
            "function": {
                "name": "_search_subtitles_with_multiple_keywords",
                "description": "Return subtitle sentences that contain any of the keywords within the current scope",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "keywords": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "1-4 keywords; matching sentences are returned. If there are less than four words, you can add some synonyms or words of different forms appropriately.",
                        },
                    },
                    "required": ["keywords"],
                },
            },
        }
        subtitles_scope_tool = {
            "type": "function",
            "function": {
                "name": "_search_subtitles_just_by_scope",
                "description": "Retrieve all subtitle sentences within the current scope. ",
                "parameters": {"type": "object", "properties": {}},
            },
        }

        self.scope_tools_1 = [period_tool, event_tool]
        self.scope_tools_2 = [keyword_scope_tool]
        self.scope_tools_replan = [period_tool]
        self.search_tools_1 = [frames_entities_tool, frames_sample_tool]
        self.search_tools_2 = [subtitles_keywords_tool, subtitles_scope_tool]
        self.search_tools_replan = list(self.search_tools_1)

    def _ensure_query_encoder(self) -> None:
        if self._query_encoder is None:
            self._query_encoder = QueryVectorizer(self._config)
            self._query_encoder._set_logger()
            self._query_encoder._initialize_vectorizer()

    def _encode_text_query(self, text: str) -> np.ndarray:
        self._ensure_query_encoder()
        return self._query_encoder.encode_query_sync(text)

    def _build_metadata_list_by_ids(self, vector_ids: Sequence[int]) -> List[Dict[str, Any]]:
        metadata_list: List[Dict[str, Any]] = []
        for vector_id in vector_ids:
            rec = self.databasemap[int(vector_id)]
            metadata_list.append(
                {
                    "frame_id": rec["frame_id"],
                    "video_fps": rec.get("video_fps") or 1.0,
                    "source_path": rec.get("source_path"),
                    "total_frames": rec.get("total_frames"),
                    "i_frames": rec.get("i_frames") or [],
                }
            )
        return metadata_list

    def _parse_scope(self, scope: Sequence[int]) -> Tuple[float, float]:
        if scope is None or len(scope) != 2:
            return -15.0, 15.0
        return float(scope[0]), float(scope[1])

    def _period_to_faiss_segment(
        self, start_time: float, end_time: float
    ) -> Tuple[int, np.ndarray]:
        if self.databasemap is None or self.vector_count == 0:
            return -1, np.empty((0, 0), dtype=np.float32)
        if start_time > end_time:
            return -1, np.empty((0, 0), dtype=np.float32)

        with self.databasemap.acquire() as videos:
            if not videos:
                return -1, np.empty((0, 0), dtype=np.float32)
            fps = float(videos[0].get("video_fps") or 1.0)

        start_frameid = int(round(float(start_time) * fps))
        end_frameid = int(round(float(end_time) * fps))
        return self._get_faiss_subset(start_frameid=start_frameid, end_frameid=end_frameid)

    def _get_faiss_subset(
        self, start_frameid: Optional[int], end_frameid: Optional[int]
    ) -> Tuple[int, np.ndarray]:
        if self.index is None or self.vector_count == 0:
            return -1, np.empty((0, 0), dtype=np.float32)

        with self.databasemap.acquire() as videos:
            offset = 0
            start_id = None
            end_id_exclusive = None

            for video in videos:
                frames = video.get("frames", [])
                if not frames:
                    continue

                frames_array = np.asarray(frames)
                if start_frameid is None:
                    local_start = 0
                else:
                    local_start = int(
                        np.searchsorted(frames_array, start_frameid, side="right")
                    )

                if end_frameid is None:
                    local_end_exclusive = len(frames)
                else:
                    local_end_exclusive = int(
                        np.searchsorted(frames_array, end_frameid, side="left")
                    )

                if local_start < local_end_exclusive:
                    start_id = offset + local_start
                    end_id_exclusive = offset + local_end_exclusive
                    break

                offset += len(frames)

        if start_id is None or end_id_exclusive is None:
            return -1, np.empty((0, 0), dtype=np.float32)

        subset_count = end_id_exclusive - start_id
        subset_vectors = self.index.subset(start_id, subset_count)
        if subset_vectors.size == 0:
            return -1, np.empty((0, 0), dtype=np.float32)
        if subset_vectors.ndim == 1:
            subset_vectors = subset_vectors.reshape(1, -1)
        return start_id, subset_vectors

    def _subset_to_time_range(self, faiss_subset: Optional[FaissSubset]) -> Tuple[Optional[float], Optional[float]]:
        if faiss_subset is None or faiss_subset.is_empty():
            return None, None
        return faiss_subset.subset_to_time_range(self.databasemap, self.vector_count)

    def _collect_sentence_ranges_for_keywords(
        self, keywords: Sequence[str]
    ) -> List[Tuple[float, float]]:
        if self.srt is None:
            return []
        uniq_kw: List[str] = []
        seen_kw = set()
        for raw in keywords:
            word = str(raw).strip()
            if not word or word in seen_kw:
                continue
            seen_kw.add(word)
            uniq_kw.append(word)
            if len(uniq_kw) >= 4:
                break
        if not uniq_kw:
            return []

        sentence_keys = set()
        ranges: List[Tuple[float, float]] = []
        for word in uniq_kw:
            times = self.srt.search_word_time(word)
            for t in times:
                sent = self.srt.get_sentence_by_time(float(t))
                if sent is None:
                    continue
                key = (int(sent["start_idx"]), int(sent["end_idx"]))
                if key in sentence_keys:
                    continue
                sentence_keys.add(key)
                ranges.append((float(sent["start_time"]), float(sent["end_time"])))
        return merge_time_ranges(ranges)

    def _query_faiss_subset_vectors(
        self,
        query_vector: np.ndarray,
        top_k: int,
        subset_vectors: np.ndarray,
        global_ids: Sequence[int],
    ) -> Tuple[List[int], List[float]]:
        if subset_vectors.size == 0 or top_k <= 0:
            return [], []

        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        query_vector = query_vector.astype(np.float32)
        candidate_vectors = subset_vectors.astype(np.float32)
        if candidate_vectors.ndim == 1:
            candidate_vectors = candidate_vectors.reshape(1, -1)

        if self.faiss_index_type == "FlatIP":
            score_vec = np.matmul(candidate_vectors, query_vector[0])
            order = np.argsort(-score_vec)[: min(top_k, score_vec.shape[0])]
        else:
            delta = candidate_vectors - query_vector[0]
            score_vec = np.sum(delta * delta, axis=1)
            order = np.argsort(score_vec)[: min(top_k, score_vec.shape[0])]

        picked = [int(global_ids[int(i)]) for i in order.tolist()]
        scores = [float(score_vec[int(i)]) for i in order.tolist()]
        return picked, scores

    @local_tool_use
    def _get_subset_by_period(
        self, start_time: float, end_time: float, faiss_subset: Optional[FaissSubset] = None
    ) -> FaissSubset:
        start_id, subset_vectors = self._period_to_faiss_segment(start_time, end_time)
        base = faiss_subset if faiss_subset is not None else FaissSubset.empty()
        return base.union_period_segment(start_id, subset_vectors)

    @local_tool_use
    def _get_subset_by_event_frame(
        self,
        event: str,
        scope: Sequence[int],
        faiss_subset: Optional[FaissSubset] = None,
    ) -> FaissSubset:
        if not event:
            return faiss_subset if faiss_subset is not None else FaissSubset.empty()
        query_vector = self._encode_text_query(event)
        vector_ids, _ = self._query_faiss(query_vector, top_k=1)
        if not vector_ids:
            return faiss_subset if faiss_subset is not None else FaissSubset.empty()

        anchor = self.databasemap[int(vector_ids[0])]
        anchor_time = float(anchor["frame_id"]) / float(anchor.get("video_fps") or 1.0)
        left, right = self._parse_scope(scope)
        return self._get_subset_by_period(
            start_time=anchor_time + left,
            end_time=anchor_time + right,
            faiss_subset=faiss_subset,
        )

    @local_tool_use
    def _get_subset_by_keyword(
        self,
        keywords: Sequence[str],
        faiss_subset: Optional[FaissSubset] = None,
    ) -> FaissSubset:
        base = faiss_subset if faiss_subset is not None else FaissSubset.empty()
        time_ranges = self._collect_sentence_ranges_for_keywords(keywords)
        if not time_ranges:
            return base
        out = base
        for start_t, end_t in time_ranges:
            start_id, subset_vectors = self._period_to_faiss_segment(start_t, end_t)
            out = out.union_period_segment(start_id, subset_vectors)
        return out

    @local_tool_use
    def _search_frames_with_multiple_entities(
        self,
        entities: Sequence[str],
        top_k: int,
        faiss_subset: Optional[FaissSubset] = None,
    ) -> Tuple[List[float], List[Dict[str, Any]]]:
        entity_list = [str(item).strip() for item in entities if str(item).strip()]
        if not entity_list or top_k <= 0:
            return [], []

        subset = faiss_subset
        if subset is None or subset.is_empty():
            if self.index is None or self.vector_count == 0:
                return [], []
            subset = FaissSubset([(0, self.vector_count)])

        subset_vectors, global_ids = subset.fetch_vectors(self.index)
        if subset_vectors.size == 0:
            return [], []

        query_vectors = [self._encode_text_query(entity) for entity in entity_list]
        if self.faiss_index_type == "FlatIP":
            agg_scores = np.zeros(subset_vectors.shape[0], dtype=np.float32)
            for query_vector in query_vectors:
                q = query_vector[0] if query_vector.ndim > 1 else query_vector
                agg_scores += np.matmul(subset_vectors, q.astype(np.float32))
            order = np.argsort(-agg_scores)[: min(top_k, agg_scores.shape[0])]
        else:
            agg_scores = np.zeros(subset_vectors.shape[0], dtype=np.float32)
            for query_vector in query_vectors:
                q = query_vector[0] if query_vector.ndim > 1 else query_vector
                delta = subset_vectors - q.astype(np.float32)
                agg_scores += np.sum(delta * delta, axis=1)
            order = np.argsort(agg_scores)[: min(top_k, agg_scores.shape[0])]

        selected_ids = [global_ids[int(i)] for i in order.tolist()]
        scores = [float(agg_scores[int(i)]) for i in order.tolist()]
        return scores, self._build_metadata_list_by_ids(selected_ids)

    @local_tool_use
    def _search_frames_just_by_scope(
        self,
        budget: int,
        faiss_subset: Optional[FaissSubset] = None,
    ) -> List[Dict[str, Any]]:
        if budget <= 0:
            return []

        subset = faiss_subset
        if subset is None or subset.is_empty():
            if self.index is None or self.vector_count == 0:
                return []
            subset = FaissSubset([(0, self.vector_count)])

        subset_vectors, global_ids = subset.fetch_vectors(self.index)
        if subset_vectors.size == 0:
            return []

        size = subset_vectors.shape[0]
        count = min(int(budget), int(size))
        chosen = np.linspace(0, size - 1, num=count, dtype=int).tolist()
        selected_ids = [global_ids[int(i)] for i in chosen]
        return self._build_metadata_list_by_ids(selected_ids)

    def _subtitle_row_key(self, row: Dict[str, Any]) -> Tuple[Any, ...]:
        if "start_idx" in row and "end_idx" in row:
            return ("idx", int(row["start_idx"]), int(row["end_idx"]))
        return (
            "time",
            float(row.get("start_time", 0.0)),
            float(row.get("end_time", 0.0)),
            str(row.get("text") or ""),
        )

    def _sentence_contains_keyword(self, text: str, keyword: str) -> bool:
        pattern = r"\b{0}\b".format(re.escape(keyword))
        return re.search(pattern, text, flags=re.IGNORECASE) is not None

    @local_tool_use
    def _search_subtitles_with_multiple_keywords(
        self,
        keywords: Sequence[str],
        faiss_subset: Optional[FaissSubset] = None,
    ) -> List[Dict[str, Any]]:
        start_t, end_t = self._subset_to_time_range(faiss_subset)
        if start_t is None or end_t is None:
            return []

        rows = self.retrieve_segment_by_period(start_t=start_t, end_t=end_t)
        if not rows:
            return []

        uniq_kw: List[str] = []
        seen_kw = set()
        for raw in keywords:
            word = str(raw).strip()
            if not word or word in seen_kw:
                continue
            seen_kw.add(word)
            uniq_kw.append(word)
            if len(uniq_kw) >= 4:
                break
        if not uniq_kw:
            return []

        matched: List[Dict[str, Any]] = []
        seen_rows = set()
        for row in rows:
            text = str(row.get("text") or "")
            if not text:
                continue
            hit = False
            for word in uniq_kw:
                if self._sentence_contains_keyword(text, word):
                    hit = True
                    break
            if not hit:
                continue
            key = self._subtitle_row_key(row)
            if key in seen_rows:
                continue
            seen_rows.add(key)
            matched.append(row)
        return matched

    @local_tool_use
    def _search_subtitles_just_by_scope(
        self,
        faiss_subset: Optional[FaissSubset] = None,
        budget: Optional[int] = 100,
    ) -> List[Dict[str, Any]]:
        start_t, end_t = self._subset_to_time_range(faiss_subset)
        if start_t is None or end_t is None:
            return []

        rows = self.retrieve_segment_by_period(start_t=start_t, end_t=end_t)
        if not rows:
            return []
        if budget is None or budget <= 0 or len(rows) <= budget:
            return rows

        cap = int(budget)
        indexed = list(enumerate(rows))
        indexed.sort(
            key=lambda item: len(str(item[1].get("text") or "")),
            reverse=True,
        )
        top = indexed[:cap]
        top.sort(key=lambda item: item[0])
        return [row for _, row in top]

    @local_tool_use
    def _get_all_subtitles(
        self, faiss_subset: Optional[FaissSubset] = None
    ) -> List[Dict[str, Any]]:
        start_t, end_t = self._subset_to_time_range(faiss_subset)
        if start_t is None or end_t is None:
            with self.databasemap.acquire() as videos:
                if videos:
                    duration = float(videos[0].get("duration") or 0.0)
                    if duration > 0.0:
                        return self.retrieve_segment_by_period(start_t=0.0, end_t=duration)
            return self.retrieve_segment_by_period()
        return self.retrieve_segment_by_period(start_t=start_t, end_t=end_t)

    @staticmethod
    def _frame_time_sec(frame_meta: Dict[str, Any]) -> float:
        frame_id = frame_meta.get("frame_id")
        fps = float(frame_meta.get("video_fps") or 1.0)
        if frame_id is None:
            return 0.0
        return float(frame_id) / fps

    @staticmethod
    def _format_appear_time_list(times: Sequence[float]) -> str:
        parts = ["{0:.1f}".format(float(t)) for t in times]
        if not parts:
            return ""
        if len(parts) == 1:
            return parts[0]
        if len(parts) == 2:
            return parts[0] + " and " + parts[1]
        return ", ".join(parts[:-1]) + ", and " + parts[-1]

    @staticmethod
    def _subtitle_rows_are_adjacent(prev: Dict[str, Any], curr: Dict[str, Any]) -> bool:
        if "end_idx" in prev and "start_idx" in curr:
            return int(curr["start_idx"]) == int(prev["end_idx"]) + 1
        gap = float(curr.get("start_time", 0.0)) - float(prev.get("end_time", 0.0))
        return gap <= 0.05

    def _format_subtitle_union(self, subtitle_results: Sequence[Dict[str, Any]]) -> str:
        rows = [row for row in subtitle_results if isinstance(row, dict)]
        if not rows:
            return ""

        rows.sort(key=lambda r: (float(r.get("start_time", 0.0)), float(r.get("end_time", 0.0))))
        groups: List[List[str]] = []
        current_texts: List[str] = []
        prev_row: Optional[Dict[str, Any]] = None

        for row in rows:
            text = str(row.get("text") or "").strip()
            if not text:
                continue
            if not current_texts:
                current_texts = [text]
                prev_row = row
                continue
            if prev_row is not None and self._subtitle_rows_are_adjacent(prev_row, row):
                current_texts.append(text)
            else:
                groups.append(current_texts)
                current_texts = [text]
            prev_row = row

        if current_texts:
            groups.append(current_texts)

        chunks: List[str] = []
        for texts in groups:
            chunks.append(" ".join(texts))
        return '"..."'.join(['"{0}"'.format(chunk) for chunk in chunks if chunk])

    def _format_retrieval_context(
        self,
        frame_results: Sequence[Dict[str, Any]],
        subtitle_results: Sequence[Dict[str, Any]],
    ) -> str:
        parts: List[str] = []

        frame_rows = [row for row in frame_results if isinstance(row, dict)]
        if frame_rows:
            frame_rows.sort(key=self._frame_time_sec)
            times = [self._frame_time_sec(row) for row in frame_rows]
            appear_time = self._format_appear_time_list(times)
            parts.append(
                "We provide {cnt} most relevant frames as visual evidence. "
                "They have been arranged in chronological order, appearing at {appear_time} "
                "seconds respectively in the video.".format(
                    cnt=len(frame_rows), appear_time=appear_time
                )
            )

        subtitle_union = self._format_subtitle_union(subtitle_results)
        if subtitle_union:
            parts.append(
                "We also provide relevant subtitles of this video:\n{subtitle_union}\n"
                "You can use this subtitle as an auxiliary criterion.".format(
                    subtitle_union=subtitle_union
                )
            )

        return "\n".join(parts)

    def _parse_plan_steps(self, raw_plan: str) -> List[str]:
        text = str(raw_plan or "").strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        return [line.strip() for line in text.splitlines() if line.strip()]

    def _resolve_plan_json_path(self) -> Optional[str]:
        map_path = getattr(self, "databasemap_file_path", None) or ""
        map_path = str(map_path).strip()
        if not map_path:
            return None
        json_dir = os.path.dirname(os.path.abspath(map_path))
        base_dir = os.path.dirname(json_dir)
        plan_dir = os.path.join(base_dir, "plan")
        basename = os.path.basename(map_path)
        return os.path.join(plan_dir, basename)

    def _serialize_tool_result_for_plan(
        self, result: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        if not result:
            return None
        out: Dict[str, Any] = {}
        for key, val in result.items():
            if key == "faiss_subset" and isinstance(val, FaissSubset):
                out[key] = {
                    "segments": val.segments,
                    "total_count": val.total_count(),
                }
                continue
            if key == "frame_results" and isinstance(val, list):
                stripped: List[Any] = []
                for row in val:
                    if isinstance(row, dict) and "i_frames" in row:
                        stripped.append(
                            {k: v for k, v in row.items() if k != "i_frames"}
                        )
                    else:
                        stripped.append(row)
                out[key] = stripped
                continue
            out[key] = val
        return out

    def _append_plan_trace_json(
        self,
        plan_json_path: str,
        run_record: Dict[str, Any],
    ) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(plan_json_path)), exist_ok=True)
        data: Dict[str, Any] = {"runs": []}
        if os.path.isfile(plan_json_path):
            with open(plan_json_path, "r", encoding="utf-8") as fp:
                text = fp.read()
            if text.strip():
                data = json.loads(text)
        if "runs" not in data or not isinstance(data["runs"], list):
            data["runs"] = []
        data["databasemap_file_path"] = self.databasemap_file_path
        data["runs"].append(run_record)
        with open(plan_json_path, "w", encoding="utf-8") as fp:
            json.dump(data, fp, ensure_ascii=False, indent=2)

    def _log_replan_budget(
        self,
        budget_log: List[Dict[str, Any]],
        event: str,
        budget_before: int,
        budget_after: int,
        loop_round: Optional[int] = None,
        note: Optional[str] = None,
    ) -> None:
        entry: Dict[str, Any] = {
            "event": event,
            "budget_before": int(budget_before),
            "budget_after": int(budget_after),
        }
        if loop_round is not None:
            entry["loop_round"] = int(loop_round)
        if note:
            entry["note"] = str(note)
        budget_log.append(entry)

    def _split_answer_or_replan_response(self, text: str) -> Dict[str, Any]:
        raw = str(text or "")
        answer_match = re.search(r"\[Answer\]", raw, flags=re.IGNORECASE)
        replan_match = re.search(r"\[Replan\]", raw, flags=re.IGNORECASE)

        answer_pos = answer_match.start() if answer_match else -1
        replan_pos = replan_match.start() if replan_match else -1

        tag_pos = -1
        outcome = "none"
        if answer_pos >= 0 and (replan_pos < 0 or answer_pos <= replan_pos):
            tag_pos = answer_pos
            outcome = "answer"
        elif replan_pos >= 0:
            tag_pos = replan_pos
            outcome = "replan"

        reasoning = raw[:tag_pos].strip() if tag_pos >= 0 else raw.strip()
        answer_letter = ""
        replan_steps: List[str] = []
        if outcome == "answer":
            answer_letter = self._extract_answer_text(raw)
        elif outcome == "replan":
            replan_steps = self._parse_replan_steps(raw)

        return {
            "raw": raw,
            "reasoning": reasoning,
            "outcome": outcome,
            "answer_letter": answer_letter,
            "replan_steps": replan_steps,
        }

    def _extract_tool_selection(
        self, assistant_message: Dict[str, Any]
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        calls = assistant_message.get("tool_calls") or []
        if not calls:
            return None, {}
        first_call = calls[0]
        func = first_call.get("function") or {}
        tool_name = func.get("name")
        arg_text = str(func.get("arguments") or "").strip()
        if not tool_name or not arg_text:
            return None, {}
        arguments = json.loads(arg_text)
        if not isinstance(arguments, dict):
            return None, {}
        return tool_name, arguments

    def _format_options_text(self, options: Optional[Sequence[str]]) -> str:
        if not options:
            return "(no options provided)"
        items = [str(item).strip() for item in options if str(item).strip()]
        if not items:
            return "(no options provided)"
        return " ".join(items)

    def _format_plan_flow_brief(self, steps: Sequence[str]) -> str:
        labels: List[str] = []
        for step in steps:
            text = str(step).strip()
            if text.startswith("[Scope]"):
                labels.append("[Scope]")
            elif text.startswith("[Search]"):
                labels.append("[Search]")
        if not labels:
            return "(空)"
        return "->".join(labels)

    def _count_subtitle_sentences(
        self, subtitle_rows: Optional[Sequence[Dict[str, Any]]]
    ) -> int:
        if not subtitle_rows:
            return 0
        total = 0
        for row in subtitle_rows:
            if not isinstance(row, dict):
                continue
            items = row.get("items")
            if isinstance(items, list) and items:
                total += len(items)
            else:
                total += 1
        return total

    def _describe_scope_tool(
        self, tool_name: Optional[str], arguments: Dict[str, Any]
    ) -> str:
        if not tool_name:
            return "定位失败"
        if tool_name == "_get_subset_by_period":
            return "时间段 %.1f-%.1fs" % (
                float(arguments.get("start_time") or 0.0),
                float(arguments.get("end_time") or 0.0),
            )
        if tool_name == "_get_subset_by_event_frame":
            scope = arguments.get("scope") or []
            left = scope[0] if len(scope) > 0 else 0
            right = scope[1] if len(scope) > 1 else 0
            return "事件'%s' 窗口[%s,%s]s" % (
                arguments.get("event") or "",
                left,
                right,
            )
        if tool_name == "_get_subset_by_keyword":
            keywords = arguments.get("keywords") or []
            if isinstance(keywords, str):
                keywords = [keywords]
            return "关键词 %s" % ",".join(str(item) for item in keywords)
        return str(tool_name)

    def _describe_search_result(self, frame_count: int, subtitle_count: int) -> str:
        if frame_count > 0 and subtitle_count > 0:
            return "%d帧、%d句字幕" % (frame_count, subtitle_count)
        if frame_count > 0:
            return "%d帧" % frame_count
        if subtitle_count > 0:
            return "%d句字幕" % subtitle_count
        return "无证据"

    def _log_plan_received(
        self, steps: Sequence[str], reused: bool = False
    ) -> None:
        flow = self._format_plan_flow_brief(steps)
        if reused:
            self.logger.info("Plan(复用): %s", flow)
        else:
            self.logger.info("Plan: %s", flow)

    def _log_scope_step(
        self,
        is_replan: bool,
        tool_name: Optional[str],
        arguments: Dict[str, Any],
    ) -> None:
        prefix = "Replan Scope" if is_replan else "Scope"
        self.logger.info(
            "%s: %s", prefix, self._describe_scope_tool(tool_name, arguments)
        )

    def _log_search_step(
        self,
        is_replan: bool,
        tool_name: Optional[str],
        frame_part: Sequence[Dict[str, Any]],
        subtitle_part: Sequence[Dict[str, Any]],
    ) -> None:
        prefix = "Replan Search" if is_replan else "Search"
        frame_count = len(frame_part) if frame_part else 0
        subtitle_count = self._count_subtitle_sentences(subtitle_part)
        if tool_name == "_get_all_subtitles":
            self.logger.info("%s: 全量字幕 %d句", prefix, subtitle_count)
            return
        self.logger.info(
            "%s: %s",
            prefix,
            self._describe_search_result(frame_count, subtitle_count),
        )

    def _log_model_decision(
        self,
        budget: int,
        parsed_response: Dict[str, Any],
        loop_round: int = 0,
        phase: str = "",
    ) -> None:
        outcome = str(parsed_response.get("outcome") or "")
        round_text = "第%d轮 " % loop_round if loop_round > 0 else ""
        phase_text = phase + " " if phase else ""
        if outcome == "answer":
            letter = str(parsed_response.get("answer_letter") or "").strip() or "?"
            self.logger.info(
                "%s%sbudget=%d 选择 Answer %s",
                phase_text,
                round_text,
                budget,
                letter,
            )
            return
        if outcome == "replan":
            replan_steps = parsed_response.get("replan_steps") or []
            flow = self._format_plan_flow_brief(replan_steps)
            self.logger.info(
                "%s%sbudget=%d 选择 Replan %s",
                phase_text,
                round_text,
                budget,
                flow,
            )
            return
        self.logger.info(
            "%s%sbudget=%d 选择未识别",
            phase_text,
            round_text,
            budget,
        )

    def _extract_answer_text(self, text: str) -> str:
        match = re.search(r"\[Answer\]\s*(.+)", str(text or ""), flags=re.IGNORECASE | re.DOTALL)
        if match is None:
            return ""
        first_line = match.group(1).strip().splitlines()[0].strip()
        return first_line

    def _parse_replan_steps(self, answer_or_replan: str) -> List[str]:
        text = str(answer_or_replan or "")
        idx = text.upper().find("[REPLAN]")
        if idx < 0:
            return []
        tail = text[idx + len("[Replan]") :].strip()
        if tail.startswith("["):
            parsed = json.loads(tail)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        steps = self._parse_plan_steps(tail)
        if steps:
            return steps
        quoted = LocalPlanParser.extract_quoted_strings(tail)
        plan_like = [
            item
            for item in quoted
            if item.startswith("[Scope]") or item.startswith("[Search]")
        ]
        return plan_like

    def _call_tool_by_name(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        faiss_subset: Optional[FaissSubset],
    ) -> Dict[str, Any]:
        if tool_name == "_get_subset_by_period":
            new_subset = self._get_subset_by_period(
                start_time=arguments["start_time"],
                end_time=arguments["end_time"],
                faiss_subset=faiss_subset,
            )
            return {
                "faiss_subset": new_subset,
                "frame_results": [],
                "subtitle_results": [],
            }

        if tool_name == "_get_subset_by_event_frame":
            new_subset = self._get_subset_by_event_frame(
                event=arguments["event"],
                scope=arguments["scope"],
                faiss_subset=faiss_subset,
            )
            return {
                "faiss_subset": new_subset,
                "frame_results": [],
                "subtitle_results": [],
            }

        if tool_name == "_get_subset_by_keyword":
            new_subset = self._get_subset_by_keyword(
                keywords=arguments["keywords"],
                faiss_subset=faiss_subset,
            )
            return {
                "faiss_subset": new_subset,
                "frame_results": [],
                "subtitle_results": [],
            }

        if tool_name == "_search_frames_with_multiple_entities":
            _, metadata_list = self._search_frames_with_multiple_entities(
                entities=arguments["entities"],
                top_k=arguments["top_k"],
                faiss_subset=faiss_subset,
            )
            return {
                "faiss_subset": faiss_subset,
                "frame_results": metadata_list,
                "subtitle_results": [],
            }

        if tool_name == "_search_frames_just_by_scope":
            metadata_list = self._search_frames_just_by_scope(
                budget=arguments["budget"],
                faiss_subset=faiss_subset,
            )
            return {
                "faiss_subset": faiss_subset,
                "frame_results": metadata_list,
                "subtitle_results": [],
            }

        if tool_name == "_search_subtitles_with_multiple_keywords":
            subtitle_rows = self._search_subtitles_with_multiple_keywords(
                keywords=arguments["keywords"],
                faiss_subset=faiss_subset,
            )
            return {
                "faiss_subset": faiss_subset,
                "frame_results": [],
                "subtitle_results": subtitle_rows,
            }

        if tool_name == "_search_subtitles_just_by_scope":
            subtitle_rows = self._search_subtitles_just_by_scope(
                faiss_subset=faiss_subset,
            )
            return {
                "faiss_subset": faiss_subset,
                "frame_results": [],
                "subtitle_results": subtitle_rows,
            }

        self.logger.warning("MemoryAgentV6 unknown tool: %s", tool_name)
        return {
            "faiss_subset": faiss_subset,
            "frame_results": [],
            "subtitle_results": [],
        }

    def _resolve_tool_via_cloud(
        self,
        prompt_template: str,
        tools: Sequence[Dict[str, Any]],
        user_query: str,
        duration: float,
        plan_step: str,
    ) -> Tuple[Optional[str], Dict[str, Any], Dict[str, Any]]:
        prompt = prompt_template.format(
            possible_tool_list="",
            output_json_format="",
            question=user_query,
            duration=duration,
            scope_plan_step=plan_step,
            search_plan_step=plan_step,
        )
        self.agentic_retriever.append_user(prompt)
        msg = self.agentic_retriever.generate_with_tools(list(tools), reset=False)
        tool_name, arguments = self._extract_tool_selection(msg)
        return tool_name, arguments, msg

    def _frame_results_to_image_paths(
        self, frame_results: Sequence[Dict[str, Any]]
    ) -> List[str]:
        rows = [row for row in frame_results if isinstance(row, dict)]
        if not rows:
            return []
        os.makedirs(self.memory_retrieve_save_dir, exist_ok=True)
        paths: List[str] = []
        for rank, meta in enumerate(rows, start=1):
            source_path = meta.get("source_path")
            frame_id = meta.get("frame_id")
            if source_path is None or frame_id is None:
                continue
            video_fps = float(meta.get("video_fps") or 1.0)
            video_name = os.path.splitext(os.path.basename(str(source_path) or "unknown"))[0]
            second = float(frame_id) / video_fps
            filename = (
                "rank{0:02d}_sec{1:.2f}_{2}_fid{3}.jpg".format(
                    rank, second, video_name, int(frame_id)
                )
            )
            save_path = os.path.join(self.memory_retrieve_save_dir, filename)
            ok = extract_save_frame_by_index(
                video_path=str(source_path),
                output_path=save_path,
                frame_index=int(frame_id),
                backend="cv2",
            )
            if ok and os.path.isfile(save_path):
                paths.append(os.path.abspath(save_path))
        return paths

    def _append_evidence(
        self,
        frame_results: List[Dict[str, Any]],
        subtitle_results: List[Dict[str, Any]],
        frame_part: Sequence[Dict[str, Any]],
        subtitle_part: Sequence[Dict[str, Any]],
    ) -> None:
        if frame_part:
            frame_results.extend(frame_part)
        if subtitle_part:
            subtitle_results.extend(subtitle_part)

    def _execute_scope_step(
        self,
        step_text: str,
        scope_index: int,
        user_query: str,
        duration: float,
        topk: int,
        faiss_subset: Optional[FaissSubset],
        is_replan: bool,
    ) -> Tuple[Optional[FaissSubset], Dict[str, Any]]:
        tool_name = None
        arguments: Dict[str, Any] = {}
        assistant_msg: Dict[str, Any] = {}
        parse_source = "none"
        execution_pass = "replan" if is_replan else "initial"

        if is_replan or scope_index == 0:
            if is_replan:
                tool_name, arguments = LocalPlanParser.try_parse_scope1_locally_replan(
                    step_text, duration, topk=topk
                )
                tools = self.scope_tools_replan
            else:
                tool_name, arguments = LocalPlanParser.try_parse_scope1_locally(
                    step_text, duration, topk=topk
                )
                tools = self.scope_tools_1
            prompt_tpl = prompt_scope_funccall_1
        else:
            tool_name, arguments = LocalPlanParser.try_parse_scope2_locally(step_text)
            tools = self.scope_tools_2
            prompt_tpl = prompt_scope_funccall_2

        if tool_name:
            parse_source = "local"
        elif not is_replan:
            tool_name, arguments, assistant_msg = self._resolve_tool_via_cloud(
                prompt_tpl, tools, user_query, duration, step_text
            )
            if tool_name:
                parse_source = "cloud"

        out = {
            "faiss_subset": faiss_subset,
            "frame_results": [],
            "subtitle_results": [],
        }
        if tool_name:
            out = self._call_tool_by_name(tool_name, arguments, faiss_subset)

        trace = {
            "phase": "scope",
            "plan_step": step_text,
            "tool_name": tool_name,
            "arguments": dict(arguments) if arguments else {},
            "result": self._serialize_tool_result_for_plan(out if tool_name else None),
            "assistant_content": (assistant_msg.get("content") or "").strip(),
            "parse_source": parse_source,
            "execution_pass": execution_pass,
            "scope_index": int(scope_index),
        }
        self._log_scope_step(is_replan, tool_name, dict(arguments) if arguments else {})
        return out.get("faiss_subset"), trace

    def _execute_search_step(
        self,
        step_text: str,
        search_index: int,
        user_query: str,
        duration: float,
        topk: int,
        faiss_subset: Optional[FaissSubset],
        is_replan: bool,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        tool_name = None
        arguments: Dict[str, Any] = {}
        assistant_msg: Dict[str, Any] = {}
        parse_source = "none"
        execution_pass = "replan" if is_replan else "initial"

        if is_replan or search_index == 0:
            if is_replan:
                tool_name, arguments = LocalPlanParser.try_parse_search1_locally_replan(
                    step_text, topk=topk
                )
                tools = self.search_tools_replan
            else:
                tool_name, arguments = LocalPlanParser.try_parse_search1_locally(
                    step_text, topk=topk
                )
                tools = self.search_tools_1
            prompt_tpl = prompt_search_funccall_1
        else:
            tool_name, arguments = LocalPlanParser.try_parse_search2_locally(step_text)
            tools = self.search_tools_2
            prompt_tpl = prompt_search_funccall_2

        if tool_name:
            parse_source = "local"
        elif not is_replan:
            tool_name, arguments, assistant_msg = self._resolve_tool_via_cloud(
                prompt_tpl, tools, user_query, duration, step_text
            )
            if tool_name:
                parse_source = "cloud"

        out = {
            "frame_results": [],
            "subtitle_results": [],
        }
        if tool_name:
            out = self._call_tool_by_name(tool_name, arguments, faiss_subset)

        frame_part = out.get("frame_results") or []
        subtitle_part = out.get("subtitle_results") or []
        trace = {
            "phase": "search",
            "plan_step": step_text,
            "tool_name": tool_name,
            "arguments": dict(arguments) if arguments else {},
            "result": self._serialize_tool_result_for_plan(out if tool_name else None),
            "assistant_content": (assistant_msg.get("content") or "").strip(),
            "parse_source": parse_source,
            "execution_pass": execution_pass,
            "search_index": int(search_index),
        }
        self._log_search_step(is_replan, tool_name, frame_part, subtitle_part)
        return (
            frame_part,
            subtitle_part,
            trace,
        )

    def _execute_plan_steps(
        self,
        steps: Sequence[str],
        user_query: str,
        duration: float,
        topk: int,
        faiss_subset: Optional[FaissSubset],
        frame_results: List[Dict[str, Any]],
        subtitle_results: List[Dict[str, Any]],
        tool_traces: List[Dict[str, Any]],
        is_replan: bool,
    ) -> Tuple[Optional[FaissSubset], bool]:
        scope_count = 0
        search_count = 0
        ran_second_search = False

        for step in steps:
            step_text = str(step)
            if step_text.startswith("[Scope]"):
                faiss_subset, trace = self._execute_scope_step(
                    step_text,
                    scope_count,
                    user_query,
                    duration,
                    topk,
                    faiss_subset,
                    is_replan,
                )
                tool_traces.append(trace)
                scope_count += 1
                continue

            if step_text.startswith("[Search]"):
                frame_part, subtitle_part, trace = self._execute_search_step(
                    step_text,
                    search_count,
                    user_query,
                    duration,
                    topk,
                    faiss_subset,
                    is_replan,
                )
                self._append_evidence(
                    frame_results, subtitle_results, frame_part, subtitle_part
                )
                tool_traces.append(trace)
                if search_count >= 1:
                    ran_second_search = True
                search_count += 1
                continue

        return faiss_subset, ran_second_search

    def agentic_retrieve_and_answer_pipeline( # _with_existing_plan
        self, user_query: str, options: Optional[Sequence[str]] = None
    ) -> Dict[str, Any]:
        self.agentic_retriever.reset_session()
        with self.databasemap.acquire() as videos:
            if videos:
                duration = float(videos[0].get("duration") or 0.0)
            else:
                duration = 0.0

        duration_for_log = max(duration, 1.0)
        self.replan_budget = int(0.5 * math.log(duration_for_log))
        initial_replan_budget = int(self.replan_budget)
        replan_budget_log: List[Dict[str, Any]] = []
        self._log_replan_budget(
            replan_budget_log,
            "init",
            initial_replan_budget,
            initial_replan_budget,
        )

        is_long_video = duration > LONG_VIDEO_THRE
        subtitle_count = len(self.srt) if self.srt is not None else 0
        has_excessive_subtitle = subtitle_count > EX_SUBTITLE_THRE
        topk = int(self.memory_topk)
        options_text = self._format_options_text(options)

        if is_long_video and has_excessive_subtitle:
            plan_variant = "long_with_srt"
            plan_prompt = prompt_generate_plan_with_faiss_and_srt_of_long_video.format(
                video_duration=duration, question=user_query
            )
        elif is_long_video:
            plan_variant = "long_faiss_only"
            plan_prompt = prompt_generate_plan_with_faiss_of_long_video.format(
                video_duration=duration, question=user_query
            )
        else:
            plan_variant = "short"
            plan_prompt = prompt_generate_plan_with_faiss_of_short_video.format(
                video_duration=duration, question=user_query
            )

        self.agentic_retriever.append_user(plan_prompt)
        plan_text = self.agentic_retriever.generate(reset=False)
        steps = self._parse_plan_steps(plan_text)
        if not steps or not str(steps[0]).startswith("[Scope]"):
            steps = ["[Scope] Pay attention to the entire video"] + steps
        self._log_plan_received(steps)

        faiss_subset: Optional[FaissSubset] = None
        frame_results: List[Dict[str, Any]] = []
        subtitle_results: List[Dict[str, Any]] = []
        tool_traces: List[Dict[str, Any]] = []

        faiss_subset, ran_second_search = self._execute_plan_steps(
            steps,
            user_query,
            duration,
            topk,
            faiss_subset,
            frame_results,
            subtitle_results,
            tool_traces,
            is_replan=False,
        )

        if not ran_second_search and not has_excessive_subtitle:
            all_subs = self._get_all_subtitles(faiss_subset=faiss_subset)
            self._append_evidence(frame_results, subtitle_results, [], all_subs)
            self._log_search_step(False, "_get_all_subtitles", [], all_subs)
            tool_traces.append(
                {
                    "phase": "search",
                    "plan_step": (
                        "[Synthetic] _get_all_subtitles "
                        "(no second [Search], subtitles not excessive)"
                    ),
                    "tool_name": "_get_all_subtitles",
                    "arguments": {},
                    "result": self._serialize_tool_result_for_plan(
                        {
                            "frame_results": [],
                            "subtitle_results": all_subs,
                        }
                    ),
                    "assistant_content": "",
                    "parse_source": "builtin",
                    "execution_pass": "initial",
                    "search_index": -1,
                }
            )

        new_frame_results = list(frame_results)
        new_subtitle_results = list(subtitle_results)
        final_answer = ""
        answer_loop_records: List[Dict[str, Any]] = []
        answer_final_record: Optional[Dict[str, Any]] = None
        loop_round = 0

        while self.replan_budget > 1:
            loop_round += 1
            budget_at_start = int(self.replan_budget)
            self._log_replan_budget(
                replan_budget_log,
                "loop_enter",
                budget_at_start,
                budget_at_start,
                loop_round=loop_round,
            )

            retrieval_context = self._format_retrieval_context(
                new_frame_results, new_subtitle_results
            )
            answer_prompt = prompt_answer_or_replan.format(
                video_time=duration,
                question=user_query,
                options_text=options_text,
                retrieval_context=retrieval_context,
            )
            image_paths = self._frame_results_to_image_paths(new_frame_results)
            self.agentic_retriever.append_user(answer_prompt, image_paths=image_paths)
            answer_or_replan = self.agentic_retriever.generate(reset=False)
            parsed_response = self._split_answer_or_replan_response(answer_or_replan)
            self._log_model_decision(
                budget_at_start, parsed_response, loop_round=loop_round
            )

            round_record: Dict[str, Any] = {
                "round": loop_round,
                "replan_budget_at_start": budget_at_start,
                "retrieval_context": retrieval_context,
                "frame_image_count": len(image_paths),
                "model": {
                    "raw": parsed_response["raw"],
                    "reasoning": parsed_response["reasoning"],
                    "outcome": parsed_response["outcome"],
                    "answer_letter": parsed_response["answer_letter"],
                    "replan_steps": parsed_response["replan_steps"],
                },
                "replan_tool_call_start_index": None,
                "replan_tool_call_end_index": None,
            }

            if parsed_response["outcome"] == "answer":
                final_answer = parsed_response["answer_letter"] or self._extract_answer_text(
                    answer_or_replan
                )
                self._log_replan_budget(
                    replan_budget_log,
                    "loop_exit_answer",
                    budget_at_start,
                    budget_at_start,
                    loop_round=loop_round,
                )
                answer_loop_records.append(round_record)
                break

            if parsed_response["outcome"] == "replan":
                replan_steps = parsed_response["replan_steps"]
                new_frame_results = []
                new_subtitle_results = []
                replan_traces: List[Dict[str, Any]] = []
                round_record["replan_tool_call_start_index"] = len(tool_traces)
                self._execute_plan_steps(
                    replan_steps,
                    user_query,
                    duration,
                    topk,
                    faiss_subset,
                    new_frame_results,
                    new_subtitle_results,
                    replan_traces,
                    is_replan=True,
                )
                tool_traces.extend(replan_traces)
                if tool_traces:
                    round_record["replan_tool_call_end_index"] = len(tool_traces) - 1
                self._append_evidence(
                    frame_results,
                    subtitle_results,
                    new_frame_results,
                    new_subtitle_results,
                )
                self._log_replan_budget(
                    replan_budget_log,
                    "loop_exit_replan",
                    budget_at_start,
                    budget_at_start,
                    loop_round=loop_round,
                )
            else:
                self._log_replan_budget(
                    replan_budget_log,
                    "loop_exit_other",
                    budget_at_start,
                    budget_at_start,
                    loop_round=loop_round,
                )

            answer_loop_records.append(round_record)

            budget_before_dec = int(self.replan_budget)
            self.replan_budget -= 1
            self._log_replan_budget(
                replan_budget_log,
                "decrement",
                budget_before_dec,
                int(self.replan_budget),
                loop_round=loop_round,
            )

        self._log_replan_budget(
            replan_budget_log,
            "final",
            int(self.replan_budget),
            int(self.replan_budget),
            note="before answer_now" if not final_answer else "answered in loop",
        )

        if not final_answer:
            retrieval_context = self._format_retrieval_context(
                new_frame_results, new_subtitle_results
            )
            answer_now_prompt = prompt_answer_now.format(
                video_time=duration,
                question=user_query,
                options_text=options_text,
                retrieval_context=retrieval_context,
            )
            image_paths = self._frame_results_to_image_paths(new_frame_results)
            self.agentic_retriever.append_user(answer_now_prompt, image_paths=image_paths)
            answer_now_text = self.agentic_retriever.generate(reset=False)
            parsed_final = self._split_answer_or_replan_response(answer_now_text)
            self._log_model_decision(
                int(self.replan_budget), parsed_final, phase="AnswerNow"
            )
            final_answer = parsed_final["answer_letter"] or self._extract_answer_text(
                answer_now_text
            )
            answer_final_record = {
                "raw": parsed_final["raw"],
                "reasoning": parsed_final["reasoning"],
                "outcome": parsed_final["outcome"],
                "answer_letter": final_answer,
                "frame_image_count": len(image_paths),
                "retrieval_context": retrieval_context,
            }

        final_retrieval_context = self._format_retrieval_context(
            new_frame_results, new_subtitle_results
        )

        option_list = None
        if options:
            option_list = [str(item).strip() for item in options if str(item).strip()]
            if not option_list:
                option_list = None

        run_record: Dict[str, Any] = {
            "ts": time.time(),
            "schema_version": "v6",
            "question": user_query,
            "options": option_list,
            "video_duration_sec": duration,
            "flags": {
                "is_long_video": is_long_video,
                "has_excessive_subtitle": has_excessive_subtitle,
                "subtitle_count": subtitle_count,
            },
            "replan_budget": {
                "initial": initial_replan_budget,
                "final": int(self.replan_budget),
                "log": replan_budget_log,
            },
            "planning": {
                "raw": plan_text,
                "steps": steps,
                "prompt_variant": plan_variant,
            },
            "tool_calls": tool_traces,
            "answer_loop": answer_loop_records,
            "answer_final": answer_final_record,
            "final_answer": final_answer,
            "retrieval_context": final_retrieval_context,
        }

        plan_json_path = self._resolve_plan_json_path()
        if plan_json_path:
            self._append_plan_trace_json(plan_json_path, run_record)
            self.logger.debug("MemoryAgentV6 wrote plan trace: %s", plan_json_path)

        return {
            "final_answer": final_answer,
            "frame_results": frame_results,
            "subtitle_results": subtitle_results,
            "retrieval_context": final_retrieval_context,
            "tool_traces": tool_traces,
            "plan_steps": steps,
            "answer_loop": answer_loop_records,
            "replan_budget_log": replan_budget_log,
        }

    def agentic_retrieve_and_answer_pipeline_with_existing_plan(
        self,
        user_query: str,
        options: Optional[Sequence[str]] = None,
        *,
        existing_plan_json_path: str,
    ) -> Dict[str, Any]:
        """Replay tool_calls from an existing plan JSON, then run answer-or-replan like
        agentic_retrieve_and_answer_pipeline without cloud planning or plan trace writes."""
        self.agentic_retriever.reset_session()
        path = str(existing_plan_json_path or "").strip()
        if not path:
            raise ValueError("existing_plan_json_path 为空")
        if not os.path.isfile(path):
            raise ValueError("plan 文件不存在: %s" % path)
        with open(path, "r", encoding="utf-8") as fp:
            plan_file_data = json.load(fp)
        if not isinstance(plan_file_data, dict):
            raise ValueError("plan 文件根节点必须是 JSON 对象")

        q = str(user_query or "").strip()
        runs = plan_file_data.get("runs")
        if not isinstance(runs, list) or not runs:
            raise ValueError("plan JSON 中缺少非空的 runs 列表")
        matched_run: Optional[Dict[str, Any]] = None
        for run in runs:
            if not isinstance(run, dict):
                continue
            rq = str(run.get("question") or "").strip()
            if rq == q:
                matched_run = run
                break
        if matched_run is None:
            previews: List[str] = []
            for run in runs:
                if isinstance(run, dict):
                    previews.append(str(run.get("question") or "")[:120])
            raise ValueError(
                "plan JSON 中未找到与 user_query 完全匹配的 question（strip 后相等）；"
                "user_query=%r；runs 条数=%d；各条 question 前 120 字预览=%r"
                % (q, len(runs), previews)
            )

        recorded_calls = matched_run.get("tool_calls")
        if not isinstance(recorded_calls, list):
            raise ValueError("匹配到的 run 缺少 tool_calls 列表")

        planning = matched_run.get("planning")
        if isinstance(planning, dict):
            steps = planning.get("steps")
            if isinstance(steps, list):
                plan_steps = [str(item).strip() for item in steps if str(item).strip()]
            else:
                plan_steps = []
        else:
            plan_steps = []

        with self.databasemap.acquire() as videos:
            if videos:
                duration = float(videos[0].get("duration") or 0.0)
            else:
                duration = 0.0

        duration_for_log = max(duration, 1.0)
        self.replan_budget = int(0.5 * math.log(duration_for_log))
        initial_replan_budget = int(self.replan_budget)
        replan_budget_log: List[Dict[str, Any]] = []
        self._log_replan_budget(
            replan_budget_log,
            "init",
            initial_replan_budget,
            initial_replan_budget,
            note="existing_plan replay",
        )
        topk = int(self.memory_topk)
        options_text = self._format_options_text(options)
        if plan_steps:
            self._log_plan_received(plan_steps, reused=True)

        faiss_subset: Optional[FaissSubset] = None
        frame_results: List[Dict[str, Any]] = []
        subtitle_results: List[Dict[str, Any]] = []

        for trace_row in recorded_calls:
            if not isinstance(trace_row, dict):
                continue
            phase = str(trace_row.get("phase") or "").strip().lower()
            step_text = str(trace_row.get("plan_step") or "")
            raw_name = trace_row.get("tool_name")
            tool_name = str(raw_name).strip() if raw_name is not None else ""
            if not tool_name:
                tool_name = None
            arguments = trace_row.get("arguments")
            if not isinstance(arguments, dict):
                arguments = {}

            if phase == "scope":
                if not tool_name:
                    self.logger.warning(
                        "MemoryAgentV6 [Scope] 复用 plan 无 tool_name: step=%s",
                        step_text[:200],
                    )
                if tool_name:
                    out = self._call_tool_by_name(tool_name, arguments, faiss_subset)
                    faiss_subset = out.get("faiss_subset", faiss_subset)
                    self._log_scope_step(False, tool_name, arguments)
                continue

            if phase == "search":
                if not tool_name:
                    self.logger.warning(
                        "MemoryAgentV6 [Search] 复用 plan 无 tool_name: step=%s",
                        step_text[:200],
                    )
                if tool_name == "_get_all_subtitles":
                    all_subs = self._get_all_subtitles(faiss_subset=faiss_subset)
                    self._append_evidence(frame_results, subtitle_results, [], all_subs)
                    self._log_search_step(False, tool_name, [], all_subs)
                elif tool_name:
                    out = self._call_tool_by_name(tool_name, arguments, faiss_subset)
                    frame_part = out.get("frame_results") or []
                    subtitle_part = out.get("subtitle_results") or []
                    self._append_evidence(
                        frame_results, subtitle_results, frame_part, subtitle_part
                    )
                    self._log_search_step(False, tool_name, frame_part, subtitle_part)
                continue

            if phase == "enhance":
                self.logger.warning(
                    "MemoryAgentV6 复用 plan 跳过 [Enhance] step=%s",
                    step_text[:200],
                )
                continue

            self.logger.warning(
                "MemoryAgentV6 复用 plan 跳过未知 phase=%r plan_step=%s",
                trace_row.get("phase"),
                step_text[:200],
            )

        new_frame_results = list(frame_results)
        new_subtitle_results = list(subtitle_results)
        final_answer = ""
        answer_loop_records: List[Dict[str, Any]] = []
        loop_round = 0

        while self.replan_budget > 1:
            loop_round += 1
            budget_at_start = int(self.replan_budget)
            self._log_replan_budget(
                replan_budget_log,
                "loop_enter",
                budget_at_start,
                budget_at_start,
                loop_round=loop_round,
            )

            retrieval_context = self._format_retrieval_context(
                new_frame_results, new_subtitle_results
            )
            answer_prompt = prompt_answer_or_replan.format(
                video_time=duration,
                question=user_query,
                options_text=options_text,
                retrieval_context=retrieval_context,
            )
            image_paths = self._frame_results_to_image_paths(new_frame_results)
            self.agentic_retriever.append_user(answer_prompt, image_paths=image_paths)
            answer_or_replan = self.agentic_retriever.generate(reset=False)
            parsed_response = self._split_answer_or_replan_response(answer_or_replan)
            self._log_model_decision(
                budget_at_start, parsed_response, loop_round=loop_round
            )

            round_record: Dict[str, Any] = {
                "round": loop_round,
                "replan_budget_at_start": budget_at_start,
                "retrieval_context": retrieval_context,
                "frame_image_count": len(image_paths),
                "model": {
                    "raw": parsed_response["raw"],
                    "reasoning": parsed_response["reasoning"],
                    "outcome": parsed_response["outcome"],
                    "answer_letter": parsed_response["answer_letter"],
                    "replan_steps": parsed_response["replan_steps"],
                },
            }

            if parsed_response["outcome"] == "answer":
                final_answer = parsed_response["answer_letter"] or self._extract_answer_text(
                    answer_or_replan
                )
                self._log_replan_budget(
                    replan_budget_log,
                    "loop_exit_answer",
                    budget_at_start,
                    budget_at_start,
                    loop_round=loop_round,
                )
                answer_loop_records.append(round_record)
                break

            if parsed_response["outcome"] == "replan":
                replan_steps = parsed_response["replan_steps"]
                new_frame_results = []
                new_subtitle_results = []
                discard_traces: List[Dict[str, Any]] = []
                self._execute_plan_steps(
                    replan_steps,
                    user_query,
                    duration,
                    topk,
                    faiss_subset,
                    new_frame_results,
                    new_subtitle_results,
                    discard_traces,
                    is_replan=True,
                )
                self._append_evidence(
                    frame_results,
                    subtitle_results,
                    new_frame_results,
                    new_subtitle_results,
                )
                self._log_replan_budget(
                    replan_budget_log,
                    "loop_exit_replan",
                    budget_at_start,
                    budget_at_start,
                    loop_round=loop_round,
                )
            else:
                self._log_replan_budget(
                    replan_budget_log,
                    "loop_exit_other",
                    budget_at_start,
                    budget_at_start,
                    loop_round=loop_round,
                )

            answer_loop_records.append(round_record)

            budget_before_dec = int(self.replan_budget)
            self.replan_budget -= 1
            self._log_replan_budget(
                replan_budget_log,
                "decrement",
                budget_before_dec,
                int(self.replan_budget),
                loop_round=loop_round,
            )

        self._log_replan_budget(
            replan_budget_log,
            "final",
            int(self.replan_budget),
            int(self.replan_budget),
            note="before answer_now" if not final_answer else "answered in loop",
        )

        if not final_answer:
            retrieval_context = self._format_retrieval_context(
                new_frame_results, new_subtitle_results
            )
            answer_now_prompt = prompt_answer_now.format(
                video_time=duration,
                question=user_query,
                options_text=options_text,
                retrieval_context=retrieval_context,
            )
            image_paths = self._frame_results_to_image_paths(new_frame_results)
            self.agentic_retriever.append_user(answer_now_prompt, image_paths=image_paths)
            answer_now_text = self.agentic_retriever.generate(reset=False)
            parsed_final = self._split_answer_or_replan_response(answer_now_text)
            self._log_model_decision(
                int(self.replan_budget), parsed_final, phase="AnswerNow"
            )
            final_answer = parsed_final["answer_letter"] or self._extract_answer_text(
                answer_now_text
            )

        final_retrieval_context = self._format_retrieval_context(
            new_frame_results, new_subtitle_results
        )

        return {
            "final_answer": final_answer,
            "frame_results": frame_results,
            "subtitle_results": subtitle_results,
            "retrieval_context": final_retrieval_context,
            "tool_traces": [],
            "plan_steps": plan_steps,
            "answer_loop": answer_loop_records,
            "replan_budget_log": replan_budget_log,
            "existing_plan_json_path": path,
        }
