import importlib.util
import json
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.agent.prompts_for_symphony import (
    ocr_template,
    output_json_format,
    possible_tool_list,
    prompt_enhance_funccall,
    prompt_generate_plan_with_faiss_and_srt,
    prompt_scope_funccall,
    prompt_search_funccall,
    rag_prompt_after_agentic_retrival,
    yolo_template,
)
from src.config import Config
from src.llm.reasoner import AgenticRetrieverAPI
from src.memory.memory_manager import MemoryManagerBase
from src.memory.query.query_vectorizer import QueryVectorizer
from src.video_utils.about_frame import extract_frame_by_index

def local_tool_use(func):
    def wrapper(*args, **kwargs):
        # 执行前的额外操作
        result = func(*args, **kwargs)
        # 执行后的额外操作
        return result
    return wrapper

class MemoryAgent(MemoryManagerBase):
    """在 MemoryManagerBase 上扩展范围检索等工具能力，以便智能体给出规划。"""

    def __init__(self, config: Config = None):
        super().__init__(config=config)
        self.agentic_retriever = AgenticRetrieverAPI(config=self._config)
        self._initialize_tools_list()
        self._initialize_enhance_tools()

    def _initialize_enhance_tools(self):
        self.ocr_language = self._config.ocr_language
        self.ocr_conf_threshold = float(self._config.ocr_conf_threshold)
        self.yolo_model_path = self._config.yolo_model_path
        self.yolo_conf_threshold = float(self._config.yolo_conf_threshold)

        self.ocr_model = None
        self.yolo_model = None
        self.yolo_names = None

        if importlib.util.find_spec("easyocr") is None:
            raise ImportError("easyocr is not installed. Please install easyocr first.")
        if importlib.util.find_spec("torch") is None:
            raise ImportError("torch is not installed. Please install torch first.")
        if importlib.util.find_spec("ultralytics") is None:
            raise ImportError("ultralytics is not installed. Please install ultralytics first.")

        import easyocr
        import torch
        from ultralytics import YOLO

        langs = [str(self.ocr_language).strip()]
        if not langs[0]:
            langs = ["en"]

        self.ocr_model = easyocr.Reader(
            langs,
            gpu=bool(torch.cuda.is_available()),
            download_enabled=False,
        )
        self.yolo_model = YOLO(self.yolo_model_path)
        self.yolo_names = self.yolo_model.names

    def _initialize_tools_list(self) -> None:
        self.scope_tools = [
            {
                "type": "function",
                "function": {
                    "name": "_get_subset_by_period",
                    "description": "Obtain the subset of semantic vector library according to the time interval of video",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "start_time": {
                                "type": "integer",
                                "description": "The start time (second) of the period",
                            },
                            "end_time": {
                                "type": "integer",
                                "description": "The end time (second) of the period",
                            },
                        },
                        "required": ["start_time", "end_time"],
                    },
                },
            },
            {
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
                                "description": "Two integers [left_seconds, right_seconds] around the anchor time, e.g. [-15, 15]",
                            },
                        },
                        "required": ["event", "scope"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "_get_subset_by_keyword_subtitle",
                    "description": "Obtain a temporal subset by anchoring at the subtitle time where the keyword appears",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "keyword": {
                                "type": "string",
                                "description": "Keyword to search in subtitles",
                            },
                            "keywords_location": {
                                "type": "string",
                                "description": "Which occurrence time to use as anchor when multiple matches exist: first | last | average",
                            },
                            "scope": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "description": "Two integers [left_seconds, right_seconds] around the anchor time, e.g. [-15, 15]",
                            },
                        },
                        "required": ["keyword", "keywords_location", "scope"],
                    },
                },
            },
        ]

        self.search_tools = [
            {
                "type": "function",
                "function": {
                    "name": "_search_frames_with_multiple_entities",
                    "description": "Search frames by multiple entity texts (their similarities are aggregated) and return top_k frame metadata",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "entities": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "A list of entity strings to search for",
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Number of top frames to return",
                            },
                        },
                        "required": ["entities", "top_k"],
                    },
                },
            },
            {
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
            },
            {
                "type": "function",
                "function": {
                    "name": "_search_subtitles_with_multiple_keywords",
                    "description": "Search subtitle segments by multiple keywords within the current scope and return top_k segments",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "keywords": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "A list of keywords to search for in subtitles",
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Number of top subtitle segments to return",
                            },
                        },
                        "required": ["keywords", "top_k"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "_search_subtitles_just_by_scope",
                    "description": "Retrieve subtitle segments just by the current scope, and subsample to budget if needed",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "budget": {
                                "type": "integer",
                                "description": "Number of subtitle segments to keep within the scope",
                            }
                        },
                        "required": ["budget"],
                    },
                },
            },
        ]

        self.enhance_tools = [
            {
                "type": "function",
                "function": {
                    "name": "_enhance_via_OCR",
                    "description": "Enhance retrieved frames via OCR and return per-frame recognized texts (interface reserved)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "retrieved_frames": {
                                "type": "array",
                                "items": {"type": "object"},
                                "description": "A list of retrieved frame metadata objects",
                            }
                        },
                        "required": ["retrieved_frames"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "_enhance_via_YOLO",
                    "description": "Enhance retrieved frames via object detection (YOLO) and return per-frame descriptions (interface reserved)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "retrieved_frames": {
                                "type": "array",
                                "items": {"type": "object"},
                                "description": "A list of retrieved frame metadata objects",
                            },
                            "objects": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Optional object categories to detect; if omitted, describe detected objects via a template",
                            },
                        },
                        "required": ["retrieved_frames"],
                    },
                },
            },
        ]

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

    def _subset_to_time_range(
        self, start_id: Optional[int], subset_vectors: Optional[np.ndarray]
    ) -> Tuple[Optional[float], Optional[float]]:
        if self.vector_count == 0:
            return None, None

        if subset_vectors is None:
            first_rec = self.databasemap[0]
            last_rec = self.databasemap[self.vector_count - 1]
        else:
            if start_id is None or start_id < 0 or subset_vectors.size == 0:
                return None, None
            end_id = int(start_id) + int(subset_vectors.shape[0]) - 1
            if end_id >= self.vector_count:
                return None, None
            first_rec = self.databasemap[int(start_id)]
            last_rec = self.databasemap[end_id]

        start_t = float(first_rec["frame_id"]) / float(first_rec.get("video_fps") or 1.0)
        end_t = float(last_rec["frame_id"]) / float(last_rec.get("video_fps") or 1.0)
        if start_t > end_t:
            return end_t, start_t
        return start_t, end_t

    def _parse_scope(self, scope: Sequence[int]) -> Tuple[float, float]:
        if scope is None or len(scope) != 2:
            return -15.0, 15.0
        return float(scope[0]), float(scope[1])

    @local_tool_use
    def _query_faiss_subset(
        self,
        query_vector: np.ndarray,
        top_k: int,
        start_id: Optional[int] = None,
        subset_vectors: Optional[np.ndarray] = None,
    ) -> Tuple[List[int], List[float]]:
        if subset_vectors is None or subset_vectors.size == 0:
            return self._query_faiss(query_vector, top_k=top_k)

        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        query_vector = query_vector.astype(np.float32)
        candidate_vectors = subset_vectors.astype(np.float32)
        if candidate_vectors.ndim == 1:
            candidate_vectors = candidate_vectors.reshape(1, -1)

        if top_k <= 0:
            return [], []

        if self.faiss_index_type == "FlatIP":
            score_vec = np.matmul(candidate_vectors, query_vector[0])
            order = np.argsort(-score_vec)[: min(top_k, score_vec.shape[0])]
        else:
            delta = candidate_vectors - query_vector[0]
            score_vec = np.sum(delta * delta, axis=1)
            order = np.argsort(score_vec)[: min(top_k, score_vec.shape[0])]

        base_id = int(start_id) if start_id is not None and start_id >= 0 else 0
        vector_ids = [base_id + int(i) for i in order.tolist()]
        scores = [float(score_vec[i]) for i in order.tolist()]
        return vector_ids, scores

    @local_tool_use
    def _get_subset(self, start_frameid: Optional[int], end_frameid: Optional[int]) -> Tuple[int, np.ndarray]:
        return self._get_faiss_subset(start_frameid=start_frameid, end_frameid=end_frameid)

    def _get_faiss_subset(self, start_frameid: Optional[int], end_frameid: Optional[int]) -> Tuple[int, np.ndarray]:
        """根据帧号范围返回 (全局起始id, 子集向量)。无结果时返回 (-1, 空数组)。"""
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
                    # 严格大于 start_frameid 的第一个位置
                    local_start = int(np.searchsorted(frames_array, start_frameid, side="right"))

                if end_frameid is None:
                    local_end_exclusive = len(frames)
                else:
                    # 严格小于 end_frameid 的区间右边界（exclusive）
                    local_end_exclusive = int(np.searchsorted(frames_array, end_frameid, side="left"))

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

    @local_tool_use
    def _get_subset_by_period(self, start_time: float, end_time: float) -> Tuple[int, np.ndarray]:
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
        return self._get_subset(start_frameid=start_frameid, end_frameid=end_frameid)

    @local_tool_use
    def _get_subset_by_event_frame(self, event: str, scope: Sequence[int]) -> Tuple[int, np.ndarray]:
        if not event:
            return -1, np.empty((0, 0), dtype=np.float32)
        query_vector = self._encode_text_query(event)
        vector_ids, _ = self._query_faiss(query_vector, top_k=1)
        if not vector_ids:
            return -1, np.empty((0, 0), dtype=np.float32)

        anchor = self.databasemap[int(vector_ids[0])]
        anchor_time = float(anchor["frame_id"]) / float(anchor.get("video_fps") or 1.0)
        left, right = self._parse_scope(scope)
        return self._get_subset_by_period(
            start_time=anchor_time + left,
            end_time=anchor_time + right,
        )

    @local_tool_use
    def _get_subset_by_keyword_subtitle(
        self, keyword: str, keywords_location: str, scope: Sequence[int]
    ) -> Tuple[int, np.ndarray]:
        if self.srt is None or not keyword:
            return -1, np.empty((0, 0), dtype=np.float32)

        times = self.srt.search_word_time(keyword)
        if not times:
            return -1, np.empty((0, 0), dtype=np.float32)

        location = (keywords_location or "average").lower()
        if location == "first":
            anchor_time = float(times[0])
        elif location == "last":
            anchor_time = float(times[-1])
        else:
            anchor_time = float(sum(times) / len(times))

        left, right = self._parse_scope(scope)
        return self._get_subset_by_period(
            start_time=anchor_time + left,
            end_time=anchor_time + right,
        )

    @local_tool_use
    def _search_frames_with_multiple_entities(
        self,
        entities: Sequence[str],
        top_k: int,
        start_id: Optional[int] = None,
        subset_vectors: Optional[np.ndarray] = None,
    ) -> Tuple[List[float], List[Dict[str, Any]]]:
        entity_list = [str(item).strip() for item in entities if str(item).strip()]
        if not entity_list or top_k <= 0:
            return [], []

        if subset_vectors is None:
            if self.index is None or self.vector_count == 0:
                return [], []
            start_id = 0
            subset_vectors = self.index.subset(0, self.vector_count)
        if subset_vectors.size == 0:
            return [], []

        if subset_vectors.ndim == 1:
            subset_vectors = subset_vectors.reshape(1, -1)
        subset_vectors = subset_vectors.astype(np.float32)

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

        base_id = int(start_id) if start_id is not None and start_id >= 0 else 0
        selected_ids = [base_id + int(i) for i in order.tolist()]
        scores = [float(agg_scores[i]) for i in order.tolist()]
        metadata_list = self._build_metadata_list_by_ids(selected_ids)
        return scores, metadata_list

    @local_tool_use
    def _search_frames_just_by_scope(
        self,
        budget: int,
        start_id: Optional[int] = None,
        subset_vectors: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        if budget <= 0:
            return []

        if subset_vectors is None:
            if self.index is None or self.vector_count == 0:
                return []
            start_id = 0
            subset_vectors = self.index.subset(0, self.vector_count)
        if subset_vectors.size == 0:
            return []
        if subset_vectors.ndim == 1:
            subset_vectors = subset_vectors.reshape(1, -1)

        size = subset_vectors.shape[0]
        count = min(int(budget), int(size))
        chosen = np.linspace(0, size - 1, num=count, dtype=int).tolist()
        base_id = int(start_id) if start_id is not None and start_id >= 0 else 0
        selected_ids = [base_id + int(i) for i in chosen]
        return self._build_metadata_list_by_ids(selected_ids)

    @local_tool_use
    def _search_subtitles_with_multiple_keywords(
        self,
        keywords: Sequence[str],
        top_k: int,
        start_id: Optional[int] = None,
        subset_vectors: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        if top_k <= 0:
            return []
        start_t, end_t = self._subset_to_time_range(start_id=start_id, subset_vectors=subset_vectors)
        if start_t is None or end_t is None:
            return []
        return self.retrieve_segment_by_word_with_scope(
            keywords=keywords,
            top_k=top_k,
            start_time=start_t,
            end_time=end_t,
        )

    @local_tool_use
    def _search_subtitles_just_by_scope(
        self,
        budget: int,
        start_id: Optional[int] = None,
        subset_vectors: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        if budget <= 0:
            return []
        start_t, end_t = self._subset_to_time_range(start_id=start_id, subset_vectors=subset_vectors)
        if start_t is None or end_t is None:
            return []

        rows = self.retrieve_segment_by_period(start_t=start_t, end_t=end_t)
        if not rows:
            return []
        if len(rows) <= budget:
            return rows

        chosen = np.linspace(0, len(rows) - 1, num=int(budget), dtype=int).tolist()
        return [rows[i] for i in chosen]

    @local_tool_use
    def _enhance_via_OCR(self, retrieved_frames: Sequence[Any]) -> List[str]:
        if not retrieved_frames:
            return []

        per_frame_lines: List[str] = []
        for idx, frame_item in enumerate(retrieved_frames):
            frame_bgr = None
            if isinstance(frame_item, np.ndarray):
                frame_bgr = frame_item
            elif isinstance(frame_item, dict):
                source_path = frame_item.get("source_path")
                frame_id = frame_item.get("frame_id")
                if source_path and frame_id is not None:
                    frame_bgr = extract_frame_by_index(
                        str(source_path), int(frame_id), backend="cv2"
                    )

            if frame_bgr is None:
                text_line = "frame {idx}: failed to load frame pixels".format(idx=idx)
                per_frame_lines.append(text_line)
                continue

            frame_rgb = frame_bgr[:, :, ::-1]
            ocr_rows = self.ocr_model.readtext(frame_rgb)
            kept_texts = [
                str(row[1]).strip()
                for row in ocr_rows
                if float(row[2]) >= self.ocr_conf_threshold and str(row[1]).strip()
            ]
            if kept_texts:
                text_line = "frame {idx}: ".format(idx=idx) + "; ".join(
                    ['"{txt}"'.format(txt=txt) for txt in kept_texts]
                )
            else:
                text_line = "frame {idx}: (no text)".format(idx=idx)
            per_frame_lines.append(text_line)

        merged_text = "\n".join(per_frame_lines)
        return [ocr_template.format(ocr_result=merged_text).strip()]

    @local_tool_use
    def _enhance_via_YOLO(
        self, retrieved_frames: Sequence[Any], objects: Optional[Sequence[str]] = None
    ) -> List[str]:
        if not retrieved_frames:
            return []

        target_objects = [str(obj).strip() for obj in (objects or []) if str(obj).strip()]
        target_set = set(target_objects)

        per_frame_lines: List[str] = []
        for idx, frame_item in enumerate(retrieved_frames):
            frame_bgr = None
            if isinstance(frame_item, np.ndarray):
                frame_bgr = frame_item
            elif isinstance(frame_item, dict):
                source_path = frame_item.get("source_path")
                frame_id = frame_item.get("frame_id")
                if source_path and frame_id is not None:
                    frame_bgr = extract_frame_by_index(
                        str(source_path), int(frame_id), backend="cv2"
                    )

            if frame_bgr is None:
                yolo_line = "frame {idx}: failed to load frame pixels".format(idx=idx)
                per_frame_lines.append(yolo_line)
                continue

            yolo_result = self.yolo_model(frame_bgr, verbose=False)
            if not yolo_result:
                continue

            boxes = yolo_result[0].boxes
            class_count: Dict[str, int] = {}
            for i in range(len(boxes)):
                conf = float(boxes.conf[i].item())
                if conf < self.yolo_conf_threshold:
                    continue
                cls_id = int(boxes.cls[i].item())
                if isinstance(self.yolo_names, dict):
                    cls_name = str(self.yolo_names.get(cls_id, cls_id))
                elif isinstance(self.yolo_names, (list, tuple)):
                    if 0 <= cls_id < len(self.yolo_names):
                        cls_name = str(self.yolo_names[cls_id])
                    else:
                        cls_name = str(cls_id)
                else:
                    cls_name = str(cls_id)
                class_count[cls_name] = class_count.get(cls_name, 0) + 1

            if not class_count:
                continue

            def _count_phrase(name: str, cnt: int) -> str:
                if cnt <= 1:
                    return "1 {name}".format(name=name)
                return "{cnt} {name}s".format(cnt=cnt, name=name)

            det_desc = ", ".join(
                [_count_phrase(name, cnt) for name, cnt in class_count.items()]
            )
            yolo_line = "frame {idx}: {desc}".format(idx=idx, desc=det_desc)

            if target_set:
                prioritized_pairs = []
                for name in target_objects:
                    count = class_count.get(name, 0)
                    if count > 0:
                        prioritized_pairs.append((name, count))

                remaining_pairs = []
                for name, count in class_count.items():
                    if name not in target_set:
                        remaining_pairs.append((name, count))

                if len(prioritized_pairs) == len(target_set):
                    ordered_pairs = prioritized_pairs
                else:
                    ordered_pairs = prioritized_pairs + remaining_pairs
                det_desc = ", ".join(
                    [_count_phrase(name, cnt) for name, cnt in ordered_pairs]
                )
                yolo_line = "frame {idx}: {desc}".format(idx=idx, desc=det_desc)

            per_frame_lines.append(yolo_line)

        merged_text = "\n".join(per_frame_lines)
        return [yolo_template.format(yolo_result=merged_text).strip()]

    def _parse_plan_steps(self, raw_plan: str) -> List[str]:
        text = str(raw_plan or "").strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        return [line.strip() for line in text.splitlines() if line.strip()]

    def _tool_list_to_prompt(self, tools: Sequence[Dict[str, Any]]) -> str:
        return possible_tool_list.format(
            tool_list=json.dumps(list(tools), ensure_ascii=False, indent=2)
        ).strip()

    def _output_schema_prompt(self, tools: Sequence[Dict[str, Any]]) -> str:
        schema = {
            "type": "object",
            "properties": {
                "tool_name": {
                    "type": "string",
                    "enum": [item["function"]["name"] for item in tools],
                },
                "arguments": {"type": "object"},
            },
            "required": ["tool_name", "arguments"],
        }
        return output_json_format.format(
            tool_input_schema=json.dumps(schema, ensure_ascii=False, indent=2)
        ).strip()

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

    def _call_tool_by_name(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        start_id: Optional[int]=None,
        subset_vectors: Optional[np.ndarray]=None,
        retrieved_frames: Sequence[Dict[str, Any]]=None,
    ) -> Dict[str, Any]:
        if tool_name == "_get_subset_by_period":
            new_start_id, new_subset = self._get_subset_by_period(
                start_time=arguments["start_time"], end_time=arguments["end_time"]
            )
            return {
                "start_id": new_start_id,
                "subset_vectors": new_subset,
                "scope_desc": "period [{s}, {e}]".format(
                    s=arguments["start_time"], e=arguments["end_time"]
                ),
            }

        if tool_name == "_get_subset_by_event_frame":
            new_start_id, new_subset = self._get_subset_by_event_frame(
                event=arguments["event"], scope=arguments["scope"]
            )
            return {
                "start_id": new_start_id,
                "subset_vectors": new_subset,
                "scope_desc": "event '{e}' with scope {s}".format(
                    e=arguments["event"], s=arguments["scope"]
                ),
            }

        if tool_name == "_get_subset_by_keyword_subtitle":
            new_start_id, new_subset = self._get_subset_by_keyword_subtitle(
                keyword=arguments["keyword"],
                keywords_location=arguments["keywords_location"],
                scope=arguments["scope"],
            )
            return {
                "start_id": new_start_id,
                "subset_vectors": new_subset,
                "scope_desc": "subtitle keyword '{k}' ({loc}) scope {s}".format(
                    k=arguments["keyword"],
                    loc=arguments["keywords_location"],
                    s=arguments["scope"],
                ),
            }

        if tool_name == "_search_frames_with_multiple_entities":
            scores, metadata_list = self._search_frames_with_multiple_entities(
                entities=arguments["entities"],
                top_k=arguments["top_k"],
                start_id=start_id,
                subset_vectors=subset_vectors,
            )
            return {"scores": scores, "frame_results": metadata_list, "subtitle_results": []}

        if tool_name == "_search_frames_just_by_scope":
            metadata_list = self._search_frames_just_by_scope(
                budget=arguments["budget"],
                start_id=start_id,
                subset_vectors=subset_vectors,
            )
            return {"scores": [], "frame_results": metadata_list, "subtitle_results": []}

        if tool_name == "_search_subtitles_with_multiple_keywords":
            subtitle_rows = self._search_subtitles_with_multiple_keywords(
                keywords=arguments["keywords"],
                top_k=arguments["top_k"],
                start_id=start_id,
                subset_vectors=subset_vectors,
            )
            return {"scores": [], "frame_results": [], "subtitle_results": subtitle_rows}

        if tool_name == "_search_subtitles_just_by_scope":
            subtitle_rows = self._search_subtitles_just_by_scope(
                budget=arguments["budget"],
                start_id=start_id,
                subset_vectors=subset_vectors,
            )
            return {"scores": [], "frame_results": [], "subtitle_results": subtitle_rows}

        if tool_name == "_enhance_via_OCR":
            enhanced_texts = self._enhance_via_OCR(retrieved_frames=retrieved_frames)
            return {"enhance_results": enhanced_texts}

        if tool_name == "_enhance_via_YOLO":
            yolo_objects = arguments.get("objects")
            enhanced_texts = self._enhance_via_YOLO(
                retrieved_frames=retrieved_frames, objects=yolo_objects
            )
            return {"enhance_results": enhanced_texts}

        return {}

    def _format_retrieval_context(
        self,
        frame_results: Sequence[Dict[str, Any]],
        subtitle_results: Sequence[Dict[str, Any]],
        enhance_results: Sequence[str],
    ) -> str:
        lines = []

        frame_count = len(frame_results)
        if frame_count > 0:
            lines.append(
                "We provide {cnt} most relevant frames as visual evidence.".format(
                    cnt=frame_count
                )
            )

        subtitle_count = len(subtitle_results)
        if subtitle_count > 0:
            subtitle_texts: List[str] = []
            for row in subtitle_results:
                if isinstance(row, dict):
                    text = (
                        row.get("text")
                        or row.get("subtitle")
                        or row.get("content")
                        or row.get("sentence")
                        or ""
                    )
                    normalized = str(text).strip()
                    if normalized:
                        subtitle_texts.append('"{text}"'.format(text=normalized))
                    else:
                        subtitle_texts.append(
                            '"{text}"'.format(text=json.dumps(row, ensure_ascii=False))
                        )
                else:
                    normalized = str(row).strip()
                    if normalized:
                        subtitle_texts.append('"{text}"'.format(text=normalized))
            merged_subtitles = " | ".join([item for item in subtitle_texts if item])
            lines.append(
                "We also provide {cnt} most relevant subtitle snippets as language evidence: {subs}".format(
                    cnt=subtitle_count,
                    subs=merged_subtitles or "(empty subtitles)",
                )
            )
        else:
            lines.append("No relevant subtitle evidence was retrieved.")

        if enhance_results:
            lines.extend([str(item).strip() for item in enhance_results if str(item).strip()])
        else:
            lines.append("No enhancement model output is provided.")

        return "\n".join(lines)

    def _resolve_plan_json_path(self) -> Optional[str]:
        """与 databasemap 同级目录下的 plan/{视频名}.json，例如 .../short/json/a.json -> .../short/plan/a.json。"""
        map_path = getattr(self, "databasemap_file_path", None) or ""
        map_path = str(map_path).strip()
        if not map_path:
            return None
        json_dir = os.path.dirname(os.path.abspath(map_path))
        base_dir = os.path.dirname(json_dir)
        plan_dir = os.path.join(base_dir, "plan")
        basename = os.path.basename(map_path)
        return os.path.join(plan_dir, basename)

    def _serialize_tool_result_for_plan(self, result: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not result:
            return None
        out: Dict[str, Any] = {}
        for key, val in result.items():
            if key == "subset_vectors" and isinstance(val, np.ndarray):
                shape = list(val.shape)
                out[key] = {
                    "shape": shape,
                    "dtype": str(val.dtype),
                    "note": "numpy array omitted",
                }
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

    def agentic_retrieve_pipeline(
        self, user_query: str, options: Optional[Sequence[str]] = None
    ) -> Dict[str, Any]:
        with self.databasemap.acquire() as videos:
            if videos:
                first_video = videos[0]
                duration = float(first_video.get("duration") or 0.0)
            else:
                duration = 0.0

        plan_prompt = prompt_generate_plan_with_faiss_and_srt.format(
            video_duration=duration, question=user_query
        )
        self.agentic_retriever.append_user(plan_prompt)
        plan_text = self.agentic_retriever.generate(reset=True)
        steps = self._parse_plan_steps(plan_text)

        start_id = None
        subset_vectors = None
        scope_desc = ""
        frame_results: List[Dict[str, Any]] = []
        subtitle_results: List[Dict[str, Any]] = []
        enhance_results: List[str] = []
        tool_traces: List[Dict[str, Any]] = []

        # 以下是我预留的参数，不要改动
        need_tool_desc = False
        scope_tool_prompt = ""
        search_tool_prompt = ""
        enhance_tool_prompt = ""
        scope_schema_prompt = ""
        search_schema_prompt = ""
        enhance_schema_prompt = ""
        if need_tool_desc:
            scope_tool_prompt = self._tool_list_to_prompt(self.scope_tools)
            search_tool_prompt = self._tool_list_to_prompt(self.search_tools)
            enhance_tool_prompt = self._tool_list_to_prompt(self.enhance_tools)
            scope_schema_prompt = self._output_schema_prompt(self.scope_tools)
            search_schema_prompt = self._output_schema_prompt(self.search_tools)
            enhance_schema_prompt = self._output_schema_prompt(self.enhance_tools)

        for step in steps:
            step_text = str(step)
            if step_text.startswith("[Scope]"):
                prompt = prompt_scope_funccall.format(
                    possible_tool_list=scope_tool_prompt,
                    output_json_format=scope_schema_prompt,
                    question=user_query,
                    duration=duration,
                    scope_plan_step=step_text,
                )
                self.agentic_retriever.append_user(prompt)
                msg = self.agentic_retriever.generate_with_tools(self.scope_tools, reset=True)
                tool_name, arguments = self._extract_tool_selection(msg)
                scope_out = None
                if tool_name:
                    scope_out = self._call_tool_by_name(
                        tool_name=tool_name,
                        arguments=arguments,
                        start_id=start_id,
                        subset_vectors=subset_vectors,
                        retrieved_frames=frame_results,
                    )
                    start_id = scope_out.get("start_id", start_id)
                    subset_vectors = scope_out.get("subset_vectors", subset_vectors)
                    scope_desc = scope_out.get("scope_desc", scope_desc)
                tool_traces.append(
                    {
                        "phase": "scope",
                        "plan_step": step_text,
                        "tool_name": tool_name,
                        "arguments": dict(arguments) if arguments else {},
                        "result": self._serialize_tool_result_for_plan(scope_out),
                        "assistant_content": (msg.get("content") or "").strip(),
                    }
                )
                continue

            elif step_text.startswith("[Search]"):
                prompt = prompt_search_funccall.format(
                    possible_tool_list=search_tool_prompt,
                    output_json_format=search_schema_prompt,
                    question=user_query,
                    duration=duration,
                    search_plan_step=step_text,
                )
                self.agentic_retriever.append_user(prompt)
                msg = self.agentic_retriever.generate_with_tools(self.search_tools, reset=True)
                tool_name, arguments = self._extract_tool_selection(msg)
                search_out = None
                if tool_name:
                    search_out = self._call_tool_by_name(
                        tool_name=tool_name,
                        arguments=arguments,
                        start_id=start_id,
                        subset_vectors=subset_vectors,
                        retrieved_frames=frame_results,
                    )
                    frame_part = search_out.get("frame_results") or []
                    subtitle_part = search_out.get("subtitle_results") or []
                    frame_results.extend(frame_part)
                    subtitle_results.extend(subtitle_part)
                tool_traces.append(
                    {
                        "phase": "search",
                        "plan_step": step_text,
                        "tool_name": tool_name,
                        "arguments": dict(arguments) if arguments else {},
                        "result": self._serialize_tool_result_for_plan(search_out),
                        "assistant_content": (msg.get("content") or "").strip(),
                    }
                )
                continue

            elif step_text.startswith("[Enhance]"):
                prompt = prompt_enhance_funccall.format(
                    possible_tool_list=enhance_tool_prompt,
                    output_json_format=enhance_schema_prompt,
                    question=user_query,
                    duration=duration,
                    enhance_plan_step=step_text,
                )
                self.agentic_retriever.append_user(prompt)
                msg = self.agentic_retriever.generate_with_tools(self.enhance_tools, reset=True)
                tool_name, arguments = self._extract_tool_selection(msg)
                enhance_out = None
                if tool_name:
                    enhance_out = self._call_tool_by_name(
                        tool_name=tool_name,
                        arguments=arguments,
                        start_id=start_id,
                        subset_vectors=subset_vectors,
                        retrieved_frames=frame_results,
                    )
                    enhance_results.extend(enhance_out.get("enhance_results") or [])
                tool_traces.append(
                    {
                        "phase": "enhance",
                        "plan_step": step_text,
                        "tool_name": tool_name,
                        "arguments": dict(arguments) if arguments else {},
                        "result": self._serialize_tool_result_for_plan(enhance_out),
                        "assistant_content": (msg.get("content") or "").strip(),
                    }
                )

            else:
                pass

        retrieval_context = self._format_retrieval_context(
            frame_results=frame_results,
            subtitle_results=subtitle_results,
            enhance_results=enhance_results,
        )
        options_text = "(no options provided)"
        if options:
            option_items = [str(item).strip() for item in options if str(item).strip()]
            if option_items:
                options_text = " ".join(option_items)

        rag_prompt = rag_prompt_after_agentic_retrival.format(
            video_time=duration,
            question=user_query,
            options_text=options_text,
            retrieval_context=retrieval_context,
        )

        plan_json_path = self._resolve_plan_json_path()
        if plan_json_path:
            option_list = None
            if options:
                option_list = [str(item).strip() for item in options if str(item).strip()]
                if not option_list:
                    option_list = None
            run_record = {
                "ts": time.time(),
                "question": user_query,
                "options": option_list,
                "video_duration_sec": duration,
                "planning": {"raw": plan_text, "steps": steps},
                "tool_calls": tool_traces,
            }
            self._append_plan_trace_json(plan_json_path, run_record)

        return {
            "rag_prompt": rag_prompt,
            "metadata_list": frame_results,
        }

