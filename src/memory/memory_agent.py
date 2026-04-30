import importlib.util
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.agent.prompts_for_symphony import ocr_template, yolo_template
from src.config import Config
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


#   基础实现：
#   1. self._query_faiss_subset(self, query_vector, top_k, start_id=None, subset_vectors=None) 
#       根据已有subset_vectors，实现MemoryManagerBase的_query_faiss相同的功能。或者如果没有传入就直接调用_query_faiss

#   有关[Scope]的功能：
#   1. self._get_subset_by_period(self, start_time, end_time):
#       该函数通过start_time和end_time去计算start_frameid和end_frameid，然后去调用_get_subset获得子集和初始id
#   2. self._get_subset_by_event_frame(self, event, scope):
#       该函数通过一个event字符串去计算query_vector，然后去全局的向量库查询top1帧的位置，作为锚点，scope是一个两个int元素的列表，
#       比如[-15, 15], 代表锚点的前后15秒的内容作为子集返回（调用_get_subset_by_period）。
#   3. self._get_subset_by_keyword_subtitle(self, keyword, keywords_location, scope):
#       该函数通过一个keyword字符串尝试去字幕库搜索出现的时间（使用ThreadSafeSRT的search_word_time函数）。
#       如果出现了多次，则使用 keywords_location(一个字符串，"first"|"last"|"average")来选择哪一个时间作为锚点，"average"代表所有时间取平均
#       scope是一个两个int元素的列表，比如[-15, 15], 代表锚点的前后15秒的内容作为子集返回（调用_get_subset_by_period）

#   有关[Search]的功能：
#   1. self._search_frames_with_multiple_entities(self, entities, top_k, start_id=None, subset_vectors=None)
#       entities是一个字符串列表，首先会通过文本编码器对里面的所有字符串进行编码获得等列表长度的向量列表。
#       然后对于其中的所有向量都去和subset_vectors做相似度，最后相加获得总的相似度。
#       然后去取top_k，最后返回和_retrieve函数类似的scores，metadata_list
#   2. self._search_frames_just_by_scope(self, budget, start_id=None, subset_vectors=None)
#       通过start_id和subset_vectors可以算得视频的时间区间。然后该函数均匀采样返回budget个区间内的视频帧的metadata_list
#   3. self._search_subtitles_with_multiple_keywords(self, keywords, top_k, start_id=None, subset_vectors=None)
#       通过start_id和subset_vectors可以算得视频的时间区间。然后该函数对于在该时间区间内去执行retrieve_segment_by_word_with_scope
#   4. self._search_subtitles_just_by_scope(self, budget, start_id=None, subset_vectors=None)
#       通过start_id和subset_vectors可以算得视频的时间区间。然后该函数在该时间区间内去执行MemoryManagerBase的retrieve_segment_by_period

#   有关[Enhance]的功能：
#   1. self._enhance_via_OCR(self, retrieved_frames)
#       该函数通过OCR模型去识别retrieved_frames，返回retrieved_frames等长的列表，列表元素是通过OCR模型识别出来的每帧的文本
#       不过，由于我还没有选好OCR模型，所以你可以先写好接口，先不实现。
#   2. self._enhance_via_YOLO(self, retrieved_frames, objects=None)
#       该函数通过YOLO模型去对retrieved_frames进行目标识别，如果没有提供objects则通过一个文本模板描述识别到的物体。
#       如果提供了objects(一个列表，元素是物体名称的字符串)，则专门去识别在这上面的物体。
#       最终返回retrieved_frames等长的列表，列表元素是每帧通过文本模板描述识别到的物体的句子
#       不过，由于我还没有选好YOLO模型，所以你可以先写好接口，先不实现。
class MemoryAgent(MemoryManagerBase):
    """在 MemoryManagerBase 上扩展范围检索等工具能力，以便智能体给出规划。"""

    def __init__(self, config: Config = None):
        super().__init__(config=config)
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
                per_frame_lines.append(
                    "frame {idx}: (no objects detected)".format(idx=idx)
                )
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
                yolo_line = "frame {idx}: (no objects detected)".format(idx=idx)
                per_frame_lines.append(yolo_line)
                continue

            det_desc = ", ".join(
                ["{name}({cnt})".format(name=name, cnt=cnt) for name, cnt in class_count.items()]
            )
            yolo_line = "frame {idx}: detected {desc}".format(idx=idx, desc=det_desc)

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
                    [
                        "{name}({cnt})".format(name=name, cnt=cnt)
                        for name, cnt in ordered_pairs
                    ]
                )
                yolo_line = "frame {idx}: detected {desc}".format(idx=idx, desc=det_desc)

            per_frame_lines.append(yolo_line)

        merged_text = "\n".join(per_frame_lines)
        return [yolo_template.format(yolo_result=merged_text).strip()]

