#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Symphony benchmark system (v3).

Inject 使用 ``SymVideoInputByStreamWindow``（ffprobe 媒体时间窗 + 非 I 包触发 + 预算选帧），
不再使用 ``make_sym_video_input`` 的按 GOP + ``select_strategy`` 路径。

在 v1 检索编排基础上：
- 使用配置 ``memory_manager.retrieve_item_type``：``frame``（默认）或 ``clip``；
- ``clip`` 时由 ``MemoryManager`` 将 GOP mp4 写入 ``logs/memory/retrieve/clips/dialog_{id}/``，
  新 ``dialog_id`` 会删除上一对话对应子目录；同对话内每次检索会清空该对话子目录再写入。
- ``clip`` 模式下 Reasoner 的 ``memory_results`` 为上述 mp4 按 GOP 顺序拼接的**全部帧**（RGB），
  RAG 文案使用 ``build_rag_prompt_with_clips``；``frame`` 模式仍为单帧解码 + ``build_rag_prompt_with_frames``。
"""

import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

import faiss

# 项目根：symphony/v3 -> 上四级到 HelloWorld
_project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
sys.path.insert(0, _project_root)

from src.benchmark.retrieve_clip_frames import (
    count_existing_clip_mp4s,
    decode_clip_info_all_frames_bgr,
)
from src.benchmark.prompt_template import build_rag_prompt_with_clips, build_rag_prompt_with_frames
from src.memory.frame_vectorizer import SymFrameVectorizerForV3
from src.memory.memory_manager import MemoryManagerBase
from src.memory.query_vectorizer import QueryVectorizer
from src.system.symphony.v1.benchmark import SymphonySystemBench
from src.video_input.video_input import SymVideoInputByStreamWindow


class SymphonySystemBenchV3(SymphonySystemBench):
    """
    Inject：``SymVideoInputByStreamWindow`` + ``SymFrameVectorizerForV3.encode_sym_frames_list``；
    ``register_i_frames`` 仍由 ffprobe 元数据提供。检索 clip 由 ``MemoryManager.retrieve_sync`` 导出。
    """

    def _setup_logger(self):
        self.logger = logging.getLogger("SymphonySystemBenchV3")
        self.logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        self.logger.addHandler(console_handler)
        self.logger.propagate = False

    def _init_components(
        self,
        video_path: Optional[str] = None,
        faiss_path: Optional[str] = None,
        map_path: Optional[str] = None,
    ):
        """与 v1 相同，但视频输入固定为 ``SymVideoInputByStreamWindow``。"""
        self.config.memory_mode = "both"
        if faiss_path is not None:
            self.config.memory_faiss_file_path = faiss_path
        if map_path is not None:
            self.config.memory_databasemap_file_path = map_path

        self.memory_manager = MemoryManagerBase(self.config)
        self.query_vectorizer = QueryVectorizer(self.config)

        self.memory_manager.init_sync()
        self.query_vectorizer._initialize_vectorizer()

        if video_path:
            self.config.video_file_path = video_path
            self.video_input = SymVideoInputByStreamWindow(self.config)
            self.frame_vectorizer = SymFrameVectorizerForV3(self.config)
            self.video_input.init_for_file(video_path)
            ixs = getattr(self.video_input, "i_frame_indices", None)
            if ixs:
                self.memory_manager.register_i_frames(video_path, list(ixs))
            self.frame_vectorizer._initialize_vectorizer()
        else:
            self.video_input = None
            self.frame_vectorizer = None

    def _run_inject_phase(
        self, video_path: str, video_id: str, dataset_name: str, subset: Optional[str] = None
    ) -> Dict[str, Any]:
        """按媒体时间窗迭代，窗内策略已选好帧，直接编码入库。"""
        faiss_path, map_path = self._get_db_paths(dataset_name, video_id, subset)
        if os.path.isfile(faiss_path):
            self.logger.info("向量库已存在，跳过 inject: %s", faiss_path)
            self._init_components(video_path=None, faiss_path=faiss_path, map_path=map_path)
            idx = faiss.read_index(faiss_path)
            return {
                "total_frames": idx.ntotal,
                "total_vectors": idx.ntotal,
                "elapsed_sec": 0,
                "batch_size": self.batch_size,
                "skipped": True,
            }

        self._init_components(video_path=video_path, faiss_path=faiss_path, map_path=map_path)

        total_video_frames = int(getattr(self.video_input, "total_frames", 0) or 0)
        total_vectors = 0
        t0 = time.time()

        if not hasattr(self.video_input, "iter_frames_by_stream_window"):
            self.logger.error("video_input 非 SymVideoInputByStreamWindow，无法 inject")
            return {
                "total_frames": 0,
                "total_vectors": 0,
                "elapsed_sec": 0.0,
                "batch_size": self.batch_size,
                "skipped": False,
            }

        for batch in self.video_input.iter_frames_by_stream_window():
            if not batch:
                continue
            vector_data_list = self.frame_vectorizer.encode_sym_frames_list(batch)
            if vector_data_list:
                self.memory_manager.add_vectors_batch(vector_data_list)
                total_vectors += len(vector_data_list)

        self.memory_manager.current_video_name = None
        self.memory_manager.save_database_sync()
        elapsed = time.time() - t0
        self.logger.info(
            "Inject 完成(流式窗): 视频总帧约 %s -> %s 向量, 耗时 %.2fs",
            total_video_frames,
            total_vectors,
            elapsed,
        )
        return {
            "total_frames": total_video_frames,
            "total_vectors": total_vectors,
            "elapsed_sec": elapsed,
            "batch_size": self.batch_size,
            "skipped": False,
        }

    def _run_query_single(
        self,
        question: str,
        sample_id: str = "",
        sample: Optional[Dict[str, Any]] = None,
        video_time: Optional[float] = None,
        dialog_id: int = 0,
    ) -> Dict[str, Any]:
        """编码查询向量、检索；clip 路径由 MemoryManager 写入 logs/memory/retrieve/clips/。"""
        t0 = time.time()
        query_vector = self.query_vectorizer.encode_query_sync(question)
        scores, frames_metadata, clip_info = self.memory_manager.retrieve_sync(
            query_vector, dialog_id=dialog_id
        )
        retrieve_time = time.time() - t0

        result = {
            "question": question,
            "retrieve_time_sec": retrieve_time,
            "scores": scores,
        }

        rit = getattr(self.memory_manager, "memory_retrieve_item_type", None) or getattr(
            self.config, "memory_retrieve_item_type", "frame"
        )
        if isinstance(rit, str):
            rit = rit.strip().lower()
        else:
            rit = "frame"
        if rit not in ("frame", "clip"):
            rit = "frame"
        result["retrieve_item_type"] = rit

        if clip_info and clip_info.get("error"):
            result["retrieve_clip_error"] = clip_info["error"]

        query_text = question
        select_frame_num = len(frames_metadata) if frames_metadata else 0
        assert video_time, "video_time 必须要有才能创建ragprompt"

        clip_paths_ok = (
            rit == "clip"
            and clip_info
            and not clip_info.get("error")
            and count_existing_clip_mp4s(clip_info) > 0
        )
        clip_bgr_list = (
            decode_clip_info_all_frames_bgr(clip_info, logger=self.logger)
            if clip_paths_ok
            else []
        )
        num_existing_clips = count_existing_clip_mp4s(clip_info) if clip_info else 0

        if video_time is not None and frames_metadata and sample is not None:
            options = sample.get("options", [])
            if not isinstance(options, list):
                options = list(options) if options else []
            if clip_paths_ok and clip_bgr_list:
                query_text = build_rag_prompt_with_clips(
                    video_time=video_time,
                    num_selected_clips=num_existing_clips,
                    question=question,
                    options=options,
                )
            else:
                query_text = build_rag_prompt_with_frames(
                    video_time=video_time,
                    num_selected_frame=len(frames_metadata),
                    question=question,
                    options=options,
                )

        result["rag_question"] = query_text
        result["select_frame_num"] = select_frame_num
        if rit == "clip":
            result["select_clip_num"] = num_existing_clips
            result["reasoner_input_frame_count"] = (
                len(clip_bgr_list) if clip_bgr_list else 0
            )
        else:
            result["select_clip_num"] = 0
            result["reasoner_input_frame_count"] = select_frame_num

        if rit == "clip" and clip_bgr_list:
            self._fill_reasoner_from_bgr_frames(
                t0, retrieve_time, query_text, clip_bgr_list, str(sample_id), result
            )
        else:
            self._fill_reasoner_result(
                t0, retrieve_time, query_text, frames_metadata, str(sample_id), result
            )

        return result

    def _build_video_entry(
        self,
        video_id: str,
        query_results: List[Dict],
        create_v_db_time: float = 0,
    ) -> Dict[str, Any]:
        """在父类 JSON 结构上为每条 question 附加 retrieve_item_type / clip 路径等。"""
        entry = super()._build_video_entry(video_id, query_results, create_v_db_time)
        questions = entry.get("questions") or []
        for i, q in enumerate(questions):
            if i >= len(query_results):
                break
            r = query_results[i]
            for key in (
                "retrieve_item_type",
                "retrieve_clip_error",
                "select_clip_num",
                "reasoner_input_frame_count",
            ):
                if key in r:
                    q[key] = r[key]
        return entry
