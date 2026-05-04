#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Symphony benchmark system (v5)."""

import os
import time
from typing import Any, Dict, List, Optional

import cv2

from src.config import (
    system_mode_wants_memory_reinject,
    system_mode_wants_new_plan,
    system_mode_wants_vlm_qa,
)
from src.llm.reasoner import QueryRequest
from src.memory.frame.frame_vectorizer import SymFrameVectorizerForV3
from src.memory.memory_agent import MemoryAgent
from src.memory.query.query_vectorizer import QueryVectorizer
from src.system.symphony.v4.benchmark import SymphonySystemBenchV4
from src.video_input.video_input import SymVideoInputByStreamWindow


class SymphonySystemBenchV5(SymphonySystemBenchV4):
    """v5 Benchmark 编排：MemoryAgent 的 agentic 检索与 is_local_vlm 推理分支。"""

    def _benchmark_uses_api_vlm(self) -> bool:
        """与外层 is_local_vlm 语义对齐：优先读 is_local_vlm，否则 benchmark_is_local_vlm。"""
        uses_local = bool(
            getattr(self.config, "is_local_vlm", getattr(self.config, "benchmark_is_local_vlm", True))
        )
        return bool(self.use_cloud and not uses_local)

    def _init_components(
        self,
        video_path: Optional[str] = None,
        faiss_path: Optional[str] = None,
        map_path: Optional[str] = None,
        srt_path: Optional[str] = None,
    ):
        if faiss_path is not None:
            self.config.memory_faiss_file_path = faiss_path
        if map_path is not None:
            self.config.memory_databasemap_file_path = map_path
        if srt_path is not None:
            self.config.memory_srt_file_path = srt_path

        self.memory_manager = MemoryAgent(self.config)
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

    def _run_query_single(
        self,
        question: str,
        sample_id: str = "",
        sample: Optional[Dict[str, Any]] = None,
        video_time: Optional[float] = None,
        dialog_id: int = 0,
    ) -> Dict[str, Any]:
        t0 = time.time()
        options: List[str] = []
        if sample is not None:
            maybe_options = sample.get("options", [])
            if isinstance(maybe_options, list):
                options = maybe_options
            elif maybe_options:
                options = list(maybe_options)

        if (
            system_mode_wants_memory_reinject(self.config.system_mode)
            and not system_mode_wants_new_plan(self.config.system_mode)
            and not system_mode_wants_vlm_qa(self.config.system_mode)
        ):
            return {
                "question": question,
                "retrieve_time_sec": 0.0,
                "scores": [],
                "retrieved_frames": [],
                "retrieved_frames_metadata": [],
                "rag_question": question,
                "retrieve_item_type": "frame",
                "select_frame_num": 0,
                "select_clip_num": 0,
                "reasoner_input_frame_count": 0,
                "cloud_result": None,
                "cloud_error": (
                    "system_mode 为仅记忆重注入（未启用新 plan 与 VLM），跳过检索与推理"
                ),
                "total_time_sec": time.time() - t0,
            }

        if system_mode_wants_new_plan(self.config.system_mode):
            retrieve_pack = self.memory_manager.agentic_retrieve_pipeline(
                question, options=options
            )
        else:
            plan_path = self.memory_manager._resolve_plan_json_path()
            if not plan_path or not os.path.isfile(plan_path):
                raise ValueError(
                    "system_mode 未启用新 plan，但未找到已有 plan 文件（期望与 databasemap 同名的 "
                    "plan/*.json）: %r" % (plan_path,)
                )
            retrieve_pack = self.memory_manager.agentic_retrieve_pipeline_with_existing_plan(
                question,
                options=options,
                existing_plan_json_path=plan_path,
            )
        retrieve_time = time.time() - t0

        frames_metadata = retrieve_pack.get("metadata_list") or []
        query_text = str(retrieve_pack.get("rag_prompt") or question)

        result: Dict[str, Any] = {
            "question": question,
            "retrieve_time_sec": retrieve_time,
            "scores": [],
            "retrieved_frames": [],
            "retrieved_frames_metadata": frames_metadata,
            "rag_question": query_text,
            "retrieve_item_type": "frame",
            "select_frame_num": len(frames_metadata),
            "select_clip_num": 0,
            "reasoner_input_frame_count": len(frames_metadata),
        }

        if video_time is not None and frames_metadata and sample is not None:
            gathered = self._build_subtitle_context(frames_metadata, video_time)
            if gathered:
                result["gathered_subtitles"] = gathered

        if not system_mode_wants_vlm_qa(self.config.system_mode):
            result["cloud_result"] = None
            result["cloud_error"] = "system_mode 未启用 VLM 问答，仅检索"
            result["total_time_sec"] = time.time() - t0
            return result

        is_local_vlm = bool(
            getattr(self.config, "is_local_vlm", getattr(self.config, "benchmark_is_local_vlm", True))
        )
        if not is_local_vlm:
            self._fill_reasoner_result(
                t0,
                retrieve_time,
                query_text,
                frames_metadata,
                str(sample_id),
                result,
                clip_info=None,
            )
            return result

        frame_list_bgr = self._decode_retrieval_frames_bgr(frames_metadata)
        result["reasoner_input_frame_count"] = len(frame_list_bgr)

        if self.use_cloud and frame_list_bgr:
            frames_rgb = [
                cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                if f.ndim == 3
                else cv2.cvtColor(f, cv2.COLOR_GRAY2RGB)
                for f in frame_list_bgr
            ]
            qid = (hash(sample_id) % (2**31)) if sample_id else int(time.time())
            query_request = QueryRequest(
                query_text=query_text,
                memory_results=frames_rgb,
                query_id=qid,
                dialog_id=int(dialog_id),
            )
            reasoner = self._get_reasoner()
            response = reasoner.infer_sync(query_request)
            result["cloud_result"] = response.result
            result["cloud_error"] = response.error
            result["total_time_sec"] = time.time() - t0
        elif self.use_cloud:
            result["cloud_result"] = None
            result["cloud_error"] = "use_cloud=True 但无可用图像（agentic 检索未解码到有效帧）"
            result["total_time_sec"] = time.time() - t0
        else:
            result["cloud_result"] = None
            result["cloud_error"] = "use_cloud=False，跳过推理"
            result["total_time_sec"] = retrieve_time

        return result
