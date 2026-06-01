#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Symphony motivation system (v6)."""

import os
import time
from typing import Any, Dict, List, Optional

from src.config import (
    system_mode_is_inject_only_no_query,
    system_mode_wants_any_plan_retrieval,
    system_mode_wants_existing_plan,
    system_mode_wants_new_plan,
    system_mode_wants_vlm_qa,
)
from src.memory.frame.frame_vectorizer import SymFrameVectorizerForV3
from src.memory.memory_agent_v6 import MemoryAgentV6
from src.memory.query.query_vectorizer import QueryVectorizer
from src.system.symphony.v5.motivation import SymphonySystemMotiV5
from src.video_input.video_input import SymVideoInputByStreamWindow


class SymphonySystemMotiV6(SymphonySystemMotiV5):
    """v6 Motivation 编排：MemoryAgentV6 的 agentic 检索作答管线（VLM 内嵌，无需本地大模型）。"""

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

        self.memory_manager = MemoryAgentV6(self.config)
        self.query_vectorizer = QueryVectorizer(self.config)

        self.memory_manager.init_sync()
        self.query_vectorizer._initialize_vectorizer()

        if video_path:
            self.config.video_file_path = video_path
            self.video_input = SymVideoInputByStreamWindow(self.config)
            self.frame_vectorizer = SymFrameVectorizerForV3(self.config)
            for fn, need_hs, need_attn in self._encode_hooks:
                self.frame_vectorizer.register_encode_hook(
                    fn, need_hidden_states=need_hs, need_attentions=need_attn
                )
            self.video_input.init_for_file(video_path)
            ixs = getattr(self.video_input, "i_frame_indices", None)
            if ixs:
                self.memory_manager.register_i_frames(video_path, list(ixs))
            self.frame_vectorizer._initialize_vectorizer()
        else:
            self.video_input = None
            self.frame_vectorizer = None

    def _call_agentic_retrieve_and_answer(
        self, question: str, options: List[str]
    ) -> Optional[Dict[str, Any]]:
        sm = int(self.config.system_mode)
        if not system_mode_wants_any_plan_retrieval(sm):
            return None
        if system_mode_wants_new_plan(sm):
            return self.memory_manager.agentic_retrieve_and_answer_pipeline(
                question, options=options
            )
        plan_path = self.memory_manager._resolve_plan_json_path()
        if plan_path and os.path.isfile(plan_path):
            return (
                self.memory_manager.agentic_retrieve_and_answer_pipeline_with_existing_plan(
                    question,
                    options=options,
                    existing_plan_json_path=plan_path,
                )
            )
        return None

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

        if system_mode_is_inject_only_no_query(self.config.system_mode):
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
                    "system_mode 为仅帧+ASR 注入（值 3，未启用 plan 与 VLM），跳过检索与推理"
                ),
                "total_time_sec": time.time() - t0,
            }

        retrieve_pack = self._call_agentic_retrieve_and_answer(question, options)
        if retrieve_pack and retrieve_pack.get("agent_failed"):
            phase = str(retrieve_pack.get("failure_phase") or "unknown")
            return {
                "question": question,
                "retrieve_time_sec": time.time() - t0,
                "scores": [],
                "retrieved_frames": [],
                "retrieved_frames_metadata": [],
                "rag_question": question,
                "retrieve_item_type": "frame",
                "select_frame_num": 0,
                "select_clip_num": 0,
                "reasoner_input_frame_count": 0,
                "cloud_result": None,
                "cloud_error": "Agent 任务失败: 云端输出解析失败 (%s)" % phase,
                "total_time_sec": time.time() - t0,
            }
        if retrieve_pack is None:
            sm = int(self.config.system_mode)
            if not system_mode_wants_any_plan_retrieval(sm):
                msg = "system_mode plan 位为 00（未启用复用已有 plan 也未启用新 plan），跳过检索"
            elif system_mode_wants_existing_plan(sm):
                plan_path = self.memory_manager._resolve_plan_json_path()
                msg = (
                    "system_mode 启用复用已有 plan，但未找到 plan 文件（期望 plan/*.json）: %r"
                    % (plan_path,)
                )
            else:
                msg = "system_mode 未执行 plan 检索，跳过"
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
                "cloud_error": msg,
                "total_time_sec": time.time() - t0,
            }

        retrieve_time = time.time() - t0

        frames_metadata = retrieve_pack.get("frame_results") or []
        query_text = str(retrieve_pack.get("retrieval_context") or question)
        final_answer = str(retrieve_pack.get("final_answer") or "").strip()

        result = {
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

        if not system_mode_wants_vlm_qa(self.config.system_mode):
            result["cloud_result"] = None
            result["cloud_error"] = "system_mode 未启用 VLM 问答，仅检索"
            result["total_time_sec"] = time.time() - t0
            return result

        if not self.use_cloud:
            result["cloud_result"] = None
            result["cloud_error"] = "use_cloud=False，跳过推理"
            result["total_time_sec"] = time.time() - t0
            return result

        result["cloud_result"] = final_answer or None
        if final_answer:
            result["cloud_error"] = None
        else:
            result["cloud_error"] = "agentic 检索作答未得到有效答案"
        result["total_time_sec"] = time.time() - t0
        return result
