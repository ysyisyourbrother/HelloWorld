#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v5 agentic 检索：按 system_mode plan 位选择生成/复用/跳过。"""

import os
from typing import Any, Dict, List, Optional

from src.config import (
    system_mode_wants_any_plan_retrieval,
    system_mode_wants_existing_plan,
    system_mode_wants_new_plan,
)


def run_agentic_retrieve(
    memory_manager,
    system_mode: int,
    question: str,
    options: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """
    按 system_mode 执行 agentic 检索。
    返回 None 表示本模式不做 plan 检索（x00xx）或复用 plan 时本地无文件（x01xx 跳过）。
    """
    sm = int(system_mode)
    opts = options or []

    if not system_mode_wants_any_plan_retrieval(sm):
        return None

    if system_mode_wants_new_plan(sm):
        return memory_manager.agentic_retrieve_pipeline(question, options=opts)

    plan_path = memory_manager._resolve_plan_json_path()
    if not plan_path or not os.path.isfile(plan_path):
        return None

    return memory_manager.agentic_retrieve_pipeline_with_existing_plan(
        question,
        options=opts,
        existing_plan_json_path=plan_path,
    )


def empty_retrieve_skip_result(question: str, cloud_error: str, t0: float) -> Dict[str, Any]:
    import time

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
        "cloud_error": cloud_error,
        "total_time_sec": time.time() - t0,
    }
