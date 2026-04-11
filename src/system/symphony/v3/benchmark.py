#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Symphony benchmark system (v3).

在 v1（SymVideoInput + SymFrameVectorizer 按 GOP 注入）基础上：
- 使用配置 ``memory_manager.retrieve_item_type``：``frame``（默认）或 ``clip``；
- ``clip`` 时在每次检索后对 topk 元数据调用 ``ffmpeg_utils`` 导出去重 GOP 临时 mp4。
"""

import logging
import os
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional

# 项目根：symphony/v3 -> 上四级到 HelloWorld
_project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)
sys.path.insert(0, _project_root)

from src.benchmark.utils import build_rag_prompt
from src.system.symphony.v1.benchmark import SymphonySystemBench
from src.video_utils import ffmpeg_utils


class SymphonySystemBenchV3(SymphonySystemBench):
    """
    继承 v1 ``SymphonySystemBench``（GOP 选帧注入 + ``register_i_frames``），
    查询阶段根据 ``config.memory_retrieve_item_type`` 支持 clip 导出。
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

    def _run_query_single(
        self,
        question: str,
        sample_id: str = "",
        sample: Optional[Dict[str, Any]] = None,
        video_time: Optional[float] = None,
    ) -> Dict[str, Any]:
        """编码查询向量、检索；``clip`` 模式下导出 GOP 临时 mp4 路径列表。"""
        t0 = time.time()
        query_vector = self.query_vectorizer.encode_query_sync(question)
        scores, frames_metadata = self.memory_manager.retrieve_sync(query_vector)
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

        result["retrieve_clip_paths"] = []
        result["retrieve_clip_temp_dir"] = None
        if rit == "clip" and frames_metadata:
            tmpdir = tempfile.mkdtemp(prefix="symphony_v3_clip_")
            result["retrieve_clip_temp_dir"] = tmpdir
            try:
                pairs = ffmpeg_utils.export_unique_gop_mp4s_from_memory_records(
                    frames_metadata, output_dir=tmpdir, prefix="hit"
                )
                result["retrieve_clip_paths"] = [
                    {"gop_start": int(rng[0]), "gop_end": int(rng[1]), "path": p}
                    for rng, p in pairs
                ]
            except Exception as e:
                self.logger.warning("检索 clip 导出失败: %s", e)
                result["retrieve_clip_error"] = str(e)

        query_text = question
        select_frame_num = len(frames_metadata) if frames_metadata else 0
        assert video_time, "video_time 必须要有才能创建ragprompt"
        if video_time is not None and frames_metadata and sample is not None:
            options = sample.get("options", [])
            if not isinstance(options, list):
                options = list(options) if options else []
            query_text = build_rag_prompt(
                video_time=video_time,
                num_selected_frame=len(frames_metadata),
                question=question,
                options=options,
            )
        result["rag_question"] = query_text
        result["select_frame_num"] = select_frame_num

        result["cloud_result"] = None
        result["cloud_error"] = "实时模式已切换为仅返回检索元数据，不再返回检索帧"
        result["total_time_sec"] = retrieve_time

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
                "retrieve_clip_paths",
                "retrieve_clip_temp_dir",
                "retrieve_clip_error",
            ):
                if key in r:
                    q[key] = r[key]
        return entry
