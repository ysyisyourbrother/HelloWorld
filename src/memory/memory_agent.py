from typing import List, Optional, Tuple

import numpy as np

from src.config import Config
from src.memory.memory_manager import MemoryManagerBase

def local_tool_use(func):
    def wrapper(*args, **kwargs):
        # 执行前的额外操作
        result = func(*args, **kwargs)
        # 执行后的额外操作
        return result
    return wrapper


class MemoryAgent(MemoryManagerBase):
    """在 MemoryManagerBase 上扩展范围检索等能力。"""

    def __init__(self, config: Config = None):
        super().__init__(config=config)

    def _get_subset(self, start_frameid: Optional[int], end_frameid: Optional[int]) -> Tuple[int, np.ndarray]:
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
    def _query_faiss_by_range(
        self,
        query_vector: np.ndarray,
        top_k: int,
        start_frameid: int,
        end_frameid: int,
    ) -> Tuple[List[int], List[float]]:
        """
        在指定帧号范围内检索向量。

        Args:
            query_vector: 查询向量，shape=[D] 或 [1, D]
            top_k: 返回前 k 个结果
            start_frameid: 起始帧号（含）
            end_frameid: 结束帧号（含）

        Returns:
            (vector_ids, scores):
                vector_ids: 原始 FAISS 全局向量 id
                scores: 与 vector_ids
        """
        if self.index is None or self.vector_count == 0 or top_k <= 0:
            return [], []

        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)

        query_row = query_vector[0]

        start_id, subset_vectors = self._get_subset(start_frameid, end_frameid)
        if start_id < 0 or subset_vectors.size == 0:
            return [], []

        # 与 _query_faiss 保持一致：FlatL2 返回距离，FlatIP 返回内积
        if self.faiss_index_type == "FlatIP":
            subset_scores = np.dot(subset_vectors, query_row)
            rank_idx = np.argsort(subset_scores)[::-1]
        else:
            diff = subset_vectors - query_row
            subset_scores = np.sum(diff * diff, axis=1)
            rank_idx = np.argsort(subset_scores)

        k = min(top_k, len(rank_idx))
        top_idx = rank_idx[:k]

        vector_ids = (top_idx + start_id).tolist()
        scores = subset_scores[top_idx].astype(float).tolist()
        return vector_ids, scores

    def _retrieve_tasks_loop(self):
        f'''
        已有向量数据库和字幕数据库
        initial_planning: 根据问题文本 query, 定问题类型 -> 定时间范围 -> 定事件范围 -> 定查询实体
        问题类型:    
                利用特定图像和文本可解决: 
                    1. 物体常规属性类(事实型1, 强图片弱文本), 
                    2. 字幕类(事实型2, 弱图片强文本), 
                    3. 概括事件类(事实型3, 强图片强文本), 
                    4. 推断或评价类(推理型1, 弱图片强文本), 
                    5. 溯因类(推理型2, 强图片弱文本)
                需要利用额外工具类: 
                    1. 画面内计数类(事实型4, 利用YoLo探测的数量)
                    2. 画面内位置类(事实型5, 利用YoLo探测框的位置)
                    3. OCR类(事实型6, 利用OCR模型识别画面中的文本)。
        '''
        pass