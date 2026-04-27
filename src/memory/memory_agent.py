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
