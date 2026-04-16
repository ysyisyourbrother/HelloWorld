import threading
from contextlib import contextmanager
from typing import Tuple

import faiss
import numpy as np


class ThreadSafeFaiss:
    """线程安全的Faiss索引类"""

    def __init__(self, index: faiss.Index):
        self._index = index
        self._lock = threading.RLock()

    @contextmanager
    def acquire(self):
        """上下文管理器，用于自动获取和释放锁"""
        try:
            self._lock.acquire()
            yield self._index
        finally:
            self._lock.release()

    def save_local(self, path: str = None):
        with self.acquire():
            faiss.write_index(self._index, path)

    def delete(self):
        ret = []
        with self.acquire():
            # Faiss的IndexFlatL2/IndexFlatIP没有docstore属性，直接重置索引
            self._index.reset()
        return ret

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """线程安全的搜索方法"""
        with self.acquire():
            return self._index.search(query_vector, top_k)
