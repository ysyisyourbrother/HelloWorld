import multiprocessing as mp
import numpy as np
import time
from collections import defaultdict
from config import Config


class VectorMemory():
    """基于向量的记忆模块"""
    def __init__(self, max_size=10000):
        self.max_size = max_size
        self.vectors = []
        self.metadata = []
        print("向量记忆模块已初始化")
    
    def add_memory(self, vector_data):
        """添加记忆"""
        if len(self.vectors) >= self.max_size:
            # 移除最旧的记忆
            self.vectors.pop(0)
            self.metadata.pop(0)
        
        self.vectors.append(vector_data['vector'])
        self.metadata.append({
            'timestamp': vector_data['timestamp'],
            'frame_id': vector_data['frame_id'],
            'is_keyframe': vector_data.get('is_keyframe', False)
        })
    
    def retrieve(self, query_vector, top_k=5):
        """基于相似度检索记忆"""
        if not self.vectors:
            return []
        
        # 计算相似度
        similarities = []
        for i, vector in enumerate(self.vectors):
            similarity = np.dot(query_vector, vector) / (
                np.linalg.norm(query_vector) * np.linalg.norm(vector)
            )
            similarities.append((similarity, i, self.metadata[i]))
        
        # 按相似度排序并返回top-k
        similarities.sort(reverse=True, key=lambda x: x[0])
        return similarities[:top_k]

class GraphMemory(MemoryBase):
    """基于图结构的记忆模块"""
    def __init__(self):
        self.graph = defaultdict(list)
        self.nodes = {}
        print("图记忆模块已初始化")
    
    def add_memory(self, vector_data):
        """添加记忆到图结构"""
        node_id = vector_data['frame_id']
        self.nodes[node_id] = vector_data
        
        # TODO: 实现图结构连接逻辑
        # 这里简单添加时间连接
        if len(self.nodes) > 1:
            prev_node_id = list(self.nodes.keys())[-2]
            self.graph[prev_node_id].append(node_id)
    
    def retrieve(self, query_vector, top_k=5):
        """基于图结构检索记忆"""
        # TODO: 实现图检索逻辑
        # 这里简单返回最近的节点
        recent_nodes = list(self.nodes.items())[-top_k:]
        return [(0.8, node_id, data) for node_id, data in recent_nodes]

class MemoryManager:
    def __init__(self, config=None):
        """
        初始化MemoryManager模块
        负责管理记忆模块的存储和检索
        
        Args:
            config (Config): 配置对象实例
        """
        if config is None:
            config = Config()
        
        self.config = config
        
        # 从Config对象获取配置
        self.database_type = config.memory_database_type
        self.retrieval_strategy = config.memory_retrieval_strategy
        self.max_memory_size = config.memory_max_size
        
        self.memory = self._initialize_memory()
        
        # 两个队列：一个用于接收向量，一个用于接收查询
        self.vector_queue = None
        self.query_queue = None
        self.result_queue = mp.Queue(maxsize=100)  # 使用默认队列大小
        
        self.running = False
        
    def _initialize_memory(self):
        """初始化记忆模块"""
        if self.database_type == "vector":
            return VectorMemory(self.max_memory_size)
        elif self.database_type == "graph":
            return GraphMemory()
        else:
            raise ValueError(f"不支持的记忆模块类型: {self.database_type}")
    
    def set_vector_queue(self, vector_queue):
        """设置向量队列（来自FrameVectorizer）"""
        self.vector_queue = vector_queue
    
    def set_query_queue(self, query_queue):
        """设置查询队列（来自QueryVectorizer）"""
        self.query_queue = query_queue
    
    def manage_memory(self):
        """记忆管理主循环"""
        while self.running:
            try:
                # 处理新的向量数据
                if self.vector_queue and not self.vector_queue.empty():
                    vector_data = self.vector_queue.get(timeout=0.1)
                    self.memory.add_memory(vector_data)
                
                # 处理查询请求
                if self.query_queue and not self.query_queue.empty():
                    query_data = self.query_queue.get(timeout=0.1)
                    query_vector = query_data['vector']
                    query_id = query_data['query_id']
                    
                    # 检索相关记忆
                    if self.retrieval_strategy == "similarity":
                        results = self.memory.retrieve(query_vector, top_k=5)
                    else:
                        results = self.memory.retrieve(query_vector, top_k=5)
                    
                    # 创建结果数据
                    result_data = {
                        'query_id': query_id,
                        'results': results,
                        'timestamp': time.time()
                    }
                    
                    if not self.result_queue.full():
                        self.result_queue.put(result_data)
                
                time.sleep(0.001)
            except Exception as e:
                print(f"记忆管理时出错: {e}")
                time.sleep(0.1)
    
    def start(self):
        """启动记忆管理进程"""
        self.running = True
        self.process = mp.Process(target=self.manage_memory)
        self.process.start()
        print(f"MemoryManager进程已启动，数据库类型: {self.database_type}")
    
    def stop(self):
        """停止记忆管理进程"""
        self.running = False
        if hasattr(self, 'process'):
            self.process.join(timeout=5)
        print("MemoryManager进程已停止")
    
    def get_result_queue(self):
        """获取结果队列供APIServerE使用"""
        return self.result_queue

if __name__ == "__main__":
    # 测试代码
    memory_manager = MemoryManager()
    try:
        memory_manager.start()
        time.sleep(5)  # 运行5秒进行测试
    except KeyboardInterrupt:
        pass
    finally:
        memory_manager.stop()