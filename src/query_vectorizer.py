import multiprocessing as mp
import numpy as np
import time
from abc import ABC, abstractmethod
from config import Config

class QueryEncoderBase(ABC):
    """查询编码器基类"""
    @abstractmethod
    def encode(self, query_text):
        pass

class BERTEncoder(QueryEncoderBase):
    """BERT查询编码器"""
    def __init__(self):
        # TODO: 初始化BERT模型
        self.model = None  # 这里应该加载实际的BERT模型
        print("BERT查询编码器已初始化")
    
    def encode(self, query_text):
        # TODO: 实现实际的BERT编码
        # 返回模拟的向量
        return np.random.rand(768)  # BERT-base的输出维度

class SentenceBERTEncoder(QueryEncoderBase):
    """Sentence-BERT查询编码器"""
    def __init__(self):
        # TODO: 初始化Sentence-BERT模型
        self.model = None  # 这里应该加载实际的Sentence-BERT模型
        print("Sentence-BERT查询编码器已初始化")
    
    def encode(self, query_text):
        # TODO: 实现实际的Sentence-BERT编码
        # 返回模拟的向量
        return np.random.rand(384)  # Sentence-BERT的输出维度

class QueryVectorizer:
    def __init__(self, config=None):
        """
        初始化QueryVectorizer模块
        负责把用户的自然语言查询转换为语义向量
        
        Args:
            config (Config): 配置对象实例
        """
        if config is None:
            config = Config()
        
        self.config = config
        
        # 从Config对象获取配置
        self.model_type = config.query_model_type
        self.max_length = config.query_max_length
        
        self.encoder = self._initialize_encoder()
        self.query_queue = mp.Queue(maxsize=50)  # 使用默认队列大小
        self.vector_queue = mp.Queue(maxsize=50)  # 使用默认队列大小
        self.running = False
        
    def _initialize_encoder(self):
        """初始化查询编码器"""
        if self.model_type == "BERT":
            return BERTEncoder()
        elif self.model_type == "SentenceBERT":
            return SentenceBERTEncoder()
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
    
    def process_queries(self):
        """处理查询的主循环"""
        while self.running:
            try:
                if not self.query_queue.empty():
                    query_data = self.query_queue.get(timeout=1)
                    query_text = query_data['text']
                    query_id = query_data['query_id']
                    
                    # 向量化查询文本
                    vector = self.encoder.encode(query_text)
                    
                    # 创建向量数据
                    vector_data = {
                        'vector': vector,
                        'query_text': query_text,
                        'query_id': query_id,
                        'timestamp': time.time()
                    }
                    
                    if not self.vector_queue.full():
                        self.vector_queue.put(vector_data)
                
                time.sleep(0.001)
            except Exception as e:
                print(f"处理查询时出错: {e}")
                time.sleep(0.1)
    
    def add_query(self, query_text, query_id=None):
        """添加查询到队列"""
        if query_id is None:
            query_id = int(time.time() * 1000)
        
        query_data = {
            'text': query_text,
            'query_id': query_id
        }
        
        if not self.query_queue.full():
            self.query_queue.put(query_data)
            return query_id
        else:
            print("查询队列已满")
            return None
    
    def start(self):
        """启动查询向量化进程"""
        self.running = True
        self.process = mp.Process(target=self.process_queries)
        self.process.start()
        print(f"QueryVectorizer进程已启动，模型: {self.model_type}")
    
    def stop(self):
        """停止查询向量化进程"""
        self.running = False
        if hasattr(self, 'process'):
            self.process.join(timeout=5)
        print("QueryVectorizer进程已停止")
    
    def get_vector_queue(self):
        """获取向量队列供MemoryManager使用"""
        return self.vector_queue
    
    def get_query_queue(self):
        """获取查询队列供APIServerE使用"""
        return self.query_queue

if __name__ == "__main__":
    # 测试代码
    query_vectorizer = QueryVectorizer()
    try:
        query_vectorizer.start()
        
        # 测试查询
        query_vectorizer.add_query("这个视频中有什么？")
        time.sleep(2)
        
        query_vectorizer.add_query("展示一下关键帧")
        time.sleep(2)
        
    except KeyboardInterrupt:
        pass
    finally:
        query_vectorizer.stop()