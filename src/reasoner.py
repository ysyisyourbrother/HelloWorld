import multiprocessing as mp
import time
from abc import ABC, abstractmethod
from config import Config

class ReasonerBase(ABC):
    """推理器基类"""
    @abstractmethod
    def process_query(self, query_text, memory_results):
        pass
    
    @abstractmethod
    def generate_stream(self, query_text, memory_results):
        pass

class VLMReasoner(ReasonerBase):
    """VLM推理器"""
    def __init__(self):
        # TODO: 初始化VLM模型
        self.model = None  # 这里应该加载实际的VLM模型
        self.max_tokens = 2048
        self.temperature = 0.7
        print("VLM推理器已初始化")
    
    def process_query(self, query_text, memory_results):
        """处理查询并返回完整结果"""
        # TODO: 实现实际的VLM推理
        # 这里返回模拟结果
        response = f"根据查询'{query_text}'和{len(memory_results)}个记忆片段，我分析了视频内容。"
        
        return {
            'response': response,
            'memory_used': len(memory_results),
            'processing_time': 0.5
        }
    
    def generate_stream(self, query_text, memory_results):
        """流式生成响应"""
        # TODO: 实现实际的VLM流式生成
        # 这里返回模拟的流式响应
        response_parts = [
            f"正在分析查询：{query_text}\n\n",
            f"找到了{len(memory_results)}个相关记忆片段。\n\n",
            "基于视频内容分析，",
            "我发现了以下关键信息：\n",
            "1. 视频中包含丰富的视觉信息\n",
            "2. 关键帧显示了重要场景\n",
            "3. 时间序列提供了上下文\n\n",
            "综合分析后，",
            "我认为这个问题的答案是：",
            "需要结合具体的视频内容来确定。"
        ]
        
        for part in response_parts:
            yield {
                'chunk': part,
                'finished': False,
                'timestamp': time.time()
            }
            time.sleep(0.1)  # 模拟流式延迟
        
        # 发送结束标记
        yield {
            'chunk': '',
            'finished': True,
            'timestamp': time.time()
        }

class Reasoner:
    def __init__(self, config=None):
        """
        初始化Reasoner模块
        负责接收查询数据，使用大型VLM进行推理
        
        Args:
            config (Config): 配置对象实例
        """
        if config is None:
            config = Config()
        
        self.config = config
        
        # 从Config对象获取配置
        self.model_type = config.reasoner_model_type
        self.max_tokens = config.reasoner_max_tokens
        self.temperature = config.reasoner_temperature
        
        self.reasoner = self._initialize_reasoner()
        self.query_queue = mp.Queue(maxsize=100)
        self.result_queue = mp.Queue(maxsize=100)
        self.running = False
        
    def _initialize_reasoner(self):
        """初始化推理器"""
        if self.model_type == "VLM":
            return VLMReasoner()
        else:
            raise ValueError(f"不支持的推理器类型: {self.model_type}")
    
    def process_queries(self):
        """处理查询的主循环"""
        while self.running:
            try:
                if not self.query_queue.empty():
                    query_data = self.query_queue.get(timeout=1)
                    query_text = query_data['query']
                    memory_results = query_data['memory_results']
                    query_id = query_data['query_id']
                    stream_mode = query_data.get('stream_mode', False)
                    
                    if stream_mode:
                        # 流式处理
                        self._process_stream_query(query_id, query_text, memory_results)
                    else:
                        # 批量处理
                        result = self.reasoner.process_query(query_text, memory_results)
                        
                        result_data = {
                            'query_id': query_id,
                            'result': result,
                            'timestamp': time.time()
                        }
                        
                        if not self.result_queue.full():
                            self.result_queue.put(result_data)
                
                time.sleep(0.001)
            except Exception as e:
                print(f"处理查询时出错: {e}")
                time.sleep(0.1)
    
    def _process_stream_query(self, query_id, query_text, memory_results):
        """处理流式查询"""
        try:
            for chunk_data in self.reasoner.generate_stream(query_text, memory_results):
                stream_data = {
                    'query_id': query_id,
                    'chunk': chunk_data,
                    'timestamp': time.time()
                }
                
                if not self.result_queue.full():
                    self.result_queue.put(stream_data)
                
                if chunk_data.get('finished', False):
                    break
                    
        except Exception as e:
            error_data = {
                'query_id': query_id,
                'error': str(e),
                'timestamp': time.time()
            }
            
            if not self.result_queue.full():
                self.result_queue.put(error_data)
    
    def add_query(self, query_text, memory_results, query_id=None, stream_mode=False):
        """添加查询到队列"""
        if query_id is None:
            query_id = int(time.time() * 1000)
        
        query_data = {
            'query': query_text,
            'memory_results': memory_results,
            'query_id': query_id,
            'stream_mode': stream_mode
        }
        
        if not self.query_queue.full():
            self.query_queue.put(query_data)
            return query_id
        else:
            print("查询队列已满")
            return None
    
    def start(self):
        """启动推理器进程"""
        self.running = True
        self.process = mp.Process(target=self.process_queries)
        self.process.start()
        print(f"Reasoner进程已启动，模型: {self.model_type}")
    
    def stop(self):
        """停止推理器进程"""
        self.running = False
        if hasattr(self, 'process'):
            self.process.join(timeout=5)
        print("Reasoner进程已停止")
    
    def get_query_queue(self):
        """获取查询队列供APIServerC使用"""
        return self.query_queue
    
    def get_result_queue(self):
        """获取结果队列供APIServerC使用"""
        return self.result_queue

if __name__ == "__main__":
    # 测试代码
    reasoner = Reasoner()
    try:
        reasoner.start()
        
        # 测试查询
        memory_results = [{'similarity': 0.8, 'frame_id': 1, 'data': {}}]
        reasoner.add_query("这个视频中有什么？", memory_results, stream_mode=True)
        time.sleep(5)
        
    except KeyboardInterrupt:
        pass
    finally:
        reasoner.stop()