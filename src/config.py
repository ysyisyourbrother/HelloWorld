import json
import os

class Config:
    def __init__(self, config_path="configs/config.json"):
        """
        初始化配置类
        从config.json文件读取配置到类属性中
        
        Args:
            config_path (str): 配置文件路径
        """
        self.config_path = config_path
        self._config = {}
        self._load_config()
        self._set_attributes()
    
    def _load_config(self):
        """加载配置文件"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self._config = json.load(f)
            else:
                print(f"配置文件 {self.config_path} 不存在，使用默认配置")
                self._set_default_config()
        except Exception as e:
            print(f"加载配置文件失败: {e}，使用默认配置")
            self._set_default_config()
    
    def _set_default_config(self):
        """设置默认配置"""
        self._config = {
            "stream_input": {
                "fps": 30,
                "video_source": 0,
                "video_path": "",
                "queue_size": 100
            },
            "frame_vectorizer": {
                "model_type": "clip",
                "extraction_strategy": "uniform",
                "frame_interval": 1,
                "frame_queue_size": 200
            },
            "query_vectorizer": {
                "model_type": "bert",
                "query_queue_size": 50
            },
            "memory_manager": {
                "memory_type": "vector",
                "retrieval_strategy": "similarity",
                "max_memory_size": 10000,
                "retrieve_queue_size": 100
            },
            "reasoner": {
                "model_type": "VLM",
                "max_tokens": 2048,
                "temperature": 0.7
            },
            "api_server_e": {
                "host": "localhost",
                "port": 8000,
                "max_connections": 10
            },
            "api_server_c": {
                "host": "localhost",
                "port": 9000,
                "max_connections": 10
            }
        }
    
    def _set_attributes(self):
        """将配置设置为类属性"""
        # Stream Input配置
        stream_config = self._config.get("stream_input", {})
        self.stream_fps = stream_config.get("fps", 30)
        self.stream_video_source = stream_config.get("video_source", 0)
        self.stream_video_path = stream_config.get("video_path", "")
        self.stream_queue_size = stream_config.get("queue_size", 100)
        
        # Frame Vectorizer配置
        frame_config = self._config.get("frame_vectorizer", {})
        self.frame_model_type = frame_config.get("model_type", "clip")
        self.frame_extraction_strategy = frame_config.get("extraction_strategy", "uniform")
        self.frame_interval = frame_config.get("frame_interval", 1)
        self.frame_queue_size = frame_config.get("frame_queue_size", 200)
        
        # Query Vectorizer配置
        query_config = self._config.get("query_vectorizer", {})
        self.query_model_type = query_config.get("model_type", "bert")
        self.query_queue_size = query_config.get("query_queue_size", 50)
        
        # Memory Manager配置
        memory_config = self._config.get("memory_manager", {})
        self.memory_type = memory_config.get("memory_type", "vector")
        self.memory_retrieval_strategy = memory_config.get("retrieval_strategy", "similarity")
        self.memory_max_size = memory_config.get("max_memory_size", 10000)
        self.memory_retrieve_queue_size = memory_config.get("retrieve_queue_size", 100)
        
        # Reasoner配置
        reasoner_config = self._config.get("reasoner", {})
        self.reasoner_model_type = reasoner_config.get("model_type", "VLM")
        self.reasoner_max_tokens = reasoner_config.get("max_tokens", 2048)
        self.reasoner_temperature = reasoner_config.get("temperature", 0.7)
        
        # API Server E配置
        api_e_config = self._config.get("api_server_e", {})
        self.api_host = api_e_config.get("host", "localhost")
        self.api_port = api_e_config.get("port", 8000)
        
        # API Server C配置
        api_c_config = self._config.get("api_server_c", {})
        self.server_host = api_c_config.get("host", "localhost")
        self.server_port = api_c_config.get("port", 9000)
    
    def reload(self):
        """重新加载配置"""
        self._load_config()
        self._set_attributes()
    
    def get_config(self, section=None):
        """获取配置字典"""
        if section:
            return self._config.get(section, {})
        return self._config
    
    def get_stream_config(self):
        """获取stream_input配置"""
        return self._config.get("stream_input", {})
    
    def get_frame_config(self):
        """获取frame_vectorizer配置"""
        return self._config.get("frame_vectorizer", {})
    
    def get_query_config(self):
        """获取query_vectorizer配置"""
        return self._config.get("query_vectorizer", {})
    
    def get_memory_config(self):
        """获取memory_manager配置"""
        return self._config.get("memory_manager", {})
    
    def get_reasoner_config(self):
        """获取reasoner配置"""
        return self._config.get("reasoner", {})
    
    def get_api_e_config(self):
        """获取api_server_e配置"""
        return self._config.get("api_server_e", {})
    
    def get_api_c_config(self):
        """获取api_server_c配置"""
        return self._config.get("api_server_c", {})