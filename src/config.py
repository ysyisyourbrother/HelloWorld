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
                print(f"配置文件 {self.config_path} 不存在")
        except Exception as e:
            print(f"加载配置文件失败: {e}")
    
    def _set_attributes(self):
        """将配置设置为类属性"""
        # Stream Input配置
        stream_config = self._config.get("stream_input", {})
        self.stream_log_file = stream_config.get("log_file", "logs/stream_input.log")
        self.stream_video_source = stream_config.get("video_source", "camera") # "camera" or "file"
        self.stream_video_file_path = stream_config.get("video_file_path", "demo/assets/cooking.mp4")
        self.stream_reader_type = stream_config.get("reader_type", "cv2")  # 视频读取器类型: "cv2" 或 "decord"
        
        # Frame Vectorizer配置
        frame_config = self._config.get("frame_vectorizer", {})
        self.frame_log_file = frame_config.get("log_file", "logs/frame_vectorizer.log")
        self.frame_model_type = frame_config.get("model_type", "BGE")
        self.frame_extraction_strategy = frame_config.get("extraction_strategy", "every_frame")
        self.frame_interval = frame_config.get("frame_interval", 1)
        self.frame_use_vlm = frame_config.get("use_vlm", False)
        self.frame_device = frame_config.get("device", "cuda")
        self.frame_model_path = frame_config.get("model_path", "/mnt/share/cache/models/BGE-VL-base")
        
        # Query Vectorizer配置
        query_config = self._config.get("query_vectorizer", {})
        self.query_model_type = query_config.get("model_type", "BGE")
        self.query_model_path = query_config.get("model_path", "/mnt/share/cache/models/BGE-VL-base")
        self.query_dimension = query_config.get("dimension", 512)
        self.query_device = query_config.get("device", "cuda")
        
        # Memory Manager配置
        memory_config = self._config.get("memory_manager", {})
        self.memory_database_type = memory_config.get("database_type", "vector")
        self.memory_retrieval_strategy = memory_config.get("retrieval_strategy", "similarity")
        self.memory_max_size = memory_config.get("max_memory_size", 10000)
        
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
        self.server_host = api_c_config.get("host", "0.0.0.0")
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