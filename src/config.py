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
        self.stream_original_fps = stream_config.get("original_fps", True)  # 是否按原帧率入队
        
        # Frame Vectorizer配置
        frame_config = self._config.get("frame_vectorizer", {})
        self.frame_log_file = frame_config.get("log_file", "logs/frame_vectorizer.log")
        self.frame_model_type = frame_config.get("model_type", "BGE")
        self.frame_extraction_strategy = frame_config.get("extraction_strategy", "every_frame") # "every_frame", "interval"
        self.frame_interval = frame_config.get("frame_interval", 1)
        self.frame_use_vlm = frame_config.get("use_vlm", False)
        self.frame_device = frame_config.get("device", "cuda")
        self.frame_model_path = frame_config.get("model_path", "/mnt/share/cache/models/BGE-VL-base")
        
        # Query Vectorizer配置
        query_config = self._config.get("query_vectorizer", {})
        self.query_log_file = query_config.get("log_file", "logs/query_vectorizer.log")
        self.query_model_type = query_config.get("model_type", "BGE")
        self.query_model_path = query_config.get("model_path", "/mnt/share/cache/models/BGE-VL-base")
        self.query_dimension = query_config.get("dimension", 512)
        self.query_device = query_config.get("device", "cuda")
        
        # Memory Manager配置
        memory_config = self._config.get("memory_manager", {})
        self.memory_log_file = memory_config.get("log_file", "logs/memory_manager.log")
        self.memory_resume = memory_config.get("resume", False)
        self.memory_database_type = memory_config.get("database_type", "vector")
        self.memory_faiss_index_type = memory_config.get("faiss_index_type", "FlatIP")
        self.memory_faiss_file_path = memory_config.get("faiss_file_path", "database/database.faiss")
        self.memory_databasemap_file_path = memory_config.get("databasemap_file_path", "database/databasemap.json")
        self.memory_dimension = memory_config.get("dimension", 512)
        self.memory_retrieval_strategy = memory_config.get("retrieval_strategy", "topk")
        self.memory_topk = memory_config.get("topk", 5)
        self.memory_max_size = memory_config.get("max_memory_size", 10000)
        memory_mode = memory_config.get("mode", "both")
        
        self.memory_mode = memory_mode #  ["only_query", "only_inject", "both"]
        
        # Reasoner配置
        reasoner_config = self._config.get("reasoner", {})
        self.reasoner_log_file = reasoner_config.get("log_file", "logs/reasoner.log")
        self.reasoner_test_mode = reasoner_config.get("test_mode", False)
        self.reasoner_test_response = reasoner_config.get("test_response", "这是测试模式的响应文本")
        self.reasoner_model_path = reasoner_config.get("model_path", "/mnt/share/cache/models/LLaVA-Video-7B-Qwen2")
        self.reasoner_model_name = reasoner_config.get("model_name", "llava_qwen")
        self.reasoner_model_base = reasoner_config.get("model_base", None)
        self.reasoner_torch_dtype = reasoner_config.get("torch_dtype", "bfloat16")
        self.reasoner_load_in_8bit = reasoner_config.get("load_in_8bit", False)
        self.reasoner_load_in_4bit = reasoner_config.get("load_in_4bit", False)
        self.reasoner_device_map = reasoner_config.get("device_map", "auto")
        self.reasoner_attn_implementation = reasoner_config.get("attn_implementation", "eager")
        self.reasoner_mm_spatial_pool_mode = reasoner_config.get("mm_spatial_pool_mode", "average")
        self.reasoner_conv_template = reasoner_config.get("conv_template", "qwen_1_5")
        self.reasoner_device = reasoner_config.get("device", "cuda")
        self.reasoner_max_new_tokens = reasoner_config.get("max_new_tokens", 128)
        self.reasoner_temperature = reasoner_config.get("temperature", 0.0)
        self.reasoner_top_p = reasoner_config.get("top_p", 0.1)
        self.reasoner_num_beams = reasoner_config.get("num_beams", 1)
        self.reasoner_do_sample = reasoner_config.get("do_sample", False)
        self.reasoner_max_history_turns = reasoner_config.get("max_history_turns", None)  # None 表示不限制轮数
        
        # API Server E配置
        api_e_config = self._config.get("api_server_e", {})
        self.communication_mode = api_e_config.get("communication_mode", "unary")  # "unary", "server_stream", "bidi_stream"
        self.cloud_server_url = api_e_config.get("cloud_server_url", "localhost:9000")
        self.api_e_log_file = api_e_config.get("log_file", "logs/api_server_e.log")
        self.api_e_test_mode = api_e_config.get("test_mode", False)
        self.api_e_dialog_mode = api_e_config.get("dialog_mode", "single")  # "single" 或 "multi"
        
        # API Server C配置
        api_c_config = self._config.get("api_server_c", {})
        self.api_c_log_file = api_c_config.get("log_file", "logs/api_server_c.log")
        self.server_host = api_c_config.get("host", "0.0.0.0")
        self.server_port = api_c_config.get("port", 9000)
        
        # Cloud Server配置
        cloud_config = self._config.get("cloud_server", {})

        # Edge 配置
        edge_config = self._config.get("edge", {})
        self.edge_mode = edge_config.get("mode", "query_while_inject") # "query_while_inject", "query_with_memory", "only_inject", "benchmark"
    
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
    
    def get_cloud_config(self):
        """获取cloud_server配置"""
        return self._config.get("cloud_server", {})