import json
import os


def _default_bge_vl_model_path():
    """
    本地 BGE-VL 权重目录默认值。
    优先环境变量 BGE_VL_MODEL_PATH 或 HELLOWORLD_BGE_VL_MODEL_PATH；
    否则在常见部署路径中选第一个已存在的目录（如 Jetson Orin 上常为 SSD 路径）。
    """
    env = os.environ.get("BGE_VL_MODEL_PATH") or os.environ.get(
        "HELLOWORLD_BGE_VL_MODEL_PATH"
    )
    if env:
        return env
    candidates = (
        "/mnt/share/cache/models/BGE-VL-base",
        "/mnt/ssd/huggingface/model/BGE-VL-base",
    )
    for p in candidates:
        if os.path.isdir(p):
            return p
    return candidates[0]


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
        # Video Input 配置（video_input 段；兼容旧键名 stream_input）
        video_config = self._config.get("video_input") or self._config.get("stream_input", {})
        self.video_log_file = video_config.get("log_file", "logs/video_input.log")
        self.video_file_path = video_config.get("video_file_path", "demo/assets/cooking.mp4")
        self.video_original_fps = video_config.get("original_fps", True)  # 是否按原帧率入队
        self.video_target_fps = video_config.get("target_fps", 8)  # 特定帧率
        # 本地 mp4 解码：auto（有 decord 用 decord，否则 cv2）| decord | cv2；可被环境变量 VIDEO_READER_BACKEND 覆盖（auto 时）
        self.video_reader_backend = video_config.get("reader_backend", "auto")
        # SymVideoInput 实现：V1=ffprobe 扫 GOP；V2=GStreamer（Jetson 等无 ffprobe 时用）
        self.video_input_version = video_config.get("version", "V2")
        # V2 GOP 扫描子进程所用 Python（需含 gi）；未配置时用环境变量 GST_GOP_SCAN_PYTHON 或 /usr/bin/python3
        self.video_gop_scan_python = video_config.get("gop_scan_python")
        # 多进程跨队列是否传输 RGB 整帧（True=传 ndarray；False=仅元数据，由接收端按路径+帧号再解码）
        self.video_ipc_send_frame = video_config.get("ipc_send_frame", False)

        # Frame Vectorizer配置
        frame_config = self._config.get("frame_vectorizer", {})
        self.frame_log_file = frame_config.get("log_file", "logs/frame_vectorizer.log")
        self.frame_model_type = frame_config.get("model_type", "BGE")
        self.frame_interval = frame_config.get("frame_interval", 10)
        self.frame_device = frame_config.get("device", "cuda")
        self.frame_model_path = frame_config.get(
            "model_path", _default_bge_vl_model_path()
        )
        self.frame_attn_implementation = frame_config.get(
            "attn_implementation", "eager"
        )

        # Query Vectorizer配置
        query_config = self._config.get("query_vectorizer", {})
        self.query_log_file = query_config.get("log_file", "logs/query_vectorizer.log")
        self.query_model_type = query_config.get("model_type", "BGE")
        self.query_model_path = query_config.get(
            "model_path", _default_bge_vl_model_path()
        )
        self.query_device = query_config.get("device", "cuda")
        
        # Memory Manager配置
        memory_config = self._config.get("memory_manager", {})
        self.memory_log_file = memory_config.get("log_file", "logs/memory_manager.log")
        self.memory_database_type = memory_config.get("database_type", "vector")
        self.memory_faiss_index_type = memory_config.get("faiss_index_type", "FlatIP")
        self.memory_faiss_file_path = memory_config.get("faiss_file_path", "database/database.faiss")
        self.memory_databasemap_file_path = memory_config.get("databasemap_file_path", "database/databasemap.json")
        self.memory_dimension = memory_config.get("dimension", 512)
        self.memory_topk = memory_config.get("topk", 5)
        self.memory_mode = memory_config.get("mode", "both")  # ["only_query", "only_inject", "both"]
        self.memory_save_retrieved_frames = memory_config.get("save_retrieved_frames", True)
        self.memory_save_injected_frames = memory_config.get("save_injected_frames", False)
        
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
        self.reasoner_max_history_turns = reasoner_config.get("max_history_turns", 10)  # None 表示不限制轮数
        
        # API Server E配置
        api_e_config = self._config.get("api_server_e", {})
        self.cloud_server_url = api_e_config.get("cloud_server_url", "127.0.0.1:9000")
        self.api_e_log_file = api_e_config.get("log_file", "logs/api_server_e.log")
        self.api_e_test_mode = api_e_config.get("test_mode", False)
        
        # API Server C配置
        api_c_config = self._config.get("api_server_c", {})
        self.api_c_log_file = api_c_config.get("log_file", "logs/api_server_c.log")
        self.server_host = api_c_config.get("host", "127.0.0.1")
        self.server_port = api_c_config.get("port", 9000)
        
        # Cloud Server配置
        cloud_config = self._config.get("cloud_server", {})

        # Edge 配置
        edge_config = self._config.get("edge", {})
        self.edge_mode = edge_config.get("mode", "query_while_inject") # "query_while_inject", "query_with_memory", "only_inject", "benchmark"

        # Benchmark 配置
        bench_config = self._config.get("benchmark", {})
        self.benchmark_dataset = bench_config.get("dataset", "Video-MME")  # "egoschema" | "Video-MME"
        self.benchmark_subset = bench_config.get("subset", "short")  # egoschema: "Subset"; Video-MME: "short"|"medium"|"long"
        self.benchmark_dataset_path = bench_config.get("dataset_path", "local_datasets")
        self.benchmark_batch_size = bench_config.get("batch_size", 16)
        self.benchmark_result_dir = bench_config.get("result_dir", "benchmark_results/venus")
        self.benchmark_use_cloud = bench_config.get("use_cloud", True)
        self.benchmark_video_dir_egoschema = bench_config.get("video_dir_egoschema", "local_datasets/egoschema/videos")
        self.benchmark_video_dir_videomme = bench_config.get("video_dir_videomme", "local_datasets/Video-MME/data")
        self.benchmark_db_dir_egoschema = bench_config.get("db_dir_egoschema", "database/egoschema/venus")
        self.benchmark_db_dir_videomme = bench_config.get("db_dir_videomme", "database/videomme/venus")

    def reload(self):
        """重新加载配置"""
        self._load_config()
        self._set_attributes()

    def get_config(self, section=None):
        """获取配置字典"""
        if section:
            return self._config.get(section, {})
        return self._config
    
    def get_video_config(self):
        """获取 video_input 配置"""
        return self._config.get("video_input") or self._config.get("stream_input", {})
    
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


class SymConfig(Config):
    """继承 Config，默认使用 symconfig.json，额外解析 select_strategy"""

    def __init__(self, config_path="configs/symconfig.json"):
        super().__init__(config_path)
        frame_config = self._config.get("frame_vectorizer", {})
        self.frame_select_strategy = frame_config.get("select_strategy", "random")  # "random" | "first" | "pktsize"