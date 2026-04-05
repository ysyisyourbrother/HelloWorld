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
        video_config = self._config.get("video_input") or {}
        stream_config = self._config.get("stream_input") or {}

        def _vs(key, default=None):
            """stream_input 优先，其次 video_input 内同名键（兼容旧版全写在 video_input 下）。"""
            if key in stream_config:
                return stream_config[key]
            return video_config.get(key, default)

        # Video Input：仅 video_input 段（本地文件 / 离线解码等）
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
        self.video_ipc_send_frame = bool(_vs("ipc_send_frame", False))

        # Stream Input：独立 stream_input 段，缺省时回退到 video_input 旧键（_vs）
        # stream 专用日志：有 stream_input.log_file 用它；否则若仅有旧版 video_input.log_file 则共用；否则默认 logs/stream_input.log
        self.stream_log_file = stream_config.get("log_file") or video_config.get(
            "log_file", "logs/stream_input.log"
        )
        self.stream_uri = _vs("stream_uri", "") or ""
        self.stream_window_duration_sec = float(_vs("stream_window_duration_sec", 1.0))
        self.stream_target_decode_fps = float(_vs("stream_target_decode_fps", 2.0))
        self.stream_trigger_ratio = float(_vs("stream_trigger_ratio", 1.35))
        self.stream_baseline_ewma_alpha = float(_vs("stream_baseline_ewma_alpha", 0.08))
        self.stream_warmup_windows = int(_vs("stream_warmup_windows", 3))
        self.stream_queue_maxsize = int(_vs("stream_queue_maxsize", 100))
        self.stream_rtsp_depay = _vs("stream_rtsp_depay", "h264")  # auto | h264 | h265

        # 流录制：RTSP 下用 GStreamer splitmuxsink 分段写 MP4（见 stream_input）
        self.stream_record_enable = bool(_vs("stream_record_enable", False))
        self.stream_record_segment_minutes = float(_vs("stream_record_segment_minutes", 1.0))
        self.stream_record_dir = _vs("stream_record_dir", "logs/stream_recordings")
        self.stream_record_max_seconds = float(_vs("stream_record_max_seconds", 0.0))
        self.stream_record_latency_ms = int(_vs("stream_record_latency_ms", 200))
        self.stream_record_rtsp_tcp = bool(_vs("stream_record_rtsp_tcp", False))
        # 兼容旧配置，当前实现已忽略
        self.stream_record_merge_on_exit = bool(_vs("stream_record_merge_on_exit", True))
        self.stream_record_fps = float(_vs("stream_record_fps", 25.0))
        self.stream_record_keep_segments = bool(_vs("stream_record_keep_segments", True))
        self.stream_record_final_basename = _vs("stream_record_final_basename", "") or ""

        self.edge_use_stream_input = bool(_vs("edge_use_stream_input", False))

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
        # MemoryManagerOnlineV2：流式会话最大时长（秒），>0 到时停止流并保存 faiss/json；0 不自动结束
        self.memory_stream_max_seconds = float(
            memory_config.get("memory_stream_max_seconds", 0.0)
        )
        # MemoryManagerOnlineV3：短期记忆 TTL = 分段时长(秒) × 该系数（略大于 1）
        self.memory_short_memory_ttl_ratio = float(
            memory_config.get("short_memory_ttl_ratio", 1.1)
        )
        # V3：RTSP 源时是否按录制目录 segment_*.mp4 + 墙钟启发式取长期像素
        self.memory_segment_pixel_heuristic = bool(
            memory_config.get("segment_pixel_heuristic", True)
        )
        # 流式边端是否使用 OnlineV3（短记忆 + 分段取帧）；False 则仍为 OnlineV2
        self.memory_online_v3 = bool(memory_config.get("online_v3", True))
        
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
        # query_* 经 APIServerE+gRPC；retrieve_* 仅本地 query_text_sync，无云端
        self.edge_mode = edge_config.get("mode", "query_while_inject")

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
        """获取 video_input 配置字典（不含 stream_input）。"""
        return dict(self._config.get("video_input") or {})

    def get_stream_config(self):
        """获取 stream_input 配置字典。"""
        return dict(self._config.get("stream_input") or {})
    
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