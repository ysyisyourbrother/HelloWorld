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


# system_mode 位掩码（与 Config.system_mode 约定一致，供其他模块引用）
# bit0=1：重新注入记忆；0 表示依赖已有本地记忆、不做本轮注入流水线
# bit1=2：通过云端 API 拉取新的检索 plan；0 表示使用已有 plan（如 agentic_retrieve_pipeline_with_existing_plan）
# bit2=4：用 VLM 对问题与检索结果作答；0 表示仅检索、不跑 VLM 问答
# 三 bit 全开 = 1+2+4 = 7（二进制 0b111）；注意 8 为 0b1000，不是三 bit 全开
SYSTEM_MODE_REINJECT_MEMORY = 1
SYSTEM_MODE_NEW_PLAN = 2
SYSTEM_MODE_VLM_QA = 4


def system_mode_wants_memory_reinject(system_mode):
    return (int(system_mode) & SYSTEM_MODE_REINJECT_MEMORY) != 0


def system_mode_wants_new_plan(system_mode):
    return (int(system_mode) & SYSTEM_MODE_NEW_PLAN) != 0


def system_mode_wants_vlm_qa(system_mode):
    return (int(system_mode) & SYSTEM_MODE_VLM_QA) != 0


def system_mode_online_memory_needs_query_encoder(system_mode):
    """在线 MemoryManager：仅 system_mode==1（仅注入位）时不初始化 QueryVectorizer。"""
    return int(system_mode) != SYSTEM_MODE_REINJECT_MEMORY


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

    def _parse_system_mode(self, raw):
        """顶层 system_mode：十进制整型，由至多三 bit 组合，见模块级 SYSTEM_MODE_* 常量。"""
        if raw is None:
            return 7
        return int(raw)

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

        # 系统运行模式：顶层 system_mode，三 bit 掩码见模块常量 SYSTEM_MODE_*
        self.system_mode = self._parse_system_mode(self._config.get("system_mode"))

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
        self.stream_trigger_ratio = float(_vs("stream_trigger_ratio", 1.25))
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

        # Audio Input（供 Memory / ASR 等消费；与 video_input 并列的独立配置段）
        audio_config = self._config.get("audio_input") or {}
        self.audio_log_file = audio_config.get("log_file", "logs/audio_input.log")
        # 留空则使用本配置中已解析的 video_file_path（通常与当前解码视频一致）
        self.audio_source_path = (audio_config.get("source_path") or "").strip() or None
        self.audio_stream_as_chunks = bool(audio_config.get("stream_as_chunks", False))
        self.audio_chunk_duration_sec = float(audio_config.get("chunk_duration_sec", 1.0))
        # True：固定按 chunk_duration_sec 步进（类似 whisper_online --comp_unaware）；False：按墙钟+sleep（simultaneous 默认可计算感知）
        self.audio_sim_comp_unaware = bool(audio_config.get("sim_comp_unaware", False))

        # ASR / 字幕识别（与 frame_vectorizer 同级独立段，供后续 asr 模块使用）
        asr_config = self._config.get("asr") or {}
        self.asr_log_file = asr_config.get("log_file", "logs/asr.log")
        self.asr_language = asr_config.get("language", "en") # ["auto", "en"]
        self.asr_model_size = asr_config.get("model_size", "tiny")
        self.asr_model_path = asr_config.get(
            "model_path", "/mnt/share/cache/models/whisper-tiny"
        )
        self.asr_backend = asr_config.get(
            "backend", "faster-whisper"
        )  # ["transformers", "faster-whisper"]
        self.asr_device = asr_config.get("device", "auto")
        self.asr_compute_type = asr_config.get("compute_type", "float16")
        self.asr_beam_size = int(asr_config.get("beam_size", 5))
        self.asr_buffer_trimming = asr_config.get(
            "buffer_trimming", "sentence"
        )  # ["sentence", "segment"]
        self.asr_buffer_trimming_sec = float(asr_config.get("buffer_trimming_sec", 15.0))

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
        self.memory_srt_file_path = memory_config.get("srt_file_path", "database/subtitles.srt")
        self.memory_dimension = memory_config.get("dimension", 512)
        self.memory_topk = memory_config.get("topk", 5)
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
        # 检索产出类型：frame=仅帧元数据（默认）；clip=按 topk 命中 GOP 导出临时 mp4（需 i_frames 与 ffmpeg）
        self.memory_retrieve_item_type = memory_config.get("retrieve_item_type", "frame")
        
        # 增强工具的设置
        enhance_tool_config = self._config.get("enhance_tool", {})
        self.ocr_language = enhance_tool_config.get("ocr_language", "en") # en, ch_sim
        self.ocr_conf_threshold = enhance_tool_config.get("ocr_conf_threshold", 0.4) 
        self.yolo_model_path = enhance_tool_config.get("yolo_model_path", "/mnt/share/cache/models/YOLO/yolo26l.pt") 
        self.yolo_conf_threshold = enhance_tool_config.get("yolo_conf_threshold", 0.3) 

        # ReasonerLocal配置
        reasoner_local_config = self._config.get("reasoner_local", {})
        self.reasoner_local_log_file = reasoner_local_config.get("log_file", "logs/reasoner.log")
        self.reasoner_local_test_mode = reasoner_local_config.get("test_mode", False)
        self.reasoner_local_test_response = reasoner_local_config.get("test_response", "这是测试模式的响应文本")
        self.reasoner_local_model_path = reasoner_local_config.get("model_path", "/mnt/share/cache/models/LLaVA-Video-7B-Qwen2")
        self.reasoner_local_model_name = reasoner_local_config.get("model_name", "llava_qwen")
        self.reasoner_local_model_base = reasoner_local_config.get("model_base", None)
        self.reasoner_local_torch_dtype = reasoner_local_config.get("torch_dtype", "bfloat16")
        self.reasoner_local_load_in_8bit = reasoner_local_config.get("load_in_8bit", False)
        self.reasoner_local_load_in_4bit = reasoner_local_config.get("load_in_4bit", False)
        self.reasoner_local_device_map = reasoner_local_config.get("device_map", "auto")
        self.reasoner_local_attn_implementation = reasoner_local_config.get("attn_implementation", "eager")
        self.reasoner_local_mm_spatial_pool_mode = reasoner_local_config.get("mm_spatial_pool_mode", "average")
        self.reasoner_local_conv_template = reasoner_local_config.get("conv_template", "qwen_1_5")
        self.reasoner_local_device = reasoner_local_config.get("device", "cuda")
        self.reasoner_local_max_new_tokens = reasoner_local_config.get("max_new_tokens", 128)
        self.reasoner_local_temperature = reasoner_local_config.get("temperature", 0.0)
        self.reasoner_local_top_p = reasoner_local_config.get("top_p", 0.1)
        self.reasoner_local_num_beams = reasoner_local_config.get("num_beams", 1)
        self.reasoner_local_do_sample = reasoner_local_config.get("do_sample", False)
        # 单次推理最多送入的图像张数；超出则沿序列均匀稀疏采样。None 或 <=0 表示不限制
        self.reasoner_local_max_img_num = reasoner_local_config.get("max_img_num", 20)

        # APIServerE配置
        client_config = self._config.get("client", {})
        self.client_simu_url_of_server = client_config.get("cloud_server_url", "127.0.0.1:9000")
        self.client_log_file = client_config.get("log_file", "logs/client.log")
        self.client_test_mode = client_config.get("test_mode", False)
        
        # APIServerC配置
        server_simu_config = self._config.get("server_simulation", {})
        self.server_simu_log_file = server_simu_config.get("log_file", "logs/server_simulation.log")
        self.server_simu_host = server_simu_config.get("host", "127.0.0.1")
        self.server_simu_port = server_simu_config.get("port", 9000)
        
        # Cloud Server配置：manual=自建 APIServerC+本地 Reasoner；api=厂商云端 HTTP 多模态 API（如 DashScope）
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
        self.benchmark_result_dir = bench_config.get("result_dir", "benchmark_results/vrag")
        self.benchmark_video_dir_egoschema = bench_config.get("video_dir_egoschema", "local_datasets/egoschema/videos")
        self.benchmark_video_dir_videomme = bench_config.get("video_dir_videomme", "local_datasets/Video-MME/data")
        self.benchmark_db_dir_egoschema = bench_config.get("db_dir_egoschema", "database/egoschema/vrag")
        self.benchmark_db_dir_videomme = bench_config.get("db_dir_videomme", "database/videomme/vrag")
        
        self.benchmark_is_local_vlm = bench_config.get("is_local_vlm", True)

        # Venus 编排层（场景切分 + 聚类 + OCR/YOLO 增强 + 渐进式检索）
        venus_config = self._config.get("venus", {})
        self.venus_k_clusters = int(venus_config.get("k_clusters", 5))
        self.venus_retrieve_strategy = venus_config.get(
            "retrieve_strategy", "progressive"
        )  # "progressive" | "threshold"
        self.venus_retrieve_tau = float(venus_config.get("retrieve_tau", 0.07))
        self.venus_retrieve_theta = float(venus_config.get("retrieve_theta", 0.9))
        self.venus_retrieve_beta = float(venus_config.get("retrieve_beta", 1.0))
        self.venus_retrieve_n_max = int(venus_config.get("retrieve_n_max", 32))
        self.venus_retrieve_threshold = float(venus_config.get("retrieve_threshold", 0.2))

        # 大模型提供商 API（benchmark_is_local_vlm=False 时 VLM 走 ReasonerVLMAPI；LLM 走 ReasonerLLMAPI）
        api_config = self._config.get("api", {})
        self.api_vlm_model_name = api_config.get("vlm_model_name", "qwen3.6-plus")
        self.api_vlm_key = api_config.get("vlm_key", "")
        self.api_vlm_base_url = api_config.get(
            "vlm_base_url", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.api_llm_model_name = api_config.get("llm_model_name", "deepseek-v4-flash") # deepseek-v4-flash, deepseek-v4-pro
        self.api_llm_key = api_config.get("llm_key", "")
        self.api_llm_base_url = api_config.get("llm_base_url", "https://api.deepseek.com")

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

    def get_audio_config(self):
        """获取 audio_input 配置字典。"""
        return dict(self._config.get("audio_input") or {})

    def get_asr_config(self):
        """获取 asr 配置字典。"""
        return dict(self._config.get("asr") or {})
    
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