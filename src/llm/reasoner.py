import multiprocessing as mp
import threading
import numpy as np
import torch
import logging
from logging.handlers import RotatingFileHandler
import time
import queue
import glob
import os
import copy
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

# 本项目
from src.config import Config

# LLaVA相关导入
from models.llava.model.builder import load_pretrained_model
from models.llava.mm_utils import (
    get_model_name_from_path, 
    process_images, 
    tokenizer_image_token, 
    KeywordsStoppingCriteria
)
from models.llava.constants import (
    IMAGE_TOKEN_INDEX, 
    DEFAULT_IMAGE_TOKEN, 
    DEFAULT_IM_START_TOKEN, 
    DEFAULT_IM_END_TOKEN, 
    IGNORE_INDEX
)
from models.llava.conversation import SeparatorStyle, conv_qwen


def _uniform_subsample_image_list(images, max_n):
    # type: (List[Any], Optional[int]) -> List[Any]
    """
    沿时间顺序对图像列表做均匀下采样，最多保留 max_n 张（含首尾附近索引）。
    max_n 为 None 或 <=0 时返回原列表的浅拷贝；len(images) <= max_n 时全量返回。
    """
    if not images:
        return list(images)
    if max_n is None or max_n <= 0:
        return list(images)
    n = len(images)
    k = int(max_n)
    if n <= k:
        return list(images)
    if k == 1:
        return [images[n // 2]]
    indices = [int(round(j * (n - 1) / float(k - 1))) for j in range(k)]
    seen = set()
    out = []
    for i in indices:
        if i not in seen:
            seen.add(i)
            out.append(images[i])
    return out


@dataclass
class QueryRequest:
    """查询请求结构体"""
    query_text: str              # 查询文本
    memory_results: List[Any]    # 记忆检索结果（帧数据列表）
    query_id: int                # 查询ID
    dialog_id: int = 0          # 对话ID（多轮对话时用于云端记忆）


@dataclass
class QueryResponse:
    """查询响应结构体"""
    query_id: int                # 查询ID
    result: Optional[str]        # 完整结果
    error: Optional[str]         # 错误信息
    timestamp: float             # 时间戳


class ReasonerBase:
    def __init__(self, config: Config = None):
        """
        初始化Reasoner模块
        负责基于检索到的视频帧和用户查询进行推理，生成回答
        
        Args:
            config (Config): 配置对象实例
        """
        if config is None:
            config = Config()
        
        # 测试模式配置
        self.test_mode = config.reasoner_test_mode
        self.test_response = config.reasoner_test_response
        
        # 从Config对象获取配置
        self.log_file = config.reasoner_log_file
        self.model_path = config.reasoner_model_path
        self.model_name = config.reasoner_model_name
        self.model_base = config.reasoner_model_base
        self.torch_dtype = config.reasoner_torch_dtype
        self.load_in_8bit = config.reasoner_load_in_8bit
        self.load_in_4bit = config.reasoner_load_in_4bit
        self.device_map = config.reasoner_device_map
        self.attn_implementation = config.reasoner_attn_implementation
        self.mm_spatial_pool_mode = config.reasoner_mm_spatial_pool_mode
        self.conv_template = config.reasoner_conv_template
        self.device = config.reasoner_device
        
        # 生成参数
        self.max_new_tokens = config.reasoner_max_new_tokens
        self.temperature = config.reasoner_temperature
        self.top_p = config.reasoner_top_p
        self.num_beams = config.reasoner_num_beams
        self.do_sample = config.reasoner_do_sample
        self.max_history_turns = getattr(config, "reasoner_max_history_turns", None)
        self.max_img_num = getattr(config, "reasoner_max_img_num", None)

        # 模型相关
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.max_length = None
        
        # 按 dialog_id 维护对话历史：(user_turn, assistant_turn) 列表
        self.dialog_histories: Dict[int, List[Tuple[str, str]]] = {}
        
    def _set_logger(self):
        """设置日志记录器"""
        log_file = self.log_file
        pattern = log_file.replace(".log", "*")
        log_files = glob.glob(pattern)
        for f in log_files:
            try:
                os.remove(f)
            except Exception as e:
                pass

        self.logger = logging.getLogger(name='Reasoner')
        self.logger.handlers.clear()
        self.logger.setLevel(logging.DEBUG)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.INFO)
        self.logger.addHandler(console_handler)

        # 文件处理器
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        self.logger.propagate = False
    
    def _initialize_model(self):
        """初始化LLaVA模型"""
        self.logger.info(f"开始加载LLaVA模型: {self.model_path}")
        
        # 准备overwrite_config
        overwrite_config = {}
        overwrite_config["mm_spatial_pool_mode"] = self.mm_spatial_pool_mode
        
        # 记录显存使用
        mem_before = torch.cuda.max_memory_allocated()
        
        # 加载模型
        self.tokenizer, self.model, self.image_processor, self.max_length = load_pretrained_model(
            model_path=self.model_path,
            model_base=self.model_base,
            model_name=self.model_name,
            torch_dtype=self.torch_dtype,
            load_in_8bit=self.load_in_8bit,
            load_in_4bit=self.load_in_4bit,
            device_map=self.device_map,
            attn_implementation=self.attn_implementation,
            overwrite_config=overwrite_config
        )
        
        mem_after = torch.cuda.max_memory_allocated()
        mem_used_gb = (mem_after - mem_before) / 1024 / 1024 / 1024
        
        self.logger.info(f"模型加载完成，显存使用: {mem_used_gb:.2f} GB")
        
        # 设置为评估模式
        self.model.eval()
        
        # 设置pad_token_id（针对Qwen模型）
        if self.tokenizer.pad_token_id is None:
            if "qwen" in self.tokenizer.name_or_path.lower():
                self.logger.info("为Qwen模型设置pad_token_id")
                self.tokenizer.pad_token_id = 151643
    
    def _preprocess_frames(self, frames: List[np.ndarray]) -> torch.Tensor:
        """
        预处理帧数据
        
        Args:
            frames: 帧列表（numpy数组，RGB格式）
            
        Returns:
            处理后的张量
        """
        # 使用image_processor处理帧
        processed_frames = self.image_processor.preprocess(
            frames, 
            return_tensors="pt"
        )["pixel_values"].to(self.device)
        
        # 转换为bfloat16
        if self.torch_dtype == "bfloat16":
            processed_frames = processed_frames.bfloat16()
        elif self.torch_dtype == "float16":
            processed_frames = processed_frames.half()
        
        return processed_frames
    
    def _build_prompt(self, query_text: str, has_frames: bool = True) -> str:
        """
        构建提示词
        
        Args:
            query_text: 用户查询文本
            has_frames: 是否包含视频帧
            
        Returns:
            构建好的提示词
        """
        if has_frames:
            if self.model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + query_text
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + query_text
        else:
            qs = query_text
        
        return qs
    
    @staticmethod
    def _strip_image_placeholders_for_history(user_msg: str) -> str:
        """
        从要写入历史的 user 消息中移除图像占位符，避免下一轮无图时
        prompt 中仍含 <image> 导致 input_ids 与 images 数量不一致触发 CUDA assert。
        """
        if not user_msg:
            return user_msg
        s = user_msg
        for token in (DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_TOKEN):
            s = s.replace(token, "")
        return s.strip()
    
    def _inference_with_history(
        self,
        question: str,
        frames: Optional[torch.Tensor] = None,
        history: Optional[List[Tuple[str, str]]] = None,
    ) -> str:
        """
        带对话历史的推理。先按历史拼 conv，再拼当前轮，再生成。
        
        Args:
            question: 当前轮用户提示（已含图像 token 等）
            frames: 当前轮视频帧张量（仅当前轮带帧，历史轮仅文本）
            history: 历史轮列表 [(user_msg, assistant_msg), ...]
            
        Returns:
            生成的文本
        """
        if history is None:
            history = []
        conv = copy.deepcopy(conv_qwen)
        for user_msg, assistant_msg in history:
            conv.append_message(conv.roles[0], user_msg)
            conv.append_message(conv.roles[1], assistant_msg)
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        
        self.logger.debug(f"对话提示长度: {len(prompt_question)} 字符, 历史轮数: {len(history)}")

        input_ids = tokenizer_image_token(
            prompt_question,
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt"
        ).unsqueeze(0).to(self.device)
        attention_masks = input_ids.ne(self.tokenizer.pad_token_id).long().to(self.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        
        with torch.inference_mode():
            output_ids = self.model.generate(
                inputs=input_ids,
                images=frames,
                attention_mask=attention_masks,
                modalities="video" if frames is not None else None,
                do_sample=self.do_sample,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                top_p=self.top_p,
                num_beams=self.num_beams,
                use_cache=True,
                stopping_criteria=[stopping_criteria]
            )
        text_outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return text_outputs
    
    def infer_sync(self, query_request: QueryRequest) -> QueryResponse:
        """
        同步推理，直接返回结果（用于 benchmark 等单进程场景，无需 gRPC/队列）。
        需先调用 _initialize_model() 或通过 test_mode 跳过模型加载。
        """
        if not hasattr(self, "logger") or self.logger is None:
            self._set_logger()
        if not self.test_mode and self.model is None:
            self._initialize_model()
        try:
            query_id = query_request.query_id
            query_text = query_request.query_text
            memory_results = query_request.memory_results
            dialog_id = getattr(query_request, "dialog_id", 0)

            self.logger.info(f"开始处理查询 {query_id}: {query_text} (dialog_id={dialog_id})")

            if self.test_mode:
                self.logger.info(f"测试模式：查询 {query_id} 直接返回测试文本")
                return QueryResponse(
                    query_id=query_id,
                    result=self.test_response,
                    error=None,
                    timestamp=time.time()
                )

            if memory_results and len(memory_results) > 0 and self.max_img_num:
                n_before = len(memory_results)
                memory_results = _uniform_subsample_image_list(
                    memory_results, self.max_img_num
                )
                if len(memory_results) < n_before:
                    self.logger.info(
                        "Reasoner max_img_num=%d: 检索图像 %d -> %d（均匀稀疏采样）",
                        int(self.max_img_num),
                        n_before,
                        len(memory_results),
                    )

            frames_tensor = None
            if memory_results and len(memory_results) > 0:
                frames_tensor = self._preprocess_frames(memory_results)
                self.logger.debug(f"预处理了 {len(memory_results)} 帧，张量形状: {frames_tensor.shape}")

            question = self._build_prompt(query_text, has_frames=(frames_tensor is not None))
            history = self.dialog_histories.get(dialog_id, [])
            if getattr(self, "max_history_turns", None) is not None and self.max_history_turns > 0:
                history = history[-self.max_history_turns:]

            start_time = time.time()
            result_text = self._inference_with_history(question, frames_tensor, history=history)
            inference_time = time.time() - start_time

            self.logger.debug(f"查询 {query_id} 推理完成，耗时: {inference_time:.2f}s")
            self.logger.info(f"回答: {result_text}")

            if dialog_id != 0:
                history_user_msg = self._strip_image_placeholders_for_history(question)
                self.dialog_histories.setdefault(dialog_id, []).append((history_user_msg, result_text))
                if getattr(self, "max_history_turns", None) is not None and self.max_history_turns > 0:
                    self.dialog_histories[dialog_id] = self.dialog_histories[dialog_id][-self.max_history_turns:]

            return QueryResponse(
                query_id=query_id,
                result=result_text,
                error=None,
                timestamp=time.time()
            )
        except Exception as e:
            self.logger.error(f"处理查询 {query_request.query_id} 时出错: {e}", exc_info=True)
            return QueryResponse(
                query_id=query_request.query_id,
                result=None,
                error=str(e),
                timestamp=time.time()
            )

class ReasonerOnline(ReasonerBase):
    """在线推理类：在 Base 同步能力上提供队列与子进程处理。"""

    def __init__(self, config: Config = None):
        super().__init__(config)
        self.prompt_queue = mp.Queue(maxsize=100)
        self.result_queue = mp.Queue(maxsize=100)
        self.running = False
        self.running_event = None

    def _process_query(self, query_request: QueryRequest):
        response = self.infer_sync(query_request)
        self.result_queue.put(response)

    def _process_queries(self):
        """处理查询的主循环"""
        self.logger.info("查询处理线程启动")

        while self.running_event.is_set():
            try:
                query_request: QueryRequest = self.prompt_queue.get(timeout=1.0)
                self._process_query(query_request)
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"处理查询时出错: {e}", exc_info=True)

        self.logger.info("查询处理线程退出")

    def _process_main(self):
        """主处理函数"""
        self._set_logger()
        self.logger.info(f"Reasoner进程启动, 进程ID: {mp.current_process().pid}")

        if self.test_mode:
            self.logger.info("测试模式：不加载模型，直接返回测试文本")
        else:
            self._initialize_model()

        self._process_queries()

    def start(self):
        """启动Reasoner进程"""
        self.running_event = mp.Event()
        self.running_event.set()

        self.process = mp.Process(target=self._process_main, daemon=True)
        if hasattr(self.process, "name"):
            self.process.name = "Reasoner-Processor"
        self.process.start()
        self.running = True

    def start_single_process(self):
        """启动单进程模式"""
        self.running_event = mp.Event()
        self.running_event.set()
        self._process_main()

    def stop(self):
        """停止Reasoner进程"""
        if hasattr(self, "running_event"):
            self.running_event.clear()

        self.running = False

        if hasattr(self, "process") and self.process.is_alive():
            self.process.join(timeout=5)

    def add_query(
        self,
        query_text: str,
        memory_results: List[Any],
        query_id: int,
        dialog_id: int = 0,
    ):
        """添加查询请求到队列"""
        query_request = QueryRequest(
            query_text=query_text,
            memory_results=memory_results,
            query_id=query_id,
            dialog_id=dialog_id,
        )
        self.prompt_queue.put(query_request)

    def get_result_queue(self):
        """获取结果队列"""
        return self.result_queue

