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
from typing import List, Optional, Dict, Any

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


@dataclass
class QueryRequest:
    """查询请求结构体"""
    query_text: str              # 查询文本
    memory_results: List[Any]    # 记忆检索结果（帧数据列表）
    query_id: int                # 查询ID


@dataclass
class QueryResponse:
    """查询响应结构体"""
    query_id: int                # 查询ID
    result: Optional[str]        # 完整结果
    error: Optional[str]         # 错误信息
    timestamp: float             # 时间戳


class Reasoner:
    def __init__(self, config: Config = None):
        """
        初始化Reasoner模块
        负责基于检索到的视频帧和用户查询进行推理，生成回答
        
        Args:
            config (Config): 配置对象实例
        """
        if config is None:
            config = Config()
        self.config = config
        
        # 从Config对象获取配置
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
        
        # 模型相关
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.max_length = None
        
        # 队列相关
        self.query_queue = mp.Queue(maxsize=100)
        self.result_queue = mp.Queue(maxsize=100)
        
        self.running = False
        self.running_event = None
        
    def _set_logger(self):
        """设置日志记录器"""
        log_file = self.config.reasoner_log_file
        pattern = log_file.replace(".log", "*")
        log_files = glob.glob(pattern)
        for f in log_files:
            try:
                os.remove(f)
            except Exception as e:
                pass

        self.logger = logging.getLogger(name='Reasoner')
        self.logger.setLevel(logging.DEBUG)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.INFO)
        self.logger.addHandler(console_handler)

        # 文件处理器
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
    
    def _inference(
        self, 
        query_text: str, 
        frames: Optional[torch.Tensor] = None
    ) -> str:
        """
        执行推理
        
        Args:
            query_text: 查询文本
            frames: 预处理后的帧张量
            
        Returns:
            生成的文本
        """
        # 构建提示词
        question = self._build_prompt(query_text, has_frames=(frames is not None))
        
        self.logger.debug(f"查询提示: {question}")
        
        # 构建对话
        conv = copy.deepcopy(conv_qwen)
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        
        # Tokenize
        input_ids = tokenizer_image_token(
            prompt_question, 
            self.tokenizer, 
            IMAGE_TOKEN_INDEX,
            return_tensors="pt"
        ).unsqueeze(0).to(self.device)
        
        # 创建attention mask
        attention_masks = input_ids.ne(self.tokenizer.pad_token_id).long().to(self.device)
        
        # 停止条件
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        
        # 生成
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
        
        # 解码
        text_outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        
        return text_outputs
    
    def _process_query(self, query_request: QueryRequest):
        """
        处理单个查询请求
        
        Args:
            query_request: 查询请求对象
        """
        try:
            query_id = query_request.query_id
            query_text = query_request.query_text
            memory_results = query_request.memory_results
            
            self.logger.info(f"开始处理查询 {query_id}: {query_text}")
            
            # 预处理帧数据
            frames_tensor = None
            if memory_results and len(memory_results) > 0:
                # 假设memory_results是numpy数组列表（RGB格式）
                frames_tensor = self._preprocess_frames(memory_results)
                self.logger.debug(f"预处理了 {len(memory_results)} 帧，张量形状: {frames_tensor.shape}")
            
            # 执行推理
            start_time = time.time()
            result_text = self._inference(query_text, frames_tensor)
            inference_time = time.time() - start_time
            
            self.logger.info(f"查询 {query_id} 推理完成，耗时: {inference_time:.2f}s")
            self.logger.info(f"回答: {result_text}")
            
            # 构建响应
            response = QueryResponse(
                query_id=query_id,
                result=result_text,
                error=None,
                timestamp=time.time()
            )
            self.result_queue.put(response)
            
        except Exception as e:
            self.logger.error(f"处理查询 {query_request.query_id} 时出错: {e}", exc_info=True)
            # 发送错误响应
            response = QueryResponse(
                query_id=query_request.query_id,
                result=None,
                error=str(e),
                timestamp=time.time()
            )
            self.result_queue.put(response)
    
    def _process_queries(self):
        """处理查询的主循环"""
        self.logger.info("查询处理线程启动")
        
        while self.running_event.is_set():
            try:
                # 从队列获取查询请求
                query_request: QueryRequest = self.query_queue.get(timeout=1.0)
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
        
        # 初始化模型
        self._initialize_model()
        
        # 启动查询处理循环
        self._process_queries()
    
    def start(self):
        """启动Reasoner进程"""
        self.running_event = mp.Event()
        self.running_event.set()
        
        self.process = mp.Process(target=self._process_main, daemon=True)
        if hasattr(self.process, 'name'):
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
        if hasattr(self, 'running_event'):
            self.running_event.clear()
        
        self.running = False
        
        if hasattr(self, 'process') and self.process.is_alive():
            self.process.join(timeout=5)
    
    def add_query(
        self, 
        query_text: str, 
        memory_results: List[Any], 
        query_id: int
    ):
        """
        添加查询请求到队列
        
        Args:
            query_text: 查询文本
            memory_results: 记忆检索结果
            query_id: 查询ID
        """
        query_request = QueryRequest(
            query_text=query_text,
            memory_results=memory_results,
            query_id=query_id
        )
        self.query_queue.put(query_request)
    
    def get_result_queue(self):
        """获取结果队列"""
        return self.result_queue


if __name__ == "__main__":
    # 测试代码
    import cv2
    
    config = Config()
    reasoner = Reasoner(config)
    
    try:
        reasoner.start()
        
        # 读取测试图片
        img_path = "test_frame.png"
        if os.path.exists(img_path):
            frame = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 添加查询
            reasoner.add_query(
                query_text="Hello, please describe this image.",
                memory_results=[frame],
                query_id=1
            )
            
            # 等待结果
            result = reasoner.result_queue.get(timeout=30)
            print(f"查询结果: {result}")
        else:
            print(f"测试图片 {img_path} 不存在")
        
        time.sleep(5)
        
    except KeyboardInterrupt:
        pass
    finally:
        reasoner.stop()
