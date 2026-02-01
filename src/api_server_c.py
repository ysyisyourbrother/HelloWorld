import pickle
import time
import queue
import logging
from logging.handlers import RotatingFileHandler
import glob
import os
import grpc
from concurrent import futures
from typing import Optional
import multiprocessing as mp

# 本项目
from src.config import Config
from src.reasoner import QueryRequest, QueryResponse

# 导入生成的 gRPC 代码
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from proto import query_service_pb2
from proto import query_service_pb2_grpc


class QueryServiceServicer(query_service_pb2_grpc.QueryServiceServicer):
    """gRPC 查询服务实现"""
    
    def __init__(self, prompt_queue: mp.Queue, result_queue: mp.Queue, logger: logging.Logger):
        self.prompt_queue = prompt_queue
        self.result_queue = result_queue
        self.logger = logger
    
    def Query(self, request: query_service_pb2.QueryRequest, context):
        """
        一元RPC - 处理单轮对话查询
        
        Args:
            request: 查询请求
            context: gRPC 上下文
            
        Returns:
            查询响应
        """
        query_id = request.query_id
        query_text = request.query_text
        dialog_id = getattr(request, "dialog_id", 0)
        
        self.logger.info(f"收到查询请求 {query_id}: {query_text} (dialog_id={dialog_id})")
        
        # 反序列化帧数据
        memory_results = []
        if request.memory_results:
            memory_results = pickle.loads(request.memory_results)
            self.logger.debug(f"反序列化了 {len(memory_results)} 帧数据")
        
        # 创建 QueryRequest 并添加到查询队列
        query_request = QueryRequest(
            query_text=query_text,
            memory_results=memory_results,
            query_id=query_id,
            dialog_id=dialog_id
        )
        self.prompt_queue.put(query_request)
        
        # 等待结果
        response = self.result_queue.get(timeout=300)  # 5分钟超时
        
        # 构建 gRPC 响应
        grpc_response = query_service_pb2.QueryResponse(
            query_id=response.query_id,
            result=response.result if response.result else "",
            error=response.error if response.error else "",
            timestamp=response.timestamp
        )
        
        self.logger.info(f"查询 {query_id} 处理完成")
        return grpc_response
    
    def QueryStream(self, request: query_service_pb2.QueryRequest, context):
        """
        服务器端流式RPC - 多轮对话（预留接口）
        
        Args:
            request: 查询请求
            context: gRPC 上下文
            
        Yields:
            查询响应流
        """
        # TODO: 实现服务器端流式RPC
        self.logger.warning("QueryStream 方法尚未实现")
        yield query_service_pb2.QueryResponse(
            query_id=request.query_id,
            result="",
            error="QueryStream 方法尚未实现",
            timestamp=time.time()
        )
    
    def QueryBidiStream(self, request_iterator, context):
        """
        双向流式RPC - 多轮对话（预留接口）
        
        Args:
            request_iterator: 请求迭代器
            context: gRPC 上下文
            
        Yields:
            查询响应流
        """
        # TODO: 实现双向流式RPC
        self.logger.warning("QueryBidiStream 方法尚未实现")
        for request in request_iterator:
            yield query_service_pb2.QueryResponse(
                query_id=request.query_id,
                result="",
                error="QueryBidiStream 方法尚未实现",
                timestamp=time.time()
            )


class APIServerC:
    """云端 gRPC API 服务器"""
    
    def __init__(self, config: Config = None):
        """
        初始化云端 API 服务器
        
        Args:
            config (Config): 配置对象实例
        """
        if config is None:
            config = Config()
        self.log_file = config.api_c_log_file
        self.server_host = config.server_host
        self.server_port = config.server_port
        
        self.prompt_queue: Optional[mp.Queue] = None # reasoner的输入
        self.result_queue: Optional[mp.Queue] = None # reasoner的输出
        self.server: Optional[grpc.Server] = None
        self.running = False
        
        self._set_logger()
    
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

        self.logger = logging.getLogger(name='APIServerC')
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
    
    def set_prompt_queue(self, prompt_queue: mp.Queue):
        """
        设置查询队列
        
        Args:
            prompt_queue: Reasoner 的查询队列
        """
        self.prompt_queue = prompt_queue
    
    def set_result_queue(self, result_queue: mp.Queue):
        """
        设置结果队列
        
        Args:
            result_queue: Reasoner 的结果队列
        """
        self.result_queue = result_queue
    
    def start(self):
        """启动 gRPC 服务器"""
        if self.prompt_queue is None or self.result_queue is None:
            raise ValueError("查询队列或结果队列未设置，请先调用 set_query_queue() 和 set_result_queue()")
        
        # 创建 gRPC 服务器
        self.server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
        
        # 添加服务
        query_service_pb2_grpc.add_QueryServiceServicer_to_server(
            QueryServiceServicer(self.prompt_queue, self.result_queue, self.logger),
            self.server
        )
        
        # 监听端口
        server_host = self.server_host
        server_port = self.server_port
        listen_addr = f"{server_host}:{server_port}"
        self.server.add_insecure_port(listen_addr)
        
        # 启动服务器
        self.server.start()
        self.running = True
        self.logger.info(f"gRPC 服务器已启动，监听地址: {listen_addr}")
        
        # 等待服务器关闭
        self.server.wait_for_termination()
    
    def stop(self):
        """停止 gRPC 服务器"""
        if self.server is not None:
            self.server.stop(grace=5)
            self.running = False
            self.logger.info("gRPC 服务器已停止")
