import pickle
import time
import queue
import logging
from logging.handlers import RotatingFileHandler
import glob
import os
import grpc
from typing import Optional
import threading
import multiprocessing as mp

import cv2

# 本项目
from src.config import Config
from src.query_vectorizer import QueryVectorizer, QueryData
from src.memory_manager import MemoryManager, MemoryResult

# 导入生成的 gRPC 代码
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from proto import query_service_pb2
from proto import query_service_pb2_grpc


class APIServerE:
    """边端 gRPC API 客户端"""
    
    def __init__(self, config: Config = None):
        """
        初始化边端 API 服务器
        
        Args:
            config (Config): 配置对象实例
        """
        if config is None:
            config = Config()
        self.log_file = config.api_e_log_file
        self.test_mode = config.api_e_test_mode

        self.cloud_server_url = config.cloud_server_url
        
        self.grpc_channel: Optional[grpc.Channel] = None
        self.grpc_stub: Optional[query_service_pb2_grpc.QueryServiceStub] = None
        self.running = False
        self.query_id_counter = 0
        self.query_id_lock = threading.Lock()

        self.query_queue = mp.Queue(maxsize=100)  # 用户查询队列
        self.query_result_queue = None  # 需要由memory_manager设置

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

        self.logger = logging.getLogger(name='APIServerE')
        self.logger.setLevel(logging.DEBUG)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.INFO)
        self.logger.addHandler(console_handler)

        # 文件处理器（确保日志目录存在）
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        self.logger.propagate = False
    
    def set_query_result_queue(self, query_result_queue):
        """
        """
        self.query_result_queue = query_result_queue
    
    def _get_next_query_id(self) -> int:
        """获取下一个查询ID"""
        with self.query_id_lock:
            self.query_id_counter += 1
            return self.query_id_counter
    
    def _connect_to_cloud(self):
        """连接到云端 gRPC 服务器"""
        cloud_server_url = self.cloud_server_url
        # 从 URL 中提取主机和端口
        # 假设格式为 "grpc://host:port" 或 "http://host:port" 或 "host:port"
        if cloud_server_url.startswith("grpc://"):
            cloud_server_url = cloud_server_url[7:]
        elif cloud_server_url.startswith("http://"):
            cloud_server_url = cloud_server_url[7:]
        elif cloud_server_url.startswith("https://"):
            cloud_server_url = cloud_server_url[8:]
        
        # 视频帧数据可能较大，提高 gRPC 消息大小限制（默认 4MB）
        max_msg_size = 50 * 1024 * 1024  # 50MB
        options = [
            ('grpc.max_send_message_length', max_msg_size),
            ('grpc.max_receive_message_length', max_msg_size),
        ]
        self.grpc_channel = grpc.insecure_channel(cloud_server_url, options=options)
        self.grpc_stub = query_service_pb2_grpc.QueryServiceStub(self.grpc_channel)
        self.logger.info(f"已连接到云端服务器: {cloud_server_url}")
    
    def query(self, query_text: str, dialog_id: int = 0) -> dict:
        """
        处理用户查询（单轮对话）
        
        Args:
            query_text: 用户查询文本
            dialog_id: 对话ID（用于多轮对话，单轮对话时默认为0）
            
        Returns:
            包含查询结果的字典
        """
        if self.test_mode:
            self.logger.debug("测试模式：委托 query_test 处理")
            return self.query_test(query_text=query_text, dialog_id=dialog_id)

        if self.grpc_stub is None:
            self._connect_to_cloud()

        query_id = self._get_next_query_id()
        timestamp = time.time()

        self.logger.info(f"处理查询 {query_id}: {query_text}")

        query_data = QueryData(
            query=query_text,
            query_id=query_id,
            dialog_id=dialog_id,
            timestamp=timestamp
        )
        self.query_queue.put(query_data)
        self.logger.debug(f"查询 {query_id} 已放入向量化队列")

        query_result: MemoryResult = self.query_result_queue.get(timeout=300)
        if query_result.query_id != query_id:
            self.logger.warning(f"查询ID不匹配: 期望 {query_id}, 收到 {query_result.query_id}")

        frame_data_list = query_result.frame_data_list
        memory_results_bytes = pickle.dumps(frame_data_list) if frame_data_list else b""
        self.logger.debug(f"序列化了 {len(frame_data_list) if frame_data_list else 0} 帧数据")

        return self._send_grpc_and_return(
            query_text=query_text,
            memory_results_bytes=memory_results_bytes,
            query_id=query_id,
            dialog_id=dialog_id
        )

    def query_test(self, query_text: str, image_path: Optional[str] = None, dialog_id: int = 0) -> dict:
        """
        测试模式查询：跳过 MemoryManager，直接发送 gRPC 请求。
        可传入图片地址，将该图片放入 memory_results 供云端推理使用。
        
        Args:
            query_text: 用户查询文本
            image_path: 可选，图片文件路径，读取后放入 memory_results
            dialog_id: 对话ID（用于多轮对话，单轮对话时默认为0）
            
        Returns:
            包含查询结果的字典
        """
        if self.grpc_stub is None:
            self._connect_to_cloud()

        query_id = self._get_next_query_id()
        self.logger.info(f"测试查询 {query_id}: {query_text}")

        if image_path:
            frame = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if frame is None:
                self.logger.warning(f"无法读取图片: {image_path}，使用空的 memory_results")
                memory_results_bytes = b""
            else:
                if frame.ndim == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                else:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_list = [frame]
                memory_results_bytes = pickle.dumps(frame_list)
                self.logger.debug(f"已将图片 {image_path} 放入 memory_results")
        else:
            memory_results_bytes = b""
            self.logger.debug("无图片路径，使用空的 memory_results")

        return self._send_grpc_and_return(
            query_text=query_text,
            memory_results_bytes=memory_results_bytes,
            query_id=query_id,
            dialog_id=dialog_id
        )

    def _send_grpc_and_return(
        self,
        query_text: str,
        memory_results_bytes: bytes,
        query_id: int,
        dialog_id: int
    ) -> dict:
        """构建 gRPC 请求、发送并返回结果"""
        grpc_request = query_service_pb2.QueryRequest(
            query_text=query_text,
            memory_results=memory_results_bytes,
            query_id=query_id,
            dialog_id=dialog_id
        )
        self.logger.info(f"发送查询 {query_id} 到云端")
        grpc_response = self.grpc_stub.Query(grpc_request, timeout=300)
        return {
            "query_id": grpc_response.query_id,
            "result": grpc_response.result,
            "error": grpc_response.error if grpc_response.error else None,
            "timestamp": grpc_response.timestamp
        }
    
    def query_stream(self, query_text: str, dialog_id: int = 0):
        """
        服务器端流式RPC查询（预留接口）
        
        Args:
            query_text: 用户查询文本
            dialog_id: 对话ID
            
        Yields:
            查询响应流
        """
        # TODO: 实现服务器端流式RPC
        self.logger.warning("query_stream 方法尚未实现")
        yield {
            "query_id": 0,
            "result": "",
            "error": "query_stream 方法尚未实现",
            "timestamp": time.time()
        }
    
    def query_bidi_stream(self, query_texts: list, dialog_id: int = 0):
        """
        双向流式RPC查询（预留接口）
        
        Args:
            query_texts: 查询文本列表
            dialog_id: 对话ID
            
        Yields:
            查询响应流
        """
        # TODO: 实现双向流式RPC
        self.logger.warning("query_bidi_stream 方法尚未实现")
        for query_text in query_texts:
            yield {
                "query_id": 0,
                "result": "",
                "error": "query_bidi_stream 方法尚未实现",
                "timestamp": time.time()
            }
    
    def start(self):
        """启动边端 API 服务器"""
        self.running = True
        self.logger.info("边端 API 服务器已启动")
        # 连接到云端
        if self.grpc_stub is None:
            self._connect_to_cloud()
    
    def stop(self):
        """停止边端 API 服务器"""
        if self.grpc_channel is not None:
            self.grpc_channel.close()
            self.logger.info("已关闭云端连接")
        self.running = False
        self.logger.info("边端 API 服务器已停止")

    def check_cloud_status(self):
        return self.running
