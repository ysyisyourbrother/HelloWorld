import multiprocessing as mp
import signal
import sys
import os
import logging
from typing import Optional

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.api_server_c import APIServerC
from src.reasoner import Reasoner


class VenusSystemCloud:
    """云端系统 - 整合 gRPC API 服务器和推理模块"""
    
    def __init__(self, config: Config = None):
        """
        初始化云端系统
        
        Args:
            config: 配置对象，如果为None则从默认路径加载
        """
        if config is None:
            config = Config()
        self.config = config
        
        # 组件
        self.api_server: Optional[APIServerC] = None
        self.reasoner: Optional[Reasoner] = None
        
        # 状态
        self.running = False
        
        # 设置信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # 设置日志
        self._setup_logger()
    
    def _setup_logger(self):
        """设置系统日志"""
        self.logger = logging.getLogger('VenusSystemCloud')
        self.logger.setLevel(logging.INFO)
        
        # 控制台输出
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        self.logger.propagate = False
    
    def _signal_handler(self, signum, frame):
        """处理退出信号"""
        self.logger.debug(f"收到信号 {signum}，正在关闭系统...")
        self.stop()
        sys.exit(0)
    
    def _initialize_components(self):
        """初始化系统组件"""
        self.logger.info("正在初始化云端系统组件...")
        
        # 启动 Reasoner
        self.logger.debug("初始化 Reasoner")
        self.reasoner = Reasoner(self.config)
        # 从 Reasoner 获取队列
        prompt_queue = self.reasoner.prompt_queue
        result_queue = self.reasoner.result_queue
        
        # 初始化 API 服务器并设置队列
        self.api_server = APIServerC(self.config)
        self.api_server.set_prompt_queue(prompt_queue)
        self.api_server.set_result_queue(result_queue)
    
    def start(self):
        """启动云端系统"""
        if self.running:
            self.logger.warning("系统已在运行中")
            return
        
        self.logger.info("=" * 50)
        self.logger.info("启动云端系统")
        self.logger.info("=" * 50)
        
        # 初始化组件
        self._initialize_components()
        
        # 启动 Reasoner（如果启用）
        if self.reasoner is not None:
            self.logger.debug("启动推理模块...")
            self.reasoner.start()
            self.logger.debug("推理模块已启动")
        
        # 启动 API 服务器（阻塞调用）
        self.logger.info(f"启动 gRPC API 服务器 (监听 {self.config.server_host}:{self.config.server_port})...")
        self.running = True
        
        try:
            self.api_server.start()
        except KeyboardInterrupt:
            self.logger.debug("收到中断信号")
        except Exception as e:
            self.logger.error(f"API 服务器运行出错: {e}", exc_info=True)
        finally:
            self.stop()
    
    def stop(self):
        """停止云端系统"""
        if not self.running:
            return
        
        self.logger.info("正在关闭云端系统...")
        self.running = False
        
        # 停止 API 服务器
        if self.api_server is not None:
            try:
                self.api_server.stop()
                self.logger.debug("API 服务器已停止")
            except Exception as e:
                self.logger.error(f"停止 API 服务器时出错: {e}")
        
        # 停止 Reasoner
        if self.reasoner is not None:
            try:
                self.reasoner.stop()
                self.logger.debug("推理模块已停止")
            except Exception as e:
                self.logger.error(f"停止推理模块时出错: {e}")
        
        self.logger.info("云端系统已关闭")


if __name__ == "__main__":
    # 直接运行时的测试代码
    system = VenusSystemCloud()
    system.start()

