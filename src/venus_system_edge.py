import signal
import sys
import os
import logging
from typing import Optional

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config
from src.api.api_server_e import APIServerE
from src.video_input.video_input import VideoInput
from src.memory.memory_manager import MemoryManager


# 支持的边端模式
EDGE_MODE_QUERY_WHILE_INJECT = "query_while_inject"
EDGE_MODE_QUERY_WITH_MEMORY = "query_with_memory"
EDGE_MODE_ONLY_INJECT = "only_inject"
EDGE_MODE_BENCHMARK = "benchmark"


class VenusSystemEdge:
    """边端系统 - 根据 edge_mode 整合视频编码、检索与 gRPC 客户端"""
    
    def __init__(self, config: Config = None):
        """
        初始化边端系统
        
        Args:
            config: 配置对象，如果为 None 则从默认路径加载
        """
        if config is None:
            config = Config()
        self.config = config
        self.edge_mode = getattr(config, "edge_mode", EDGE_MODE_QUERY_WHILE_INJECT)
        
        # 组件
        self.api_server: Optional[APIServerE] = None
        self.video_input: Optional[VideoInput] = None
        self.memory_manager: Optional[MemoryManager] = None
        
        # 状态
        self.running = False
        
        # 设置信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # 设置日志
        self._setup_logger()
    
    def _setup_logger(self):
        """设置系统日志"""
        self.logger = logging.getLogger('VenusSystemEdge')
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
    
    def _check_vector_and_map_files(self) -> bool:
        """检测向量文件和 map 文件是否存在（用于 query_with_memory）"""
        faiss_path = self.config.memory_faiss_file_path
        map_path = self.config.memory_databasemap_file_path
        if not os.path.isfile(faiss_path):
            self.logger.error(f"向量文件不存在: {faiss_path}，query_with_memory 模式需要已有向量库")
            return False
        if not os.path.isfile(map_path):
            self.logger.error(f"databasemap 文件不存在: {map_path}，query_with_memory 模式需要已有 map 文件")
            return False
        return True
    
    def _initialize_components(self):
        """根据 edge_mode 初始化对应组件"""
        self.logger.info("正在初始化边端系统组件...")
        self.logger.info(f"边端模式: {self.edge_mode}")
        
        if self.edge_mode == EDGE_MODE_QUERY_WHILE_INJECT:
            self._initialize_query_while_inject()
        elif self.edge_mode == EDGE_MODE_QUERY_WITH_MEMORY:
            self._initialize_query_with_memory()
        elif self.edge_mode == EDGE_MODE_ONLY_INJECT:
            self._initialize_only_inject()
        elif self.edge_mode == EDGE_MODE_BENCHMARK:
            self._initialize_benchmark()
        else:
            raise ValueError(f"不支持的 edge_mode: {self.edge_mode}")
    
    def _initialize_query_while_inject(self):
        """query_while_inject：VideoInput, MemoryManager（内含帧/查询编码）, APIServerE"""
        self.video_input = VideoInput(self.config)
        self.memory_manager = MemoryManager(self.config)
        self.api_server = APIServerE(self.config)
        
        # VideoInput -> MemoryManager 注入线程；APIServerE 同步调用 MemoryManager 查询
        self.memory_manager.set_frame_queue(self.video_input.frame_queue)
        self.api_server.set_memory_manager(self.memory_manager)
        
        self.logger.info("query_while_inject 组件初始化完成")
    
    def _initialize_query_with_memory(self):
        """query_with_memory：MemoryManager, APIServerE；必须已有向量文件和 map 文件"""
        if not self._check_vector_and_map_files():
            raise FileNotFoundError("query_with_memory 模式需要已存在的向量文件和 databasemap 文件")

        self.memory_manager = MemoryManager(self.config)
        self.api_server = APIServerE(self.config)
        # 查询模式无需启动 MemoryManager 子线程，按需在 APIServerE 查询时同步检索
        self.memory_manager.init_sync()
        self.api_server.set_memory_manager(self.memory_manager)
        
        self.logger.info("query_with_memory 组件初始化完成")
    
    def _initialize_only_inject(self):
        """only_inject：VideoInput, MemoryManager；仅编码与建索引"""
        self.video_input = VideoInput(self.config)
        self.memory_manager = MemoryManager(self.config)
        
        self.memory_manager.set_frame_queue(self.video_input.frame_queue)
        
        self.logger.info("only_inject 组件初始化完成")
    
    def _initialize_benchmark(self):
        """benchmark：预留接口，暂不实现"""
        self.logger.warning("benchmark 模式尚未实现，仅预留接口")
        # 不创建任何组件
    
    def _run_query_loop(self, dialog_id: int = 1):
        """像 launch_client_onlyText 一样的循环提问。输入空行或 quit 退出。"""
        self.logger.info(f"对话 ID={dialog_id}，输入问题后回车；空行或 quit 退出")
        while True:
            try:
                query_text = input("你: ").strip()
            except EOFError:
                break
            if not query_text or query_text.lower() == "quit":
                break
            result = self.api_server.query(query_text=query_text, dialog_id=dialog_id)
            print(f"答: {result.get('result', '')}\n")
    
    def start(self):
        """根据 edge_mode 启动对应组件与流程"""
        if self.running:
            self.logger.warning("系统已在运行中")
            return
        
        self.logger.info("=" * 50)
        self.logger.info("启动边端系统")
        self.logger.info("=" * 50)
        
        self._initialize_components()
        
        if self.edge_mode == EDGE_MODE_BENCHMARK:
            self._start_benchmark()
            return
        
        try:
            if self.edge_mode == EDGE_MODE_QUERY_WHILE_INJECT:
                self._start_query_while_inject()
            elif self.edge_mode == EDGE_MODE_QUERY_WITH_MEMORY:
                self._start_query_with_memory()
            elif self.edge_mode == EDGE_MODE_ONLY_INJECT:
                self._start_only_inject()
        except KeyboardInterrupt:
            self.logger.debug("收到中断信号")
        except Exception as e:
            self.logger.error(f"运行出错: {e}", exc_info=True)
        finally:
            self.stop()
    
    def _start_query_while_inject(self):
        """启动一边编码一边询问流程"""
        self.running = True
        
        if self.memory_manager is not None:
            self.logger.debug("启动 MemoryManager...")
            self.memory_manager.start()
        if self.video_input is not None:
            self.logger.debug("启动 VideoInput...")
            self.video_input.start()
        
        if self.api_server is not None:
            self.api_server.start()
            self.logger.info("边端已就绪，可一边编码一边提问")
            self._run_query_loop(dialog_id=1)
    
    def _start_query_with_memory(self):
        """启动基于已有向量库的检索与询问流程"""
        self.running = True
        
        if self.api_server is not None:
            self.api_server.start()
            self.logger.info("边端已就绪，可对已有向量库进行提问")
            self._run_query_loop(dialog_id=1)
    
    def _start_only_inject(self):
        """仅对一个视频进行编码与索引构建，等待完成后退出"""
        self.running = True
        
        if self.memory_manager is not None:
            self.logger.debug("启动 MemoryManager...")
            self.memory_manager.start()
        if self.video_input is not None:
            self.logger.debug("启动 VideoInput...")
            self.video_input.start()
        
        self.logger.info("only_inject：正在对视频进行编码与建索引，等待完成...")
        if self.video_input is not None and hasattr(self.video_input, 'process') and self.video_input.process is not None:
            self.video_input.process.join()
        # 给流水线一点时间排空
        import time
        time.sleep(3.0)
        self.logger.info("视频编码与索引构建完成，运行中，按 Ctrl+C 退出")
        while True:
            time.sleep(1)
    
    def _start_benchmark(self):
        """benchmark 模式预留接口"""
        self.logger.info("benchmark 模式暂未实现")
    
    def stop(self):
        """停止边端系统及已启动的组件"""
        if not self.running and self.edge_mode != EDGE_MODE_BENCHMARK:
            return
        
        self.logger.info("正在关闭边端系统...")
        self.running = False
        
        if self.video_input is not None:
            try:
                self.video_input.stop()
                self.logger.debug("VideoInput 已停止")
            except Exception as e:
                self.logger.error(f"停止 VideoInput 时出错: {e}")
        
        if self.memory_manager is not None:
            try:
                self.memory_manager.stop()
                self.logger.debug("MemoryManager 已停止")
            except Exception as e:
                self.logger.error(f"停止 MemoryManager 时出错: {e}")
        
        if self.api_server is not None:
            try:
                self.api_server.stop()
                self.logger.debug("APIServerE 已停止")
            except Exception as e:
                self.logger.error(f"停止 APIServerE 时出错: {e}")
        
        self.logger.info("边端系统已关闭")


if __name__ == "__main__":
    # 直接运行时的测试代码
    system = VenusSystemEdge()
    system.start()
