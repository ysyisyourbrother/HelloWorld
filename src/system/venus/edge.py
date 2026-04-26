import signal
import sys
import os
import logging
import time
from typing import Optional

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from src.config import Config
from src.video_input.video_input import VideoInputOnline
from src.video_input.stream_input import StreamVideoInput
from src.memory.memory_manager import (
    MemoryManagerOnline,
    MemoryManagerOnlineV2,
    MemoryManagerOnlineV3,
)

# 支持的边端模式
EDGE_MODE_QUERY_WHILE_INJECT = "query_while_inject"
EDGE_MODE_QUERY_WITH_MEMORY = "query_with_memory"
EDGE_MODE_RETRIEVE_WHILE_INJECT = "retrieve_while_inject"
EDGE_MODE_RETRIEVE_WITH_MEMORY = "retrieve_with_memory"
EDGE_MODE_ONLY_INJECT = "only_inject"
EDGE_MODE_BENCHMARK = "benchmark"


class VenusSystemEdge:
    """边端系统 - 根据 edge_mode 整合视频编码、检索；query_* 经 APIServerE 与云端 gRPC，retrieve_* 仅本地检索"""
    
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
        
        # 组件（query_* 模式才实例化 APIServerE；retrieve_* 保持 None）
        self.api_server = None
        self.video_input: Optional[VideoInputOnline] = None
        self.memory_manager: Optional[MemoryManagerOnline] = None
        
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
            self.logger.error(
                f"向量文件不存在: {faiss_path}，query_with_memory / retrieve_with_memory 需要已有向量库"
            )
            return False
        if not os.path.isfile(map_path):
            self.logger.error(
                f"databasemap 文件不存在: {map_path}，query_with_memory / retrieve_with_memory 需要已有 map 文件"
            )
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
        elif self.edge_mode == EDGE_MODE_RETRIEVE_WHILE_INJECT:
            self._initialize_retrieve_while_inject()
        elif self.edge_mode == EDGE_MODE_RETRIEVE_WITH_MEMORY:
            self._initialize_retrieve_with_memory()
        elif self.edge_mode == EDGE_MODE_ONLY_INJECT:
            self._initialize_only_inject()
        elif self.edge_mode == EDGE_MODE_BENCHMARK:
            self._initialize_benchmark()
        else:
            raise ValueError(f"不支持的 edge_mode: {self.edge_mode}")
    
    def _use_stream_input(self):
        return getattr(self.config, "edge_use_stream_input", False)

    def _initialize_query_while_inject(self):
        """query_while_inject：VideoInput, MemoryManager（内含帧/查询编码）, APIServerE"""
        if self._use_stream_input():
            self.video_input = StreamVideoInput(self.config)
            if getattr(self.config, "memory_online_v3", True):
                self.memory_manager = MemoryManagerOnlineV3(self.config)
            else:
                self.memory_manager = MemoryManagerOnlineV2(self.config)
            self.memory_manager.set_frame_queue(self.video_input.frame_queue)
            self.memory_manager.set_stream_input(self.video_input)
        else:
            self.video_input = VideoInputOnline(self.config)
            self.memory_manager = MemoryManagerOnline(self.config)
            self.memory_manager.set_frame_queue(self.video_input.frame_queue)
        from src.api.client import QueryClient

        self.api_server = QueryClient(self.config)
        # VideoInput -> MemoryManager 注入线程；APIServerE 同步调用 MemoryManager 查询
        self.api_server.set_memory_manager(self.memory_manager)
        
        self.logger.info("query_while_inject 组件初始化完成")
    
    def _initialize_query_with_memory(self):
        """query_with_memory：MemoryManager, APIServerE；必须已有向量文件和 map 文件"""
        if not self._check_vector_and_map_files():
            raise FileNotFoundError("query_with_memory 模式需要已存在的向量文件和 databasemap 文件")

        self.memory_manager = MemoryManagerOnline(self.config)
        from src.api.client import QueryClient

        self.api_server = QueryClient(self.config)
        # 查询模式无需启动 MemoryManager 子线程，按需在 APIServerE 查询时同步检索
        self.memory_manager.init_sync()
        self.api_server.set_memory_manager(self.memory_manager)
        
        self.logger.info("query_with_memory 组件初始化完成")

    def _initialize_retrieve_while_inject(self):
        """retrieve_while_inject：与 query_while_inject 相同流水线，但不创建 APIServerE（无云端）。"""
        if self._use_stream_input():
            self.video_input = StreamVideoInput(self.config)
            if getattr(self.config, "memory_online_v3", True):
                self.memory_manager = MemoryManagerOnlineV3(self.config)
            else:
                self.memory_manager = MemoryManagerOnlineV2(self.config)
            self.memory_manager.set_frame_queue(self.video_input.frame_queue)
            self.memory_manager.set_stream_input(self.video_input)
        else:
            self.video_input = VideoInputOnline(self.config)
            self.memory_manager = MemoryManagerOnline(self.config)
            self.memory_manager.set_frame_queue(self.video_input.frame_queue)
        self.api_server = None
        self.logger.info("retrieve_while_inject 组件初始化完成（仅本地检索）")

    def _initialize_retrieve_with_memory(self):
        """retrieve_with_memory：与 query_with_memory 相同向量库，但不创建 APIServerE。"""
        if not self._check_vector_and_map_files():
            raise FileNotFoundError(
                "retrieve_with_memory 模式需要已存在的向量文件和 databasemap 文件"
            )
        self.memory_manager = MemoryManagerOnline(self.config)
        self.memory_manager.init_sync()
        self.api_server = None
        self.logger.info("retrieve_with_memory 组件初始化完成（仅本地检索）")

    def _initialize_only_inject(self):
        """only_inject：VideoInput, MemoryManager；仅编码与建索引"""
        if self._use_stream_input():
            self.video_input = StreamVideoInput(self.config)
            if getattr(self.config, "memory_online_v3", True):
                self.memory_manager = MemoryManagerOnlineV3(self.config)
            else:
                self.memory_manager = MemoryManagerOnlineV2(self.config)
            self.memory_manager.set_frame_queue(self.video_input.frame_queue)
            self.memory_manager.set_stream_input(self.video_input)
        else:
            self.video_input = VideoInputOnline(self.config)
            self.memory_manager = MemoryManagerOnline(self.config)
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

    def _run_retrieve_loop(self, dialog_id: int = 1):
        """控制台循环：仅调用 MemoryManager.query_text_sync，不经过 APIServerE / gRPC。"""
        self.logger.info(
            "本地检索模式（无云端 gRPC）；输入查询文本后回车，空行或 quit 退出"
        )
        if self.memory_manager is None:
            self.logger.error("memory_manager 未设置，无法检索")
            return
        query_id = 0
        while True:
            try:
                query_text = input("检索: ").strip()
            except EOFError:
                break
            if not query_text or query_text.lower() == "quit":
                break
            query_id += 1
            ts = time.time()
            qres = self.memory_manager.query_text_sync(
                query_text=query_text,
                query_id=query_id,
                dialog_id=dialog_id,
                timestamp=ts,
                trace_ts={"retrieve_enqueued_at": ts},
            )
            scores = qres.scores or []
            meta = qres.metadata_list or []
            rframes = getattr(qres, "retrieval_frames", None)
            print("检索命中 %d 条:" % len(meta))
            for rank, (s, m) in enumerate(zip(scores, meta), start=1):
                print(
                    "  #%d score=%.6f frame_id=%s fps=%s path=%s"
                    % (
                        rank,
                        float(s),
                        m.get("frame_id"),
                        m.get("video_fps"),
                        m.get("source_path"),
                    )
                )
            if rframes is not None:
                npx = sum(1 for f in rframes if f is not None)
                print("  像素帧（BGR）: %d / %d" % (npx, len(rframes)))
            print("")

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
            elif self.edge_mode == EDGE_MODE_RETRIEVE_WHILE_INJECT:
                self._start_retrieve_while_inject()
            elif self.edge_mode == EDGE_MODE_RETRIEVE_WITH_MEMORY:
                self._start_retrieve_with_memory()
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

    def _start_retrieve_while_inject(self):
        """一边注入一边本地检索（无 APIServerE）。"""
        self.running = True
        if self.memory_manager is not None:
            self.logger.debug("启动 MemoryManager...")
            self.memory_manager.start()
        if self.video_input is not None:
            self.logger.debug("启动 VideoInput...")
            self.video_input.start()
        self.logger.info("边端已就绪（仅本地检索），可一边编码一边输入检索文本")
        self._run_retrieve_loop(dialog_id=1)

    def _start_retrieve_with_memory(self):
        """基于已有向量库仅本地检索。"""
        self.running = True
        self.logger.info("边端已就绪（仅本地检索），可对已有向量库输入检索文本")
        self._run_retrieve_loop(dialog_id=1)

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
