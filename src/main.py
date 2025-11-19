import multiprocessing as mp
import time
import signal
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from stream_input import StreamInput
from frame_vectorizer import FrameVectorizer
from query_vectorizer import QueryVectorizer
from memory_manager import MemoryManager
from api_server_e import APIServerE
from reasoner import Reasoner
from api_server_c import APIServerC

class VideoAnalysisSystem:
    def __init__(self, config_path="configs/config.json"):
        """
        初始化视频分析系统
        协调所有模块的启动和停止
        """
        self.config_path = config_path
        self.modules = {}
        self.running = False
        
        # 设置信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def initialize_client_modules(self):
        """初始化客户端模块"""
        print("初始化客户端模块...")
        
        # 创建客户端模块实例
        stream_input = StreamInput(self.config_path)
        frame_vectorizer = FrameVectorizer(self.config_path)
        query_vectorizer = QueryVectorizer(self.config_path)
        memory_manager = MemoryManager(self.config_path)
        api_server_e = APIServerE(self.config_path)
        
        # 设置模块间的连接
        frame_vectorizer.set_frame_queue(stream_input.get_frame_queue())
        memory_manager.set_vector_queue(frame_vectorizer.get_vector_queue())
        memory_manager.set_query_queue(query_vectorizer.get_vector_queue())
        
        # 设置APIServerE的依赖
        api_server_e.set_query_vectorizer(query_vectorizer)
        api_server_e.set_memory_manager(memory_manager)
        
        self.modules.update({
            'stream_input': stream_input,
            'frame_vectorizer': frame_vectorizer,
            'query_vectorizer': query_vectorizer,
            'memory_manager': memory_manager,
            'api_server_e': api_server_e
        })
        
        print("客户端模块初始化完成")
    
    def initialize_server_modules(self):
        """初始化服务器端模块"""
        print("初始化服务器端模块...")
        
        # 创建服务器端模块实例
        reasoner = Reasoner(self.config_path)
        api_server_c = APIServerC(self.config_path)
        
        # 设置模块间的连接
        api_server_c.set_reasoner(reasoner)
        
        self.modules.update({
            'reasoner': reasoner,
            'api_server_c': api_server_c
        })
        
        print("服务器端模块初始化完成")
    
    def start_client(self):
        """启动客户端模块"""
        print("启动客户端模块...")
        
        # 按顺序启动客户端模块
        start_order = [
            'memory_manager',    # 先启动记忆管理器
            'query_vectorizer',  # 再启动查询向量化器
            'frame_vectorizer',  # 然后启动帧向量化器
            'stream_input',      # 接着启动流输入
            'api_server_e'       # 最后启动API服务器
        ]
        
        for module_name in start_order:
            if module_name in self.modules:
                self.modules[module_name].start()
                time.sleep(0.5)  # 给每个模块一些启动时间
        
        print("客户端模块启动完成")
    
    def start_server(self):
        """启动服务器端模块"""
        print("启动服务器端模块...")
        
        # 按顺序启动服务器端模块
        start_order = [
            'reasoner',      # 先启动推理器
            'api_server_c'   # 然后启动API服务器
        ]
        
        for module_name in start_order:
            if module_name in self.modules:
                self.modules[module_name].start()
                time.sleep(0.5)  # 给每个模块一些启动时间
        
        print("服务器端模块启动完成")
    
    def start(self, mode='both'):
        """启动系统"""
        print(f"启动视频分析系统，模式: {mode}")
        
        self.running = True
        
        if mode in ['client', 'both']:
            self.initialize_client_modules()
            self.start_client()
        
        if mode in ['server', 'both']:
            self.initialize_server_modules()
            self.start_server()
        
        print("系统启动完成")
        
        if mode == 'both':
            print("客户端和服务器端都已启动")
            print(f"客户端API: http://localhost:8000")
            print(f"服务器端API: http://localhost:9000")
        elif mode == 'client':
            print("客户端已启动")
            print(f"客户端API: http://localhost:8000")
        elif mode == 'server':
            print("服务器端已启动")
            print(f"服务器端API: http://localhost:9000")
    
    def stop(self):
        """停止系统"""
        print("正在停止视频分析系统...")
        
        self.running = False
        
        # 按相反顺序停止模块
        stop_order = [
            'api_server_e', 'api_server_c',  # 先停止API服务器
            'stream_input', 'reasoner',       # 然后停止输入和推理器
            'frame_vectorizer', 'query_vectorizer',  # 接着停止向量化器
            'memory_manager'                 # 最后停止记忆管理器
        ]
        
        for module_name in stop_order:
            if module_name in self.modules:
                try:
                    self.modules[module_name].stop()
                except Exception as e:
                    print(f"停止模块 {module_name} 时出错: {e}")
        
        print("视频分析系统已停止")
    
    def _signal_handler(self, signum, frame):
        """信号处理器"""
        print(f"\n收到信号 {signum}，正在停止系统...")
        self.stop()
        sys.exit(0)
    
    def run(self, mode='both'):
        """运行系统"""
        try:
            self.start(mode)
            
            # 保持运行
            while self.running:
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\n收到键盘中断信号")
        except Exception as e:
            print(f"运行时出错: {e}")
        finally:
            self.stop()

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='视频分析系统')
    parser.add_argument('--mode', choices=['client', 'server', 'both'], 
                       default='both', help='运行模式')
    parser.add_argument('--config', default='configs/config.json', 
                       help='配置文件路径')
    
    args = parser.parse_args()
    
    # 检查配置文件是否存在
    if not os.path.exists(args.config):
        print(f"配置文件不存在: {args.config}")
        return
    
    # 创建并运行系统
    system = VideoAnalysisSystem(args.config)
    system.run(args.mode)

if __name__ == "__main__":
    main()