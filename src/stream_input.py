import cv2
import numpy as np
import multiprocessing as mp
import time
from config import Config

class StreamInput:
    def __init__(self, config=None):
        """
        初始化StreamInput模块
        负责从视频流中提取帧，支持摄像头实时视频流或本地视频文件
        
        Args:
            config (Config): 配置对象实例
        """
        if config is None:
            config = Config()
        
        self.config = config
        
        # 从Config对象获取配置
        self.fps = config.stream_fps
        self.video_source = config.stream_video_source
        self.video_file_path = config.stream_video_path
        
        self.frame_queue = mp.Queue(maxsize=config.stream_queue_size)
        self.running = False
        
    def initialize_video_source(self):
        """初始化视频源"""
        if self.video_source == "camera":
            self.cap = cv2.VideoCapture(0)
        else:
            self.cap = cv2.VideoCapture(self.video_file_path)
        
        if not self.cap.isOpened():
            raise Exception(f"无法打开视频源: {self.video_source}")
    
    def extract_frames(self):
        """提取帧的主循环"""
        frame_interval = 1.0 / self.fps
        last_time = time.time()
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                if self.video_source != "camera":
                    break
                continue
            
            current_time = time.time()
            if current_time - last_time >= frame_interval:
                timestamp = current_time
                frame_data = {
                    'frame': frame,
                    'timestamp': timestamp,
                    'frame_id': int(timestamp * 1000)  # 使用毫秒作为帧ID
                }
                
                if not self.frame_queue.full():
                    self.frame_queue.put(frame_data)
                    last_time = current_time
            
            time.sleep(0.001)  # 避免CPU占用过高
    
    def start(self):
        """启动帧提取进程"""
        self.running = True
        self.initialize_video_source()
        
        self.process = mp.Process(target=self.extract_frames)
        self.process.start()
        print(f"StreamInput进程已启动，FPS: {self.fps}")
    
    def stop(self):
        """停止帧提取进程"""
        self.running = False
        if hasattr(self, 'process'):
            self.process.join(timeout=5)
        if hasattr(self, 'cap'):
            self.cap.release()
        print("StreamInput进程已停止")
    
    def get_frame_queue(self):
        """获取帧队列供其他模块使用"""
        return self.frame_queue

if __name__ == "__main__":
    # 测试代码
    stream_input = StreamInput()
    try:
        stream_input.start()
        time.sleep(10)  # 运行10秒进行测试
    except KeyboardInterrupt:
        pass
    finally:
        stream_input.stop()