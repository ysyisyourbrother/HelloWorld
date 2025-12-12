from config import Config

class APIServerE:
    def __init__(self, config=None):
        """
        初始化APIServerE模块
        边端API服务器, 提供API接口接收用户请求
        
        Args:
            config (Config): 配置对象实例
        """
        if config is None:
            config = Config()
        
        self.config = config
        