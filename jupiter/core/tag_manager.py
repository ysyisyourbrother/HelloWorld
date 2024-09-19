 

class Tag:
    _instance = None  # 用于存储单例实例

    def __new__(cls):
        if cls._instance is None:
            # 创建单例实例
            cls._instance = super(Tag, cls).__new__(cls)
            cls._instance.tag_id = 0  # 初始化 tag_id
        return cls._instance

    def get_next_tag(self):
        # 返回当前的 tag_id，然后递增
        current_tag = self.tag_id
        self.tag_id += 1
        return current_tag
