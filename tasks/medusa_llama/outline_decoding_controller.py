from jupiter.core.threadsafe_queue import Queue
from .kv_cache import initialize_past_key_values
class OutlineDecodingController:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(OutlineDecodingController, cls).__new__(cls)
        return cls._instance
    def __init__(self, point_num, config, model):
        # 确保初始化只发生一次
        if not hasattr(self, 'initialized'):
            self.point_num = point_num
            self.config = config
            self.model = model
            
            self.medusa_logits_for_point = []
            self.logits_for_point = []
            self.past_key_values_for_point = []
            self.past_key_values_data_for_point = []
            self.current_length_data_for_point = []
            self.initialized = True  # 标记已初始化
    def add_request(self,medusa_logits, logits, point_id):
        assert self.config.is_last_stage
        self.medusa_logits_for_point[point_id].add(medusa_logits)
        self.logits_for_point[point_id].add(logits)
    def remove_request(self,point_id ):
        assert self.config.is_last_stage
        if  self.medusa_logits_for_point[point_id].len() == 0:
            return None 
        else:
            medusa_logits = self.medusa_logits_for_point[point_id].remove()
            logits = self.logits_for_point[point_id].remove()
            return {
                        "point_id": point_id,
                        "medusa_logits": medusa_logits,
                        "logits": logits
                }
    def set_request_queue(self):
        if self.config.is_last_stage:
            self.medusa_logits_for_point=[]
            self.logits_for_point=[]
            for i in range(self.point_num):
                medusa_logits_queue = Queue()
                self.medusa_logits_for_point.append(medusa_logits_queue)
                logits_queue = Queue()
                self.logits_for_point.append(logits_queue)
    def prepare_point_kv_cache(self):
        print("prepare point kv cache")
        for i in range(self.point_num):
            past_key_values, past_key_values_data, current_length_data = initialize_past_key_values(self.model)
            self.past_key_values_for_point.append(past_key_values)
            self.past_key_values_data_for_point.append(past_key_values_data)
            self.current_length_data_for_point.append(current_length_data)
            #TODO: kv cache占用内存是原本的 point_num+1倍，在initialize_past_key_values 修改max_length 修改预分配空间
    def check(self):
        for past_key_values in self.past_key_values_for_point:
            print(past_key_values[0][0].shape[2])
    def get_point_past_key_values(self,point_id ):
        return self.past_key_values_for_point[point_id]
    def get_point_current_length_data(self,point_id):
        return self.current_length_data_for_point[point_id]
controller = None