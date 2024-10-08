"""
decoding的任务队列
1. last stage 给 first stage 发送 tree_candidates
    1. last stage: tree_candidates_send_queues
    2. first stage: tree_candidates_receive_queues
2. tree decoding 计算
    2.1 发送activation (除最后一个stage)
    2.2 接收activation (除第一个stage)
3. last stage broadcast  new_token ,其他接收  new_token  
"""
import inspect

import threading
import torch
import torch.distributed as dist

from . import threadsafe_queue,tag_manager
from .communication import recv_helper_thread,send_helper_thread
class CommunicationHandler():
    """ Handles communication between stages. """
    def __init__(self, config):
        self.rank = config.stage
        self.world_size = config.total_stage
        self.next_rank = config.next_rank
        self.pre_rank = config.pre_rank
        self.if_first_rank = config.is_first_stage
        self.if_last_rank = config.is_last_stage

        '''
        因为未知的new token数量,第一个元素 new_token_len,
        select_indices 为 [:,1:new_token_len+1],
        new_input_ids 为[:, new_token_len+1:2*new_token_len+1] 
        new token数量 一定小于等于 config.medusa_num_heads
        '''
        # 用于 reshape
        self.tensor_shape = {
                            "tree_decoding": (1,  64, config.hidden_size), 
                            "tree_candidates": (1,64),
                            "new_token":(1,1+2*config.medusa_num_heads) 
                            }
        # 用于recv 申请空间 
        self.tensor_shape_for_recv = {
                            "tree_decoding": (1, int (1 + 64*config.hidden_size)), 
                            "tree_candidates": (1, int (1+1*64)),
                            "new_token":(1, int (1+ 1+2*config.medusa_num_heads)  )
                            }
        self.tensor_type = {
                            "tree_decoding": config.torch_dtype,
                            "tree_candidates":torch.int64,
                            "new_token":torch.int64
                            } 
        self.tag_manager = tag_manager.Tag()
        self.tensor_tag = {
                            "tree_decoding":  self.tag_manager.get_next_tag(),  # 获取累加的tag，和其他communication_handler保持tag不一样
                            "tree_candidates":self.tag_manager.get_next_tag(),
                            "new_token": self.tag_manager.get_next_tag()
                            }
        
        self.device = config.device
        self.setup_queue()
        # Stop event to signal threads to stop
        self.stop_event = threading.Event()
        # List to keep track of helper threads
        self.helper_threads = []
        self.start_helper_threads()

    def setup_queue(self):
        """
        Setup queues for communication between main compute thread
        and helper communication threads. One queue per tensor
        in forward / backward direction.
        """
        # for tree candidates
        if self.if_first_rank:
            self.tree_candidates_receive_queues = threadsafe_queue.Queue()
        if self.if_last_rank:
            self.tree_candidates_send_queues = threadsafe_queue.Queue()
        # for tree decoding
        if not self.if_first_rank:
            self.tree_decoding_receive_queues = threadsafe_queue.Queue()
        if not self.if_last_rank:
            self.tree_decoding_send_queues = threadsafe_queue.Queue()
        # for new token
        if self.if_last_rank:
            self.new_token_send_queues = threadsafe_queue.Queue()
        else:
            self.new_token_receive_queues = threadsafe_queue.Queue()
    def start_helper_threads(self):
        # for tree candidates
        if self.if_first_rank:
            self.start_helper_thread(func=recv_helper_thread, 
                                    args=(self.tree_candidates_receive_queues, 
                                        self.tensor_shape_for_recv["tree_candidates"], 
                                        self.world_size-1,  # from last stage
                                        self.tensor_tag["tree_candidates"],
                                        self.tensor_type["tree_candidates"],
                                        self.stop_event))
        if self.if_last_rank:
            self.start_helper_thread(func=send_helper_thread, 
                                    args=(self.tree_candidates_send_queues, 
                                    0, # to first stage
                                    self.tensor_tag["tree_candidates"],
                                    self.stop_event))
        # for tree decoding
        if not self.if_first_rank:
            self.start_helper_thread(func=recv_helper_thread, 
                                    args=(self.tree_decoding_receive_queues, 
                                        self.tensor_shape_for_recv["tree_decoding"], 
                                        self.pre_rank, 
                                        self.tensor_tag["tree_decoding"],
                                        self.tensor_type["tree_decoding"],
                                        self.stop_event))
        if not self.if_last_rank:
            self.start_helper_thread(func=send_helper_thread, 
                                    args=(self.tree_decoding_send_queues, 
                                    self.next_rank,
                                    self.tensor_tag["tree_decoding"],
                                    self.stop_event))

        # for new token
        if self.if_last_rank:
            self.start_helper_thread(func=broadcast_send_helper_thread, 
                                    args=(self.new_token_send_queues, 
                                    self.world_size-1, #  src=self.world_size-1
                                    self.stop_event))
        else:
            self.start_helper_thread(
                                    func=broadcast_recv_helper_thread,
                                    args=(self.new_token_receive_queues, 
                                    self.tensor_shape_for_recv["new_token"],
                                    self.world_size-1, #  src=self.world_size-1
                                    self.tensor_type["new_token"],
                                    self.stop_event))
                
    def start_helper_thread(self, func, args):
        helper_thread = threading.Thread(target=func, args=args,daemon=True)
        helper_thread.start()
        self.helper_threads.append(helper_thread)  # Track the thread
        
    def stop_helper_threads(self):
        # Signal all helper threads to stop
        # self.stop_event.set()
        pass
        
    def flatten_before_send(self,tensor, point_id):
        flattened_tensor = tensor.reshape(1,-1)
        point_id_tensor = torch.tensor([point_id], dtype=flattened_tensor.dtype, device=flattened_tensor.device)
        point_id_tensor = point_id_tensor.reshape(1,-1)
        result_tensor = torch.cat((point_id_tensor, flattened_tensor),dim=1)
        return result_tensor
    def reshape_after_recv(self,tensor,tag):
        if tag == self.tensor_tag["tree_decoding"]:
            shape = self.tensor_shape["tree_decoding"]
        elif tag == self.tensor_tag["tree_candidates"]:
            shape = self.tensor_shape["tree_candidates"]
        elif tag == self.tensor_tag["new_token"]:
            shape =  self.tensor_shape["new_token"]
        point_id = int(tensor[0][0].item())
        reshaped_tensor = tensor[:, 1:].reshape(shape)
        return reshaped_tensor,point_id
    def send(self, tensor, tag, point_id): 
        tensor = self.flatten_before_send(tensor, point_id)
        if tag == self.tensor_tag["tree_decoding"]:
            self.tree_decoding_send_queues.add(tensor)
        elif tag == self.tensor_tag["tree_candidates"]:
            self.tree_candidates_send_queues.add(tensor)
        elif tag == self.tensor_tag["new_token"]:
            self.new_token_send_queues.add(tensor)
        else:
            raise NotImplementedError
    def recv(self, tag):
        if tag == self.tensor_tag["tree_decoding"]:
            tensor =  self.tree_decoding_receive_queues.remove()
            # tensor = tensor.requires_grad_()
        elif tag == self.tensor_tag["tree_candidates"]:
            tensor =  self.tree_candidates_receive_queues.remove()
        elif tag == self.tensor_tag["new_token"]:
            tensor =  self.new_token_receive_queues.remove()
        else:
            raise NotImplementedError
        if self.device == "cuda":
            tensor = tensor.cuda()
        tensor, point_id = self.reshape_after_recv(tensor,tag)
        return tensor, point_id
    

def broadcast_send_helper_thread(send_queue, src,stop_event):
    """
    负责广播发送张量的线程
    Arguments:
        - send_queue: 等待被发送的张量队列
        - src_rank: 发送张量的rank
        - stop_event: 停止事件
    """
    while not stop_event.is_set():  # Check if stop signal is set
        # 当send_queue为空时，队列阻塞
        tensor = send_queue.remove()
        _broadcast_send(tensor, src, )
        
def broadcast_recv_helper_thread(recv_queue, tensor_shape, src_rank, dtype, stop_event):
    """负责接收张量的线程
    Arguments:
        - recv_queue:  接收队列
        - tensor_shape: 张量的形状
        - src_rank: 发送张量的rank
        - dtype: 发送张量的dtype
        - stop_event: 停止事件
    """
    while not stop_event.is_set():  # Check if stop signal is set
        tensor = _broadcast_recv(tensor_shape, src_rank,dtype)
        recv_queue.add(tensor)

def _broadcast_send(tensor, src_rank ):
    if tensor.device != torch.device("cpu"): # for gloo 
        tensor = tensor.cpu()
    dist.broadcast(tensor=tensor , src=src_rank)


def _broadcast_recv(tensor_shape, src_rank,dtype):
    tensor = torch.zeros(tensor_shape, dtype=dtype) 
    dist.broadcast(tensor, src=src_rank)
    return tensor