"""
每个设备都需要一直从前后两个设备接收数据，并加入到forward和backward
两条任务队列中
"""

import threading
import torch
import torch.distributed as dist
from . import threadsafe_queue,tag_manager

class CommunicationHandler():
    """ Handles communication between stages. """
    def __init__(self, config):
        self.rank = config.stage
        self.world_size = config.total_stage
        self.next_rank = config.next_rank
        self.pre_rank = config.pre_rank
        self.if_first_rank = config.is_first_stage
        self.if_last_rank = config.is_last_stage
        self.tag_manager = tag_manager.Tag()
        self.tensor_tag = {"forward": self.tag_manager.get_next_tag()}
        self.torch_dtype = config.torch_dtype
        # 针对sequence length可变问题,这里设置的是max_sub_sequence_len
        self.tensor_shape = {"forward": (1, config.max_sub_sequence_len, config.hidden_size), }  
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
        if not self.if_first_rank:
            self.forward_receive_queues = threadsafe_queue.Queue()
        if not self.if_last_rank:
            self.forward_send_queues = threadsafe_queue.Queue()


    def start_helper_threads(self):
        if not self.if_first_rank:
            # 启动 recv forward helper thread 
            self.start_helper_thread(func=recv_helper_thread, 
                                    args=(self.forward_receive_queues, 
                                        self.tensor_shape["forward"], 
                                        self.pre_rank, 
                                        self.tensor_tag["forward"],
                                        self.torch_dtype,
                                        self.stop_event))  # Pass stop_event
        if not self.if_last_rank:
            # 启动 send forward helper thread
            self.start_helper_thread(func=send_helper_thread, 
                                    args=(self.forward_send_queues, 
                                    self.next_rank,
                                    self.tensor_tag["forward"],
                                    self.stop_event))  # Pass stop_event


    def start_helper_thread(self, func, args):
        helper_thread = threading.Thread(target=func, args=args,daemon=True)
        helper_thread.start()
        self.helper_threads.append(helper_thread)  # Track the thread
    def stop_helper_threads(self):
            # Signal all helper threads to stop
            self.stop_event.set()
    def send(self, tensor):
        self.forward_send_queues.add(tensor)

    def recv(self):
        tensor = self.forward_receive_queues.remove()
        tensor = tensor.requires_grad_()
        if self.device == "cuda":
            tensor = tensor.cuda()
        return tensor


def recv_helper_thread(recv_queue, tensor_shape, src_rank, tag,dtype,stop_event):
    """负责接收张量的线程
    Arguments:
        - send_queue: 等待被发送的张量队列
        - num_iterations: 需要发送多少次张量
    """
    while not stop_event.is_set():  # Check if stop signal is set
        tensor = _recv(tensor_shape, src_rank, tag,dtype)
        # print(f"recv tensor from {src_rank}.")
        recv_queue.add(tensor)


def send_helper_thread(send_queue, dst_rank, tag,stop_event):
    """负责发送张量的线程
    Arguments:
        - send_queue: 等待被发送的张量队列
        - num_iterations: 需要发送多少次张量
    """
    while not stop_event.is_set():  # Check if stop signal is set
        # 当send_queue为空时，队列阻塞
        tensor = send_queue.remove()
        # print(f"send tensor to {dst_rank}.")
        _send(tensor, dst_rank, tag,)
#TODO: define backend :gloo,  only support cpu
def _send(tensor, dst_rank, tag ):
    if tensor.device != torch.device("cpu"):
        tensor = tensor.cpu()
    dist.send(tensor=tensor, dst=dst_rank, tag=tag)
    # print("_send to rank {}, tag {}".format(dst_rank, tag))



def _recv(tensor_shape, src_rank, tag,dtype):
    tensor = torch.zeros(tensor_shape, dtype=dtype) #TODO: 由config决定
    # print("try _recv from rank {}, tag {}".format(src_rank, tag))
    dist.recv(tensor, src=src_rank, tag=tag)
    return tensor