"""
每个设备都需要一直从前后两个设备接收数据，并加入到forward和backward
两条任务队列中
"""

import threading
import torch
import torch.distributed as dist

from . import threadsafe_queue

class CommunicationHandler():
    """ Handles communication between stages. """
    def __init__(self, config):
        self.rank = config.stage
        self.world_size = config.total_stage
        self.next_rank = config.next_rank
        self.pre_rank = config.pre_rank
        self.if_first_rank = config.is_first_stage
        self.if_last_rank = config.is_last_stage
        self.tensor_tag = {"forward": 0}
        # TODO: 解决sequence length可变问题
        # 在执行下面_recv方法的时候，先recv一个shape的长度，然后再创建接收的tensor = torch.zeros(shape)
        self.tensor_shape = {"forward": (1, 47, config.hidden_size), 
                             }  
        self.setup_queue()
        self.start_helper_threads()

    def setup_queue(self):
        """
        Setup queues for communication between main compute thread
        and helper communication threads. One queue per tensor
        in forward / backward direction.
        """
        self.forward_receive_queues = threadsafe_queue.Queue()
        self.forward_send_queues = threadsafe_queue.Queue()


    def start_helper_threads(self):
         
        if not self.if_first_rank:
            # 启动 recv forward helper thread 
            self.start_helper_thread(func=recv_helper_thread, 
                                    args=(self.forward_receive_queues, 
                                        self.tensor_shape["forward"], 
                                        self.pre_rank, 
                                        self.tensor_tag["forward"]))
        if not self.if_last_rank:
            # 启动 send forward helper thread
            self.start_helper_thread(func=send_helper_thread, 
                                    args=(self.forward_send_queues, 
                                    self.next_rank,
                                    self.tensor_tag["forward"]))


    def start_helper_thread(self, func, args):
        helper_thread = threading.Thread(target=func, args=args,daemon=True)
        helper_thread.start()


    def send(self, tensor):
        self.forward_send_queues.add(tensor)

    
    def recv(self):
        tensor = self.forward_receive_queues.remove()
        tensor = tensor.requires_grad_()
        return tensor


def recv_helper_thread(recv_queue, tensor_shape, src_rank, tag):
    """负责接收张量的线程
    Arguments:
        - send_queue: 等待被发送的张量队列
        - num_iterations: 需要发送多少次张量
    """
    while True:
        tensor = _recv(tensor_shape, src_rank, tag)
        print(f"recv tensor from {src_rank}.")
        recv_queue.add(tensor)


def send_helper_thread(send_queue, dst_rank, tag):
    """负责发送张量的线程
    Arguments:
        - send_queue: 等待被发送的张量队列
        - num_iterations: 需要发送多少次张量
    """
    while True:
        # 当send_queue为空时，队列阻塞
        tensor = send_queue.remove()
        print(f"send tensor to {dst_rank}.")
        _send(tensor, dst_rank, tag)

def _send(tensor, dst_rank, tag):
    tensor = tensor.cpu()
    dist.send(tensor=tensor, dst=dst_rank, tag=tag)


def _recv(tensor_shape, src_rank, tag):
    tensor = torch.zeros(tensor_shape, dtype=torch.float16)
    dist.recv(tensor, src=src_rank, tag=tag)
    return tensor