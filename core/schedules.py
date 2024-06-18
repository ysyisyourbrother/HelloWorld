import torch
from core.communication import CommunicationHandler


class PipelineRuntime():
    def __init__(self, stage_model, config, args):
        self.config = config
        self.args = args
        self.stage = config.stage
        self.next_rank = config.next_rank
        self.pre_rank = config.pre_rank
        self.total_stage = config.total_stage
        self.stage_model = stage_model
        self.comm_handler = CommunicationHandler(config)

    def send_activation_forward(self, tensor):
        """Forward pass of the activations.
        """
        # 最后一个stage直接返回
        if self.stage == self.total_stage-1:
            return 
        self.comm_handler.send(tensor)
    
    def receive_activation_forward(self, input_sample = None):
        if self.stage == 0: # 从input_sample获取输入
            if input_sample is not None:
                tensor = input_sample.cuda()
                return tensor
            else:
                raise Exception("Missing input.")
        else: # 从上一台机器接收tensor
            tensor = self.comm_handler.recv()
            tensor = tensor.cuda()

        return tensor
        

class PrefillingPipeline(PipelineRuntime):
    def __init__(self, stage_model, config, args):
        super().__init__(stage_model, config, args)
    
    def pipeline_forward(self, input_ids = None):
        """ Forward pass of prefilling stage.
        """
        # bs == 1, seq_len == 47
        bs,seq_len = input_ids.shape
        assert bs == 1

        if self.config.is_first_stage:  # 第一个stage
            if not self.config.is_last_stage: # world > 1
                hidden_states = self.stage_model.prefilling(input_ids=input_ids, inputs_embeds=None, temperature=self.config.temperature)
                self.send_activation_forward(hidden_states)
            else: # world == 1
                raise NotImplementedError("暂不支持单机推理")
        else:
            hidden_states = self.receive_activation_forward()
            hidden_states = hidden_states.cuda()
            if not self.config.is_last_stage:   # 不是第一个也不是最后一个stage
                hidden_states = self.stage_model.prefilling(input_ids=None, inputs_embeds=hidden_states, temperature=self.config.temperature)
                self.send_activation_forward(hidden_states)
            else: # 最后一个stage
                medusa_logits, logits = self.stage_model.prefilling(input_ids=None, inputs_embeds=hidden_states, temperature=self.config.temperature)
                print("finish prefilling")
                return medusa_logits, logits



        
