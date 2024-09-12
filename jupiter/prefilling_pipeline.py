import torch

from jupiter.core.schedules import PipelineRuntime

class PrefillingPipeline(PipelineRuntime):
    def __init__(self, stage_model, config, args):
        super().__init__(stage_model, config, args)
    
    def pipeline_forward(self, input_ids = None): # 完整句子 pipeline
        """ Forward pass of prefilling stage.
        """
        # bs == 1, seq_len == 47
        if self.config.is_first_stage:
            bs,seq_len = input_ids.shape
            assert bs == 1

        if self.config.is_first_stage:  # 第一个stage
            if not self.config.is_last_stage: # world > 1
                hidden_states = self.stage_model.prefilling(input_ids=input_ids, inputs_embeds=None )
                self.send_activation_forward(self.padding_before_send(hidden_states))
            else: # world == 1
                raise NotImplementedError("暂不支持单机推理")
        else:
            hidden_states = self.receive_activation_forward()
            hidden_states = self.cliping_after_recv(hidden_states)
            print("hidden_states", hidden_states.shape)
            if not self.config.is_last_stage:   # 不是第一个也不是最后一个stage
                hidden_states = self.stage_model.prefilling(input_ids=None, inputs_embeds=hidden_states )
                self.send_activation_forward(self.padding_before_send(hidden_states))
            else: # 最后一个stage
                medusa_logits, logits = self.stage_model.prefilling(input_ids=None, inputs_embeds=hidden_states )
                print("finish prefilling")
                return medusa_logits, logits


    def split_tensor_along_dim(self,tensor, num_splits, dim=1):
        shape = list(tensor.size())
        assert dim < len(shape), "Dimension out of range for the tensor"
        split_size = shape[dim] // num_splits
        remainder = shape[dim] % num_splits   
        assert split_size +1 <= self.config.max_sub_sequence_len # +1是一个要拼接sub_seq length的信息
        assert remainder +1 <= self.config.max_sub_sequence_len
        # 划分张量
        splits = []
        start = 0
        for i in range(num_splits):
            length = split_size + 1 if i < remainder else split_size
            splits.append(tensor.narrow(dim, start, length))
            start += length
        
        return splits
    
    
    def padding_before_send(self,tensor):
        # 原本的tensor为 [1,sub_len,hidden_size], padding为 [1,sub_len+1,hidden_size]
        # 其中tensor[0][0][0] = sub_len
        shape = tensor.size()
        sub_len = shape[1]
        zero_part = torch.zeros(shape[0], 1, shape[2], dtype=tensor.dtype, device=tensor.device)
        modified_tensor = torch.cat((zero_part, tensor), dim=1)
        sub_len = torch.tensor(sub_len).to(modified_tensor.dtype)

        modified_tensor[0][0][0] =  sub_len
        return modified_tensor


    def cliping_after_recv(self,tensor):
        sub_len = int(tensor[0, 0, 0]) 
        print("sub_len", sub_len)
        return tensor[:,1:sub_len+1,:]


    def pipeline_sub_forward(self, sub_input_ids = None):
        # 对llamamodel forward (不包括head)
        if self.config.is_first_stage:
            if not self.config.is_last_stage:
                hidden_states = self.stage_model.forward_sub_sequences(input_ids=sub_input_ids, inputs_embeds=None )
                self.send_activation_forward(self.padding_before_send(hidden_states))
            else:
                raise NotImplementedError("暂不支持单机推理")
        else:
            hidden_states = self.receive_activation_forward()
            hidden_states = self.cliping_after_recv(hidden_states)
            if not self.config.is_last_stage:   # 不是第一个也不是最后一个stage
                hidden_states = self.stage_model.forward_sub_sequences(input_ids=None, inputs_embeds=hidden_states )
                self.send_activation_forward(self.padding_before_send(hidden_states))
            else:#最后一个stage
                hidden_states = self.stage_model.forward_sub_sequences(input_ids=None, inputs_embeds=hidden_states )
                print("hidden_states", hidden_states.shape)
                return hidden_states


    def pipeline_with_sequence_slicing(self ,input_ids = None):
        if self.config.is_first_stage:
            bs,_ = input_ids.shape
            assert bs == 1
        if self.config.is_last_stage:
            sub_hidden_states = []
        # Step 0 : init prefilling (init kv cache)
        self.stage_model.prefilling_init()
        # Step1: 划分原本的sequence为多个sub-sequence
        if self.config.is_first_stage:
            assert input_ids is not None
            sub_sequences = self.split_tensor_along_dim(input_ids, self.config.num_sub_sequences, dim=1)
        for i in range (self.config.num_sub_sequences):
            if self.config.is_first_stage:
                sub_input_ids = sub_sequences[i]
                self.pipeline_sub_forward(sub_input_ids)
            else:
                # 最后一个stage 得到hidden_states
                if self.config.is_last_stage:
                    hidden_states = self.pipeline_sub_forward( None)
                    sub_hidden_states.append(hidden_states)
                else:
                    self.pipeline_sub_forward( None)
        if self.config.is_last_stage:
            # hidden_states = torch.cat(sub_hidden_states, dim=1)
            print("hidden_states", hidden_states.shape)
            # Step2: 得到medusa_logits和logits
            medusa_logits,logits = self.stage_model.prefilling_finish(hidden_states) 
            print("finish prefilling")
            return medusa_logits, logits
        else:
            self.stage_model.prefilling_finish( )
            print("finish prefilling")
            
            
            