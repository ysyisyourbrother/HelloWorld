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
            bs,_ = input_ids.shape
            assert bs == 1

        if self.config.is_first_stage:  # 第一个stage
            if not self.config.is_last_stage: # world > 1
                hidden_states = self.stage_model.prefilling(input_ids=input_ids, inputs_embeds=None )
                # self.send_activation_forward(self.padding_before_send(hidden_states))
                seq_len = torch.tensor(hidden_states.shape[1])
                print("seq_len", seq_len)
                self.send_seq_len(seq_len)
                self.send_activation_forward(hidden_states)
            else: # world == 1
                raise NotImplementedError("暂不支持单机推理")
        else:
            seq_len = self.receive_seq_len().item()
            print("seq_len", seq_len)
            hidden_states = self.receive_activation_forward()
            hidden_states = hidden_states[:,1:seq_len+1,:]
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
    
    


    def pipeline_with_sequence_slicing(self ,input_ids = None):
        if self.config.is_first_stage:
            bs,_ = input_ids.shape
            assert bs == 1
        # Step 0 : init prefilling (init kv cache)
        self.stage_model.prefilling_init()
        # Step 1: 划分原本的sequence为多个sub-sequence
        if self.config.is_first_stage:
            assert input_ids is not None
            sub_sequences = self.split_tensor_along_dim(input_ids, self.config.num_sub_sequences, dim=1)
        for i in range (self.config.num_sub_sequences):
            if self.config.is_first_stage:
                if self.config.is_last_stage:
                    raise NotImplementedError("暂不支持单机推理")
                sub_input_ids = sub_sequences[i]
                seq_len = torch.tensor( int (sub_input_ids.shape[1])).reshape(1,1)
                print("seq_len", seq_len)
                self.send_seq_len(seq_len)
                hidden_states = self.stage_model.forward_sub_sequences(input_ids=sub_input_ids, inputs_embeds=None )
                self.send_activation_forward(hidden_states)
            else:
                seq_len =  int (self.receive_seq_len().item())
                print("seq_len", seq_len)
                hidden_states = self.receive_activation_forward()
                hidden_states = hidden_states[:,:seq_len,:]
                hidden_states = self.stage_model.forward_sub_sequences(input_ids=None, inputs_embeds=hidden_states )
                if not self.config.is_last_stage:   # 不是第一个也不是最后一个stage
                    seq_len = torch.tensor( int (hidden_states.shape[1])).reshape(1,1)
                    self.send_seq_len(seq_len)
                    self.send_activation_forward(hidden_states)
        
        if self.config.is_last_stage:
            # hidden_states = torch.cat(sub_hidden_states, dim=1)
            print("hidden_states", hidden_states.shape)
            # Step 2: 得到medusa_logits和logits
            medusa_logits,logits = self.stage_model.prefilling_finish(hidden_states) 
            print("finish prefilling")
            return medusa_logits, logits
        else:
            self.stage_model.prefilling_finish( )
            print("finish prefilling")
    
    def pipeline_with_sequence_slicing_no_finish(self ,input_ids = None):
        if self.config.is_first_stage:
            bs,_ = input_ids.shape
            assert bs == 1
        # Step 0 : init prefilling (init kv cache)
        self.stage_model.prefilling_init()
        # Step 1: 划分原本的sequence为多个sub-sequence
        if self.config.is_first_stage:
            assert input_ids is not None
            sub_sequences = self.split_tensor_along_dim(input_ids, self.config.num_sub_sequences, dim=1)
        for i in range (self.config.num_sub_sequences):
            if self.config.is_first_stage:
                sub_input_ids = sub_sequences[i]
                if self.config.is_last_stage:
                    raise NotImplementedError("暂不支持单机推理")
                seq_len = torch.tensor( int (sub_input_ids.shape[1])).reshape(1,1)
                self.send_seq_len(seq_len)
                hidden_states = self.stage_model.forward_sub_sequences(input_ids=sub_input_ids, inputs_embeds=None )
                self.send_activation_forward(hidden_states)
            else:
                seq_len =  int (self.receive_seq_len().item())
                hidden_states = self.receive_activation_forward()
                hidden_states = hidden_states[:,:seq_len,:]
                hidden_states = self.stage_model.forward_sub_sequences(input_ids=None, inputs_embeds=hidden_states )
                if not self.config.is_last_stage:   # 不是第一个也不是最后一个stage
                    seq_len = torch.tensor( int (hidden_states.shape[1])).reshape(1,1)
                    self.send_seq_len(seq_len)
                    self.send_activation_forward(hidden_states)
        
        
    def points_saturation(self,points_input_ids):
        medusa_logits_list = []
        logits_list = []
        extra_kwargs = {
            'is_point': True,
            'point_id':0,
                    }
        for point_id, input_ids in enumerate(points_input_ids)  :
            if self.config.is_first_stage:
                if  self.config.is_last_stage:
                    raise NotImplementedError("暂不支持单机推理")
                # Set is_point = True
                extra_kwargs["point_id"]=point_id
                hidden_states = self.stage_model.forward_sub_sequences(input_ids=input_ids, inputs_embeds=None, **extra_kwargs) 
                seq_len = torch.tensor( int (hidden_states.shape[1])).reshape(1,1)
                self.send_seq_len(seq_len)    
                self.send_activation_forward(hidden_states)
            else:
                seq_len =  int (self.receive_seq_len().item())
                hidden_states = self.receive_activation_forward()
                hidden_states = hidden_states[:,:seq_len,:]
                extra_kwargs["point_id"]=point_id
                hidden_states = self.stage_model.forward_sub_sequences(input_ids=None, inputs_embeds=hidden_states, **extra_kwargs) 

                if not self.config.is_last_stage:   # 不是第一个也不是最后一个stage
                    seq_len = torch.tensor( int (hidden_states.shape[1])).reshape(1,1)
                    self.send_seq_len(seq_len)    
                    self.send_activation_forward(hidden_states)
                else: #最后一个stage
                    medusa_logits = []
                    with torch.inference_mode():
                        logits =  self.stage_model.lm_head(hidden_states)
                        for i in range(self.config.medusa_num_heads):
                            medusa_logits.append(self.stage_model.medusa_head[i](hidden_states))
                        medusa_logits_list.append( torch.stack(medusa_logits, dim=0))
                        logits_list.append(logits)
                    
        if self.config.is_last_stage:  
            return  medusa_logits_list,logits_list
        else:
            return None