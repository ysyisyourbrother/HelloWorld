from core.schedules import PipelineRuntime

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

    def pipeline_forward_sub_sequences(self):
        # TODO: 实现此函数

        # Step1: 划分原本的sequence为多个sub-sequence
        # Step2: 同时注入多个sub_sequence进入pipeline: 
            # for sub_seq in xxx:
            #     self.pipeline_forward()
        # Step3: 实现不同sub_sequence长度不同下的通信能力，在core.communication 23行
        # Step4: 处理不同sub_sequence推理过程中的kv cache关系
        pass

        
