# Source: https://github.com/huggingface/transformers/blob/v4.34-release/src/transformers/models/mistral/modeling_mistral.py
# Modifications are denoted by the symbol: [MODIFIED]
# There are mainly two modifications:
# 1. Using preallocated GPU memory for KVCache
# 2. Modifying attention mask for integration with Medusa
""" PyTorch Mistral model."""
import inspect
import math
from typing import List, Optional, Tuple, Union
import warnings
from .utils import *

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
# [MODIFIED] Import from transformer library
from medusa.model.modeling_mistral_kv import MistralPreTrainedModel,MistralModel
from transformers import AutoTokenizer
from medusa.model.kv_cache import initialize_past_key_values
from medusa.model.medusa_choices import *
from .utils import *

class ResBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        # Initialize as an identity mapping
        torch.nn.init.zeros_(self.linear.weight)
        # Use SiLU activation to keep consistent with the Llama model
        self.act = nn.SiLU()
    def forward(self, x):
        return x + self.act(self.linear(x))
    
class MedusaMistralForCausalLM(MistralPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = MistralModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
        #[modified]
        self.medusa_head = nn.ModuleList(
            [
                nn.Sequential(
                    *([ResBlock(config.hidden_size)] * config.medusa_num_layers),
                    nn.Linear(config.hidden_size, config.vocab_size, bias=False),
                )
                for _ in range(config.medusa_num_heads)
            ]
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model
    def get_tokenizer(self):
        return self.tokenizer
    def get_medusa_choice(self, model_name):
        if 'vicuna' in model_name:
            if '7b' in model_name:
                return vicuna_7b_stage2
            elif '13b' in model_name:
                return vicuna_13b_stage2
            elif '33b' in model_name:
                return vicuna_33b_stage2
        elif 'zephyr' in model_name:
            return zephyr_stage2
        warnings.warn('Please specify medusa choice configuration!')
        return mc_sim_7b_63
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        past_key_values=None,
        output_orig=False,
        position_ids=None,
        medusa_forward=False,
        **kwargs,
    ):
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        with torch.inference_mode():
            # MistralModel.forward() 
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                **kwargs,
        )
            if output_orig: # 这里指的是MedusaMistralForCausalLM的结果, 经过model(MistralModel) + lm_head(Linear)
                orig = self.lm_head(outputs[0])
        # Clone the output hidden states
        # Clone the output hidden states
        hidden_states = outputs[0].clone()
        medusa_logits = []
        # TODO: Consider parallelizing this loop for efficiency?
        for i in range(self.config.medusa_num_heads):
            medusa_logits.append(self.medusa_head[i](hidden_states))
        if output_orig:
            return torch.stack(medusa_logits, dim=0), outputs, orig
        return torch.stack(medusa_logits, dim=0)

    def reset_medusa_mode(self):
        self.medusa_mask = None
        self.medusa_mode = None
    
    def medusa_generate(
            self,
            input_ids,
            attention_mask=None,
            temperature=0.0,
            max_steps=512,
            # The hyperparameters below are for the Medusa
            # top-1 prediciton for the next token, top-7 predictions for the next token, top-6 predictions for the next next token.
            medusa_choices=None,
            posterior_threshold=0.09,  # threshold validation of Medusa output
            # another threshold hyperparameter, recommended to be sqrt(posterior_threshold)
            posterior_alpha=0.3,
            top_p=0.8, 
            sampling = 'typical', 
            fast = True
        ): 
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"

      # Avoid modifying the input_ids in-place
        input_ids = input_ids.clone()
        if medusa_choices is None:
            medusa_choices = self.get_medusa_choice(self.config.base_model_name_or_path)
        # Cache medusa buffers (the fixed patterns for tree attention)
        if medusa_choices is None:
            medusa_choices = self.get_medusa_choice(self.config.base_model_name_or_path)

        if hasattr(self, "medusa_choices") and self.medusa_choices == medusa_choices:
            # Load the cached medusa buffer
            medusa_buffers = self.medusa_buffers
        else:
            # Initialize the medusa buffer
            # 参考：https://github.com/FasterDecoding/Medusa/blob/main/notebooks/medusa_configuration_explained.ipynb
            medusa_buffers = generate_medusa_buffers(
                medusa_choices, device=self.base_model.device
            )
        self.medusa_buffers = medusa_buffers
        self.medusa_choices = medusa_choices
        
        # Initialize the past key and value states
        if hasattr(self, "past_key_values"):
            past_key_values = self.past_key_values
            past_key_values_data = self.past_key_values_data
            current_length_data = self.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else: # 为每一个decoder层都创建KVCache存储
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(self)
            self.past_key_values = past_key_values
            self.past_key_values_data = past_key_values_data
            self.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        
        reset_medusa_mode(self) #TODO:
        # Initialize tree attention mask and process prefill tokens
        # 处理prefilling的tokens，同时生成初始medusa_logits
        medusa_logits, logits = initialize_medusa(
            input_ids, self, self.medusa_buffers["medusa_attn_mask"], past_key_values
        )
        new_token = 0
        last_round_token = 0
        print("init ")
        print("medusa_logits.shape: {}, logits.shape: {}".format(medusa_logits.shape, logits.shape))
        # 使用medusa_logits生成投机采样的多个candidates
        for idx in range(max_steps):
            # print("\nstep: ", idx)
            # print("mdedusa_logits.shape: {}, logits.shape: {}".format(medusa_logits.shape, logits.shape))
            # Generate candidates with topk predictions from Medusa heads
            candidates, tree_candidates = generate_candidates(
                medusa_logits,
                logits,
                self.medusa_buffers["tree_indices"],
                self.medusa_buffers["retrieve_indices"],
                temperature=temperature,
                posterior_alpha=posterior_alpha,
                posterior_threshold=posterior_threshold,
                top_p=top_p,
                sampling=sampling,
                fast=fast,
            )
            # print("\ncandidates.shape: {}, tree_candidates.shape: {}".format(candidates.shape, tree_candidates.shape))
            # print("INPUT OF tree_decoding : mdedusa_logits.shape: {}, logits.shape: {}".format(medusa_logits.shape, logits.shape))
            # 将token tree输入model再次执行前向传播，验证candidate的可行性
            # Use tree attention to verify the candidates and get predictions
            medusa_logits, logits, outputs = tree_decoding(
                self,   
                tree_candidates,
                self.past_key_values, # [modified]
                self.medusa_buffers["medusa_position_ids"], # [modified]
                input_ids,
                self.medusa_buffers["retrieve_indices"], # [modified]
            )
            # print("\nafter tree decoding: mdedusa_logits.shape: {}, logits.shape: {}".format(medusa_logits.shape, logits.shape))
            # 选择最终被接受的tokens
            best_candidate, accept_length = evaluate_posterior(
                logits, candidates, temperature, posterior_threshold, posterior_alpha, top_p=top_p, sampling=sampling, fast=fast
            )
            # Update the input_ids and logits
            # 使用上一轮采样产生的medusa logits用来下一轮的token candidate生成
            input_ids, logits, medusa_logits, new_token,_ = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                medusa_buffers["retrieve_indices"],
                None, # [modified] outputs 实际上没用上
                logits,
                medusa_logits,
                new_token,
                self.past_key_values_data,
                self.current_length_data,
            )
            
            # 返回generator用来返回推理结果
            yield {
                "text": self.tokenizer.decode(
                    input_ids[0, input_len:],
                    skip_special_tokens=True,
                    spaces_between_special_tokens=False,
                    clean_up_tokenization_spaces=True,
                )
            }

            if self.tokenizer.eos_token_id in input_ids[0, input_len:]:
                print("\nFinal step: ", idx+1 )
                break

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past


 