import os
from huggingface_hub import hf_hub_download
import warnings
# Source: https://github.com/huggingface/transformers/blob/v4.34-release/src/transformers/models/llama/modeling_llama.py
# Modifications are denoted by the symbol: [MODIFIED]
# There are mainly two modifications:
# 1. Using preallocated GPU memory for KVCache
# 2. Modifying attention mask for integration with Medusa

""" PyTorch LLaMA model."""
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from medusa.model.kv_cache import initialize_past_key_values
from medusa.model.medusa_choices import *
from .utils import *

# [MODIFIED] Import from transformer library
from transformers import AutoTokenizer
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, SequenceClassifierOutputWithPast 
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    # is_flash_attn_available,
    logging,
    replace_return_docstrings,
)
from  .llama_config import LlamaConfig
from medusa.model.modeling_llama_kv  import    LlamaRMSNorm, LlamaPreTrainedModel,LlamaDecoderLayer,_make_causal_mask,_expand_mask

# if is_flash_attn_available():
#     from flash_attn import flash_attn_func, flash_attn_varlen_func
#     from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"

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

ALL_LAYERNORM_LAYERS.append(LlamaRMSNorm)


 
class PPLlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        # [modified]
        if config.is_first_stage:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_pp_hidden_layers)]) # [MODIFIED]
        # 
        if config.is_last_stage:
            self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # Copied from transformers.models.bart.modeling_bart.BartDecoder._prepare_decoder_attention_mask
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        # [MODIFIED] add medusa mask
        if hasattr(self, "medusa_mask") and self.medusa_mask is not None:
            medusa_mask = self.medusa_mask
            medusa_len = medusa_mask.size(-1)
            combined_attention_mask[:, :, -medusa_len:, -medusa_len:][
                medusa_mask == 0
            ] = combined_attention_mask.min()
            if hasattr(self, "medusa_mode"):
                # debug mode
                if self.medusa_mode == "debug":
                    torch.save(combined_attention_mask, "medusa_mask.pt")

        return combined_attention_mask

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,  # [MODIFIED] past_key_value is KVCache class
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            # 获取KVCache的current_length
            past_key_values_length = past_key_values[0][0].shape[2]
            # 包括当前推理序列的总seq_length
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            # 生成待推理新序列的position_ids
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        # prefilling 推理步骤的时候attention_mask为None
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
            padding_mask = None
        else:
            if 0 in attention_mask:
                padding_mask = attention_mask
            else:
                padding_mask = None

        # decoding推理步骤的时候为medusa tree attention mask
        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )
        # [MODIFIED] 
        self.attention_mask = attention_mask
        self.position_ids = position_ids

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        # 遍历所有decoder层
        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, past_key_value, output_attentions, padding_mask=padding_mask)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer), hidden_states, attention_mask, position_ids
                )
            else:
                # 带着past_key_value进行decoding阶段计算
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    padding_mask=padding_mask,
                )

            hidden_states = layer_outputs[0]

            # 收集新产生的KVCache
            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)
        # [modified]
        if self.config.is_last_stage:
            hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

class PPMedusaLlamaForCausalLM(LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        # [modified]
        self.config = config
        self.model = PPLlamaModel(config)
        self.vocab_size = config.vocab_size
        # [modified]
        if config.is_last_stage:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
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

        # Initialize weights and apply final processing
        self.post_init()

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
    # @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    # [mmodified]
    def forward(
        self,
        input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        past_key_values=None,
        output_orig=False,
        position_ids=None,
        medusa_forward=False,
        **kwargs,
    ):
        """Forward pass of the MedusaModel.

        Args:
            input_ids (torch.Tensor, optional): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            labels (torch.Tensor, optional): Ground truth labels for loss computation.
            past_key_values (tuple, optional): Tuple containing past key and value states for attention.
            output_orig (bool, optional): Whether to also output predictions from the original LM head.
            position_ids (torch.Tensor, optional): Position IDs.

        Returns:
            torch.Tensor: A tensor containing predictions from all Medusa heads.
            (Optional) Original predictions from the base model's LM head.
        """
        with torch.inference_mode():
            # 注意：执行的是LlamaModel.forward(),不是LlamaForCausalLM.forward()
            outputs = self.model(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                **kwargs,
            )
            # [modified] 不是last stage, 直接返回outputs:BaseModelOutputWithPast
            if not self.config.is_last_stage:
                return outputs
            if output_orig: # 这里指的是LlamaForCausalLM的结果, 经过model(LlamaModel) + lm_head(Linear)
                orig = self.lm_head(outputs[0])
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
    def prefilling(
        self,
        input_ids,
        inputs_embeds=None,
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
        if self.config.is_first_stage: # 第一个stage，输入的是input_ids [1,seq]
            assert input_ids != None
            assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
            # Avoid modifying the input_ids in-place
            input_ids = input_ids.clone()
        else: # 之后的stage，输入的是inputs_embeds [1,seq,hidden_size]
            assert inputs_embeds != None
            assert inputs_embeds.shape[0] == 1, "Only support batch size 1 for now!!"
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
        #TODO: self.past_key_values  作为模型属性，pipline的时候 不同输入的past_key_values不同 
        
        reset_medusa_mode(self) #TODO:
        # Initialize tree attention mask and process prefill tokens
        # 处理prefilling的tokens，同时生成初始medusa_logits
        if self.config.is_last_stage: # 最后一个stage 得到medusa_logits和logits
            medusa_logits, _, logits = self(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds, # None
                past_key_values=past_key_values, 
                output_orig=True, 
                medusa_forward=True
            )            
            self.model.medusa_mask =  medusa_buffers["medusa_attn_mask"] #TODO: 是所有stage吗
            return medusa_logits, logits # [num_medusa_head.1,seq_len,vocab_size], [1,seq_len,vocab_size]
        else: # 其他stage，只得llama_model的outputs：BaseModelOutputWithPast
            outputs = self(
                input_ids=input_ids,  # None
                inputs_embeds=inputs_embeds,
                past_key_values=past_key_values, 
                output_orig=True, 
                medusa_forward=True
            )
            self.model.medusa_mask =  medusa_buffers["medusa_attn_mask"] #TODO: 是所有stage吗
            assert isinstance(outputs, BaseModelOutputWithPast)
            hidden_states = outputs.last_hidden_state
            return hidden_states # [1,seq_len,hidden_size] 
    
    def tree_decoding(self, tree_candidates, tree_candidates_embeds,input_ids):
        if self.config.is_first_stage:
            assert tree_candidates != None
            assert tree_candidates_embeds == None
        else:
            assert tree_candidates == None
            assert tree_candidates_embeds != None
        position_ids =  self.medusa_buffers["medusa_position_ids"] + input_ids.shape[1]
        if  self.config.is_last_stage:
            tree_medusa_logits, outputs, tree_logits = self(
                    input_ids = tree_candidates,
                    inputs_embeds = tree_candidates_embeds,
                    output_orig=True,
                    past_key_values=self.past_key_values,
                    position_ids=position_ids,  # can not be none !
                    medusa_forward=True,
                )
            logits = tree_logits[0, self.medusa_buffers["retrieve_indices"]]
            medusa_logits = tree_medusa_logits[:, 0, self.medusa_buffers["retrieve_indices"]]
            return  medusa_logits, logits 
        else:
            outputs = self(
                    input_ids = tree_candidates,
                    inputs_embeds = tree_candidates_embeds,
                    output_orig=True,
                    past_key_values=self.past_key_values,
                    position_ids=position_ids,  # can not be none !
                    medusa_forward=True,
                )
            assert isinstance(outputs, BaseModelOutputWithPast)
            hidden_states = outputs.last_hidden_state
            return hidden_states # [1,64,hidden_size]
            
    def generate_candidates(
        self,
        medusa_logits, 
        logits, ):
            assert self.config.is_last_stage
            candidates, tree_candidates = generate_candidates(
                    medusa_logits,
                    logits,
                    self.medusa_buffers["tree_indices"],
                    self.medusa_buffers["retrieve_indices"],
                    temperature=self.config.temperature,
                    posterior_alpha=self.config.posterior_alpha,
                    posterior_threshold=self.config.posterior_threshold,
                    top_p=self.config.top_p,
                    sampling=self.config.sampling,
                    fast=self.config.fast,
                )
            return candidates, tree_candidates
    def evaluate_posterior(self,
                           logits,
                           candidates,        
):
        assert  self.config.is_last_stage
        best_candidate, accept_length = evaluate_posterior(
                    logits, 
                    candidates, 
                    temperature=self.config.temperature,
                    posterior_alpha=self.config.posterior_alpha,
                    posterior_threshold=self.config.posterior_threshold,
                    top_p=self.config.top_p,
                    sampling=self.config.sampling,
                    fast=self.config.fast,
                    )
        return best_candidate, accept_length
    def update_inference_inputs(
        self,
            input_ids,
            candidates,
            best_candidate,
            accept_length,
            logits,
            medusa_logits,
            new_token,
    ):
        assert  self.config.is_last_stage
        input_ids, logits, medusa_logits, new_token ,select_indices= update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                self.medusa_buffers["retrieve_indices"],
                None, # [modified]
                logits,
                medusa_logits,
                new_token,
                self.past_key_values_data,
                self.current_length_data,
            )
        return input_ids, logits, medusa_logits, new_token,select_indices
    def update_kv_cache(self,input_ids, select_indices):
        assert not self.config.is_last_stage
        prev_input_len = input_ids.shape[1] 
        tgt = self.past_key_values_data[..., select_indices, :]
        dst =  self.past_key_values_data[..., prev_input_len : prev_input_len + tgt.shape[-2], :]
        dst.copy_(tgt, non_blocking=True)
        self.current_length_data.fill_(prev_input_len + tgt.shape[-2])



        

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
        """
        Args:
            input_ids (torch.Tensor, optional): Input token IDs.
            attention_mask (torch.Tensor, optional): Attention mask.
            temperature (float, optional): Temperature for typical acceptance.
            medusa_choices (list, optional): A list of integers indicating the number of choices for each Medusa head.
            posterior_threshold (float, optional): Threshold for posterior validation.
            posterior_alpha (float, optional): Another threshold hyperparameter, recommended to be sqrt(posterior_threshold).
            top_p (float, optional): Cumulative probability threshold for nucleus sampling. Defaults to 0.8.
            sampling (str, optional): Defines the sampling strategy ('typical' or 'nucleus'). Defaults to 'typical'.
            fast (bool, optional): If True, enables faster, deterministic decoding for typical sampling. Defaults to False.
        Returns:
            torch.Tensor: Output token IDs.

        Warning: Only support batch size 1 for now!!
        """
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
            input_ids, self, medusa_buffers["medusa_attn_mask"], past_key_values
        )
        # print("init")
        # print("mdedusa_logits.shape : {}, logits.shape : {}".format(medusa_logits.shape, logits.shape))
        new_token = 0
        last_round_token = 0
        
        # 使用medusa_logits生成投机采样的多个candidates
        for idx in range(max_steps):
            # print("\nstep: ", idx)
            # print("mdedusa_logits.shape: {}, logits.shape: {}".format(medusa_logits.shape, logits.shape))
            # Generate candidates with topk predictions from Medusa heads
            candidates, tree_candidates = generate_candidates(
                medusa_logits,
                logits,
                medusa_buffers["tree_indices"],
                medusa_buffers["retrieve_indices"],
                temperature=temperature,
                posterior_alpha=posterior_alpha,
                posterior_threshold=posterior_threshold,
                top_p=top_p,
                sampling=sampling,
                fast=fast,
            )
            # print("INPUT OF tree_decoding : mdedusa_logits.shape: {}, logits.shape: {}".format(medusa_logits.shape, logits.shape))
            # 将token tree输入model再次执行前向传播，验证candidate的可行性
            # Use tree attention to verify the candidates and get predictions
            medusa_logits, logits, outputs = tree_decoding(
                self,
                tree_candidates,
                past_key_values,
                medusa_buffers["medusa_position_ids"],
                input_ids,
                medusa_buffers["retrieve_indices"],
            )
            # print("After Decoding")
            # print("mdedusa_logits.shape: {}, logits.shape: {}".format(medusa_logits.shape, logits.shape))
            # Evaluate the posterior of the candidates to select the accepted candidate prefix
            # 选择最终被接受的tokens
            best_candidate, accept_length = evaluate_posterior(
                logits, candidates, temperature, posterior_threshold, posterior_alpha, top_p=top_p, sampling=sampling, fast=fast
            )

            # Update the input_ids and logits
            # 使用上一轮采样产生的medusa logits用来下一轮的token candidate生成
            input_ids, logits, medusa_logits, new_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                medusa_buffers["retrieve_indices"],
                # outputs, outputs 实际上没用上
                None,
                logits,
                medusa_logits,
                new_token,
                past_key_values_data,
                current_length_data,
            )
            # print("after update")
            # print("input_ids: ", input_ids.shape)
            # print("new_token: ", new_token)
            # print("mdedusa_logits.shape: {}, logits.shape: {}".format(medusa_logits.shape, logits.shape))
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
 