# coding=utf-8
# Copyright 2023 Mistral AI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch Mixtral model."""

from collections import defaultdict
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import DynamicCache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import (
    MoeCausalLMOutputWithPast,
    MoeModelOutputWithPast,
)
from transformers.processing_utils import Unpack
from transformers.utils import (
    LossKwargs,
    logging,
)
from modeling_mistral import (
    MistralAttention,
    MistralForCausalLM,
    MistralForQuestionAnswering,
    MistralForSequenceClassification,
    MistralForTokenClassification,
    MistralModel,
    MistralPreTrainedModel,
    MistralRMSNorm,
    MistralRotaryEmbedding,
)
from .configuration_mixtral import MixtralConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "mistralai/Mixtral-8x7B-v0.1"
_CONFIG_FOR_DOC = "MixtralConfig"


def load_balancing_loss_func(
    gate_logits: Union[torch.Tensor, Tuple[torch.Tensor], None],
    num_experts: Optional[int] = None,
    top_k=2,
    attention_mask: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, int]:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits:
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        num_experts:
            Number of experts
        top_k:
            The number of experts to route per-token, can be also interpreted as the `top-k` routing
            parameter.
        attention_mask (`torch.Tensor`, *optional*):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.

    Returns:
        The auxiliary loss.
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)

    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)

    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)

        # Compute the mask that masks all padding tokens as 0 with the same shape of expert_mask
        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand((num_hidden_layers, batch_size, sequence_length, top_k, num_experts))
            .reshape(-1, top_k, num_experts)
            .to(compute_device)
        )

        # Compute the percentage of tokens routed to each experts
        tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(
            expert_attention_mask, dim=0
        )

        # Compute the mask that masks all padding tokens as 0 with the same shape of tokens_per_expert
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(compute_device)
        )

        # Compute the average probability of routing to these experts
        router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0) / torch.sum(
            router_per_expert_attention_mask, dim=0
        )

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts


class MixtralBlockSparseTop2MLP(nn.Module):
    def __init__(self, config: MixtralConfig):
        super().__init__()
        self.ffn_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size

        self.w1 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.w2 = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.w3 = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)

        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states):
        current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(hidden_states)
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states


class MixtralSparseMoeBlock(nn.Module):
    """
    This implementation is
    strictly equivalent to standard MoE with full capacity (no
    dropped tokens). It's faster since it formulates MoE operations
    in terms of block-sparse operations to accommodate imbalanced
    assignments of tokens to experts, whereas standard MoE either
    (1) drop tokens at the cost of reduced performance or (2) set
    capacity factor to number of experts and thus waste computation
    and memory on padding.
    """

    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok

        # gating
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)

        self.experts = nn.ModuleList([MixtralBlockSparseTop2MLP(config) for _ in range(self.num_experts)])

        # Jitter parameters
        self.jitter_noise = config.router_jitter_noise
        
        #Request Level, 记录推理整个数据集每个专家的激活情况
        #Sequence Level，记录单次推理每个专家的激活情况（整个数据集只有一个request的特殊情况）
        self.expert_activation_counts = defaultdict(int)  # 记录路由专家的激活次数
        self.total_tokens_nor = 0
        self.top_avg = np.zeros((self.top_k)) #记录topk的score均值

        
        #Cache hit rate ，记录当前层在一次推理中的缓存命中率
        self.continue_cnt = defaultdict(int) # 记录专家当前连续次数
        # self.expert_activation_counts_max_continue = defaultdict(int) # 记录路由专家被连续激活的最大次数
        self.cache_hit_cnt = 0 # 缓存命中次数
        self.cache_request_cnt = 0 # 缓存请求次数
        
        
        #Token Level,记录一次推理中的每个token激活专家的情况
        self.token_frequencies = defaultdict(lambda: np.zeros((self.num_experts)))# 记录每个token的激活情况
        self.total_tokens = 0  # 记录一次推理时的token 数量


    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        if self.training and self.jitter_noise > 0:
            hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - self.jitter_noise, 1.0 + self.jitter_noise)
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1,sorted=True)
        
        if(sequence_length==1):#只统计decode阶段
                    # 统计路由专家激活次数
            for i in range(batch_size * sequence_length):
                scores = router_logits.softmax(dim=-1, dtype=torch.float)
                self.token_frequencies[self.total_tokens+1] = scores[i].cpu().detach().numpy()
                for k in range(self.top_k):
                    expert_idx = selected_experts[i, k].item()
                    self.top_avg[k] += scores[i, expert_idx].item()

                    self.expert_activation_counts[expert_idx] += 1
                    # self.token_frequencies[self.total_tokens][expert_idx] = topk_weight[i,k].item()
                    
                    
                    if self.continue_cnt[expert_idx]>=1:#被cache
                        self.cache_hit_cnt+=1
                    
                    self.continue_cnt[expert_idx] +=1 #连续次数+1
                    # self.expert_activation_counts_max_continue[expert_idx]=max(self.expert_activation_counts_max_continue[expert_idx],self.continue_cnt[expert_idx])
                    
                for k in range(self.num_experts):
                    if k not in set(selected_experts[i].tolist()):# 没被激活的
                        self.continue_cnt[k] = 0 #连续中断,offload
                    
            self.cache_request_cnt += self.top_k # 缓存请求次数
            self.total_tokens += 1
            self.total_tokens_nor +=1
        
        
        
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states, router_logits

    def get_expert_frequencies(self):
        """计算每个专家的激活概率"""
        if self.total_tokens == 0:
            return  {}
        routed_frequencies = {}
        # 路由专家的激活概率
        for expert_idx in range(self.num_experts):
            count = self.expert_activation_counts[expert_idx]
            frequency = count / (self.total_tokens)  
            routed_frequencies[expert_idx] = frequency
        # 共享专家的激活概率（总是 1.0，因为每个 token 都会经过共享专家）
        return  routed_frequencies

    def reset_counts(self):
        """重置计数器"""
        self.expert_activation_counts.clear()
        self.total_tokens = 0
        
    def get_layer_top_avg(self):
        """计算每层的top分数均值"""
        if self.total_tokens_nor == 0:
            return  np.zeros((self.num_experts))            
        for k in range(self.top_k):
            self.top_avg[k] = self.top_avg[k]/self.total_tokens_nor
        return self.top_avg
        # 共享专家的激活概率（总是 1.0，因为每个 token 都会经过共享专家）
        # return  routed_frequencies

    def reset_top_avg(self):
        """重置计数器"""
        self.total_tokens_nor = 0
        self.top_avg = np.zeros((self.top_k))
        
    # def get_expert_max_continue(self):
    #     routed_max_continue = {}
    #     # 路由专家的连续激活最大次数
    #     for expert_idx in range(self.num_experts):
    #         routed_max_continue[expert_idx] = self.expert_activation_counts_max_continue[expert_idx]/self.total_tokens
    #     # 共享专家的激活概率（总是 1.0，因为每个 token 都会经过共享专家）
    #     return  routed_max_continue
    
    # def reset_continue_counts(self):
    #     """重置计数器"""
    #     self.expert_activation_counts_max_continue.clear()
    #     self.continue_cnt.clear()
        
    def get_expert_hit_rate(self):
        routed_hit_rate = {}
        for expert_idx in range(self.num_experts):
            # routed_hit_rate[expert_idx] = self.cache_hit_cnt[expert_idx]/self.total_tokens
            routed_hit_rate[expert_idx] = self.cache_hit_cnt[expert_idx]/self.expert_activation_counts[expert_idx]
        # 共享专家的激活概率（总是 1.0，因为每个 token 都会经过共享专家）
        return  routed_hit_rate
    
    def reset_hit_counts(self):
        """重置计数器"""
        self.cache_hit_cnt.clear()
        
    def get_token_frequency(self):
        # 共享专家的激活概率（总是 1.0，因为每个 token 都会经过共享专家）
        return  self.token_frequencies
    
    def reset_token_frequency(self):
        """重置计数器"""
        self.token_frequencies.clear()


class MixtralRMSNorm(MistralRMSNorm):
    pass


class MixtralAttention(MistralAttention):
    pass


class MixtralDecoderLayer(nn.Module):
    def __init__(self, config: MixtralConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = MixtralAttention(config, layer_idx)

        self.block_sparse_moe = MixtralSparseMoeBlock(config)
        self.input_layernorm = MixtralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MixtralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_router_logits (`bool`, *optional*):
                Whether or not to return the logits of all the routers. They are useful for computing the router loss, and
                should not be returned during inference.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, router_logits = self.block_sparse_moe(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs


class MixtralRotaryEmbedding(MistralRotaryEmbedding):
    pass


class MixtralPreTrainedModel(MistralPreTrainedModel):
    _supports_static_cache = False  # MoE models don't work with torch.compile (`torch.where(condition)` not supported)


class MixtralModel(MistralModel):
    def __init__(self, config: MixtralConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [MixtralDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

    def get_all_expert_frequencies(self):
        """获取所有层的专家激活概率"""
        all_frequencies = {}
        for layer_idx, layer in enumerate(self.layers):
            
            moe_block = layer.block_sparse_moe  # 假设每层有一个 moe_block 属性
            routed_freq = moe_block.get_expert_frequencies()
            all_frequencies[layer_idx] = {"routed": routed_freq}
            # print(f'sum:{moe_block.total_tokens}')
        return all_frequencies

    def reset_all_expert_counts(self):
        """重置所有层的计数器"""
        for layer_idx, layer in enumerate(self.layers):
            
            moe_block = layer.block_sparse_moe
            moe_block.reset_counts()
        
    def get_all_layer_top_avg(self):
        """获取所有层的专家激活概率"""
        all_frequencies = {}
        for layer_idx, layer in enumerate(self.layers):
            moe_block = layer.block_sparse_moe  # 假设每层有一个 moe_block 属性
            # routed_freq = moe_block.get_layer_top_avg()
            all_frequencies[layer_idx] =  moe_block.get_layer_top_avg()
            # print(f'sum:{moe_block.total_tokens}')
        return all_frequencies

    def reset_all_layer_top_avg(self):
        """重置所有层的计数器"""
        for layer_idx, layer in enumerate(self.layers):
            moe_block = layer.block_sparse_moe
            moe_block.reset_top_avg()
        
    # def get_all_expert_continue(self):
    #     """获取所有层的专家连续激活次数"""
    #     all_frequencies = {}
    #     for layer_idx, layer in enumerate(self.layers):
            
    #         moe_block = layer.block_sparse_moe  # 假设每层有一个 moe_block 属性
    #         routed_freq = moe_block.get_expert_max_continue()
    #         all_frequencies[layer_idx] = {"routed": routed_freq}
    #         # print(f'sum:{moe_block.total_tokens}')
    #     return all_frequencies

    # def reset_all_expert_continue(self):
    #     """重置所有层的计数器"""
    #     for layer_idx, layer in enumerate(self.layers):
            
    #         moe_block = layer.block_sparse_moe
    #         moe_block.reset_continue_counts()
            
    def get_all_expert_hit_rate(self):
        """获取所有层的缓存命中率的情况"""
        all_frequencies = {}
        for layer_idx, layer in enumerate(self.layers):
            
            moe_block = layer.block_sparse_moe  # 假设每层有一个 moe_block 属性
            routed_freq = moe_block.get_expert_hit_rate()
            all_frequencies[layer_idx] = {"routed": routed_freq}
            # print(f'sum:{moe_block.total_tokens}')
        return all_frequencies

    def reset_all_expert_hit_rate(self):
        """重置所有层的计数器"""
        for layer_idx, layer in enumerate(self.layers):
            
            moe_block = layer.block_sparse_moe
            moe_block.reset_hit_counts()   
            
    def get_all_token_frequency(self):
        """获取所有层的专家token维度的激活概率"""
        all_frequencies = {}
        for layer_idx, layer in enumerate(self.layers):
            
            moe_block = layer.block_sparse_moe  # 假设每层有一个 moe_block 属性
            routed_freq = moe_block.get_token_frequency()
            all_frequencies[layer_idx] = {"routed": routed_freq}
            # print(f'sum:{moe_block.total_tokens}')
        return all_frequencies

    def reset_all_token_frequency(self):
        """重置所有层的计数器"""
        for layer_idx, layer in enumerate(self.layers):
            
            moe_block = layer.block_sparse_moe
            moe_block.reset_token_frequency()   


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, MoeModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    output_router_logits,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    output_router_logits=output_router_logits,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if output_router_logits:
                all_router_logits += (layer_outputs[-1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
        )
        return output if return_dict else output.to_tuple()


class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...


class MixtralForCausalLM(MistralForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = MixtralModel(config)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_local_experts
        self.num_experts_per_tok = config.num_experts_per_tok

    
    def get_all_expert_frequencies(self):
        return self.model.get_all_expert_frequencies()
    def reset_all_expert_counts(self):
        self.model.reset_all_expert_counts()
        
    def get_all_layer_top_avg(self):
        return self.model.get_all_layer_top_avg()
    def reset_all_layer_top_avg(self):
        self.model.reset_all_layer_top_avg()     
        
    def get_all_expert_continue(self):
        return self.model.get_all_expert_continue()
    def reset_all_expert_continue(self):
        self.model.reset_all_expert_continue()

    def get_all_expert_hit_rate(self):
        return self.model.get_all_expert_hit_rate()
    def reset_all_expert_hit_rate(self):
        self.model.reset_all_expert_hit_rate()
        
    def get_all_token_frequency(self):
        return self.model.get_all_token_frequency()
    def reset_all_token_frequency(self):
        self.model.reset_all_token_frequency()


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, MoeCausalLMOutputWithPast]:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, MixtralForCausalLM

        >>> model = MixtralForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
        >>> tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits if return_dict else outputs[-1],
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)  # make sure to reside in the same device

        if not return_dict:
            output = (logits,) + outputs[1:]
            if output_router_logits:
                output = (aux_loss,) + output
            return (loss,) + output if loss is not None else output

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )


class MixtralForSequenceClassification(MistralForSequenceClassification):
    pass


class MixtralForTokenClassification(MistralForTokenClassification):
    pass


class MixtralForQuestionAnswering(MistralForQuestionAnswering):
    pass