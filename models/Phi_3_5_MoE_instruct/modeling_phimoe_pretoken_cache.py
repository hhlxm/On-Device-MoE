# coding=utf-8
# Copyright 2024 Microsoft and the HuggingFace Inc. team. All rights reserved.
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

""" PyTorch PhiMoE model."""
from collections import defaultdict
import inspect
import math
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import (
    MoeCausalLMOutputWithPast,
    MoeModelOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import is_torch_greater_or_equal_than_1_13
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)
from transformers.utils.import_utils import is_torch_fx_available
from .configuration_phimoe import PhiMoEConfig

from einops import rearrange
from flash_attn.layers.rotary import RotaryEmbedding as FlashRotaryEmbedding

from collections import OrderedDict, defaultdict, deque
import heapq
import time

class CacheEntry:
    def __init__(self, expert_id, frequency=0, last_access_time=None):
        self.expert_id = expert_id
        self.frequency = frequency
        self.last_access_time = last_access_time if last_access_time is not None else time.time()

    # For LFU heap comparison
    def __lt__(self, other):
        if self.frequency == other.frequency:
            return self.last_access_time < other.last_access_time
        return self.frequency < other.frequency

class CacheSimulator:
    def __init__(self, capacity, strategy='lru', max_expert_id=64):
        if capacity <= 0:
            raise ValueError("Cache capacity must be positive.")
        self.capacity = capacity
        self.max_expert_id = max_expert_id
        self.strategy = strategy.lower()
        self.hits = 0
        self.misses = 0
        self.requests = 0

        if self.strategy == 'lru' or self.strategy == 'fifo':
            self.cache_store = OrderedDict() # Stores expert_id directly
        elif self.strategy == 'lfu':
            self.cache_store = {} # expert_id -> CacheEntry object
            self.frequency_heap = [] # Min-heap: (frequency, last_access_time, expert_id)
        else:
            raise ValueError(f"Unknown cache strategy: {strategy}. Supported: 'lru', 'fifo', 'lfu'")

    def _evict_fifo(self):
        if not self.cache_store:
            return None
        return self.cache_store.popitem(last=False)[0] # Remove the first item added

    def _evict_lru(self):
        if not self.cache_store:
            return None
        return self.cache_store.popitem(last=False)[0] # Remove the least recently used (first item)

    def _evict_lfu(self):
        if not self.frequency_heap:
            return None
        # Pop the item with the lowest frequency (and oldest access time in case of ties)
        _, _, expert_id_to_evict = heapq.heappop(self.frequency_heap)
        if expert_id_to_evict in self.cache_store:
            del self.cache_store[expert_id_to_evict]
        return expert_id_to_evict

    def update(self, expert_id):# only update not access
        if self.strategy == 'lru':
            if expert_id in self.cache_store:
                # self.cache_store.move_to_end(expert_id)
                pass # 已经再缓存中
            else:
                if len(self.cache_store) >= self.capacity:
                    self._evict_lru()
                self.cache_store[expert_id] = None
        elif self.strategy == 'fifo':
            if expert_id in self.cache_store:
                # No change in order for FIFO on hit
                pass
            else:
                if len(self.cache_store) >= self.capacity:
                    self._evict_fifo()
                self.cache_store[expert_id] = None
        elif self.strategy == 'lfu':
            current_time = time.time()
            if expert_id in self.cache_store:
                pass
            else:
                if len(self.cache_store) >= self.capacity:
                    evicted_expert_id = self._evict_lfu()
                    # print(f"LFU Evicted: {evicted_expert_id}")
                new_entry = CacheEntry(expert_id, frequency=1, last_access_time=current_time)
                self.cache_store[expert_id] = new_entry
                heapq.heappush(self.frequency_heap, (new_entry.frequency, new_entry.last_access_time, expert_id))

    def access(self, expert_id):
        self.requests += 1
        is_hit = False

        if self.strategy == 'lru':
            if expert_id in self.cache_store:
                self.hits += 1
                is_hit = True
                self.cache_store.move_to_end(expert_id) # Mark as recently used
            else:
                self.misses += 1
                if len(self.cache_store) >= self.capacity:
                    self._evict_lru()
                self.cache_store[expert_id] = None # Value doesn't matter for LRU/FIFO here
        
        elif self.strategy == 'fifo':
            if expert_id in self.cache_store:
                self.hits += 1
                is_hit = True
                # No change in order for FIFO on hit
            else:
                self.misses += 1
                if len(self.cache_store) >= self.capacity:
                    self._evict_fifo()
                self.cache_store[expert_id] = None

        elif self.strategy == 'lfu':
            current_time = time.time()
            if expert_id in self.cache_store:
                self.hits += 1
                is_hit = True
                entry = self.cache_store[expert_id]

                old_heap_entry = None
                for i, item in enumerate(self.frequency_heap):
                    if item[2] == expert_id:
                        old_heap_entry = self.frequency_heap.pop(i)
                        heapq.heapify(self.frequency_heap) # Re-heapify after removal
                        break
                
                entry.frequency += 1
                entry.last_access_time = current_time
                heapq.heappush(self.frequency_heap, (entry.frequency, entry.last_access_time, expert_id))

            else:
                self.misses += 1
                if len(self.cache_store) >= self.capacity:
                    evicted_expert_id = self._evict_lfu()
                    # print(f"LFU Evicted: {evicted_expert_id}")
                
                new_entry = CacheEntry(expert_id, frequency=1, last_access_time=current_time)
                self.cache_store[expert_id] = new_entry
                heapq.heappush(self.frequency_heap, (new_entry.frequency, new_entry.last_access_time, expert_id))
        
        return is_hit


    def get_hit_rate(self):
        if self.requests == 0:
            return 0.0
        return self.hits / self.requests

    def get_stats(self):
        return {
            "hits": self.hits,
            "misses": self.misses,
            "requests": self.requests,
            "hit_rate": self.get_hit_rate(),
            "capacity": self.capacity,
            "strategy": self.strategy,
            "current_size": len(self.cache_store)
        }

    def reset_stats(self):
        if self.strategy == 'lru' or self.strategy == 'fifo':
            self.cache_store.clear()
        elif self.strategy == 'lfu':
            self.cache_store.clear()
            self.frequency_heap = []

    def reset_num(self):
        self.hits = 0
        self.misses = 0
        self.requests = 0

    def reset(self):
        self.reset_stats()
        self.reset_num()
        # self.hits = 0
        # self.misses = 0
        # self.requests = 0
        # if self.strategy == 'lru' or self.strategy == 'fifo':
        #     self.cache_store.clear()
        # elif self.strategy == 'lfu':
        #     self.cache_store.clear()
        #     self.frequency_heap = []
    
    def random_fill(self):
        self.reset()
        # Fill the cache with random expert IDs
        import random
        # 获取 capacity 个不同的专家序号
        expert_ids = random.sample(range(self.max_expert_id), self.capacity)
        # 根据不同缓存策略插入缓存
        if self.strategy in ['lru', 'fifo']:
            for expert_id in expert_ids:
                self.cache_store[expert_id] = None
        elif self.strategy == 'lfu':
            current_time = time.time()
            for expert_id in expert_ids:
                entry = CacheEntry(expert_id, frequency=1, last_access_time=current_time)
                self.cache_store[expert_id] = entry
                heapq.heappush(self.frequency_heap, (entry.frequency, entry.last_access_time, expert_id))

class Predictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        
        # self.linear = nn.Linear(self.hidden_size, self.config.num_local_experts, bias=False,dtype=torch.float32)
        # nn.init.kaiming_uniform_(self.linear.weight, a=math.sqrt(5))
        intermediate_size = int((self.hidden_size + config.num_local_experts) * 5 / 3) 
        self.linear = nn.Sequential(
            nn.Linear(self.hidden_size, intermediate_size, bias=False, dtype=torch.float32),
            nn.ReLU(),  # 或者使用 nn.GELU()
            nn.Linear(intermediate_size, config.num_local_experts, bias=False,dtype=torch.float32),
        
            
        )
        # 初始化权重
        nn.init.kaiming_uniform_(self.linear[0].weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.linear[2].weight, a=math.sqrt(5))
        # 初始化权重为常数
        # nn.init.constant_(self.linear[0].weight, 0.01)
        # nn.init.constant_(self.linear[2].weight, 0.01)
        # self.layer_norm = DeepseekV2RMSNorm(
        #     self.input_dim, eps=config.rms_norm_eps
        # )


        # nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))

    
    def forward(self,hidden_state):
        output = self.linear(hidden_state)
        return output

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

    _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)

# This makes `_prepare_4d_causal_attention_mask` a leaf function in the FX graph.
# It means that the function will not be traced through and simply appear as a node in the graph.
if is_torch_fx_available():
    if not is_torch_greater_or_equal_than_1_13:
        import torch.fx

    _prepare_4d_causal_attention_mask = torch.fx.wrap(_prepare_4d_causal_attention_mask)


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "PhiMoEConfig"


def load_balancing_loss_func(gate_logits: torch.Tensor, num_experts: torch.Tensor = None, top_k=2, attention_mask: Optional[torch.Tensor] = None) -> float:
    r"""
    Computes auxiliary load balancing loss as in Switch Transformer - implemented in Pytorch.

    See Switch Transformer (https://arxiv.org/abs/2101.03961) for more details. This function implements the loss
    function presented in equations (4) - (6) of the paper. It aims at penalizing cases where the routing between
    experts is too unbalanced.

    Args:
        gate_logits (Union[`torch.Tensor`, Tuple[torch.Tensor]):
            Logits from the `gate`, should be a tuple of model.config.num_hidden_layers tensors of
            shape [batch_size X sequence_length, num_experts].
        attention_mask (`torch.Tensor`, None):
            The attention_mask used in forward function
            shape [batch_size X sequence_length] if not None.
        num_experts (`int`, *optional*):
            Number of experts

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


# Copied from transformers.models.llama.modeling_llama._get_unpad_data
def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->PhiMoE
##https://dl.acm.org/doi/pdf/10.5555/3454287.3455397 The following is the implementation of layernorm


class PhiMoERotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)

        freqs = torch.outer(t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class Phi3LongRoPEScaledRotaryEmbedding(nn.Module):

    def __init__(self, dim, config):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = config.max_position_embeddings
        self.base = config.rope_theta
        self.short_factor = config.rope_scaling["short_factor"]
        self.long_factor = config.rope_scaling["long_factor"]
        self.short_mscale = config.rope_scaling["short_mscale"]
        self.long_mscale = config.rope_scaling["long_mscale"]
        self.original_max_position_embeddings = config.rope_scaling["original_max_position_embeddings"]

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[-2]

        if seq_len > self.original_max_position_embeddings:
            rescale_factors = torch.tensor(self.long_factor, dtype=torch.float32, device=x.device)
            mscale = self.long_mscale
        else:
            rescale_factors = torch.tensor(self.short_factor, dtype=torch.float32, device=x.device)
            mscale = self.short_mscale
        assert rescale_factors.shape == (self.dim // 2, ), \
            f"misaligned shape for LongRoPE rescale factors: {rescale_factors.shape}"

        inv_freq = 1.0 / (rescale_factors * (self.base ** (torch.arange(0, self.dim, 2).float().to(x.device) / self.dim)))

        t = torch.arange(seq_len, device=x.device, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)

        emb = torch.cat((freqs, freqs), dim=-1)
        return (emb.cos() * mscale).to(x.dtype), (emb.sin() * mscale).to(x.dtype)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)



def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)



class PhiMoEAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """

    def __init__(self, config: PhiMoEConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=self.config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=self.config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=self.config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=self.config.attention_bias)

        if getattr(config, 'rope_scaling', None) is None:
            self.rotary_emb = PhiMoERotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            if scaling_type == "longrope":
                self.rotary_emb = Phi3LongRoPEScaledRotaryEmbedding(self.head_dim, self.config)
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        
        # print ("before apply rotary pos_emb", len(kv_seq_len),torch.norm(value_states).items(),\
        #         torch.norm(query_states).items(), torch.norm(key_states).items(), position_ids)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        # print ('after pos emb', torch.norm(query_states).item(), torch.norm(key_states).items())
        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value



class PhiMoEFlashAttention2(PhiMoEAttention):
    """
    PhiMoE flash attention module. This module inherits from `PhiMoEAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    # Copied from transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ):
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )

            # overwrite attention_mask with padding_mask
            attention_mask = kwargs.pop("padding_mask")
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

        # Because the input can be padded, the absolute sequence length depends on the max position id.
        rotary_seq_len = max(kv_seq_len, position_ids[:, -1].max().item() + 1)
        cos, sin = self.rotary_emb(value_states, seq_len=rotary_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        use_sliding_windows = (
            _flash_supports_window_size
            and getattr(self.config, "sliding_window", None) is not None
            and kv_seq_len > self.config.sliding_window
        )

        if not _flash_supports_window_size:
            logger.warning_once(
                "The current flash attention version does not support sliding window attention, for a more memory efficient implementation"
                " make sure to upgrade flash-attn library."
            )

        if past_key_value is not None:
            # Activate slicing cache only if the config has a value `sliding_windows` attribute
            cache_has_contents = past_key_value.get_seq_length(self.layer_idx) > 0
            if (
                getattr(self.config, "sliding_window", None) is not None
                and kv_seq_len > self.config.sliding_window
                and cache_has_contents
            ):
                slicing_tokens = 1 - self.config.sliding_window

                past_key = past_key_value[self.layer_idx][0]
                past_value = past_key_value[self.layer_idx][1]

                past_key = past_key[:, :, slicing_tokens:, :].contiguous()
                past_value = past_value[:, :, slicing_tokens:, :].contiguous()

                if past_key.shape[-2] != self.config.sliding_window - 1:
                    raise ValueError(
                        f"past key must have a shape of (`batch_size, num_heads, self.config.sliding_window-1, head_dim`), got"
                        f" {past_key.shape}"
                    )

                if attention_mask is not None:
                    attention_mask = attention_mask[:, slicing_tokens:]
                    attention_mask = torch.cat([attention_mask, torch.ones_like(attention_mask[:, -1:])], dim=-1)

            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # Reashape to the expected shape for Flash Attention
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        attn_output = self._flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            use_sliding_windows=use_sliding_windows,
        )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    def _flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        dropout=0.0,
        softmax_scale=None,
        use_sliding_windows=False,
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            attention_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`float`):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
            use_sliding_windows (`bool`, *optional*):
                Whether to activate sliding window attention.
        """
        if not self._flash_attn_uses_top_left_mask:
            causal = self.is_causal
        else:
            # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
            causal = self.is_causal and query_length != 1

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, attention_mask, query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            if not use_sliding_windows:
                attn_output_unpad = flash_attn_varlen_func(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_in_batch_q,
                    max_seqlen_k=max_seqlen_in_batch_k,
                    dropout_p=dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                )
            else:
                attn_output_unpad = flash_attn_varlen_func(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_in_batch_q,
                    max_seqlen_k=max_seqlen_in_batch_k,
                    dropout_p=dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=(self.config.sliding_window, 0),
                )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
        else:
            if not use_sliding_windows:
                attn_output = flash_attn_func(
                    query_states,
                    key_states,
                    value_states,
                    dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                )
            else:
                attn_output = flash_attn_func(
                    query_states,
                    key_states,
                    value_states,
                    dropout,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    window_size=(self.config.sliding_window, 0),
                )

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, attention_mask, query_length):
        batch_size, kv_seq_len, num_heads, head_dim = key_layer.shape

        # On the first iteration we need to properly re-create the padding mask
        # by slicing it on the proper place
        if kv_seq_len != attention_mask.shape[-1]:
            attention_mask_num_tokens = attention_mask.shape[-1]
            attention_mask = attention_mask[:, attention_mask_num_tokens - kv_seq_len :]

        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(attention_mask)

        key_layer = index_first_axis(key_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)
        value_layer = index_first_axis(value_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k)

        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            attention_mask = attention_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(query_layer, attention_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )



class PhiMoESdpaAttention(PhiMoEAttention):
    """
    PhiMoE attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `PhiMoEAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from PhiMoEAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "PhiMoEModel is using PhiMoESdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
            is_causal=self.is_causal and attention_mask is None and q_len > 1,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_value


PHIMOE_ATTENTION_CLASSES = {
    "eager": PhiMoEAttention,
    "flash_attention_2": PhiMoEFlashAttention2,
    "sdpa": PhiMoESdpaAttention,
}


class PhiMoEBlockSparseTop2MLP(nn.Module):
    def __init__(self, config: PhiMoEConfig):
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


class PhiMoEBLockSparseTop2MLP(PhiMoEBlockSparseTop2MLP):
    def __init__(self, *args, **kwargs):
        logger.warning_once(
            "PhiMoEBLockSparseTop2MLP is deprecated by PhiMoEBlockSparseTop2MLP and will be removed in v4.40."
        )
        super().__init__(*args, **kwargs)


class mp(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, 
        scores: torch.Tensor, 
        multiplier: torch.Tensor, 
        selected_experts: torch.Tensor,
        masked_gates: torch.Tensor,
        mask_for_one: torch.Tensor,
    ):
        ctx.save_for_backward(multiplier, selected_experts, masked_gates)
        return multiplier * mask_for_one
        
    @staticmethod
    def backward(
        ctx, 
        grad_at_output: torch.Tensor, 
    ):
        multiplier, selected_experts, masked_gates = ctx.saved_tensors
        
        grad_at_output = grad_at_output * multiplier
        
        grad_at_scores_expaned = masked_gates * grad_at_output.mul(-1)
        grad_at_scores_expaned.scatter_add_(
            dim=-1,
            index=selected_experts,
            src=grad_at_output,
        )
        
        return (
            grad_at_scores_expaned, 
            None, 
            None, 
            None, 
            None, 
        )
    
def sparsemixer(scores, top_k, jitter_eps, training):
    # 确保 top_k 为 1
    # assert top_k == 1, "Modified sparsemixer only supports top-1 selection"
    if top_k == 1:
        ################ First Expert (Top-1) ################
        
        with torch.no_grad():
            # 计算稀疏掩码
            mask_logits_threshold, max_ind = scores.max(dim=-1, keepdim=True)
            factor = scores.abs().clamp(min=mask_logits_threshold)
            mask_logits_threshold = (
                (mask_logits_threshold - scores) / factor
            ) > (2 * jitter_eps)

        # 应用掩码
        masked_gates = scores.masked_fill(mask_logits_threshold, float('-inf'))
        if training:
            # 使用 Gumbel 采样选择专家
            selected_experts = (
                masked_gates - torch.empty_like(masked_gates, memory_format=torch.legacy_contiguous_format).exponential_().log()
            ).max(dim=-1)[1].unsqueeze(-1)  # Gumbel 采样
        else:
            selected_experts = max_ind
            
        # 计算梯度所需的 scores
        masked_gates = torch.softmax(masked_gates, dim=-1)
        multiplier_o = masked_gates.gather(dim=-1, index=selected_experts)
        
        if training:
            # 计算中点掩码（用于 Heun's third-order 方法）
            max_scores, max_ind = masked_gates.max(dim=-1, keepdim=True)
            mask_for_one = torch.logical_or(
                selected_experts == max_ind,
                torch.rand_like(max_scores) > 0.75  # Heun's third-order 方法
            )
            # 1 -> 1.0 & 0 -> 1./3: lambda x: (x + 0.5) / 1.5
            mask_for_one = torch.add(0.3333, mask_for_one, alpha=0.6667).type_as(masked_gates)

            multiplier = mp.apply(
                scores,
                multiplier_o,
                selected_experts,
                masked_gates,
                mask_for_one,
            )
        else:
            multiplier = multiplier_o

        return multiplier, selected_experts

    elif top_k == 2:
    
        ################ first expert ################
        
        with torch.no_grad():
            # compute mask for sparsity
            mask_logits_threshold, max_ind = scores.max(dim=-1, keepdim=True)
            factor = scores.abs().clamp(min=mask_logits_threshold)
            mask_logits_threshold = (
                (mask_logits_threshold - scores) / factor
            ) > (2 * jitter_eps)

        # apply mask 
        masked_gates = scores.masked_fill(mask_logits_threshold, float('-inf'))
        if training:
            selected_experts = (
                masked_gates - torch.empty_like(masked_gates, memory_format=torch.legacy_contiguous_format).exponential_().log()
            ).max(dim=-1)[1].unsqueeze(-1) # gumbel sampling, more robust than than the multinomial method
        else:
            selected_experts = max_ind
            
        # compute scores for gradients
        masked_gates = torch.softmax(masked_gates, dim=-1)
        multiplier_o = masked_gates.gather(dim=-1, index=selected_experts)
        
        if training:
            # compute midpoint mask 
            max_scores, max_ind = masked_gates.max(dim=-1, keepdim=True)
            mask_for_one = torch.logical_or(
                selected_experts == max_ind,
                torch.rand_like(max_scores) > 0.75 # Heun's third-order method: f(x) - f(0) = .25 f'(x) + .75 f'(x/3.)
            ) 
            # 1 -> 1.0 & 0 -> 1./3: lambda x: (x + 0.5) / 1.5
            mask_for_one = torch.add(0.3333, mask_for_one, alpha=0.6667).type_as(masked_gates)

            multiplier = mp.apply(
                scores, 
                multiplier_o, 
                selected_experts, 
                masked_gates, 
                mask_for_one,
            )
        else:
            multiplier = multiplier_o

        # masked out first expert 
        masked_scores = torch.scatter(
            scores,
            -1,
            selected_experts,
            float('-inf'),
        )
        with torch.no_grad():
            # compute mask for sparsity
            mask_logits_threshold, max_ind = masked_scores.max(dim=-1, keepdim=True)
            factor = scores.abs().clamp(min=mask_logits_threshold)
            mask_logits_threshold = (
                (mask_logits_threshold - scores) / factor
            ) > (2 * jitter_eps)

        # apply mask 
        masked_gates_top2 = masked_scores.masked_fill(mask_logits_threshold, float('-inf'))
        if training:
            selected_experts_top2 = (
                masked_gates_top2 - torch.empty_like(masked_gates_top2, memory_format=torch.legacy_contiguous_format).exponential_().log()
            ).max(dim=-1)[1].unsqueeze(-1) # gumbel sampling, more robust than than the multinomial method
        else:
            selected_experts_top2 = max_ind
        # compute scores for gradients
        masked_gates_top2 = torch.softmax(masked_gates_top2, dim=-1)
        multiplier_top2_o = masked_gates_top2.gather(dim=-1, index=selected_experts_top2)
        
        if training: 
            # compute midpoint mask 
            max_scores, max_ind = masked_gates_top2.max(dim=-1, keepdim=True)
            mask_for_one_top2 = torch.logical_or(
                selected_experts_top2 == max_ind,
                torch.rand_like(max_scores).uniform_() > 0.75 # Heun's third-order method: f(x) - f(0) = .25 f'(x) + .75 f'(x/3.)
            ) 
            # 1 -> 1.0 & 0 -> 1./3: lambda x: (x + 0.5) / 1.5
            mask_for_one_top2 = torch.add(0.3333, mask_for_one_top2, alpha=0.6667).type_as(masked_gates_top2)

            multiplier_top2 = mp.apply(
                scores, 
                multiplier_top2_o, 
                selected_experts_top2, 
                masked_gates_top2, 
                mask_for_one_top2,
            )
        else:
            multiplier_top2 = multiplier_top2_o
        
        multiplier = torch.concat((multiplier, multiplier_top2), dim=-1)
        selected_experts = torch.concat((selected_experts, selected_experts_top2), dim=-1)
        
        return (
            multiplier, 
            selected_experts,
        )

iterations = 0
class PhiMoESparseMoeBlock(nn.Module):
    """
    This implementation is
    strictly equivalent to standard MoE with full capacity (no
    dropped tokens). It's faster since it formulates MoE operations
    in terms of block-sparse operations to accomodate imbalanced
    assignments of tokens to experts, whereas standard MoE either
    (1) drop tokens at the cost of reduced performance or (2) set
    capacity factor to number of experts and thus waste computation
    and memory on padding.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        global iterations
        iterations +=1
        self.iter = iterations
        # gating
        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=False)

        self.experts = nn.ModuleList([PhiMoEBlockSparseTop2MLP(config) for _ in range(self.num_experts)])

        # Jitter parameters
        self.router_jitter_noise = config.router_jitter_noise
        self.input_jitter_noise = config.input_jitter_noise
        
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
        
        #pre
        self.last_topk_idx = None

        # Cache Simulators
        cache_ratio = self.config.cache_ratio
        self.cache_capacity = int(cache_ratio*self.config.num_local_experts) # 缓存容量
        print(f"Cache capacity: {self.cache_capacity}")
        self.lru_cache_sim = CacheSimulator(capacity=self.cache_capacity, strategy='lru',max_expert_id=self.config.num_local_experts)
        self.fifo_cache_sim = CacheSimulator(capacity=self.cache_capacity, strategy='fifo',max_expert_id=self.config.num_local_experts)
        self.lfu_cache_sim = CacheSimulator(capacity=self.cache_capacity, strategy='lfu',max_expert_id=self.config.num_local_experts)
        self.lru_cache_sim_real = CacheSimulator(capacity=self.cache_capacity, strategy='lru',max_expert_id=self.config.num_local_experts)
        self.fifo_cache_sim_real = CacheSimulator(capacity=self.cache_capacity, strategy='fifo',max_expert_id=self.config.num_local_experts)
        self.lfu_cache_sim_real = CacheSimulator(capacity=self.cache_capacity, strategy='lfu',max_expert_id=self.config.num_local_experts)

        
        
    def forward(self, hidden_states: torch.Tensor, prediction_for_current_token_scores: Optional[torch.Tensor] = None) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        if self.training and self.input_jitter_noise > 0:
            hidden_states *= torch.empty_like(hidden_states).uniform_(1.0 - self.input_jitter_noise, 1.0 + self.input_jitter_noise)
        hidden_states = hidden_states.view(-1, hidden_dim)

        router_logits = self.gate(hidden_states)

        routing_weights, selected_experts = sparsemixer(
            router_logits, 
            top_k=self.top_k, 
            jitter_eps=self.router_jitter_noise, 
            training=self.training,
        )

        self.last_topk_idx = selected_experts.detach()

        if(sequence_length==1):#只统计decode阶段
                    # 统计路由专家激活次数
            for i in range(batch_size * sequence_length):
                scores = router_logits.softmax(dim=-1, dtype=torch.float32)
                self.token_frequencies[self.total_tokens+1] = scores[i].cpu().detach().numpy()
                current_token_activated_experts = set() # To avoid double counting for a single token if k > 1

                for k in range(self.top_k):
                    expert_idx = selected_experts[i, k].item()
                    current_token_activated_experts.add(expert_idx) # Add to set for this token


                    self.top_avg[k] += scores[i, expert_idx].item()
                    self.expert_activation_counts[expert_idx] += 1
                    
                    if self.continue_cnt[expert_idx]>=1:#被cache
                        self.cache_hit_cnt+=1 # 缓存命中次数+1
                    self.continue_cnt[expert_idx] +=1 #连续次数+1
                    # self.expert_activation_counts_max_continue[expert_idx]=max(self.expert_activation_counts_max_continue[expert_idx],self.continue_cnt[expert_idx])
                    
                # --- Predictive Cache Logic --- 模拟t-1的预测当前t的cache的影响
                if prediction_for_current_token_scores is not None:
                    # prediction_for_current_token_scores has shape [bsz, 1, num_local_experts]
                    current_item_prediction_scores = prediction_for_current_token_scores[i].squeeze(0) # Shape [num_local_experts]
                    num_to_predict = min(self.cache_capacity, int(self.config.num_experts_per_tok*self.config.pre_ratio)) # 预测的专家数量
                    if current_item_prediction_scores.numel() > 0 : 
                        predicted_expert_indices_for_token = torch.topk(current_item_prediction_scores, k=num_to_predict, dim=-1,sorted = True)[1]
                        for predicted_expert_idx in predicted_expert_indices_for_token[:num_to_predict].tolist():
                            # "Update" or "Prime" the predictive cache based on prediction
                            self.lru_cache_sim.update(predicted_expert_idx)
                            self.fifo_cache_sim.update(predicted_expert_idx)
                            self.lfu_cache_sim.update(predicted_expert_idx)


                # Access cache simulators for each unique expert activated by this token
                for expert_idx_sim in current_token_activated_experts:
                    self.lru_cache_sim.access(expert_idx_sim)
                    self.fifo_cache_sim.access(expert_idx_sim)
                    self.lfu_cache_sim.access(expert_idx_sim)
                    self.lru_cache_sim_real.access(expert_idx_sim)
                    self.fifo_cache_sim_real.access(expert_idx_sim)
                    self.lfu_cache_sim_real.access(expert_idx_sim)


                for k  in range(self.num_experts):
                    if k not in set(selected_experts[i].tolist()):# 没被激活的
                        self.continue_cnt[k] = 0 #连续中断,offload
                    
            self.cache_request_cnt += self.top_k # 缓存请求次数
            self.total_tokens += 1
            self.total_tokens_nor +=1

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0:
                continue

            # in torch it is faster to index using lists than torch tensors
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x_list].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x_list, idx_list, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        # print ( 'moe', self.iter, torch.norm(final_hidden_states).item())
        return final_hidden_states, router_logits

    # Add methods to get stats from simulators
    def get_lru_cache_stats(self):
        return self.lru_cache_sim.get_stats()

    def get_fifo_cache_stats(self):
        return self.fifo_cache_sim.get_stats()

    def get_lfu_cache_stats(self):
        return self.lfu_cache_sim.get_stats()

    def get_lru_cache_real_stats(self):
        return self.lru_cache_sim_real.get_stats()

    def get_fifo_cache_real_stats(self):
        return self.fifo_cache_sim_real.get_stats()
    
    def get_lfu_cache_real_stats(self):
        return self.lfu_cache_sim_real.get_stats()
    

    def reset_cache_simulators(self):
        self.lru_cache_sim.reset()
        self.fifo_cache_sim.reset()
        self.lfu_cache_sim.reset()
        self.lru_cache_sim_real.reset()
        self.fifo_cache_sim_real.reset()
        self.lfu_cache_sim_real.reset()
    
    def random_fill_cache(self):
        self.lru_cache_sim.random_fill()
        self.fifo_cache_sim.random_fill()
        self.lfu_cache_sim.random_fill()
        self.lru_cache_sim_real.random_fill()
        self.fifo_cache_sim_real.random_fill()
        self.lfu_cache_sim_real.random_fill()

    def get_expert_frequencies(self):
        """计算每个专家的激活概率"""
        if self.total_tokens_nor == 0:
            return  {}
        routed_frequencies = {}
        # 路由专家的激活概率
        for expert_idx in range(self.num_experts):
            count = self.expert_activation_counts[expert_idx]
            frequency = count / (self.total_tokens_nor)  
            routed_frequencies[expert_idx] = frequency
        # 共享专家的激活概率（总是 1.0，因为每个 token 都会经过共享专家）
        return  routed_frequencies

    def reset_counts(self):
        """重置计数器"""
        self.expert_activation_counts.clear()
        self.total_tokens_nor = 0
        
        
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
    #     self.total_tokens = 0
        
    def get_expert_hit_rate(self):
        # routed_hit_rate = {}
        # for expert_idx in range(self.config.num_local_experts):
        #     # routed_hit_rate[expert_idx] = self.cache_hit_cnt[expert_idx]/self.total_tokens
        #     routed_hit_rate[expert_idx] = self.cache_hit_cnt[expert_idx]/self.expert_activation_counts[expert_idx]
        # # 共享专家的激活概率（总是 1.0，因为每个 token 都会经过共享专家）
        return  self.cache_hit_cnt/self.cache_request_cnt #缓存命中次数/缓存请求次数
    def reset_hit_counts(self):
        """重置缓存命中次数，缓存请求次数，缓存连续队列"""
        self.cache_hit_cnt=0
        self.cache_request_cnt = 0
        self.continue_cnt.clear()
        
    def get_token_frequency(self):
        # 共享专家的激活概率（总是 1.0，因为每个 token 都会经过共享专家）
        return  self.token_frequencies
    
    def reset_token_frequency(self):
        """重置计数器"""
        self.token_frequencies.clear()
        self.total_tokens = 0

class PhiMoEDecoderLayer(nn.Module):
    def __init__(self, config: PhiMoEConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        self.self_attn = PHIMOE_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)

        self.block_sparse_moe = PhiMoESparseMoeBlock(config)
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps, elementwise_affine=True)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps, elementwise_affine=True)

        self.predictor = Predictor(config)

        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        prediction_for_current_token_scores: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
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
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states_input = self.post_attention_layernorm(hidden_states)
        
        #pos2
        predictor_output = None
        if self.predictor is not None :
            predictor_output = self.predictor(hidden_states_input.float())
        
        hidden_states, router_logits = self.block_sparse_moe(hidden_states_input,prediction_for_current_token_scores)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_router_logits:
            outputs += (router_logits,)
            
        if predictor_output is not None:
            outputs += (predictor_output,)

        return outputs


PHIMOE_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`PhiMoEConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare PhiMoE Model outputting raw hidden-states without any specific head on top.",
    PHIMOE_START_DOCSTRING,
)

class PhiMoEPreTrainedModel(PreTrainedModel):
    config_class = PhiMoEConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["PhiMoEDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        pass
        # std = self.config.initializer_range
        # if isinstance(module, nn.Linear):
        #     module.weight.data.normal_(mean=0.0, std=std)
        #     if module.bias is not None:
        #         module.bias.data.zero_()
        # elif isinstance(module, nn.Embedding):
        #     module.weight.data.normal_(mean=0.0, std=std)
        #     if module.padding_idx is not None:
        #         module.weight.data[module.padding_idx].zero_()


PHIMOE_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `decoder_input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

            Contains pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        output_router_logits (`bool`, *optional*):
            Whether or not to return the logits of all the routers. They are useful for computing the router loss, and
            should not be returned during inference.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare PhiMoE Model outputting raw hidden-states without any specific head on top.",
    PHIMOE_START_DOCSTRING,
)

class PhiMoEModel(PhiMoEPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`PhiMoEDecoderLayer`]

    Args:
        config: PhiMoEConfig
    """

    def __init__(self, config: PhiMoEConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [PhiMoEDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._attn_implementation = config._attn_implementation
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps, elementwise_affine=True)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()
        self.pre_dis = 0
        self.pre_ahead = 1
        
        self.iou_scores = []  # 存储每一层的 IoU
        self.accuracy_scores = []
        self.tokens = 0  # 统计总的 token 数量

        # 添加存储前一 token predictor 输出的属性
        self.previous_predictor_outputs = [None] * config.num_hidden_layers
        
    # block_sparse_moe

    def get_all_lru_cache_stats(self):
        all_stats = {}
        for layer_idx, layer in enumerate(self.layers):
            if hasattr(layer, 'block_sparse_moe') and hasattr(layer.block_sparse_moe, 'get_lru_cache_stats'):
                all_stats[layer_idx] = layer.block_sparse_moe.get_lru_cache_stats()
        return all_stats
    
    def get_all_fifo_cache_stats(self):
        all_stats = {}
        for layer_idx, layer in enumerate(self.layers):
            if hasattr(layer, 'block_sparse_moe') and hasattr(layer.block_sparse_moe, 'get_fifo_cache_stats'):
                all_stats[layer_idx] = layer.block_sparse_moe.get_fifo_cache_stats()
        return all_stats

    def get_all_lfu_cache_stats(self):
        all_stats = {}
        for layer_idx, layer in enumerate(self.layers):
            if hasattr(layer, 'block_sparse_moe') and hasattr(layer.block_sparse_moe, 'get_lfu_cache_stats'):
                all_stats[layer_idx] = layer.block_sparse_moe.get_lfu_cache_stats()
        return all_stats

    def get_all_lru_cache_real_stats(self):
        all_stats = {}
        for layer_idx, layer in enumerate(self.layers):
            if hasattr(layer, 'block_sparse_moe') and hasattr(layer.block_sparse_moe, 'get_lru_cache_real_stats'):
                all_stats[layer_idx] = layer.block_sparse_moe.get_lru_cache_real_stats()
        return all_stats
    
    def get_all_fifo_cache_real_stats(self):
        all_stats = {}
        for layer_idx, layer in enumerate(self.layers):
            if hasattr(layer, 'block_sparse_moe') and hasattr(layer.block_sparse_moe, 'get_fifo_cache_real_stats'):
                all_stats[layer_idx] = layer.block_sparse_moe.get_fifo_cache_real_stats()
        return all_stats
    
    def get_all_lfu_cache_real_stats(self):
        all_stats = {}
        for layer_idx, layer in enumerate(self.layers):
            if hasattr(layer, 'block_sparse_moe') and hasattr(layer.block_sparse_moe, 'get_lfu_cache_real_stats'):
                all_stats[layer_idx] = layer.block_sparse_moe.get_lfu_cache_real_stats()
        return all_stats
    
    def reset_all_cache_simulators(self):
        for layer_idx, layer in enumerate(self.layers):
             if hasattr(layer, 'block_sparse_moe') and hasattr(layer.block_sparse_moe, 'reset_cache_simulators'):
                layer.block_sparse_moe.reset_cache_simulators()
    
    def random_fill_all_cache(self):
        for layer_idx, layer in enumerate(self.layers):
             if hasattr(layer, 'block_sparse_moe') and hasattr(layer.block_sparse_moe, 'random_fill_cache'):
                layer.block_sparse_moe.random_fill_cache()

    def get_all_expert_frequencies(self):
        """获取所有层的专家激活概率"""
        all_frequencies = {}
        for layer_idx, layer in enumerate(self.layers):
            moe_block = layer.block_sparse_moe  # 假设每层有一个 moe_block 属性
            routed_freq = moe_block.get_expert_frequencies()
            all_frequencies[layer_idx] = {"routed": routed_freq}
            # print(f'sum_nor:{moe_block.total_tokens_nor}')
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
            
    def get_all_expert_hit_rate(self):
        """获取所有层的专家连续激活次数"""
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


    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # Ignore copy
    @add_start_docstrings_to_model_forward(PHIMOE_INPUTS_DOCSTRING)
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

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        past_key_values_length = 0

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`transformers."
                )
                use_cache = False

        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if attention_mask is not None and self._attn_implementation == "flash_attention_2" and use_cache:
            is_padding_right = attention_mask[:, -1].sum().item() != batch_size
            if is_padding_right:
                raise ValueError(
                    "You are attempting to perform batched generation with padding_side='right'"
                    " this may lead to unexpected behaviour for Flash Attention version of PhiMoE. Make sure to "
                    " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                )
                
        ori_attention_mask = attention_mask
        if self._attn_implementation == "flash_attention_2":
            # 2d mask is passed through the layers
            attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
            
        elif self._attn_implementation == "sdpa" and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
            )
        else:
            # 4d mask is passed through the layers
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                (batch_size, seq_length),
                inputs_embeds,
                past_key_values_length,
                sliding_window=self.config.sliding_window,
            )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None
        next_decoder_cache = None

        # pre_hidden = None  # 收集每一层的预测器输出
        all_topk_idx = []  # 收集每一层的 topk_idx
        all_scores = []
        all_predictor_outputs = []  # 收集每一层的预测器输出


        sequence_length = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        # 逐层处理
        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if sequence_length != 1:
                prediction_for_current_token_scores = None
            else:
                prediction_for_current_token_scores = self.previous_predictor_outputs[layer_idx]


            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    output_router_logits,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    output_router_logits=output_router_logits,
                    use_cache=use_cache,
                    prediction_for_current_token_scores=prediction_for_current_token_scores,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            router_logits_idx = 3 if output_attentions and use_cache else (
                2 if use_cache or output_attentions else 1
            )
            if output_router_logits and len(layer_outputs) > router_logits_idx:
                all_router_logits += (layer_outputs[router_logits_idx],)
            

            self.previous_predictor_outputs[layer_idx] = layer_outputs[-1][:,-1,:]
            all_topk_idx.append(decoder_layer.block_sparse_moe.last_topk_idx)
            all_predictor_outputs.append(layer_outputs[-1]) 
            

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # 在训练时计算预测损失 Todo:什么意思
        predictor_loss = None
        predict_dis = 1
        if self.training:
            predictor_losses = []
            # print(len(all_scores))
            # print(len(all_scores[0]))
            for i, (predictor_output, moe_topk_idx) in enumerate(zip(all_predictor_outputs, all_topk_idx)):
                if predictor_output is not None and moe_topk_idx is not None:
                    # 预测器输出 logits，形状为 [batch_size, seq_len, num_local_experts]
                    pred_logits = predictor_output[:, :-predict_dis]  # 当前 token 预测下一个 token，去掉最后一个
                    # 真实 top-k 索引，形状为 [batch_size * seq_len, top_k]
                    true_topk_idx = moe_topk_idx.view(predictor_output.shape[0], -1, self.config.num_experts_per_tok)[:, predict_dis:]  # 下一个 token 的真实值，去掉第一个
                    batch_size, seq_length = pred_logits.shape[0], pred_logits.shape[1]
                    
                    # 将 true_topk_idx 转换为 one-hot 编码S
                    true_topk_onehot = torch.zeros_like(pred_logits, dtype=torch.float32)
                    for k in range(self.config.num_experts_per_tok):
                        true_topk_onehot.scatter_(2, true_topk_idx[:, :, k].view(batch_size, seq_length, 1), 1)
                    # 使用 BCEWithLogitsLoss 计算损失
                    loss_fct = nn.BCEWithLogitsLoss(reduction='none')
                    pred_logits = torch.nan_to_num(pred_logits, nan=0.0)

                    loss = loss_fct(pred_logits.reshape(-1, self.config.num_local_experts).float(), 
                        true_topk_onehot.reshape(-1, self.config.num_local_experts).float())
                    mask = ori_attention_mask[:, predict_dis:].reshape(-1, 1).expand_as(loss)  # 去掉第一个 token 的 mask
                    masked_loss = loss.to(mask.device) * mask  # 屏蔽 <pad> token 的损失
                    loss = masked_loss.sum() / mask.sum()
                    predictor_losses.append(loss)
                    
            if predictor_losses:
                predictor_loss = sum(predictor_losses) / len(predictor_losses)
            
        elif self.training == False and all_predictor_outputs[1].shape[0]*all_predictor_outputs[1].shape[1]!=1:#评测prefill 阶段，但是没有适配decode阶段
            self.iou_scores = []  # 存储每一层的 IoU
            self.accuracy_scores = []  # 存储每一层的准确率
            predictor_losses = []  # 存储每一层的预测损失

            for i in range(len(self.layers)):
                if all_predictor_outputs[i] is not None and all_topk_idx[i] is not None:  # 当前层有预测和真实值
                    # 预测器输出 logits，形状为 [batch_size, seq_len, num_local_experts]
                    pred_logits = all_predictor_outputs[i][:, :-predict_dis]  # 当前 token 预测下一个 token，去掉最后一个
                    true_topk_idx = all_topk_idx[i].view(pred_logits.shape[0], -1, self.config.num_experts_per_tok)[:, predict_dis:]  # 下一 token 的真实值，去掉第一个
                    batch_size, seq_length = pred_logits.shape[0], pred_logits.shape[1]

                    # 将 true_topk_idx 转换为 one-hot 编码，形状为 [batch_size, seq_length, num_local_experts]
                    true_topk_onehot = torch.zeros_like(pred_logits)
                    # print("true_topk_onehot max:", true_topk_onehot.max().item(), "min:", true_topk_onehot.min().item())
                    for k in range(self.config.num_experts_per_tok):
                        true_topk_onehot.scatter_(2, true_topk_idx[:, :, k].view(batch_size, seq_length, 1), 1)

                    # 使用 BCEWithLogitsLoss 计算损失
                    loss_fct = nn.BCEWithLogitsLoss(reduction='none')
                
                    loss = loss_fct(pred_logits.reshape(-1, self.config.num_local_experts).float(), 
                                    true_topk_onehot.reshape(-1, self.config.num_local_experts).float())
                    mask = ori_attention_mask[:, predict_dis:].reshape(-1, 1).expand_as(loss)  # [batch_size * seq_len - 1, num_local_experts]
                    masked_loss = loss.to(mask.device) * mask  # 屏蔽 <pad> token 的损失
                    loss = masked_loss.sum() / mask.sum()  # 平均有效 token 的损失
                    predictor_losses.append(loss)

                    # 计算 IoU
                    pred_topk = torch.topk(pred_logits, k=self.config.num_experts_per_tok, dim=-1)[1]  # [batch_size, seq_len, top_k]
                    
                    pred_mask = torch.zeros_like(pred_logits).scatter_(2, pred_topk, 1)  # [batch_size, seq_len, num_local_experts]
                    intersection = (pred_mask * true_topk_onehot).sum(dim=-1)  # [batch_size, seq_len]
                    
                    correct = intersection/self.config.num_experts_per_tok
                    masked_accuracy = (correct.float().to(ori_attention_mask.device) * ori_attention_mask[:, predict_dis:]).sum() / ori_attention_mask[:, predict_dis:].sum()  # 只计算有效 token 的准确率                                        
                    
                    union = pred_mask.sum(dim=-1) + true_topk_onehot.sum(dim=-1) - intersection  # [batch_size, seq_len]
                    iou = intersection / (union + 1e-8)  # 避免除以零，[batch_size, seq_len]
                    masked_iou = (iou.to(ori_attention_mask.device) * ori_attention_mask[:, predict_dis:]).sum() / ori_attention_mask[:, predict_dis:].sum()  # 只计算有效 token 的 IoU
                    self.iou_scores.append(masked_iou.item())

                    
                    self.accuracy_scores.append(masked_accuracy.item())
                    self.tokens = ori_attention_mask[:, predict_dis:].sum().item()

            if predictor_losses:
                predictor_loss = sum(predictor_losses) / len(predictor_losses)
            # print(f"Average IoU across layers: {avg_iou:.4f}, Predictor Loss: {predictor_loss:.4f}")

        elif self.training == False and all_predictor_outputs[1].shape[0]*all_predictor_outputs[1].shape[1]==1:#评测prefill 阶段，但是没有适配decode阶段
            # print(f"Decode all_predictor_outputs shape: {all_predictor_outputs[1].shape}")
            predictor_loss = 0
            # print(f"Average IoU across layers: {avg_iou:.4f}, Predictor Loss: {predictor_loss:.4f}")



        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_router_logits]
                if v is not None
            )
        return MoeModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            router_logits=all_router_logits,
        ),predictor_loss


class PhiMoEForCausalLM(PhiMoEPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = PhiMoEModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=self.config.lm_head_bias)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_local_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        # Initialize weights and apply final processing
        self.post_init()

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

    @add_start_docstrings_to_model_forward(PHIMOE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=MoeCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    # Ignore copy
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
    ) -> Union[Tuple, MoeCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, transformers.,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, transformers., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, PhiMoEForCausalLM

        >>> model = PhiMoEForCausalLM.from_pretrained("microsoft/Phi-3.5-moe-instruct")
        >>> tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-moe-instruct")

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
        outputs,predictor_loss = self.model(
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
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        # loss = None
        # if labels is not None:
        #     # Shift so that tokens < n predict n
        #     shift_logits = logits[..., :-1, :].contiguous()
        #     shift_labels = labels[..., 1:].contiguous()
        #     # Flatten the tokens
        #     loss_fct = CrossEntropyLoss()
        #     shift_logits = shift_logits.view(-1, self.config.vocab_size)
        #     shift_labels = shift_labels.view(-1)
        #     # Enable model parallelism
        #     shift_labels = shift_labels.to(shift_logits.device)
        #     loss = loss_fct(shift_logits, shift_labels)

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


        loss = predictor_loss

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

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        output_router_logits=False,
        **kwargs,
    ):
        # When the first time input length reached long and short factor switching point, enforce re-compute cache
        # It will cause downside of slower at this single token position, however, better than current failure.
        if past_key_values and self.config.rope_scaling and input_ids.shape[1] >= self.config.original_max_position_embeddings + 1:
            past_length = past_key_values.seen_tokens if isinstance(past_key_values, Cache) else past_key_values[0][0].shape[2]
            if past_length <= self.config.original_max_position_embeddings:
                past_key_values = None
        
        # Omit tokens covered by past_key_values
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_cache_shape()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

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
                "output_router_logits": output_router_logits,
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


@add_start_docstrings(
    """
    The PhiMoE Model transformer with a sequence classification head on top (linear layer).

    [`PhiMoEForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    PHIMOE_START_DOCSTRING,
)
# Copied from transformers.models.llama.modeling_llama.LlamaForSequenceClassification with Llama->PhiMoE, LLAMA->PHIMOE
class PhiMoEForSequenceClassification(PhiMoEPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = PhiMoEModel(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    @add_start_docstrings_to_model_forward(PHIMOE_INPUTS_DOCSTRING)
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
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, transformers.,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )