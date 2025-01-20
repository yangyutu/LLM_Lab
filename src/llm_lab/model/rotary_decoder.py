import math
from typing import List, Optional, Tuple, Union
import os
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
import json
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from functools import partial

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        # float32 is needed for numeric stability. float16 is not enough.
        hidden_states = hidden_states.to(torch.float32)
        # The variance of the hidden_states is computed along the last dimension using the pow(2).
        # mean(-1, keepdim=True) operations, which square the values, compute the mean, and 
        # retain the dimensions for broadcasting.
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.gamma * hidden_states.to(input_dtype)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2] # x1 is the first half of the hidden dims
    x2 = x[..., x.shape[-1] // 2 :] # x2 is the second half of the hidden dims
    return torch.cat((-x2, x1), dim=-1)

# cos, sin has shape of (batch_size, seq_len, dim)
# if q has shape of [batch_size, heads, seq_len, head_dim], use unsqueeze_dim=1
# if q has shape of [batch_size, seq_len, heads, head_dim], use unsqueeze_dim=2

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):

    # add a dimension to the cos and sin tensors to account for the number of heads
    # from (batch_size, seq_len, dim) to (batch_size, 1, seq_len, dim) or (batch_size, seq_len, 1, dim)
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    # Here has a different order in the frequency dimension, as described in the paper https://arxiv.org/pdf/2104.09864 page 7
    # in the paper, the order is 
    # [cos m theta 1, cos m theta 1, ..., cos m theta (d//2), cos m theta (d//2)]
    # and [sin m theta 1, sin m theta 1, ..., sin m theta (d//2), sin m theta (d//2)]
    # here the order is
    # [cos m theta 1, cos m theta 2, ...cos m theta (d//2), cos m theta 1, cos m theta 2, ...cos m theta (d//2)]
    # and [sin m theta 1, sin m theta 2, ...sin m theta (d//2), sin m theta 1, sin m theta 2, ...sin m theta (d//2)]
    # that is, the frequency order is permuted
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
    ):
        super().__init__()
        self.max_seq_len_cached = max_position_embeddings
        self.original_max_seq_len = max_position_embeddings

        #inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, **self.rope_kwargs)
        
        # inv freq is a tensor of shape (dim // 2)
        # (0, 1/10000^(2/dim),..., 1/10000^((dim-2)/dim))
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    def forward(self, position_ids, datatype):
        # Core RoPE block
        # Use None to add two new dimensions to the inv_freq
        # use expand to repeat the inv_freq along the batch dimension
        # inv_freq_expanded has shape (batch_size, dim // 2, 1), dim // 2 is the number of frequencies
        # position_ids_expanded has shape (batch_size, 1, seq_len)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        # inv_freq_expanded.float() @ position_ids_expanded.float() gives shape (batch_size, dim // 2, seq_len)
        # after transpose, we get (batch_size, seq_len, dim // 2)
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        # emb has shape (batch_size, seq_len, dim), the concat is on the frequency dimension
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()

        return cos.to(dtype=datatype), sin.to(dtype=datatype)

# utility function for Group query attention
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, seqlen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, seqlen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, seqlen, head_dim)


class GQAAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.causal_attention = config.causal_attention

        # Here supports GQA, which specifies the number of key value heads << num_heads
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.qkv_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.qkv_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.qkv_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.o_bias)
        
        if self.causal_attention:
            self.register_buffer("mask", torch.triu(torch.ones(config.max_position_embeddings, config.max_position_embeddings), diagonal=1))
        
        if config.get("use_cache", False):
            self.cache_k = torch.zeros((config.cache_max_batch_size, config.cache_max_seq_len, config.num_key_value_heads, self.head_dim))
            self.cache_v = torch.zeros((config.cache_max_batch_size, config.cache_max_seq_len, config.num_key_value_heads, self.head_dim))
            
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, 
        use_cache: bool = False,
        start_pos: int = 0
    ):
        bsz, q_len, _ = hidden_states.size()


        # projetion of the hidden states into query, key and value
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        # after transpose, the shape is (bsz, num_heads, q_len, head_dim)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim)

        # Get the rotary embeddings cosines and sines functions
        cos, sin = position_embeddings
               
        # apply the rotary embeddings to the query and key states
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, unsqueeze_dim=2)
        
        if use_cache:
            # update and retrieve kv caches
            self.cache_k = self.cache_k.to(key_states.device)
            self.cache_v = self.cache_v.to(value_states.device)
            
            self.cache_k[:bsz, start_pos: start_pos + q_len] = key_states
            self.cache_v[:bsz, start_pos: start_pos + q_len] = value_states
            
            key_states = self.cache_k[:bsz, : start_pos + q_len]
            value_states = self.cache_v[:bsz, : start_pos + q_len]

        # after transpose, the shape is (bsz, num_heads, q_len, head_dim)
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)


        # Copy kv for matching the number of heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        
        
        # applied scaled dot product attention
        # attn_weights has shape (batch_size, num_heads, seq_len, seq_len)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        # add mask
        if self.causal_attention:
            if not use_cache:
                mask = self.mask[:q_len, :q_len].bool()
            else:
                kv_len = start_pos + q_len
                mask = self.mask[kv_len - start_pos: kv_len, :kv_len ].bool()
            
            attn_weights.masked_fill(mask, -torch.inf)

        # upcast attention to fp32
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        # attn_output has shape (batch_size,  seq_len, num_heads, head_dim) after transpose
        attn_output = attn_output.transpose(1, 2).contiguous()
        # attn_output output has shape (batch_size, seq_len, num_heads * head_dim) after reshape
        # which is equivalent to concatenating the heads
        attn_output = attn_output.reshape(bsz, q_len, -1)

        # apply the output projection
        attn_output = self.o_proj(attn_output)

        return attn_output

class FeedForwardMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        # silu is the same as swish
        self.silu = torch.nn.SiLU()

    def forward(self, x):
        down_proj = self.down_proj(self.silu(self.gate_proj(x)) * self.up_proj(x))

        return down_proj
    
    
class RotaryDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = GQAAttention(config=config, layer_idx=layer_idx)
        # FFN layer
        self.mlp = FeedForwardMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        use_cache: bool = False,
        start_pos: int = 0
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
        """
        residual = hidden_states
        # pre layer norm
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            use_cache=use_cache,
            start_pos=start_pos
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        # pre layer norm before FFN layer
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
    
class RotaryDecoderModel(nn.Module):
    """
    Transformer decoder consisting of *config.num_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config):
        super().__init__()
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [RotaryDecoderLayer(config, layer_idx) for layer_idx in range(config.num_layers)]
        )
        
        # apply to last layer hidden state
        self.norm = RMSNorm(config.hidden_size, eps=config.get("rms_norm_eps",1e-6))
        # rotary embedding matrices are shared across the decoder layers
        self.rotary_emb = RotaryEmbedding( dim=config.hidden_size // config.num_heads,
                                                max_position_embeddings=config.max_position_embeddings,
                                                base=config.get("rope_theta", 10000),)

        
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: bool = False,
        start_pos: int = 0
    ):

        inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        if position_ids is None:
            bsz, seq_len = input_ids.shape
            position_ids = torch.arange(start=start_pos, end=start_pos + seq_len, dtype=torch.int64, device=hidden_states.device)
            position_ids = position_ids.expand(bsz, -1)
        position_embeddings = self.rotary_emb(position_ids, datatype=hidden_states.dtype)

        for decoder_layer in self.layers:

            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                use_cache=use_cache,
                start_pos=start_pos
            )

        hidden_states = self.norm(hidden_states)

        return hidden_states
