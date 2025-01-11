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

class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
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
    
class LlamaRotaryEmbedding(nn.Module):
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
    def forward(self, x, position_ids):
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

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)