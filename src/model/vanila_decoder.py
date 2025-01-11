import math
from typing import List, Optional, Tuple, Union
import os, urllib
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from omegaconf import OmegaConf
import tiktoken

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, context_length, num_heads, dropout, qkv_bias=False):
        super().__init__()
        
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.W_Q = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_model, bias=qkv_bias)
        
        self.W_O = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Buffers are not updated during training but are part of the module's state.
        # Use diagonal= 1 to set diagonal elements to be zero
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))
        
    def forward(self, x):
        # x shape: (batch_size, num_tokens, d_model)
        (batch_size, num_tokens, d_model) = x.shape
        
        queries = self.W_Q(x)
        keys = self.W_K(x)
        values = self.W_V(x)
        
        # change to shape easier for multi-head attention
        # (batch_size, num_tokens, num_heads, d_head)
        keys = keys.view(batch_size, num_tokens, self.num_heads,  self.head_dim)
        queries = queries.view(batch_size, num_tokens, self.num_heads,  self.head_dim)
        values = values.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        
        # to shape: (batch_size, num_heads, num_tokens,  d_head)
        queries = queries.transpose(1, 2) # 
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        
        # key to shape: (batch_size, num_heads,  d_head, num_tokens)
        # attention_logits : (batch_size, num_heads, num_tokens, num_tokens)
        attention_logits = queries @ keys.transpose(2, 3) / queries.shape[-1] ** 0.5
        
        mask_bool = self.mask.bool()[:num_tokens,:num_tokens]
        
        attention_logits.masked_fill(mask_bool, -torch.inf)
        
        attention_weights = torch.softmax(attention_logits, dim=-1)
        
        # attention_weights: (batch_size, num_heads, num_tokens, num_tokens)
        attention_weights = self.dropout(attention_weights)
        
        # context_vec: (batch_size, num_heads, num_tokens, d_head)
        context_vec = (attention_weights @ values)
        
        context_vec = context_vec.transpose(2, 3).contiguous().view(batch_size, num_tokens, self.d_model)
        
        out_vec = self.W_O(context_vec)
        
        return out_vec
    
class LayerNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(d_model))
        self.shift = nn.Parameter(torch.zeros(d_model))
        
    def forward(self, x):
        mean = torch.mean(x, dim=-1, keepdim=True)
        var = torch.var(x, dim=-1, keepdim=True)
        norm_x = (x - mean) / (var + self.eps) * self.scale + self.shift
        return norm_x
    
class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.up_proj = nn.Linear(d_model, 4 * d_model)
        self.act = nn.GELU()
        self.down_proj = nn.Linear(4 * d_model, d_model)
        
    def forward(self, x):
        
        x = self.up_proj(x)
        x = self.act(x)
        x = self.down_proj(x)
        
        return x

class TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        self.att = MultiHeadAttention(d_model = config.d_model, 
                                      context_length=config.context_length, 
                                      num_heads=config.num_heads,
                                      dropout=config.dropout,
                                      qkv_bias=config.qkv_bias)
        
        self.ff = FeedForward(d_model=config.d_model)
        self.norm1 = LayerNorm(d_model=config.d_model)
        self.norm2 = LayerNorm(d_model=config.d_model)
        
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        
        shortcut = x
        # pre-norm
        x = self.norm1(x)
        x = self.att(x)
        x = self.dropout(x)
        
        x = x + shortcut
        
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout(x)
        x = x + shortcut
        
        return x
    
class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # both token embedding and position embedding are learnable
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.context_length, config.d_model)
        self.register_buffer("position_ids", torch.arange(config.context_length).view(1, -1))
        self.emb_dropout = nn.Dropout(config.dropout)
        
        self.transformer_backbone = nn.Sequential(*[TransformerLayer(self.config) for _ in range(self.config.num_layers)])
        
        self.final_norm = LayerNorm(config.d_model)


    def forward(self, input_idx):
        (batch_size, num_tokens) = input_idx.shape
        
        token_embedings = self.token_emb(input_idx)
        position_idx = self.position_ids[:, :num_tokens]
        pos_embeddings = self.pos_emb(position_idx)
        
        input_embeddings = token_embedings + pos_embeddings
        
        x = self.emb_dropout(input_embeddings)
        
        x = self.transformer_backbone(x)
        
        x = self.final_norm(x)
        
        return x
