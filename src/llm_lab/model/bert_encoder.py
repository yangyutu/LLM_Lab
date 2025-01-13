import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.models.bert.modeling_bert import BertEncoder

import math


class PositionalEncoding(nn.Module):
    # Absolute positional encoding, following the attention is all you need paper, introduces a notion of word order.
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(0)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(
            token_embedding + self.pos_embedding[:, : token_embedding.size(1), :]
        )


class LearnablePositionEncoding(nn.Module):
    # The original BERT paper states that unlike transformers, positional and segment embeddings are learned.
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super(LearnablePositionEncoding, self).__init__()
        self.pos_embedding = nn.Embedding(maxlen, emb_size)
        self.register_buffer("position_ids", torch.arange(maxlen).expand((1, -1)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, token_embedding: Tensor):
        position_ids = self.position_ids[:, : token_embedding.size(1)]
        return self.dropout(token_embedding + self.pos_embedding(position_ids))


class EmbeddingLayer(nn.Module):
    def __init__(
        self,
        embed_size,
        vocab_size,
        dropout=0.1,
        position_encoding_type: str = "learnable",
    ):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_size)
        assert position_encoding_type in ["learnable", "cosine"]
        if position_encoding_type == "learnable":
            self.pos_encoding = LearnablePositionEncoding(embed_size, dropout=dropout)
        elif position_encoding_type == "cosine":
            self.pos_encoding = PositionalEncoding(embed_size, dropout=dropout)

        self.embed_layer_norm = nn.LayerNorm(embed_size)

    def forward(self, input_ids):
        embedding = self.embed_layer_norm(
            self.pos_encoding(self.token_embed(input_ids))
        )
        return embedding


class BertEncoderModel(nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        print("custom BERT")
        self.config = config
        self.num_layers = config.num_layers
        self.dim_model = config.dim_model
        #self.pooled_output_embedding = pooled_output_embedding
        self.embedding_layer = EmbeddingLayer(
            embed_size=config.embed_size,
            vocab_size=config.vocab_size,
            dropout=config.dropout,
            position_encoding_type=config.position_encoding_type,
        )
        auto_config = AutoConfig.from_pretrained("bert-base-uncased")
        auto_config.update(
            {
                "num_attention_heads": config.nhead,
                "hidden_size": config.dim_model,
                "intermediate_size": config.dim_feedforward,
                "num_hidden_layers": config.num_layers,
            }
        )
        self.encoder = BertEncoder(auto_config)
        #self.linear = nn.Linear(config.dim_model, config.num_classes)
        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.apply(self._init_weight_fn)

    def _init_weight_fn(self, module):
        classname = module.__class__.__name__
        if "Linear" == classname:
            # m.weight.data shoud be taken from a normal distribution
            module.weight.data.normal_(0.0, 0.02)
            # m.bias.data should be 0
            module.bias.data.fill_(0)
        elif "Embedding" == classname:
            # m.weight.data shoud be taken from a normal distribution
            module.weight.data.normal_(0.0, 0.02)

    def forward(self, encoded_inputs):
        # encoded_inputs = self.tokenizer(
        #     text_list,
        #     return_tensors="pt",
        #     max_length=self.truncate,
        #     truncation="longest_first",
        #     padding="max_length",
        #     add_special_tokens=False if self.pooled_output_embedding else True,
        # )
        # encoded_inputs = {k: v.to(self.device) for k, v in encoded_inputs.items()}
        embedding = self.embedding_layer(encoded_inputs["input_ids"])
        # convert original attention mask to additive attenion score mask
        atten_mask = encoded_inputs["attention_mask"]
        mask = (
            atten_mask.float()
            .masked_fill(atten_mask == 0, float("-inf"))
            .masked_fill(atten_mask == 1, float(0.0))
        )
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        # https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py#L994
        mask = mask[:, None, None, :]
        embedding = self.encoder(embedding, attention_mask=mask)
        encoded_embeddings = embedding.last_hidden_state
        # if self.pooled_output_embedding:
        #     representation = self._mean_pooling(
        #         encoded_embeddings, encoded_inputs["attention_mask"]
        #     )
        # else:
        #     representation = encoded_embeddings[:, 0, :]
        #logits = self.linear(encoded_embeddings)

        return encoded_embeddings