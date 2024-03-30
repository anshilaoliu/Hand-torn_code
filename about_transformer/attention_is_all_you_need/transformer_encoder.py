#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project Name: Hand-torn_code
# @File Name   : transformer_encoder.py
# @author      : ahua
# @Start Date  : 2024/3/23 23:14
# @Classes     : Encoder
import torch
from torch import nn

from attention_module import MultiHeadAttention
from utils_module import PositionwiseFeedForwardNet
from utils_module import PositionalEmbedding


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, ffn_hidden, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model, n_head, dropout)

        self.ffn = PositionwiseFeedForwardNet(d_model, ffn_hidden, dropout)

    def forward(self, x, mask=None):
        x = self.attention(x, x, x, mask)
        x = self.ffn(x)

        return x


class Encoder(nn.Module):
    def __init__(self, enc_vocabulary_size, d_model=512, n_head=8, ffn_hidden=2048, max_len=5000,
                 n_layer=6, dropout=0.1):
        super(Encoder, self).__init__()

        self.token_embedding = nn.Embedding(enc_vocabulary_size, d_model, padding_idx=1)
        self.embedding = PositionalEmbedding(d_model, max_len, dropout)

        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_head, ffn_hidden, dropout) for _ in range(n_layer)]
        )

    def forward(self, x, padding_mask=None):
        x = self.token_embedding(x)
        x = self.embedding(x)

        for layer in self.layers:
            x = layer(x, padding_mask)
        return x


if __name__ == '__main__':
    pass
