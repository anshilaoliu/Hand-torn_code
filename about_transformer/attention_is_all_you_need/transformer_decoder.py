#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project Name: Hand-torn_code
# @File Name   : transformer_decoder.py
# @author      : ahua
# @Start Date  : 2024/3/26 0:17
# @Classes     :
import torch
from torch import nn

from attention_module import MultiHeadAttention
from utils_module import PositionwiseFeedForwardNet
from utils_module import PositionalEmbedding


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, ffn_hidden, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.causal_attention = MultiHeadAttention(d_model, n_head, dropout)

        self.cross_attention = MultiHeadAttention(d_model, n_head, dropout)

        self.ffn = PositionwiseFeedForwardNet(d_model, ffn_hidden, dropout)

    def forward(self, dec, enc, causal_mask, padding_mask):
        """

        :param dec: 来自decoder的输入
        :param enc: 来自encoder的输出
        :param causal_mask: 下三角掩码，防止看见未来信息
        :param padding_mask: 将输入序列中的填充部分标记为不可关注，防止模型在训练过程中对这些无意义的padding部分进行不必要的关注
        :return:
        """
        x = self.causal_attention(dec, dec, dec, causal_mask)

        x = self.cross_attention(x, enc, enc, padding_mask)

        x = self.ffn(x)

        return x


class Decoder(nn.Module):
    def __init__(self, dec_vocabulary_size, d_model=512, n_head=8, ffn_hidden=2048, max_len=5000,
                 n_layer=6, dropout=0.1):
        super(Decoder, self).__init__()

        self.token_embedding = nn.Embedding(dec_vocabulary_size, d_model, padding_idx=1)
        self.embedding = PositionalEmbedding(d_model, max_len, dropout)

        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, n_head, ffn_hidden, dropout) for _ in range(n_layer)]
        )

    def forward(self, dec, enc, causal_mask, padding_mask):
        dec = self.token_embedding(dec)
        dec = self.embedding(dec)

        for layer in self.layers:
            dec = layer(dec, enc, causal_mask, padding_mask)

        return dec


if __name__ == '__main__':
    pass
