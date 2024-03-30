#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project Name: Hand-torn_code
# @File Name   : attention_module.py
# @author      : ahua
# @Start Date  : 2024/3/23 23:11
# @Classes     : 多头注意力
import torch
from torch import nn
import math

from utils_module import LayerNorm


class ScaledDotProductAttention(nn.Module):
    """根据公式计算QkV"""
    def __init__(self, n_d):
        """

        :param n_d: 每个头的dim，用于scaling
        """
        super(ScaledDotProductAttention, self).__init__()

        self.n_d = n_d
        # 在最后一个维度上进行softmax
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, Q, K, V, mask):
        # q和k关于（2,3）维度的转置相乘并scaling
        attn_score = Q @ K.transpose(2, 3) / math.sqrt(self.n_d)

        if mask is not None:
            attn_score = attn_score.masked_fill(mask == 0, float("-inf"))

        attn_score = self.softmax(attn_score)
        attn_score = attn_score @ V

        return attn_score


class MultiHeadAttention(nn.Module):
    """多头注意力，包括残差连接和Norm"""
    def __init__(self, d_model, n_head, dropout=0.1, bias=True):
        """

        :param d_model: 输入向量embedding维度
        :param n_head:
        :param bias:
        """
        super(MultiHeadAttention, self).__init__()

        if d_model % n_head != 0:
            raise ValueError(
                "Embedding dim must be divisible by number of heads in {}. Got: embed_dim={} and num_heads={}".format(
                    self.__class__.__name__, d_model, n_head
                )
            )

        self.n_head = n_head
        self.d_model = d_model
        self.n_d = d_model // n_head

        # 投影映射矩阵
        self.w_q = nn.Linear(in_features=d_model, out_features=d_model, bias=bias)
        self.w_k = nn.Linear(in_features=d_model, out_features=d_model, bias=bias)
        self.w_v = nn.Linear(in_features=d_model, out_features=d_model, bias=bias)

        self.get_attn = ScaledDotProductAttention(self.n_d)

        # 最后多头合并之后再做一次映射
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)

        self.layer_norm = LayerNorm(d_model)

    def forward(self, x_q, x_k, x_v, mask=None):
        residual = x_q
        batch, seq_len, dimension = x_q.shape
        # 映射得到QKV矩阵
        q, k, v = self.w_q(x_q), self.w_k(x_k), self.w_v(x_v)

        # 拆分为四维张量后将（0,1,2,3）reshape为（0,2,1,3）
        q = q.view(batch, seq_len, self.n_head, self.n_d).permute(0, 2, 1, 3)
        k = k.view(batch, seq_len, self.n_head, self.n_d).permute(0, 2, 1, 3)
        v = v.view(batch, seq_len, self.n_head, self.n_d).permute(0, 2, 1, 3)

        attn_score = self.get_attn(q, k, v, mask)

        # 重新排列维度，保证内存连续型后改变为三维张量
        attn_score = attn_score.permute(0, 2, 1, 3).contiguous().view(batch, seq_len, dimension)

        output = self.w_o(attn_score)
        output = self.dropout(output)

        # 残差连接和Norm
        output = self.layer_norm(output + residual)
        return output


def test():
    d_model = 1024
    n_head = 8

    x = torch.randn(32, 64, 1024)  # Batch, Time, Dimension
    print(x.shape)

    att_model = MultiHeadAttention(d_model, n_head)
    out = att_model(x, x, x)
    print(out.shape)


if __name__ == '__main__':
    test()
