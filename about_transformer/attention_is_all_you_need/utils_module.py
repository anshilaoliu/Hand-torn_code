#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project Name: Hand-torn_code
# @File Name   : utils_module.py
# @author      : ahua
# @Start Date  : 2024/3/23 23:13
# @Classes     : LayerNorm和前馈网络以及positional embedding
import torch
from torch import nn


class LayerNorm(nn.Module):
    """也可以直接用nn.LayerNorm"""
    def __init__(self, d_model, eps=1e-9):
        super(LayerNorm, self).__init__()
        # 俩个参数，权重和偏置，防止输入激活函数的线性表示部分导致非线性效果不佳
        self.weight = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        # 防止分母0
        self.eps = eps

    def forward(self, x):
        # LayerNorm全都是对最后一维进行归一化
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        out = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight * out + self.beta


class PositionwiseFeedForwardNet(nn.Module):
    """前馈网络，包括后续残差连接与Norm"""
    def __init__(self, d_model, hidden, dropout=0.1):
        super(PositionwiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(p=dropout)
        )
        self.layer_norm = LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = self.fc(x)
        output = self.layer_norm(x+residual)
        return output


class PositionalEmbedding(nn.Module):
    """位置编码，输入token embedding返回加上位置编码后的总的embedding"""
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEmbedding, self).__init__()
        # 初始化编码
        self.pe = torch.zeros(max_len, d_model)
        # 原始论文中位置编码是直接算的，不用训练
        self.pe.requires_grad_(False)

        # 照着公式敲就行了
        # 初始化pos
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 2i
        _2i = torch.arange(0, d_model, 2)

        # 偶数计算
        self.pe[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        # 奇数计算
        self.pe[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """

        :param x: 输入的token embedding
        :return:
        """
        seq_len = x.shape[1]
        x = x + self.pe[:seq_len, :]
        return self.dropout(x)


if __name__ == '__main__':
    pass
