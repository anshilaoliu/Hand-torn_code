#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project Name: Hand-torn_code
# @File Name   : Embedding.py
# @author      : ahua
# @Start Date  : 2024/3/31 3:25
# @Classes     : 
import torch
from torch import nn


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