#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project Name: Hand-torn_code
# @File Name   : LayerNorm.py
# @author      : ahua
# @Start Date  : 2024/3/31 3:22
# @Classes     : 
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
