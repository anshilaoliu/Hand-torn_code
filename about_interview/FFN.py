#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project Name: Hand-torn_code
# @File Name   : FFN.py
# @author      : ahua
# @Start Date  : 2024/3/31 3:22
# @Classes     : 
from torch import nn


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
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = self.fc(x)
        output = self.layer_norm(x+residual)
        return output
