#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project Name: Hand-torn_code
# @File Name   : CBAM.py
# @author      : ahua
# @Start Date  : 2024/3/31 3:08
# @Classes     :
import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """通道注意力"""
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """空间注意力"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAMBlock(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=7):
        super(CBAMBlock, self).__init__()
        self.channel_attention = ChannelAttention(channel, ratio=ratio)
        self.spatial_attention = SpatialAttention(kernel_size=kernel_size)
        # 根据你的实际需求来考虑是否残差连接与下采样
        self.channel_down = nn.Conv2d(channel * 2, channel, kernel_size=1)

    def forward(self, x):
        residual = x
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)

        x = torch.cat((x, residual), dim=1)
        x = self.channel_down(x)
        return x


if __name__ == '__main__':
    pass
