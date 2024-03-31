#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project Name: Hand-torn_code
# @File Name   : SENet.py
# @author      : ahua
# @Start Date  : 2024/3/31 9:34
# @Classes     : 搭建SEBlock，以及用于SE-ResNet18/34的SEBasicBlock、用于SE-ResNet50/101/152的SEBottleNeck
#                具体SE-ResNet这里就不写了
import torch
from torch import nn


class SEBlock(nn.Module):
    def __init__(self, in_channel, r=6):
        """

        :param in_channel:
        :param r: 论文中全连接层的r，即通道数缩放因子
        """
        super(SEBlock, self).__init__()
        # 全局平均池化(Squeeze)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # 两个全连接层(Excitation)
        self.fc = nn.Sequential(
            nn.Linear(in_channel, in_channel // r, bias=False),
            nn.ReLU(),
            nn.Linear(in_channel // r, in_channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        # Squeeze,得到通道描述符，即(b, c)张量
        out = self.avg_pool(x).view(b, c)
        # Excitation，得到每个通道的权重
        out = self.fc(out).view(b, c, 1, 1)
        # 特征加权后输出
        return x * out.expand_as(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, r=6):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.SE = SEBlock(out_channel, r)

        # 防止无法连接，进行1x1下采样
        if stride != 1 or in_channel != self.expansion * out_channel:
            self.down_sample = nn.Sequential(nn.Conv2d(in_channel, self.expansion * out_channel, kernel_size=1, stride=stride, bias=False),
                                             nn.BatchNorm2d(self.expansion * out_channel))
        else:
            self.down_sample = lambda x: x

    def forward(self, x):
        residual = self.down_sample(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = self.SE(out)

        out = residual + out
        out = self.relu(out)
        return out


class SEBottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, r=6):
        super(SEBottleNeck, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.conv3 = nn.Conv2d(out_channel, out_channel * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channel * self.expansion)

        self.SE = SEBlock(self.expansion * out_channel, r)

        # 防止无法连接，进行1x1下采样
        if stride != 1 or in_channel != self.expansion * out_channel:
            self.down_sample = nn.Sequential(
                nn.Conv2d(in_channel, self.expansion * out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channel))
        else:
            self.down_sample = lambda x: x

    def forward(self, x):
        residual = self.down_sample(x)

        out = self.relu(self.bn1(self.conv1(x)))

        out = self.relu(self.bn2(self.conv2(out)))

        out = self.bn3(self.conv3(out))
        out = self.SE(out)

        out += residual
        out = self.relu(out)

        return out


def test():
    x = torch.randn(3, 3, 224, 224)
    # block = BasicBlock(3, 64)
    block = SEBottleNeck(3, 64)
    pred = block(x)
    print(x.shape)
    print(pred.shape)


if __name__ == '__main__':
    test()
