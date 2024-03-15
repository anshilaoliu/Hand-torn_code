#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project Name: Hand-torn_code
# @File Name   : UNet_pp.py
# @author      : ahua
# @Start Date  : 2024/3/4 22:52
# @Classes     : UNet++网络
from torch import nn
import torch


class DoubleConv(nn.Module):
    """同UNet定义连续的俩次卷积"""

    def __init__(self, in_channel, out_channel):
        super(DoubleConv, self).__init__()
        # 俩次卷积
        self.d_conv = nn.Sequential(
            # 相比原论文，这里加入了padding与BN
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.d_conv(x)


class UNetPP(nn.Module):
    def __init__(self, in_channel=3, out_channel=2, features=[64,128,256,512,1024], deep_supervision=True):
        """

        :param in_channel:
        :param out_channel:
        :param features: 各个采样后对应的通道数
        :param deep_supervision: 是否使用深度监督
        """
        super(UNetPP, self).__init__()

        self.deep_supervision = deep_supervision

        self.pool = nn.MaxPool2d(2, 2)
        # 双线性插值进行上采样，也可以通过ConvTranspose2d或者先ConvTranspose2d后插值实现，这里为了方便直接插值
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # 原始UNet的下采样层，每个下采样层的第0层卷积
        self.conv0_0 = DoubleConv(in_channel, features[0])
        self.conv1_0 = DoubleConv(features[0], features[1])
        self.conv2_0 = DoubleConv(features[1], features[2])
        self.conv3_0 = DoubleConv(features[2], features[3])
        self.conv4_0 = DoubleConv(features[3], features[4])

        # 每个下采样层的第一层卷积
        self.conv0_1 = DoubleConv(features[0] + features[1], features[0])
        self.conv1_1 = DoubleConv(features[1] + features[2], features[1])
        self.conv2_1 = DoubleConv(features[2] + features[3], features[2])
        self.conv3_1 = DoubleConv(features[3] + features[4], features[3])

        # 每个下采样层的第二层卷积
        self.conv0_2 = DoubleConv(features[0] * 2 + features[1], features[0])
        self.conv1_2 = DoubleConv(features[1] * 2 + features[2], features[1])
        self.conv2_2 = DoubleConv(features[2] * 2 + features[3], features[2])

        # 每个下采样层的第三层卷积
        self.conv0_3 = DoubleConv(features[0] * 3 + features[1], features[0])
        self.conv1_3 = DoubleConv(features[1] * 3 + features[2], features[1])

        # 每个下采样层的第四层卷积
        self.conv0_4 = DoubleConv(features[0] * 4 + features[1], features[0])

        # 分割头，作者原论文写了深度监督之后还过sigmoid，但是UNet没有sigmoid
        self.sigmoid = nn.Sigmoid()
        if self.deep_supervision:
            self.final1 = nn.Conv2d(features[0], out_channel, kernel_size=1)
            self.final2 = nn.Conv2d(features[0], out_channel, kernel_size=1)
            self.final3 = nn.Conv2d(features[0], out_channel, kernel_size=1)
            self.final4 = nn.Conv2d(features[0], out_channel, kernel_size=1)
        else:
            self.final = nn.Conv2d(features[0], out_channel, kernel_size=1)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))

        # 使用深度监督，返回四个分割图
        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output1 = self.sigmoid(output1)
            output2 = self.final2(x0_2)
            output2 = self.sigmoid(output2)
            output3 = self.final3(x0_3)
            output3 = self.sigmoid(output3)
            output4 = self.final4(x0_4)
            output4 = self.sigmoid(output4)
            return [output1, output2, output3, output4]

        else:
            output = self.final(x0_4)
            output = self.sigmoid(output)
            return output
