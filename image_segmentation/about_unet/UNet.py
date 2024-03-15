#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project Name: Hand-torn_code
# @File Name   : UNet.py
# @author      : ahua
# @Start Date  : 2024/3/16 5:35
# @Classes     :
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


class DoubleConv(nn.Module):
    """定义连续的俩次卷积"""

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


class UNet(nn.Module):
    def __init__(self, in_channel=3, out_channel=2, features=[64, 128, 256, 512]):
        """

        :param in_channel:
        :param out_channel:
        :param features: 各个采样后对应的通道数
        """
        super(UNet, self).__init__()
        # 记录一系列上采样和下采样操作层
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        # 最大池化下采样
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 加入
        for feature in features:
            self.downs.append(DoubleConv(in_channel, feature))
            # 下次的输入通道数变为刚刚的输出通道数
            in_channel = feature

        # 上采样最下面的一步俩次卷积
        self.final_up = DoubleConv(features[-1], features[-1]*2)

        # 上采样逆置list
        for feature in reversed(features):
            # 转置卷积上采样,因为进行了拼接，所以输入通道数x2
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=3, stride=1, padding=1))
            self.ups.append(DoubleConv(feature*2, feature))

        # 最后出结果的1x1卷积
        self.final_conv = nn.Conv2d(features[0], out_channel, kernel_size=1)

    def forward(self, x):
        # 记录跳跃连接
        skip_connections = []
        # 下采样
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        # 最下层卷积
        x = self.final_up(x)
        # 逆置跳跃连接
        skip_connections = skip_connections[::-1]
        # 上采样
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]
            if skip_connection.shape != x.shape:
                # 原论文中这里是对skip_connection做裁剪，这里对x做resize
                x = TF.resize(x, size=skip_connection.shape[2:])
            x = torch.cat((x, skip_connection), dim=1)
            x = self.ups[idx+1](x)
        output = self.final_conv(x)
        return output


def test():
    x = torch.randn(3, 3, 572, 572)
    model = UNet()
    print(x.shape)
    print(model(x).shape)


if __name__ == '__main__':
    test()
