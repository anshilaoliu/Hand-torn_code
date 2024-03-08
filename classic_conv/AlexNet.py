#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project Name: Hand-torn_code
# @Package Name: 
# @File Name   : AlexNet.py
# @author      : ahua
# @Version     : 1.0
# @Start Date  : 2024/3/9 3:54
# @Classes     :
import torch
from torch import nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        # 卷积层
        self.conv = nn.Sequential(
            # 由于LRN层已经证明无用，所以这里比起原始架构少了LRN
            # 第一层
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            # 输入通道数, 输出通道数及其他参数，input:[3, 224, 224] output:[96, 55, 55]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # output:[96, 27, 27]

            # 第二层，开始减小卷积窗口且增大输出通道数，从而提取更多特征
            nn.Conv2d(96, 256, 5, 1, 2),  # output: [256, 27, 27]
            nn.ReLU(),
            nn.MaxPool2d(3, 2),  # output: [256, 13, 13]
            # 连续3个卷积层，且使用更小的卷积窗口。除了最后的卷积层外，进一步增大了输出通道数。
            # 前两个卷积层后不使用池化层来减小输入的高和宽
            nn.Conv2d(256, 384, 3, 1, 1),  # output: [384, 13, 13]
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, 1, 1),  # output: [384, 13, 13]
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, 1),  # output: [256, 13, 13]
            nn.ReLU(),
            nn.MaxPool2d(3, 2)  # output: [256, 6, 6]
        )
        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(256 * 5 * 5, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            # 输出层
            nn.Linear(4096, num_classes),
        )

    # 前向传播
    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output

def test():
    pass


if __name__ == '__main__':
    test()
