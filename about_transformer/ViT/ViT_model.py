#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project Name: Hand-torn_code
# @File Name   : ViT_model.py
# @author      : ahua
# @Start Date  : 2024/3/10 0:38
# @Classes     : 简易版ViT模型
import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """
    切patch
    """
    def __init__(self, in_channels, patch_size, embed_dim, patch_num, dropout=0.1):
        """

        :param in_channels: 输入通道数
        :param patch_size:小方块大小，即每个小方块大小是 （patch_size x patch_size）
        :param embed_dim:embedding维度，也是卷积切分后的输出维度，等于patch_size*patch_size*in_channels
        :param patch_num:patch个数
        :param dropout:默认0.1
        """
        super(PatchEmbedding, self).__init__()
        # 卷积切patch，并拉平
        self.get_patch = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2)
        )

        # 加入CLS Token（随机初始化），为了能和patch的embedding拼在一起，第三个维度应该一致,第一维先默认使用1初始化，forward实例化时再扩充对齐
        self.cls_token = nn.Parameter(torch.randn(size=(1, 1, embed_dim)), requires_grad=True)
        # 加入位置编码，也是随机初始化，为了能和总的embedding加在一起，第二个维度应该等于patch_num+1，同样第三个维度等于embed_dim
        self.position_embedding = nn.Parameter(torch.randn(size=(1, patch_num+1, embed_dim)), requires_grad=True)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        # 切
        x = self.get_patch(x)
        # 交换后俩维
        x = x.permute(0, 2, 1)

        # 拼cls token
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)  # 第一维扩充对齐，因为第一维一般是是batch_size，运行前是不确定batch_size大小的
        x = torch.cat([x, cls_token], dim=1)
        # 加position_embedding
        x = x + self.position_embedding

        x = self.dropout(x)
        return x


class ViT(nn.Module):
    """
    ViT模型构建
    """
    def __init__(self, in_channels, patch_size, embed_dim, patch_num, heads_num, activation,
                 encoders_num, classes_num, dropout=0.1):
        """

        :param in_channels:
        :param patch_size:
        :param embed_dim:
        :param patch_num:
        :param heads_num: 多头注意力中的头
        :param activation: 激活方式
        :param encoders_num:
        :param classes_num: 类别数
        :param dropout:
        """
        super(ViT, self).__init__()
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, embed_dim, patch_num)

        # 用torch封装好的定义Transformer中的Encoder layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=heads_num, dropout=dropout, activation=activation,
                                                   batch_first=True, norm_first=True)
        # Encoder layer装入
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=encoders_num)

        # MLP分类头
        self.MLP = nn.Sequential(
            # 先做层归一化
            nn.LayerNorm(normalized_shape=embed_dim),
            nn.Linear(in_features=embed_dim, out_features=classes_num)
        )

    def forward(self, x):
        # 切
        x = self.patch_embedding(x)
        # 过encoder
        x = self.encoder(x)
        # 过MLP，因为是分类任务，只取了编码后的 CLS token（即第一个位置，也即只取第二维中索引为0的所有数据）作为输入
        x = self.MLP(x[:, 0, :])
        return x


def test():
    # 随机生成一组张量，可视为3张3通道照片，尺寸224x224
    x = torch.randn(3, 3, 224, 224)
    # 切成16x16个块，每个块14x14大小，转化维度后相当于切14x14=196个patch，8个头6个Encoder，类别假设是10
    vit_model = ViT(3, 16, 16*16*3, 14*14, 8, "gelu", 6, 10)
    pred = vit_model(x)
    print(x.shape)
    print(pred.shape)


if __name__ == '__main__':
    test()
