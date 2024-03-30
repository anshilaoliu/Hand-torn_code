#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Project Name: Hand-torn_code
# @File Name   : transformer_model.py
# @author      : ahua
# @Start Date  : 2024/3/23 23:11
# @Classes     : 构建Transformer
import torch
from torch import nn

from transformer_encoder import Encoder
from transformer_decoder import Decoder


class Transformer(nn.Module):
    def __init__(self, src_pad_idx, tgt_pad_idx, enc_vocabulary_size, dec_vocabulary_size, d_model=512,
                 n_head=8, ffn_hidden=2048, max_len=5000, n_layers=6, dropout=0.1):
        """

        :param src_pad_idx: source的pad标识符
        :param tgt_pad_idx: target的pad的标识符
        :param enc_vocabulary_size: source的词汇表大小
        :param dec_vocabulary_size: target的词汇表大小
        :param max_len:
        :param d_model:
        :param n_head:
        :param ffn_hidden:
        :param n_layers:
        :param dropout:
        """
        super(Transformer, self).__init__()

        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx

        # Encoder层
        self.encoder = Encoder(enc_vocabulary_size, d_model, n_head, ffn_hidden, max_len,
                               n_layers, dropout)
        # Decoder层
        self.decoder = Decoder(dec_vocabulary_size, d_model, n_head, ffn_hidden, max_len,
                               n_layers, dropout)
        # 输出层，做一个线性映射
        self.fc = nn.Linear(d_model, dec_vocabulary_size)

    def _make_casual_mask(self, q, k):
        # 获取第二维的seq_len, 因为是QK相乘再做mask，所以mask大小应符合QK
        len_q, len_k = q.size(1), k.size(1)
        # 生成三角mask矩阵
        mask = torch.tril(torch.ones(len_q, len_k)).type(torch.BoolTensor)
        return mask

    def _make_padding_mask(self, q, k, pad_idx_q, pad_idx_k):
        len_q, len_k = q.size(1), k.size(1)

        # mask矩阵大小应为(Batch, seq_len, len_q, len_k)
        # 不等于pad_idx时设置为True,并增加俩个维度seq_len和len_k
        q = q.ne(pad_idx_q).unsqueeze(1).unsqueeze(3)
        # 在len_k维上做重复补全
        q = q.repeat(1, 1, 1, len_k)

        k = k.ne(pad_idx_k).unsqueeze(1).unsqueeze(2)
        k = k.repeat(1, 1, len_q, 1)

        mask = q & k
        return mask

    def forward(self, src, tgt):
        # Encoder的padding_mask，此时QK都来自source
        enc_padding_mask = self._make_padding_mask(src, src, self.src_pad_idx, self.src_pad_idx)
        # Decoder的因果mask,不仅要考虑不给看未来还要考虑padding
        dec_casual_mask = self._make_padding_mask(tgt, tgt, self.tgt_pad_idx, self.tgt_pad_idx) * \
                          self._make_casual_mask(tgt, tgt)
        # 交叉注意力的padding_mask
        cross_padding_mask = self._make_padding_mask(tgt, src, self.tgt_pad_idx, self.src_pad_idx)

        enc = self.encoder(src, enc_padding_mask)
        dec = self.decoder(tgt, enc, dec_casual_mask, cross_padding_mask)
        output = self.fc(dec)
        return output


if __name__ == '__main__':
    pass
