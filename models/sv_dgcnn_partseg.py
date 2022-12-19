"""
Author: Zhuo Su
Time: 1/27/2022 21:53
"""
import os
import sys
import copy
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from .sv_layers import *
from .utils.sv_util import *

def _get_visible_value(divisor=8):
    def make_divisible(v: float) -> int:
        """
        This function is taken from the original tf repo.
        It ensures that all layers have a channel number that is divisible by 8
        It can be seen here:
        https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
        """
        min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v
    return make_divisible

def _identity(v):
    return v

_V = _get_visible_value(8)
#_V = _identity

class SV_DGCNN_PSEG(nn.Module):
    def __init__(self, args, num_part):
        super(SV_DGCNN_PSEG, self).__init__()
        self.args = args
        self.k = args.k
        self.binary = args.binary
        self.dropout = 0 if self.binary else args.dropout
        self.emb = 1024

        self.init_scalar = Vector2Scalar(2, 3)
        self.conv1 = SVBlock((6, 2), (_V(64//2), _V(64//6)))
        self.conv2 = SVBlock((_V(64//2)*2, _V(64//6)*2), (_V(64//2), _V(64//6)), self.binary)
        self.conv3 = SVBlock((_V(64//2)*2, _V(64//6)*2), (_V(128//2), _V(128//6)), self.binary)
        self.conv4 = SVBlock((_V(128//2)*2, _V(128//6)*2), (_V(256//2), _V(256//6)), self.binary)

        self.svfuse1 = SVFuse(_V(64//6)*2+_V(128//6)+_V(256//6), 3, self.binary)
        self.conv5 = SVBlock((_V(64//2)*2+_V(128//2)+_V(256//2), _V(64//6)*2+_V(128//6)+_V(256//6)), (_V(self.emb//2), _V(self.emb//6)), self.binary)
        self.conv6 = SVBlock((_V(self.emb//2), _V(self.emb//6)), (_V(self.emb//4), _V(self.emb//12)), self.binary)
        self.svfuse2 = SVFuse(_V(self.emb//12), 3, self.binary)
        self.svfuse3 = SVFuse(_V(self.emb//6), 3, self.binary)
        self.conv7 = nn.Sequential(
                nn.Conv1d(16, 64, kernel_size=1, bias=False),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(
                Conv1d(_V(self.emb//2)+_V(self.emb//4)+(_V(self.emb//6)+_V(self.emb//12))*3+64+_V(64//2)*2+_V(128//2)+_V(256//2)+(_V(64//6)*2+_V(128//6)+_V(256//6))*3, 256, self.binary),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=self.dropout)
        self.conv9 = nn.Sequential(
                Conv1d(256, 256, self.binary),
                nn.BatchNorm1d(256),
                nn.LeakyReLU(negative_slope=0.2))
        self.dp2 = nn.Dropout(p=self.dropout)
        self.conv10 = nn.Sequential(
                Conv1d(256, 128, self.binary),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(negative_slope=0.2))
        self.conv11 = nn.Conv1d(128, num_part, kernel_size=1, bias=False)

    def forward(self, x, l):
        batch_size = x.size(0)
        num_points = x.size(2)
        x = x.unsqueeze(1)

        v = get_graph_feature(x, k=self.k) # B, N_points, k, 3, 2
        s = self.init_scalar(v) # B, N_points, k, 6
        x = (s, v)
        x = self.conv1(x)
        x1 = svpool(x)

        x = get_graph_feature_sv(x1, k=self.k)
        x = self.conv2(x)
        x2 = svpool(x)

        x = get_graph_feature_sv(x2, k=self.k)
        x = self.conv3(x)
        x3 = svpool(x)

        x = get_graph_feature_sv(x3, k=self.k)
        x = self.conv4(x)
        x4 = svpool(x)

        x = svcat([x1, x2, x3, x4]) # B, N_points, [3,] dim
        x_fine = self.svfuse1(x) # B, N_points, 3*vdim+sdim(=64//2*3+64//6*6)

        x = self.conv5(x)
        x_pool = svpool(x, dim=1, keepdim=True) # B, 1, [3,] emb
        x_pool = self.conv6(x_pool) # B, 1, [3,] emb/2
        x_pool = self.svfuse2(x_pool) # B, 1, emb/2

        x = self.svfuse3(x) # B, N_points, emb=self.emb//2 + self.emb//6*3
        x = x.max(dim=1, keepdim=False)[0].unsqueeze(-1) # B, emb, 1

        l = l.view(batch_size, -1, 1)
        l = self.conv7(l) # B, 64, 1

        x = torch.cat([x, x_pool.transpose(-1, -2), l], dim=1) # B, emb+emb/2+64, 1
        x = x.repeat(1, 1, num_points) # B, emb+64, N_points

        x = torch.cat([x, x_fine.transpose(-1, -2)], dim=1)
        x = self.conv8(x)
        x = self.dp1(x)
        x = self.conv9(x)
        x = self.dp2(x)
        x = self.conv10(x)
        x = self.conv11(x)

        return x

