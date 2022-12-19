#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
import torchvision.models

from models.sv_layers import *
from models.utils.sv_util import *
from macs import get_mac, get_param

EPS = 1e-6

def _make_divisible(v: float, divisor: int=8, min_value=None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

_V = _make_divisible

class SV_DGCNN_CLS_mac(nn.Module):
    def __init__(self, args, num_class=40):
        super(SV_DGCNN_CLS_mac, self).__init__()
        self.k = args.k
        self.binary = args.binary
        p = 0 if self.binary else 0.5

        self.init_scalar = Vector2Scalar(2, 3)
        self.conv1 = SVBlock((6, 2), (64//2, 64//6))
        self.conv2 = SVBlock((64//2*2, 64//6*2), (64//2, 64//6), self.binary)
        self.conv3 = SVBlock((64//2*2, 64//6*2), (128//2, 128//6), self.binary)
        self.conv4 = SVBlock((128//2*2, 128//6*2), (256//2, 256//6), self.binary)

        self.conv5 = SVBlock((64//2*2+128//2+256//2, 64//6*2+128//6+256//6), (1024//2, 1024//6), self.binary)
        self.svfuse = SVFuse(1024//6, 3, self.binary)

        self.linear1 = Linear((1024//2+1024//6*3)*2, 512, bias=False, bw=self.binary, ba=self.binary)
        self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=p)
        self.linear2 = Linear(512, 256, bias=False, bw=self.binary, ba=self.binary)
        self.bn2 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=p)
        self.linear3 = nn.Linear(256, num_class)
        
    def forward(self, x):
        macs = (0.0, 0.0, 0.0)

        batch_size = x.size(0)
        x = x.unsqueeze(1)
        v = get_graph_feature(x, k=self.k) # B, N_points, k, 3, 2
        macs = get_mac(macs, 'Vector2Scalar', v, 3)
        s = self.init_scalar(v) # B, N_points, k, 6
        x = (s, v)
        macs = get_mac(macs, 'SVBlock', x, ((6, 2), (64//2, 64//6)))
        x = self.conv1(x)
        x1 = svpool(x)

        x = get_graph_feature_sv(x1, k=self.k)
        macs = get_mac(macs, 'SVBlock', x, ((64//2*2, 64//6*2), (64//2, 64//6)), binary=self.binary)
        x = self.conv2(x)
        x2 = svpool(x)

        x = get_graph_feature_sv(x2, k=self.k)
        macs = get_mac(macs, 'SVBlock', x, ((64//2*2, 64//6*2), (128//2, 128//6)), binary=self.binary)
        x = self.conv3(x)
        x3 = svpool(x)

        x = get_graph_feature_sv(x3, k=self.k)
        macs = get_mac(macs, 'SVBlock', x, ((128//2*2, 128//6*2), (256//2, 256//6)), binary=self.binary)
        x = self.conv4(x)
        x4 = svpool(x)

        x = svcat([x1, x2, x3, x4])
        macs = get_mac(macs, 'SVBlock', x, ((64//2*2+128//2+256//2, 64//6*2+128//6+256//6), (1024//2, 1024//6)), binary=self.binary)
        x = self.conv5(x)
        macs = get_mac(macs, 'SVFuse', x, (1024//6, 3), binary=self.binary)
        x = self.svfuse(x) # B, N_points, 1024
        x = x.transpose(-1, -2).contiguous() # B, 1024, N_points

        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        macs = get_mac(macs, 'LinearS', x, ((1024//2+1024//6*3)*2, 512), binary=self.binary)
        x = F.leaky_relu(self.bn1(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        macs = get_mac(macs, 'LinearS', x, (512, 256), binary=self.binary)
        x = F.leaky_relu(self.bn2(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        macs = get_mac(macs, 'nn_Linear', x, (256, 40))
        x = self.linear3(x)
        
        return macs

class SV_DGCNN_PSEG_mac(nn.Module):
    def __init__(self, args, num_part=50):
        super(SV_DGCNN_PSEG_mac, self).__init__()
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
        macs = (0.0, 0.0, 0.0)

        batch_size = x.size(0)
        num_points = x.size(2)
        x = x.unsqueeze(1)

        v = get_graph_feature(x, k=self.k) # B, N_points, k, 3, 2
        macs = get_mac(macs, 'Vector2Scalar', v, 3)
        s = self.init_scalar(v) # B, N_points, k, 6
        x = (s, v)
        macs = get_mac(macs, 'SVBlock', x, ((6, 2), (_V(64//2), _V(64//6))))
        x = self.conv1(x)
        x1 = svpool(x)

        x = get_graph_feature_sv(x1, k=self.k)
        macs = get_mac(macs, 'SVBlock', x, ((_V(64//2)*2, _V(64//6)*2), (_V(64//2), _V(64//6))), binary=self.binary)
        x = self.conv2(x)
        x2 = svpool(x)

        x = get_graph_feature_sv(x2, k=self.k)
        macs = get_mac(macs, 'SVBlock', x, ((_V(64//2)*2, _V(64//6)*2), (_V(128//2), _V(128//6))), binary=self.binary)
        x = self.conv3(x)
        x3 = svpool(x)

        x = get_graph_feature_sv(x3, k=self.k)
        macs = get_mac(macs, 'SVBlock', x, ((_V(128//2)*2, _V(128//6)*2), (_V(256//2), _V(256//6))), binary=self.binary)
        x = self.conv4(x)
        x4 = svpool(x)

        x = svcat([x1, x2, x3, x4]) # B, N_points, [3,] dim
        macs = get_mac(macs, 'SVFuse', x, (_V(64//6)*2+_V(128//6)+_V(256//6), 3), binary=self.binary)
        x_fine = self.svfuse1(x) # B, N_points, 3*vdim+sdim(=64//2*3+64//6*6)

        macs = get_mac(macs, 'SVBlock', x, ((_V(64//2)*2+_V(128//2)+_V(256//2), _V(64//6)*2+_V(128//6)+_V(256//6)), (_V(self.emb//2), _V(self.emb//6))), binary=self.binary)
        x = self.conv5(x)
        x_pool = svpool(x, dim=1, keepdim=True) # B, 1, [3,] emb
        macs = get_mac(macs, 'SVBlock', x_pool, ((_V(self.emb//2), _V(self.emb//6)), (_V(self.emb//4), _V(self.emb//12))), binary=self.binary)
        x_pool = self.conv6(x_pool) # B, 1, [3,] emb/2
        macs = get_mac(macs, 'SVFuse', x_pool, (_V(self.emb//12), 3), binary=self.binary)
        x_pool = self.svfuse2(x_pool) # B, 1, emb/2

        macs = get_mac(macs, 'SVFuse', x, (_V(self.emb//6), 3), binary=self.binary)
        x = self.svfuse3(x) # B, N_points, emb=self.emb//2 + self.emb//6*3
        x = x.max(dim=1, keepdim=False)[0].unsqueeze(-1) # B, emb, 1

        l = l.view(batch_size, -1, 1)
        macs = get_mac(macs, 'nn_Conv1dS', l, (16, 64))
        l = self.conv7(l) # B, 64, 1

        x = torch.cat([x, x_pool.transpose(-1, -2), l], dim=1) # B, emb+emb/2+64, 1
        x = x.repeat(1, 1, num_points) # B, emb+64, N_points

        x = torch.cat([x, x_fine.transpose(-1, -2)], dim=1)
        macs = get_mac(macs, 'Conv1dS', x, (_V(self.emb//2)+_V(self.emb//4)+(_V(self.emb//6)+_V(self.emb//12))*3+64+_V(64//2)*2+_V(128//2)+_V(256//2)+(_V(64//6)*2+_V(128//6)+_V(256//6))*3, 256), binary=self.binary)
        x = self.conv8(x)
        x = self.dp1(x)
        macs = get_mac(macs, 'Conv1dS', x, (256, 256), binary=self.binary)
        x = self.conv9(x)
        x = self.dp2(x)
        macs = get_mac(macs, 'Conv1dS', x, (256, 128), binary=self.binary)
        x = self.conv10(x)
        macs = get_mac(macs, 'nn_Conv1d', x, (128, 50))
        x = self.conv11(x)

        return macs

if __name__ == '__main__':
    class Obj(): pass
    args = Obj()
    args.k = 20
    args.binary = False
    args.dropout = 0

    model = SV_DGCNN_CLS_mac(args)
    params = get_param(model)

    x = torch.rand(2, 3, 1024)
    macs, adds, bops = model(x)
    macs /= 1e6 * 2
    adds /= 1e6 * 2
    bops /= 1e6 * 2
    print(f'Params of SV_DGCNN (FP) on ModelNet40: {params:.6f} M, MACs: {macs:.6f} M, ADDs: {adds:.6f} M, BOPs: {bops:.6f} M')

    args.binary = True

    model = SV_DGCNN_CLS_mac(args)
    params = get_param(model)

    x = torch.rand(2, 3, 1024)
    macs, adds, bops = model(x)
    macs /= 1e6 * 2
    adds /= 1e6 * 2
    bops /= 1e6 * 2
    print(f'Params of SV_DGCNN on ModelNet40: {params:.6f} M, MACs: {macs:.6f} M, ADDs: {adds:.6f} M, BOPs: {bops:.6f} M')

    args = Obj()
    args.k = 40
    args.binary = False
    args.dropout = 0

    model = SV_DGCNN_PSEG_mac(args)
    params = get_param(model)

    x = torch.rand(2, 3, 2048)
    l = torch.rand(2, 16)
    macs, adds, bops = model(x, l)
    macs /= 1e6 * 2
    adds /= 1e6 * 2
    bops /= 1e6 * 2
    print(f'Params of SV_DGCNN (FP) on ShapeNet: {params:.6f} M, MACs: {macs:.6f} M, ADDs: {adds:.6f} M, BOPs: {bops:.6f} M')

    args.binary = True

    model = SV_DGCNN_PSEG_mac(args)
    params = get_param(model)

    x = torch.rand(2, 3, 2048)
    l = torch.rand(2, 16)
    macs, adds, bops = model(x, l)
    macs /= 1e6 * 2
    adds /= 1e6 * 2
    bops /= 1e6 * 2
    print(f'Params of SV_DGCNN on ShapeNet: {params:.6f} M, MACs: {macs:.6f} M, ADDs: {adds:.6f} M, BOPs: {bops:.6f} M')
