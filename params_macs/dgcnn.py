#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Modified from https://github.com/antao97/dgcnn.pytorch
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

from models.utils.dgcnn_util import get_graph_feature

import operator
from functools import reduce

def get_param(model):
    n = sum([reduce(operator.mul, i.size(), 1) for i in model.parameters()])
    return float(n) / 1e6 * 32

class DGCNN_CLS_mac(nn.Module):
    def __init__(self, args, output_channels=40):
        super(DGCNN_CLS_mac, self).__init__()
        self.k = args.k
        self.dropout = args.dropout
        self.emb_dims = 1024
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(self.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, self.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(self.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=self.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=self.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        macs = 0.0

        batch_size, _, num_points = x.size()
        x = get_graph_feature(x, k=self.k)      # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        macs += batch_size * num_points * self.k * 6 * 64
        macs += batch_size * num_points * self.k * 64 * 2
        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        macs += batch_size * num_points * self.k * 64 * 2 * 64
        macs += batch_size * num_points * self.k * 64 * 2
        x = self.conv2(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        macs += batch_size * num_points * self.k * 64 * 2 * 128
        macs += batch_size * num_points * self.k * 128 * 2
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 128, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = get_graph_feature(x3, k=self.k)     # (batch_size, 128, num_points) -> (batch_size, 128*2, num_points, k)
        macs += batch_size * num_points * self.k * 128 * 2 * 256
        macs += batch_size * num_points * self.k * 256 * 2
        x = self.conv4(x)                       # (batch_size, 128*2, num_points, k) -> (batch_size, 256, num_points, k)
        x4 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 256, num_points, k) -> (batch_size, 256, num_points)

        x = torch.cat((x1, x2, x3, x4), dim=1)  # (batch_size, 64+64+128+256, num_points)

        macs += batch_size * num_points * (64+64+128+256) * 1024
        macs += batch_size * num_points * 1024 * 2
        x = self.conv5(x)                       # (batch_size, 64+64+128+256, num_points) -> (batch_size, emb_dims, num_points)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)           # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)           # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        x = torch.cat((x1, x2), 1)              # (batch_size, emb_dims*2)

        macs += batch_size * 1024 * 2 * 512
        macs += batch_size * 512 * 2
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2) # (batch_size, emb_dims*2) -> (batch_size, 512)
        x = self.dp1(x)
        macs += batch_size * 512 * 256
        macs += batch_size * 256 * 2
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2) # (batch_size, 512) -> (batch_size, 256)
        x = self.dp2(x)
        macs += batch_size * 256 * 40
        x = self.linear3(x)                                             # (batch_size, 256) -> (batch_size, output_channels)
        
        return macs


class Transform_Net_mac(nn.Module):
    def __init__(self, args):
        super(Transform_Net_mac, self).__init__()
        self.args = args
        self.k = 3

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn3 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn4 = nn.BatchNorm1d(256)

        self.transform = nn.Linear(256, 3*3)
        init.constant_(self.transform.weight, 0)
        init.eye_(self.transform.bias.view(3, 3))

    def forward(self, x):
        macs = 0.0

        batch_size, _, num_points, k = x.size()

        macs += batch_size * num_points * k * 6 * 64
        macs += batch_size * num_points * k * 64 * 2
        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        macs += batch_size * num_points * k * 64 * 128
        macs += batch_size * num_points * k * 128 * 2
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 128, num_points, k)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        macs += batch_size * num_points * 128 * 1024
        macs += batch_size * num_points * 1024 * 2
        x = self.conv3(x)                       # (batch_size, 128, num_points) -> (batch_size, 1024, num_points)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 1024, num_points) -> (batch_size, 1024)

        macs += batch_size * 1024 * 512
        macs += batch_size * 512 * 2
        x = F.leaky_relu(self.bn3(self.linear1(x)), negative_slope=0.2)     # (batch_size, 1024) -> (batch_size, 512)
        macs += batch_size * 512 * 256
        macs += batch_size * 256 * 2
        x = F.leaky_relu(self.bn4(self.linear2(x)), negative_slope=0.2)     # (batch_size, 512) -> (batch_size, 256)

        macs += batch_size * 256 * 9
        x = self.transform(x)                   # (batch_size, 256) -> (batch_size, 3*3)
        x = x.view(batch_size, 3, 3)            # (batch_size, 3*3) -> (batch_size, 3, 3)

        return x, macs


class DGCNN_PSEG_mac(nn.Module):
    def __init__(self, args, seg_num_all):
        super(DGCNN_PSEG_mac, self).__init__()
        self.seg_num_all = seg_num_all
        self.k = args.k
        self.dropout = args.dropout
        self.emb_dims = 1024
        self.transform_net = Transform_Net_mac(args)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(self.emb_dims)
        self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(256)
        self.bn9 = nn.BatchNorm1d(256)
        self.bn10 = nn.BatchNorm1d(128)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, self.emb_dims, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(1280, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=self.dropout)
        self.conv9 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp2 = nn.Dropout(p=self.dropout)
        self.conv10 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                   self.bn10,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv11 = nn.Conv1d(128, self.seg_num_all, kernel_size=1, bias=False)
        

    def forward(self, x, l):
        macs = 0.0
        batch_size = x.size(0)
        num_points = x.size(2)

        x0 = get_graph_feature(x, k=self.k)     # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        t, mact = self.transform_net(x0)              # (batch_size, 3, 3)
        macs += mact
        x = x.transpose(2, 1)                   # (batch_size, 3, num_points) -> (batch_size, num_points, 3)
        macs += batch_size * num_points * 9
        x = torch.bmm(x, t)                     # (batch_size, num_points, 3) * (batch_size, 3, 3) -> (batch_size, num_points, 3)
        x = x.transpose(2, 1)                   # (batch_size, num_points, 3) -> (batch_size, 3, num_points)

        x = get_graph_feature(x, k=self.k)      # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        macs += batch_size * num_points * self.k * 6 * 64
        macs += batch_size * num_points * self.k * 64 * 2
        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        macs += batch_size * num_points * self.k * 64 * 64
        macs += batch_size * num_points * self.k * 64 * 2
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        macs += batch_size * num_points * self.k * 128 * 64
        macs += batch_size * num_points * self.k * 64 * 2
        x = self.conv3(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        macs += batch_size * num_points * self.k * 64 * 64
        macs += batch_size * num_points * self.k * 64 * 2
        x = self.conv4(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)     # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        macs += batch_size * num_points * self.k * 128 * 64
        macs += batch_size * num_points * self.k * 64 * 2
        x = self.conv5(x)                       # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]    # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x1, x2, x3), dim=1)      # (batch_size, 64*3, num_points)

        macs += batch_size * num_points * 64 * 3 * 1024
        macs += batch_size * num_points * 1024 * 2
        x = self.conv6(x)                       # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]      # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        l = l.view(batch_size, -1, 1)           # (batch_size, num_categoties, 1)
        macs += batch_size * 16 * 64
        macs += batch_size * 64 * 2
        l = self.conv7(l)                       # (batch_size, num_categoties, 1) -> (batch_size, 64, 1)

        x = torch.cat((x, l), dim=1)            # (batch_size, 1088, 1)
        x = x.repeat(1, 1, num_points)          # (batch_size, 1088, num_points)

        x = torch.cat((x, x1, x2, x3), dim=1)   # (batch_size, 1088+64*3, num_points)

        macs += batch_size * num_points * (1088+64*3) * 256
        macs += batch_size * num_points * 256 * 2
        x = self.conv8(x)                       # (batch_size, 1088+64*3, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        macs += batch_size * num_points * 256 * 256
        macs += batch_size * num_points * 256 * 2
        x = self.conv9(x)                       # (batch_size, 256, num_points) -> (batch_size, 256, num_points)
        x = self.dp2(x)
        macs += batch_size * num_points * 256 * 128
        macs += batch_size * num_points * 128 * 2
        x = self.conv10(x)                      # (batch_size, 256, num_points) -> (batch_size, 128, num_points)
        macs += batch_size * num_points * 128 * 50
        x = self.conv11(x)                      # (batch_size, 128, num_points) -> (batch_size, seg_num_all, num_points)
        
        return macs

if __name__ == '__main__':
    class Obj(): pass
    args = Obj()
    args.k = 20
    args.dropout = 0

    model = DGCNN_CLS_mac(args, 20)
    params = get_param(model)

    x = torch.rand(2, 3, 1024)
    macs = model(x) / 1e6 / 2
    print(f'Params of DGCNN on ModelNet40: {params:.6f} M, MACs: {macs:.6f} M')

    args = Obj()
    args.k = 40
    args.dropout = 0

    model = DGCNN_PSEG_mac(args, 50)
    params = get_param(model)

    x = torch.rand(2, 3, 2048)
    l = torch.rand(2, 16)
    macs = model(x, l) / 1e6 / 2
    print(f'Params of DGCNN on ShapeNet: {params:.6f} M, MACs: {macs:.6f} M')
