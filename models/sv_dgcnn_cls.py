"""
Author: Zhuo Su
Time: 1/27/2022 21:53
"""


import os
import sys
import copy
import math
import numpy as np
from functools import reduce
import operator

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from .sv_layers import *
from .utils.sv_util import *

class SV_DGCNN_CLS(nn.Module):
    def __init__(self, args, num_class=40):
        super(SV_DGCNN_CLS, self).__init__()
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
        batch_size = x.size(0)
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

        x = svcat([x1, x2, x3, x4])
        x = self.conv5(x)
        x = self.svfuse(x) # B, N_points, 1024
        x = x.transpose(-1, -2).contiguous() # B, 1024, N_points

        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn1(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn2(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        
        return x

