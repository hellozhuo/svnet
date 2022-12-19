"""
Author: Zhuo Su
Time: 2/15/2022
"""

import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from .sv_layers import *
from .utils.sv_util import *

class SVPointNetEncoder(nn.Module):
    def __init__(self, k, binary):
        super(SVPointNetEncoder, self).__init__()
        self.k = k
        self.binary=binary

        self.init_scalar = Vector2Scalar(3, 3)
        self.conv_pos = SVBlock((9, 3), (64//2, 64//6))
        self.conv1 = SVBlock((64//2, 64//6), (64//2, 64//6), binary=binary)

        self.fstn = SV_STNkd((64//2, 64//6), binary=binary)

        self.conv2 = SVBlock((64//2*2, 64//6*2), (128//2, 128//6), binary=binary)
        self.conv3 = SVBlock((128//2, 128//6), (1024//2, 1024//6), binary=binary)

        self.conv_fuse = SVBlock((1024//2*2, 1024//6*2), (1024//2, 1024//6), binary=binary)

        self.svfuse = SVFuse(1024//6, 3, binary=binary)
        
    def forward(self, x):
        B, D, N = x.size()
        
        x = x.unsqueeze(1)
        v = get_graph_feature_cross(x, k=self.k) # B, N_points, k, 3, 3
        s = self.init_scalar(v) # B, N_points, k, 9
        x = (s, v)
        x = self.conv_pos(x)
        x = svpool(x)
        
        x = self.conv1(x) # B, N_points, [3,] dim
        
        x_global = self.fstn(x)
        x_global = (x_global[0].unsqueeze(1).expand_as(x[0]), x_global[1].unsqueeze(1).expand_as(x[1]))
        x = svcat([x, x_global])
        
        x = self.conv2(x)
        x = self.conv3(x) # B, N_points, [3,] 1024//(2,6)

        x_mean = svpool(x, dim=1, keepdim=True) # B, 1, [3,] 1024//(2,6)
        x_mean = (x_mean[0].expand_as(x[0]), x_mean[1].expand_as(x[1]))
        x = svcat([x, x_mean])
        x = self.conv_fuse(x)
        
        x = svpool(x, dim=1) # B, [3,] 1024//2 or 1024//6
        x = self.svfuse(x) # B, 1024//2+1024//6*3
        
        return x

class SV_PointNet_CLS(nn.Module):
    def __init__(self, args, num_class=40):
        super(SV_PointNet_CLS, self).__init__()
        self.binary = args.binary
        self.k = args.k
        p = 0 if self.binary else 0.4

        self.feat = SVPointNetEncoder(k=self.k, binary=self.binary)
        self.fc1 = Linear(1024//2+1024//6*3, 512, bias=False, bw=self.binary, ba=self.binary)
        self.fc2 = Linear(512, 256, bias=False, bw=self.binary, ba=self.binary)
        self.fc3 = nn.Linear(256, num_class)
        self.dropout = nn.Dropout(p=p)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return x
