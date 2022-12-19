"""
Author: Zhuo Su
Time: 2/17/2022 10:21
"""

import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from .sv_layers import *
from .utils.sv_util import *

class SV_PointNet_PSEG(nn.Module):
    def __init__(self, args, num_part=50):
        super(SV_PointNet_PSEG, self).__init__()
        self.k = args.k
        self.binary = args.binary

        self.init_scalar = Vector2Scalar(3, 3)
        self.conv_pos = SVBlock((9, 3), (64//2, 64//6))
        self.conv1 = SVBlock((64//2, 64//6), (64//2, 64//6), binary=self.binary)
        self.conv2 = SVBlock((64//2, 64//6), (128//2, 128//6), binary=self.binary)
        self.conv3 = SVBlock((128//2, 128//6), (128//2, 128//6), binary=self.binary)
        self.fstn = SV_STNkd((128//2, 128//6), binary=self.binary)
        self.conv4 = SVBlock((128//2*2, 128//6*2), (512//2, 512//6), binary=self.binary)
        self.conv5 = SVBlock((512//2, 512//6), (2048//2, 2048//6), binary=self.binary)

        self.svfuse = SVFuse(2048//6*2, 3, binary=self.binary, trans_back=True) # B, N, 2048//2*2+2048//6*2*3
        self.channels = 2048//2*2+2048//6*2*3
        self.conv_fuse1 = nn.Sequential(
                Conv1d(self.channels, self.channels//8, binary=self.binary),
                nn.BatchNorm1d(self.channels//8),
                nn.ReLU(inplace=True)
                )
        self.conv_fuse2 = nn.Sequential(
                Conv1d(self.channels//8, self.channels, binary=self.binary),
                nn.BatchNorm1d(self.channels),
                nn.ReLU(inplace=True)
                )
        self.convs1 = nn.Sequential(
                Conv1d(self.channels+16+64//2+128//2*2+512//2+2048//2+(64//6+128//6*2+512//6+2048//6)*3, 256, binary=self.binary),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True))
        self.convs2 = nn.Sequential(
                Conv1d(256, 256, binary=self.binary),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True))
        self.convs3 = nn.Sequential(
                Conv1d(256, 128, binary=self.binary),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True))
        self.convs4 = nn.Conv1d(128, num_part, 1)

    def forward(self, x, l):
        B, D, N = x.size()
        
        x = x.unsqueeze(1)
        v = get_graph_feature_cross(x, k=self.k) # B, N_points, k, 3, 3
        s = self.init_scalar(v) # B, N_points, k, 9
        x = (s, v)
        x = self.conv_pos(x)
        x = svpool(x)

        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)

        x_global = self.fstn(out3)
        x_global = (x_global[0].unsqueeze(1).expand_as(out3[0]), x_global[1].unsqueeze(1).expand_as(out3[1]))
        x_transformed = svcat([out3, x_global])
        out4 = self.conv4(x_transformed)
        out5 = self.conv5(out4)

        x_mean = svpool(out5, dim=1, keepdim=True, spool='mean') # B, 1, [3,] 2048//(2,6)
        x_mean = (x_mean[0].expand_as(out5[0]), x_mean[1].expand_as(out5[1]))
        x = svcat([out5, x_mean]) # B, N, [3,] 4096//(2,6)
        x, trans = self.svfuse(x) # B, N, 2048//2*2+2048//6*2*3 ~ 2048
        x = x.transpose(-1, -2).contiguous()
        x = self.conv_fuse1(x) # B, self.channels//2, N
        x = self.conv_fuse2(x) # B, self.channels, N
        if self.binary:
            x = x.mean(dim=-1)
        else:
            x, _ = x.max(dim=-1)

        x_l = torch.cat([x, l.squeeze(1)], dim=1) # B, ~self.channels+16
        x_l = x_l.view(B, -1, 1).repeat(1, 1, N) # B, ~self.channels, N

        concat = svcat([out1, out2, out3, out4, out5])
        concat_v = torch.einsum('bimj,bijk->bimk', concat[1].transpose(-1, -2), trans).view(B, N, -1)
        concat = torch.cat([concat[0], concat_v], dim=-1).transpose(-1, -2).contiguous() # B, ~64+128+128+512, N
        concat = torch.cat([x_l, concat], dim=1) # B, ~D, N
        net = self.convs1(concat)
        net = self.convs2(net)
        net = self.convs3(net)
        net = self.convs4(net)

        return net
