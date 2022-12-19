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

from .utils.sv_util import svpool

EPS = 1e-6

class Linear(nn.Linear):
    def __init__(self, in_channels, out_channels, bias, bw=False, ba=False):
        super(Linear, self).__init__(in_channels, out_channels, bias)
        self.bw, self.ba = bw, ba
        if ba:
            self.beta = nn.Parameter(torch.zeros(1, in_channels))
        if bw:
            self.scale = nn.Parameter(torch.ones(1, out_channels)/math.sqrt(in_channels))

    def forward(self, x):
        if not self.bw and not self.ba:
            return super().forward(x)
        shape_x = x.shape
        x = x.view(-1, shape_x[-1])
        w = self.weight

        if self.ba:
            x = x + self.beta.expand_as(x)
            if not self.training:
                x = torch.sign(x)
            else:
                x = torch.clamp(x, -1.2, 1.2)
                x = torch.sign(x).detach() + x - x.detach()
        if self.bw:
            if not self.training:
                w = torch.sign(w)
            else:
                w = torch.clamp(w, -1.2, 1.2)
                w = torch.sign(w).detach() + w - w.detach()
        x = F.linear(x, w) * self.scale
        if self.bias is not None:
            x = x + self.bias
        ta = x.view(shape_x[:-1] + (x.shape[-1],))
        return x.view(shape_x[:-1] + (x.shape[-1],))

class Conv1d(nn.Conv1d):
    def __init__(self, in_channels, out_channels, binary=False):
        super(Conv1d, self).__init__(in_channels, out_channels, 1, bias=False)
        print('Conv1d: ', in_channels, out_channels)
        self.binary = binary
        if binary:
            self.beta = nn.Parameter(torch.zeros(1, in_channels, 1))
            self.scale = nn.Parameter(torch.ones(1, out_channels, 1)/math.sqrt(in_channels))

    def forward(self, x):
        if not self.binary:
            return super().forward(x)
        x = x + self.beta.expand_as(x)
        if not self.training:
            x = torch.sign(x)
            w = torch.sign(self.weight)
        else:
            x = torch.clamp(x, -1.2, 1.2)
            x = torch.sign(x).detach() + x - x.detach()
            w = torch.clamp(self.weight, -1.2, 1.2)
            w = torch.sign(w).detach() + w - w.detach()

        x = F.conv1d(x, w, padding=0) * self.scale
        return x


class VectorBN(nn.Module):
    def __init__(self, dim):
        super(VectorBN, self).__init__()
        self.bn = nn.BatchNorm1d(dim)
    
    def forward(self, v):
        '''
        shape of v: B, N_points, [k,] 3, dim
        '''
        # norm = torch.sqrt((x*x).sum(2))
        shape_v = v.shape
        dim = v.size()[-1]

        norm = torch.norm(v, dim=-2) + EPS
        norm = norm.view(-1, dim)
        norm_bn = self.bn(norm)

        norm = norm.view(shape_v[:-2] + (1, dim))
        norm_bn = norm_bn.view(shape_v[:-2] + (1, dim))
        v = v / norm * norm_bn
        
        return v

class Vector2Scalar(nn.Module):
    def __init__(self, v_dim, multi, binary=False, trans_back=False):
        super(Vector2Scalar, self).__init__()

        self.trans_back = trans_back
        self.linear = Linear(v_dim, multi, bias=False, bw=binary)
        
    def forward(self, v):
        '''
        shape of v: B, N_points, [k,] 3, dim
        '''
        assert v.ndim in [3, 4, 5], 'dim of v should be in [4, 5], got {}'.format(v.ndim)
        z = self.linear(v) # B, [N_points,] [k,] 3, multi
        v = v.transpose(-1, -2) # B, [N_points,] [k,] dim, 3
        if v.ndim == 3:
            s = torch.einsum('bdi,bij->bdj', v, z)
        elif v.ndim == 4:
            s = torch.einsum('bndi,bnij->bndj', v, z)
        elif v.ndim == 5:
            s = torch.einsum('bnkdi,bnkij->bnkdj', v, z)

        s = s.view(z.shape[:-2] + (-1,))
        if self.trans_back:
            return s, z
        else:
            return s

class VectorReLU(nn.Module):
    def __init__(self):
        super(VectorReLU, self).__init__()
        self.div = 10

    def forward(self, x):
        '''
        shape of v: B, N_points, [k,] 3, dim
        '''
        shape_x = x.shape
        batch_size, v_dim = shape_x[0], shape_x[-1]

        x = x.view(batch_size, -1, 3, v_dim)
        k = x.shape[1] // self.div
        xnorm = torch.norm(x, dim=2, keepdim=True).detach()
        kx, _ = torch.kthvalue(xnorm, k, dim=1, keepdim=True)
        x = torch.where(xnorm > kx, x, torch.zeros_like(x, device=x.device))
        x = x.view(shape_x)
        return x

class SVBlock(nn.Module):
    def __init__(self, in_dims, out_dims, binary=False):
        super(SVBlock, self).__init__()
        print('SVBlock: ', in_dims, out_dims)

        self.gate = nn.Sequential(
                nn.Linear(in_dims[0], out_dims[1]//2, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(out_dims[1]//2, out_dims[1], bias=False),
                nn.Sigmoid()
                )

        self.v2s = Vector2Scalar(in_dims[1], 3, binary=binary)
        
        self.linear1 = Linear(in_dims[0] + in_dims[1] * 3, out_dims[0], bias=False, bw=binary, ba=binary)
        self.bn1 = nn.BatchNorm1d(out_dims[0])
        self.relu = nn.LeakyReLU(negative_slope=0.2)

        self.linear2 = Linear(in_dims[1], out_dims[1], bias=False, bw=binary)
        self.bn2 = VectorBN(out_dims[1])

    def forward(self, x):
        '''
        shape of s: B, N_points, [k,] s_dim
        shape of v: B, N_points, [k,] 3, v_dim
        '''
        s, v = x

        _s = s.view(s.shape[0], -1, s.shape[-1])
        _s = _s.mean(dim=1)
        v_scale = self.gate(_s)
        for i in range(v.ndim-2):
            v_scale = v_scale.unsqueeze(-2)

        s_v = self.v2s(v)
        s = torch.cat([s, s_v], dim=-1)
        s = self.linear1(s)
        shape_s = s.shape
        s = self.bn1(s.view(-1, shape_s[-1])).view(shape_s)
        s = self.relu(s)

        v = self.linear2(v)
        v = self.bn2(v)
        v = v * v_scale

        return (s, v)
        
class SVFuse(nn.Module):
    def __init__(self, v_dim, multi, binary, trans_back=False):
        super(SVFuse, self).__init__()
        print('SVFuse: ', v_dim)
        self.trans_back = trans_back

        self.v2s = Vector2Scalar(v_dim, multi, binary=binary, trans_back=trans_back)

    def forward(self, x):
        '''
        shape of s: B, N_points, [k,] s_dim
        shape of v: B, N_points, [k,] 3, v_dim
        '''
        s, v = x

        if self.trans_back:
            s_v, trans = self.v2s(v)
            s = torch.cat([s, s_v], dim=-1)
            return s, trans
        else:
            s_v = self.v2s(v)
            s = torch.cat([s, s_v], dim=-1)
            return s

class SV_STNkd(nn.Module):
    def __init__(self, dim, binary):
        super(SV_STNkd, self).__init__()

        self.conv1 = SVBlock(dim, (64//2, 64//6), binary=binary)
        self.conv2 = SVBlock((64//2, 64//6), (128//2, 128//6), binary=binary)
        self.conv3 = SVBlock((128//2, 128//6), (1024//2, 1024//6), binary=binary)

        self.fc1 = SVBlock((1024//2, 1024//6), (512//2, 512//6), binary=binary)
        self.fc2 = SVBlock((512//2, 512//6), (256//2, 256//6), binary=binary)
        self.fc3 = SVBlock((256//2, 256//6), dim, binary=binary)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x) # B, N_points, [3,] 1024//(2,6)
        x = svpool(x, dim=1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x) # B, [3,] dim
        
        return x

