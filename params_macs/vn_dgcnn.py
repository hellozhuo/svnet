#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Modified from https://github.com/FlyingGiraffe/vnn-pc
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

from models.vn_layers import *
from models.utils.vn_util import get_graph_feature
from macs import get_mac, get_param

EPS = 1e-6

class VNStdFeature_mac(nn.Module):
    def __init__(self, in_channels, dim=4, normalize_frame=False, share_nonlinearity=False, negative_slope=0.2):
        super(VNStdFeature_mac, self).__init__()
        self.dim = dim
        self.normalize_frame = normalize_frame
        self.in_channels = in_channels
        
        self.vn1 = VNLinearLeakyReLU(in_channels, in_channels//2, dim=dim, share_nonlinearity=share_nonlinearity, negative_slope=negative_slope)
        self.vn2 = VNLinearLeakyReLU(in_channels//2, in_channels//4, dim=dim, share_nonlinearity=share_nonlinearity, negative_slope=negative_slope)
        if normalize_frame:
            self.vn_lin = nn.Linear(in_channels//4, 2, bias=False)
        else:
            self.vn_lin = nn.Linear(in_channels//4, 3, bias=False)
    
    def forward(self, x, macs):
        '''
        x: point features of shape [B, N_feat, 3, N_samples, ...]
        '''
        z0 = x
        macs = get_mac(macs, 'VNLinearLeakyReLU', z0, (self.in_channels, self.in_channels//2))
        z0 = self.vn1(z0)
        macs = get_mac(macs, 'VNLinearLeakyReLU', z0, (self.in_channels//2, self.in_channels//4))
        z0 = self.vn2(z0)
        macs = get_mac(macs, 'nn_Linear', z0.transpose(1, -1), (self.in_channels//4, 3))
        z0 = self.vn_lin(z0.transpose(1, -1)).transpose(1, -1)
        
        if self.normalize_frame:
            # make z0 orthogonal. u2 = v2 - proj_u1(v2)
            v1 = z0[:,0,:]
            #u1 = F.normalize(v1, dim=1)
            v1_norm = torch.sqrt((v1*v1).sum(1, keepdims=True))
            u1 = v1 / (v1_norm+EPS)
            v2 = z0[:,1,:]
            v2 = v2 - (v2*u1).sum(1, keepdims=True)*u1
            #u2 = F.normalize(u2, dim=1)
            v2_norm = torch.sqrt((v2*v2).sum(1, keepdims=True))
            u2 = v2 / (v2_norm+EPS)

            # compute the cross product of the two output vectors        
            u3 = torch.cross(u1, u2)
            z0 = torch.stack([u1, u2, u3], dim=1).transpose(1, 2)
        else:
            z0 = z0.transpose(1, 2)
        
        macs = get_mac(macs, 'einsum', x, z0.size(2))
        if self.dim == 4:
            x_std = torch.einsum('bijm,bjkm->bikm', x, z0)
        elif self.dim == 3:
            x_std = torch.einsum('bij,bjk->bik', x, z0)
        elif self.dim == 5:
            x_std = torch.einsum('bijmn,bjkmn->bikmn', x, z0)
        
        return x_std, z0, macs

class VN_DGCNN_CLS_mac(nn.Module):
    def __init__(self, args, num_class=40):
        super(VN_DGCNN_CLS_mac, self).__init__()
        self.args = args
        self.k = args.k
        self.binary = False
        
        self.conv1 = VNLinearLeakyReLU(2, 64//3)
        self.conv2 = VNLinearLeakyReLU(64//3*2, 64//3)
        self.conv3 = VNLinearLeakyReLU(64//3*2, 128//3)
        self.conv4 = VNLinearLeakyReLU(128//3*2, 256//3)

        self.conv5 = VNLinearLeakyReLU(256//3+128//3+64//3*2, 1024//3, dim=4, share_nonlinearity=True)
        
        self.std_feature = VNStdFeature_mac(1024//3*2, dim=4, normalize_frame=False)
        self.linear1 = nn.Linear((1024//3)*12, 512)
        
        self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, num_class)
        
        if args.pooling == 'max':
            self.pool1 = VNMaxPool(64//3)
            self.pool2 = VNMaxPool(64//3)
            self.pool3 = VNMaxPool(128//3)
            self.pool4 = VNMaxPool(256//3)
        elif args.pooling == 'mean':
            self.pool1 = mean_pool
            self.pool2 = mean_pool
            self.pool3 = mean_pool
            self.pool4 = mean_pool

    def forward(self, x):
        macs = (0.0, 0.0, 0.0)

        batch_size = x.size(0)
        x = x.unsqueeze(1)
        x = get_graph_feature(x, k=self.k)
        macs = get_mac(macs, 'VNLinearLeakyReLU', x, (2, 64//3))
        x = self.conv1(x)
        x1 = self.pool1(x)
        
        x = get_graph_feature(x1, k=self.k)
        macs = get_mac(macs, 'VNLinearLeakyReLU', x, (64//3*2, 64//3))
        x = self.conv2(x)
        x2 = self.pool2(x)
        
        x = get_graph_feature(x2, k=self.k)
        macs = get_mac(macs, 'VNLinearLeakyReLU', x, (64//3*2, 128//3))
        x = self.conv3(x)
        x3 = self.pool3(x)
        
        x = get_graph_feature(x3, k=self.k)
        macs = get_mac(macs, 'VNLinearLeakyReLU', x, (128//3*2, 256//3))
        x = self.conv4(x)
        x4 = self.pool4(x)
        
        x = torch.cat((x1, x2, x3, x4), dim=1)
        macs = get_mac(macs, 'VNLinearLeakyReLU_Share', x, (256//3+128//3+64//3*2, 1024//3))
        x = self.conv5(x)
        
        num_points = x.size(-1)
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1)
        x, trans, macs = self.std_feature(x, macs)
        x = x.view(batch_size, -1, num_points)
        
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)
        
        macs = get_mac(macs, 'LinearS', x, ((1024//3)*12, 512))
        x = F.leaky_relu(self.bn1(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        macs = get_mac(macs, 'LinearS', x, (512, 256))
        x = F.leaky_relu(self.bn2(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        macs = get_mac(macs, 'nn_Linear', x, (256, 40))
        x = self.linear3(x)
        
        return macs


class VN_DGCNN_PSEG_mac(nn.Module):
    def __init__(self, args, num_part=50):
        super(VN_DGCNN_PSEG_mac, self).__init__()
        self.args = args
        self.k = args.k
        self.binary = False
        
        self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(256)
        self.bn9 = nn.BatchNorm1d(256)
        self.bn10 = nn.BatchNorm1d(128)
        
        self.conv1 = VNLinearLeakyReLU(2, 64//3)
        self.conv2 = VNLinearLeakyReLU(64//3, 64//3)
        self.conv3 = VNLinearLeakyReLU(64//3*2, 64//3)
        self.conv4 = VNLinearLeakyReLU(64//3, 64//3)
        self.conv5 = VNLinearLeakyReLU(64//3*2, 64//3)
        
        if args.pooling == 'max':
            self.pool1 = VNMaxPool(64//3)
            self.pool2 = VNMaxPool(64//3)
            self.pool3 = VNMaxPool(64//3)
        elif args.pooling == 'mean':
            self.pool1 = mean_pool
            self.pool2 = mean_pool
            self.pool3 = mean_pool
        
        self.conv6 = VNLinearLeakyReLU(64//3*3, 1024//3, dim=4, share_nonlinearity=True)
        self.std_feature = VNStdFeature_mac(1024//3*2, dim=4, normalize_frame=False)
        self.conv8 = nn.Sequential(nn.Conv1d(2299, 256, kernel_size=1, bias=False),
                               self.bn8,
                               nn.LeakyReLU(negative_slope=0.2))
        
        self.conv7 = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        
        self.dp1 = nn.Dropout(p=0.5)
        self.conv9 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp2 = nn.Dropout(p=0.5)
        self.conv10 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                   self.bn10,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv11 = nn.Conv1d(128, num_part, kernel_size=1, bias=False)
        

    def forward(self, x, l):
        macs = (0.0, 0.0, 0.0)

        batch_size = x.size(0)
        num_points = x.size(2)
        
        x = x.unsqueeze(1)
        
        x = get_graph_feature(x, k=self.k)
        macs = get_mac(macs, 'VNLinearLeakyReLU', x, (2, 64//3))
        x = self.conv1(x)
        macs = get_mac(macs, 'VNLinearLeakyReLU', x, (64//3, 64//3))
        x = self.conv2(x)
        x1 = self.pool1(x)
        
        x = get_graph_feature(x1, k=self.k)
        macs = get_mac(macs, 'VNLinearLeakyReLU', x, (64//3*2, 64//3))
        x = self.conv3(x)
        macs = get_mac(macs, 'VNLinearLeakyReLU', x, (64//3, 64//3))
        x = self.conv4(x)
        x2 = self.pool2(x)
        
        x = get_graph_feature(x2, k=self.k)
        macs = get_mac(macs, 'VNLinearLeakyReLU', x, (64//3*2, 64//3))
        x = self.conv5(x)
        x3 = self.pool3(x)
        
        x123 = torch.cat((x1, x2, x3), dim=1)
        
        macs = get_mac(macs, 'VNLinearLeakyReLU_Share', x123, (64//3*3, 1024//3))
        x = self.conv6(x123)
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1)
        x, z0, macs = self.std_feature(x, macs)
        macs = get_mac(macs, 'einsum', x123, z0.size(2))
        x123 = torch.einsum('bijm,bjkm->bikm', x123, z0).view(batch_size, -1, num_points)
        x = x.view(batch_size, -1, num_points)
        x = x.max(dim=-1, keepdim=True)[0]

        l = l.view(batch_size, -1, 1)
        macs = get_mac(macs, 'nn_Conv1dS', l, (16, 64))
        l = self.conv7(l)

        x = torch.cat((x, l), dim=1)
        x = x.repeat(1, 1, num_points)

        x = torch.cat((x, x123), dim=1)

        macs = get_mac(macs, 'nn_Conv1dS', x, (2299, 256))
        x = self.conv8(x)
        x = self.dp1(x)
        macs = get_mac(macs, 'nn_Conv1dS', x, (256, 256))
        x = self.conv9(x)
        x = self.dp2(x)
        macs = get_mac(macs, 'nn_Conv1dS', x, (256, 128))
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
    args.pooling = 'mean'

    model = VN_DGCNN_CLS_mac(args)
    params = get_param(model)

    x = torch.rand(2, 3, 1024)
    macs, adds, bops = model(x)
    macs /= 1e6 * 2
    adds /= 1e6 * 2
    bops /= 1e6 * 2
    print(f'Params VN_DGCNN on ModelNet40: {params:.6f} M, MACs: {macs:.6f} M, ADDs: {adds:.6f} M, BOPs: {bops:.6f} M')

    args = Obj()
    args.k = 40
    args.dropout = 0
    args.pooling = 'mean'

    model = VN_DGCNN_PSEG_mac(args)
    params = get_param(model)

    x = torch.rand(2, 3, 2048)
    l = torch.rand(2, 16)
    macs, adds, bops = model(x, l)
    macs /= 1e6 * 2
    adds /= 1e6 * 2
    bops /= 1e6 * 2
    print(f'Params of VN_DGCNN on ShapeNet: {params:.6f} M, MACs: {macs:.6f} M, ADDs: {adds:.6f} M, BOPs: {bops:.6f} M')
