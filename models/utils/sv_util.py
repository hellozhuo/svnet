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


def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None, x_coord=None, first=False):
    # B, 1, 3, 1024
    batch_size = x.size(0)
    num_points = x.size(3)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if x_coord is None: # dynamic knn graph
            idx = knn(x, k=k)   # (batch_size, num_points, k)
        else:          # fixed knn graph with input point coordinates
            x_coord = x_coord.view(batch_size, -1, num_points)
            idx = knn(x_coord, k=k)
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()
    num_dims = num_dims // 3

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims, 3) 
    x = x.view(batch_size, num_points, 1, num_dims, 3).repeat(1, 1, k, 1, 1)

    if first:
        feature = feature - x
        feature_mean = feature.mean(dim=2, keepdim=True).repeat(1, 1, k, 1, 1)
        feature = torch.cat((feature, feature_mean), dim=3).transpose(-1, -2).contiguous()
    else:
        feature = torch.cat((feature-x, x), dim=3).transpose(-1, -2).contiguous()
  
    return feature

def get_graph_feature_cross(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(3)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()
    num_dims = num_dims // 3

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims, 3) 
    x = x.view(batch_size, num_points, 1, num_dims, 3).repeat(1, 1, k, 1, 1)
    cross = torch.cross(feature, x, dim=-1)
    
    feature = torch.cat((feature-x, x, cross), dim=3).transpose(-1, -2).contiguous()
  
    return feature

def get_graph_feature_sv(x, k=20, idx=None):
    '''
    shape of s: B, N_points, s_dim
    shape of v: B, N_points, 3, v_dim
    '''
    s, v = x
    batch_size, num_points, s_dim = s.size()
    v_dim = v.size(-1)

    if idx is None:
        x_ = torch.cat([s, v.view(batch_size, num_points, -1)], dim=-1)
        idx = knn(x_.transpose(-1, -2), k=k)
        idx_base = torch.arange(0, batch_size, device=s.device).view(-1, 1, 1)*num_points
        idx = idx + idx_base
        idx = idx.view(-1)

    v_feature = v.view(batch_size*num_points, -1)[idx, :]
    v_feature = v_feature.view(batch_size, num_points, k, 3, v_dim) 
    v = v.view(batch_size, num_points, 1, 3, v_dim).repeat(1, 1, k, 1, 1)
    v_feature = torch.cat((v_feature-v, v), dim=-1)

    s_feature = s.view(batch_size*num_points, -1)[idx, :]
    s_feature = s_feature.view(batch_size, num_points, k, s_dim)
    s = s.view(batch_size, num_points, 1, s_dim).repeat(1, 1, k, 1)
    s_feature = torch.cat((s_feature - s, s), dim=-1)
  
    return (s_feature, v_feature)

def svpool(x, dim=2, keepdim=False, spool='max'):
    '''
    shape of s: B, N_points, k, s_dim
    shape of v: B, N_points, k, 3, v_dim
    '''
    s, v = x
    #s = s.mean(dim=2, keepdim=False)
    if spool == 'max':
        s, _ = s.max(dim=dim, keepdim=keepdim)
    elif spool == 'mean':
        s = s.mean(dim=dim, keepdim=keepdim)
    else:
        raise ValueError('not recognized pooling mean {}'.format(spool))
    v = v.mean(dim=dim, keepdim=keepdim)
    return (s, v)

def svcat(xlist):
    '''
    shape of s: B, N_points, [k,] s_dim
    shape of v: B, N_points, [k,] 3, v_dim
    '''
    slist = [x[0] for x in xlist]
    vlist = [x[1] for x in xlist]
    s = torch.cat(slist, dim=-1)
    v = torch.cat(vlist, dim=-1)

    return (s, v)

