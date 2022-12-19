'''
This code is modified from https://github.com/htqin/BiPointNet
'''
from .bipointnet_basic import MeanShift, BiLinear, BiLinearXNOR, BiLinearIRNet, BiLinearLSR, BiLinearBiReal
import math
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import numpy as np
import torch.nn.functional as F


biLinears = {
    'BiLinear': BiLinear,
    'BiLinearXNOR': BiLinearXNOR,
    'BiLinearIRNet': BiLinearIRNet,
    'BiLinearLSR': BiLinearLSR
}


seg_classes = {
    'Airplane': [0, 1, 2, 3],
    'Bag': [4, 5],
    'Cap': [6, 7],
    'Car': [8, 9, 10, 11],
    'Chair': [12, 13, 14, 15],
    'Earphone': [16, 17, 18],
    'Guitar': [19, 20, 21],
    'Knife': [22, 23],
    'Lamp': [24, 25, 26, 27],
    'Laptop': [28, 29],
    'Motorbike': [30, 31, 32, 33, 34, 35],
    'Mug': [36, 37],
    'Pistol': [38, 39, 40],
    'Rocket': [41, 42, 43],
    'Skateboard': [44, 45, 46],
    'Table': [47, 48, 49],
}


offset_map = {
    1024: -3.2041,
    2048: -3.4025,
    4096: -3.5836
}

class Conv1d(nn.Module):
    def __init__(self, inplane, outplane, Linear):
        super().__init__()
        self.lin = Linear(inplane, outplane)

    def forward(self, x):
        B, C, N = x.shape
        x = x.permute(0, 2, 1).contiguous().view(-1, C)
        x = self.lin(x).view(B, N, -1).permute(0, 2, 1).contiguous()
        return x

class BiSTN3d(nn.Module):
    def __init__(self, channel, Linear=BiLinear, pool='max', affine=True, bi_first=False):
        super(BiSTN3d, self).__init__()
        if bi_first:
            self.conv1 = Conv1d(channel, 64, Linear)
        else:
            self.conv1 = Conv1d(channel, 64, nn.Linear)
        self.conv2 = Conv1d(64, 128, Linear)
        self.conv3 = Conv1d(128, 1024, Linear)
        self.fc1 = Linear(1024, 512)
        self.fc2 = Linear(512, 256)
        self.fc3 = Linear(256, 9)

        self.bn1 = nn.BatchNorm1d(64, affine=affine)
        self.bn2 = nn.BatchNorm1d(128, affine=affine)
        self.bn3 = nn.BatchNorm1d(1024, affine=affine)
        self.bn4 = nn.BatchNorm1d(512, affine=affine)
        self.bn5 = nn.BatchNorm1d(256, affine=affine)
        self.pool = pool

    def forward(self, x):

        batchsize, D, N = x.size()
        x = F.hardtanh(self.bn1(self.conv1(x)))
        x = F.hardtanh(self.bn2(self.conv2(x)))

        if self.pool == 'max':
            x = F.hardtanh(self.bn3(self.conv3(x)))
            x = torch.max(x, 2, keepdim=True)[0]
        elif self.pool == 'mean':
            x = F.hardtanh(self.bn3(self.conv3(x)))
            x = torch.mean(x, 2, keepdim=True)
        elif self.pool == 'ema-max':
            x = self.bn3(self.conv3(x)) + offset_map[N]
            x = torch.max(x, 2, keepdim=True)[0]
            x = x.view(-1, 1024)
        x = x.view(-1, 1024)

        x = F.hardtanh(self.bn4(self.fc1(x)))
        x = F.hardtanh(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.eye(3, device=x.device).view(1, 9).repeat(batchsize, 1)
        x = x + iden
        x = x.view(-1, 3, 3)

        return x


class BiSTNkd(nn.Module):
    def __init__(self, k=64, Linear=BiLinear, pool='max', affine=True, bi_first=False):
        super(BiSTNkd, self).__init__()
        self.conv1 = Conv1d(k, 64, Linear)
        self.conv2 = Conv1d(64, 128, Linear)
        self.conv3 = Conv1d(128, 1024, Linear)
        self.fc1 = Linear(1024, 512)
        self.fc2 = Linear(512, 256)
        self.fc3 = Linear(256, k * k)

        self.bn1 = nn.BatchNorm1d(64, affine=affine)
        self.bn2 = nn.BatchNorm1d(128, affine=affine)
        self.bn3 = nn.BatchNorm1d(1024, affine=affine)
        self.bn4 = nn.BatchNorm1d(512, affine=affine)
        self.bn5 = nn.BatchNorm1d(256, affine=affine)
        self.k = k
        self.pool = pool

    def forward(self, x):
        batchsize, D, N = x.size()
        x = F.hardtanh(self.bn1(self.conv1(x)))
        x = F.hardtanh(self.bn2(self.conv2(x)))
        if self.pool == 'max':
            x = F.hardtanh(self.bn3(self.conv3(x)))
            x = torch.max(x, 2, keepdim=True)[0]
        elif self.pool == 'mean':
            x = F.hardtanh(self.bn3(self.conv3(x)))
            x = torch.mean(x, 2, keepdim=True)
        elif self.pool == 'ema-max':
            x = self.bn3(self.conv3(x)) + offset_map[N]
            x = torch.max(x, 2, keepdim=True)[0]
            x = x.view(-1, 1024)
        x = x.view(-1, 1024)

        x = F.hardtanh(self.bn4(self.fc1(x)))
        x = F.hardtanh(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.eye(self.k, device=x.device).view(1, self.k * self.k).repeat(batchsize, 1)
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class BiPointNetEncoder(nn.Module):
    def __init__(self, Linear, global_feat=True, feature_transform=False, channel=3, pool='max', affine=True, tnet=True, bi_first=False, use_bn=True):
        super(BiPointNetEncoder, self).__init__()
        self.tnet = tnet
        if self.tnet:
            self.stn = BiSTN3d(channel, Linear, pool=pool, affine=affine, bi_first=bi_first)
        if bi_first:
            self.conv1 = Conv1d(channel, 64, Linear)
        else:
            self.conv1 = Conv1d(channel, 64, nn.Linear)
        self.conv2 = Conv1d(64, 128, Linear)
        self.conv3 = Conv1d(128, 1024, Linear)
        self.bn1 = nn.BatchNorm1d(64, affine=affine)
        self.bn2 = nn.BatchNorm1d(128, affine=affine)
        self.bn3 = nn.BatchNorm1d(1024, affine=affine)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.tnet and self.feature_transform:
            self.fstn = BiSTNkd(k=64, Linear=Linear, pool=pool, affine=affine, bi_first=bi_first)
        self.pool = pool
        self.use_bn = use_bn

    def forward(self, x):
        B, D, N = x.size()
        if self.tnet:
            trans = self.stn(x)
        else:
            trans = None

        x = x.transpose(2, 1)
        if D == 6:
            x, feature = x.split(3, dim=2)
        elif D == 9:
            x, feature = x.split([3, 6], dim=2)
        if self.tnet:
            x = torch.bmm(x, trans)
        if D > 3:
            x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1)
        if self.use_bn:
            x = F.hardtanh(self.bn1(self.conv1(x)))
        else:
            x = F.hardtanh(self.conv1(x))

        if self.tnet and self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        if self.use_bn:
            x = F.hardtanh(self.bn2(self.conv2(x)))
            x = self.bn3(self.conv3(x))
        else:
            x = F.hardtanh(self.conv2(x))
            x = self.conv3(x)

        if self.pool == 'max':
            x = torch.max(x, 2, keepdim=True)[0]
        elif self.pool == 'mean':
            x = torch.mean(x, 2, keepdim=True)
        elif self.pool == 'ema-max':
            if self.use_bn:
                x = torch.max(x, 2, keepdim=True)[0] + offset_map[N]
            else:
                x = torch.max(x, 2, keepdim=True)[0] - 0.3
            x = x.view(-1, 1024)
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


class BasicBiPointNet(nn.Module):
    def __init__(self, k=40, Linear=BiLinear, pool='max', affine=True, tnet=True, bi_first=False, bi_last=False, use_bn=True):
        super(BasicBiPointNet, self).__init__()
        channel = 3
        self.feat = BiPointNetEncoder(Linear, global_feat=True, feature_transform=True, channel=channel, pool=pool, affine=affine, tnet=tnet, bi_first=bi_first, use_bn=use_bn)
        self.fc1 = Linear(1024, 512)
        self.fc2 = Linear(512, 256)
        if bi_last:
            self.fc3 = Linear(256, k)
        else:
            self.fc3 = nn.Linear(256, k)
        self.use_bn = use_bn
        self.bn1 = nn.BatchNorm1d(512, affine=affine)
        self.bn2 = nn.BatchNorm1d(256, affine=affine)

    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        if self.use_bn:
            x = F.hardtanh(self.bn1(self.fc1(x)))
            x = F.hardtanh(self.bn2(self.fc2(x)))
        else:
            x = F.hardtanh(self.fc1(x))
            x = F.hardtanh(self.fc2(x))
        x = self.fc3(x)
        return (x, trans_feat)

class BasicBiPointNetPartSeg(nn.Module):
    def __init__(self, num_part=50, Linear=BiLinear, pool='max', affine=True, Tnet=True):
        super(BasicBiPointNetPartSeg, self).__init__()
        self.pool = pool
        channel = 3
        self.stn = BiSTN3d(channel, Linear, pool=pool, affine=affine)
        self.conv1 = Conv1d(channel, 64, nn.Linear)
        self.conv2 = Conv1d(64, 128, Linear)
        self.conv3 = Conv1d(128, 128, Linear)
        self.conv4 = Conv1d(128, 512, Linear)
        self.conv5 = Conv1d(512, 2048, Linear)
        self.bn1 = nn.BatchNorm1d(64, affine=affine)
        self.bn2 = nn.BatchNorm1d(128, affine=affine)
        self.bn3 = nn.BatchNorm1d(128, affine=affine)
        self.bn4 = nn.BatchNorm1d(512, affine=affine)
        self.bn5 = nn.BatchNorm1d(2048, affine=affine)
        self.fstn = BiSTNkd(k=128, Linear=Linear, pool=pool, affine=affine)
        self.convs1 = Conv1d(4944, 256, Linear)

        self.convs2 = Conv1d(256, 256, Linear)
        self.convs3 = Conv1d(256, 128, Linear)
        self.convs4 = Conv1d(128, num_part, nn.Linear)
        self.bns1 = nn.BatchNorm1d(256, affine=affine)
        self.bns2 = nn.BatchNorm1d(256, affine=affine)
        self.bns3 = nn.BatchNorm1d(128, affine=affine)

        self.Tnet = Tnet

    def forward(self, point_cloud, label):
        B, D, N = point_cloud.size()

        if self.Tnet:
            trans = self.stn(point_cloud)
            point_cloud = point_cloud.transpose(2, 1)
            if D > 3:
                point_cloud, feature = point_cloud.split(3, dim=2)
            point_cloud = torch.bmm(point_cloud, trans)
            if D > 3:
                point_cloud = torch.cat([point_cloud, feature], dim=2)
            point_cloud = point_cloud.transpose(2, 1)

        out1 = F.hardtanh(self.bn1(self.conv1(point_cloud)))
        out2 = F.hardtanh(self.bn2(self.conv2(out1)))
        out3 = F.hardtanh(self.bn3(self.conv3(out2)))

        if self.Tnet:
            trans_feat = self.fstn(out3)
            x = out3.transpose(2, 1)
            net_transformed = torch.bmm(x, trans_feat)
            net_transformed = net_transformed.transpose(2, 1)
        else:
            net_transformed = out3

        out4 = F.hardtanh(self.bn4(self.conv4(net_transformed)))
        out5 = self.bn5(self.conv5(out4))
        if self.pool == 'max':
            out_pool = torch.max(out5, 2, keepdim=True)[0]
        elif self.pool == 'mean':
            out_pool = torch.mean(out5, 2, keepdim=True)
        elif self.pool == 'ema-max':
            out_pool = torch.max(out5, 2, keepdim=True)[0] + offset_map[N]
        out_pool = out_pool.view(-1, 2048)

        out_pool = torch.cat([out_pool, label.squeeze(1)], 1)
        expand = out_pool.view(-1, 2048+16, 1).repeat(1, 1, N)
        concat = torch.cat([expand, out1, out2, out3, out4, out5], 1)

        net = F.hardtanh(self.bns1(self.convs1(concat)))
        net = F.hardtanh(self.bns2(self.convs2(net)))
        net = F.hardtanh(self.bns3(self.convs3(net)))
        net = self.convs4(net)

        return (net, trans_feat)


class BasicBiPointNetSemSeg(nn.Module):
    def __init__(self, num_class=13, more_features=True, Linear=BiLinear, pool='max', affine=True):
        super(BasicBiPointNetSemSeg, self).__init__()
        self.more_features = more_features
        if more_features:
            channel = 9
        else:
            channel = 3
        self.k = num_class
        self.feat = BiPointNetEncoder(Linear, global_feat=False, feature_transform=True, channel=channel, pool=pool)
        self.conv1 = Conv1d(1088, 512, Linear)
        self.conv2 = Conv1d(512, 256, Linear)
        self.conv3 = Conv1d(256, 128, Linear)
        self.conv4 = Conv1d(128, self.k, nn.Linear)
        self.bn1 = nn.BatchNorm1d(512, affine=affine)
        self.bn2 = nn.BatchNorm1d(256, affine=affine)
        self.bn3 = nn.BatchNorm1d(128, affine=affine)

    def forward(self, data):
        pos = data.pos  # xyz

        # ref: https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/data_utils/S3DISDataLoader.py
        rgb = data.x  # rgb and 3 additional features

        batch = torch.max(data.batch) + 1

        pos_list = []
        rgb_list = []
        for i in range(batch):
            pos_list.append(pos[data.batch == i])
            rgb_list.append(rgb[data.batch == i])

        pos = torch.stack(pos_list).permute(0, 2, 1).contiguous()
        rgb = torch.stack(rgb_list).permute(0, 2, 1).contiguous()
        if self.more_features:
            point_cloud = torch.cat([pos, rgb], dim=1)
        else:
            point_cloud = pos

        x, trans, trans_feat = self.feat(point_cloud)
        x = F.hardtanh(self.bn1(self.conv1(x)))
        x = F.hardtanh(self.bn2(self.conv2(x)))
        x = F.hardtanh(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        x = F.log_softmax(x.view(-1,self.k), dim=-1)
        x = x.view(-1, self.k)

        return {
            'out': x,
            'trans': trans,
            'trans_feat': trans_feat,
        }


class BiPointNetLSREMax(BasicBiPointNet):
    def __init__(self, args, num_class=40):
        super().__init__(Linear=BiLinearLSR, pool='ema-max')

class BiPointNetPartSegLSREMax(BasicBiPointNetPartSeg):
    def __init__(self, args, num_part=50):
        super().__init__(num_part=num_part, Linear=BiLinearLSR, pool='ema-max')

class BiPointNetSemSegLSREMax(BasicBiPointNetSemSeg):
    def __init__(self):
        super().__init__(Linear=BiLinearLSR, pool='ema-max')
