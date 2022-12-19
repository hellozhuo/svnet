'''
This code is modified from https://github.com/htqin/BiPointNet
'''
from models.bipointnet_basic import MeanShift, BiLinear, BiLinearXNOR, BiLinearIRNet, BiLinearLSR, BiLinearBiReal
import math
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import numpy as np
import torch.nn.functional as F

from functools import reduce
import operator


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

class BiSTNkd_mac(nn.Module):
    def __init__(self, k=64, Linear=BiLinear, pool='max', affine=True, bi_first=False):
        super(BiSTNkd_mac, self).__init__()
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

    def forward(self, x, macs):
        macs, adds, bops = macs
        batchsize, D, N = x.size()

        bops += reduce(operator.mul, x.size(), 1) * 64
        x = F.hardtanh(self.bn1(self.conv1(x)))
        macs += reduce(operator.mul, x.size(), 1) * 2

        bops += reduce(operator.mul, x.size(), 1) * 128
        x = F.hardtanh(self.bn2(self.conv2(x)))
        macs += reduce(operator.mul, x.size(), 1) * 2
        if self.pool == 'max':
            x = F.hardtanh(self.bn3(self.conv3(x)))
            x = torch.max(x, 2, keepdim=True)[0]
        elif self.pool == 'mean':
            x = F.hardtanh(self.bn3(self.conv3(x)))
            x = torch.mean(x, 2, keepdim=True)
        elif self.pool == 'ema-max':
            bops += reduce(operator.mul, x.size(), 1) * 1024
            x = self.bn3(self.conv3(x)) + offset_map[N]
            macs += reduce(operator.mul, x.size(), 1) * 2
            x = torch.max(x, 2, keepdim=True)[0]
            x = x.view(-1, 1024)
        x = x.view(-1, 1024)

        bops += reduce(operator.mul, x.size(), 1) * 512
        x = F.hardtanh(self.bn4(self.fc1(x)))
        macs += reduce(operator.mul, x.size(), 1) * 2

        bops += reduce(operator.mul, x.size(), 1) * 256
        x = F.hardtanh(self.bn5(self.fc2(x)))
        macs += reduce(operator.mul, x.size(), 1) * 2

        bops += reduce(operator.mul, x.size(), 1) * self.k * self.k
        x = self.fc3(x)

        iden = torch.eye(self.k, device=x.device).view(1, self.k * self.k).repeat(batchsize, 1)
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x, (macs, adds, bops)


class BiPointNetEncoder_mac(nn.Module):
    def __init__(self, Linear, global_feat=True, feature_transform=False, channel=3, pool='max', affine=True, tnet=True, bi_first=False, use_bn=True):
        super(BiPointNetEncoder_mac, self).__init__()
        self.tnet = tnet
        if self.tnet:
            self.stn = BiSTNkd_mac(channel, Linear, pool=pool, affine=affine, bi_first=bi_first)
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
            self.fstn = BiSTNkd_mac(k=64, Linear=Linear, pool=pool, affine=affine, bi_first=bi_first)
        self.pool = pool
        self.use_bn = use_bn

    def forward(self, x, macs):
        B, D, N = x.size()
        if self.tnet:
            trans, macs = self.stn(x, macs)
        else:
            trans = None
        macs, adds, bops = macs

        x = x.transpose(2, 1)
        if D == 6:
            x, feature = x.split(3, dim=2)
        elif D == 9:
            x, feature = x.split([3, 6], dim=2)
        if self.tnet:
            macs += reduce(operator.mul, x.size(), 1) * trans.size(-1)
            x = torch.bmm(x, trans)
        if D > 3:
            x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1)
        if self.use_bn:
            macs += reduce(operator.mul, x.size(), 1) * 64
            x = F.hardtanh(self.bn1(self.conv1(x)))
            macs += reduce(operator.mul, x.size(), 1) * 2
        else:
            x = F.hardtanh(self.conv1(x))

        macs = (macs, adds, bops)

        if self.tnet and self.feature_transform:
            trans_feat, macs = self.fstn(x, macs)
            macs, adds, bops = macs
            x = x.transpose(2, 1)
            macs += reduce(operator.mul, x.size(), 1) * trans_feat.size(-1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        if self.use_bn:
            bops += reduce(operator.mul, x.size(), 1) * 128
            x = F.hardtanh(self.bn2(self.conv2(x)))
            macs += reduce(operator.mul, x.size(), 1) * 2

            bops += reduce(operator.mul, x.size(), 1) * 1024
            x = self.bn3(self.conv3(x))
            macs += reduce(operator.mul, x.size(), 1) * 2
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
            return x, trans, trans_feat, (macs, adds, bops)
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


class BasicBiPointNet_mac(nn.Module):
    def __init__(self, k=40, Linear=BiLinear, pool='max', affine=True, tnet=True, bi_first=False, bi_last=False, use_bn=True):
        super(BasicBiPointNet_mac, self).__init__()
        channel = 3
        self.feat = BiPointNetEncoder_mac(Linear, global_feat=True, feature_transform=True, channel=channel, pool=pool, affine=affine, tnet=tnet, bi_first=bi_first, use_bn=use_bn)
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
        macs = (0.0, 0.0, 0.0)

        x, trans, trans_feat, macs = self.feat(x, macs)
        macs, adds, bops = macs
        if self.use_bn:
            bops += reduce(operator.mul, x.size(), 1) * 512
            x = F.hardtanh(self.bn1(self.fc1(x)))
            macs += reduce(operator.mul, x.size(), 1) * 2

            bops += reduce(operator.mul, x.size(), 1) * 256
            x = F.hardtanh(self.bn2(self.fc2(x)))
            macs += reduce(operator.mul, x.size(), 1) * 2
        else:
            x = F.hardtanh(self.fc1(x))
            x = F.hardtanh(self.fc2(x))
        macs += reduce(operator.mul, x.size(), 1) * 40
        x = self.fc3(x)
        return (macs, adds, bops)

class BasicBiPointNetPartSeg_mac(nn.Module):
    def __init__(self, num_part=50, Linear=BiLinear, pool='max', affine=True, Tnet=True):
        super(BasicBiPointNetPartSeg_mac, self).__init__()
        self.pool = pool
        channel = 3
        self.stn = BiSTNkd_mac(channel, Linear, pool=pool, affine=affine)
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
        self.fstn = BiSTNkd_mac(k=128, Linear=Linear, pool=pool, affine=affine)
        self.convs1 = Conv1d(4944, 256, Linear)

        self.convs2 = Conv1d(256, 256, Linear)
        self.convs3 = Conv1d(256, 128, Linear)
        self.convs4 = Conv1d(128, num_part, nn.Linear)
        self.bns1 = nn.BatchNorm1d(256, affine=affine)
        self.bns2 = nn.BatchNorm1d(256, affine=affine)
        self.bns3 = nn.BatchNorm1d(128, affine=affine)

        self.Tnet = Tnet

    def forward(self, point_cloud, label):
        macs = (0.0, 0.0, 0.0)

        B, D, N = point_cloud.size()

        if self.Tnet:
            trans, macs = self.stn(point_cloud, macs)
            macs, adds, bops = macs
            point_cloud = point_cloud.transpose(2, 1)
            if D > 3:
                point_cloud, feature = point_cloud.split(3, dim=2)
            macs += reduce(operator.mul, point_cloud.size(), 1) * trans.size(-1)
            point_cloud = torch.bmm(point_cloud, trans)
            if D > 3:
                point_cloud = torch.cat([point_cloud, feature], dim=2)
            point_cloud = point_cloud.transpose(2, 1)

        macs += reduce(operator.mul, point_cloud.size(), 1) * 64
        out1 = F.hardtanh(self.bn1(self.conv1(point_cloud)))
        macs += reduce(operator.mul, out1.size(), 1) * 2

        bops += reduce(operator.mul, out1.size(), 1) * 128
        out2 = F.hardtanh(self.bn2(self.conv2(out1)))
        macs += reduce(operator.mul, out2.size(), 1) * 2

        bops += reduce(operator.mul, out2.size(), 1) * 128
        out3 = F.hardtanh(self.bn3(self.conv3(out2)))
        macs += reduce(operator.mul, out3.size(), 1) * 2

        macs = (macs, adds, bops)
        if self.Tnet:
            trans_feat, macs = self.fstn(out3, macs)
            macs, adds, bops = macs
            x = out3.transpose(2, 1)
            macs += reduce(operator.mul, x.size(), 1) * trans_feat.size(-1)
            net_transformed = torch.bmm(x, trans_feat)
            net_transformed = net_transformed.transpose(2, 1)
        else:
            net_transformed = out3

        bops += reduce(operator.mul, net_transformed.size(), 1) * 512
        out4 = F.hardtanh(self.bn4(self.conv4(net_transformed)))
        macs += reduce(operator.mul, out4.size(), 1) * 2

        bops += reduce(operator.mul, out4.size(), 1) * 2048
        out5 = self.bn5(self.conv5(out4))
        macs += reduce(operator.mul, out5.size(), 1) * 2
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

        bops += reduce(operator.mul, concat.size(), 1) * 256
        net = F.hardtanh(self.bns1(self.convs1(concat)))
        macs += reduce(operator.mul, net.size(), 1) * 2

        bops += reduce(operator.mul, net.size(), 1) * 256
        net = F.hardtanh(self.bns2(self.convs2(net)))
        macs += reduce(operator.mul, net.size(), 1) * 2

        bops += reduce(operator.mul, net.size(), 1) * 128
        net = F.hardtanh(self.bns3(self.convs3(net)))
        macs += reduce(operator.mul, net.size(), 1) * 2

        macs += reduce(operator.mul, net.size(), 1) * 50
        net = self.convs4(net)

        return (macs, adds, bops)


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


class BiPointNetLSREMax_mac(BasicBiPointNet_mac):
    def __init__(self, args):
        super().__init__(Linear=BiLinearLSR, pool='ema-max')

class BiPointNetPartSegLSREMax_mac(BasicBiPointNetPartSeg_mac):
    def __init__(self, args, num_part=50):
        super().__init__(num_part=num_part, Linear=BiLinearLSR, pool='ema-max')

class BiPointNetSemSegLSREMax(BasicBiPointNetSemSeg):
    def __init__(self):
        super().__init__(Linear=BiLinearLSR, pool='ema-max')


def get_param(model):
    n = sum([reduce(operator.mul, i.size(), 1) for i in model.parameters()])
    params = float(n)

    bparams = 0.0
    for layer in model.modules():
        if isinstance(layer, BiLinearLSR):
            bparams += reduce(operator.mul, layer.weight.size(), 1)
    params = ((params - bparams) * 32 + bparams) / 1e6

    return params

if __name__ == '__main__':
    class Obj(): pass
    args = Obj()

    model = BiPointNetLSREMax_mac(args)
    params = get_param(model)

    x = torch.rand(2, 3, 1024)
    macs, adds, bops = model(x)
    macs /= 1e6 * 2
    adds /= 1e6 * 2
    bops /= 1e6 * 2
    print(f'Params of BiPointNet on ModelNet40: {params:.6f} M, MACs: {macs:.6f} M, ADDs: {adds:.6f} M, BOPs: {bops:.6f} M')

    args = Obj()

    model = BiPointNetPartSegLSREMax_mac(args)
    params = get_param(model)

    x = torch.rand(2, 3, 2048)
    l = torch.rand(2, 16)
    macs, adds, bops = model(x, l)
    macs /= 1e6 * 2
    adds /= 1e6 * 2
    bops /= 1e6 * 2
    print(f'Params of BiPointNet on ShapeNet: {params:.6f} M, MACs: {macs:.6f} M, ADDs: {adds:.6f} M, BOPs: {bops:.6f} M')
