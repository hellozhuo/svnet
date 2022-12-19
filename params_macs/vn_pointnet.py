"""
Modified from https://github.com/FlyingGiraffe/vnn-pc
"""

import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from models.vn_layers import *
from models.utils.vn_util import get_graph_feature_cross

from macs import get_mac, get_param

class STNkd_mac(nn.Module):
    def __init__(self, args, d=64):
        super(STNkd_mac, self).__init__()
        self.args = args
        
        self.conv1 = VNLinearLeakyReLU(d, 64//3, dim=4, negative_slope=0.0)
        self.conv2 = VNLinearLeakyReLU(64//3, 128//3, dim=4, negative_slope=0.0)
        self.conv3 = VNLinearLeakyReLU(128//3, 1024//3, dim=4, negative_slope=0.0)

        self.fc1 = VNLinearLeakyReLU(1024//3, 512//3, dim=3, negative_slope=0.0)
        self.fc2 = VNLinearLeakyReLU(512//3, 256//3, dim=3, negative_slope=0.0)
        
        if args.pooling == 'max':
            self.pool = VNMaxPool(1024//3)
        elif args.pooling == 'mean':
            self.pool = mean_pool
        
        self.fc3 = VNLinear(256//3, d)
        self.d = d

    def forward(self, x, macs):
        batchsize = x.size()[0]
        macs = get_mac(macs, 'VNLinearLeakyReLU', x, (self.d, 64//3))
        x = self.conv1(x)
        macs = get_mac(macs, 'VNLinearLeakyReLU', x, (64//3, 128//3))
        x = self.conv2(x)
        macs = get_mac(macs, 'VNLinearLeakyReLU', x, (128//3, 1024//3))
        x = self.conv3(x)
        x = self.pool(x)

        macs = get_mac(macs, 'VNLinearLeakyReLU', x, (1024//3, 512//3))
        x = self.fc1(x)
        macs = get_mac(macs, 'VNLinearLeakyReLU', x, (512//3, 256//3))
        x = self.fc2(x)
        macs = get_mac(macs, 'VNLinear', x, (256//3, self.d))
        x = self.fc3(x)
        
        return x, macs

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
        ## Takes ~1.5B MACs here
        macs = get_mac(macs, 'VNLinearLeakyReLU', z0, (self.in_channels, self.in_channels//2))
        z0 = self.vn1(z0)

        ## Takes ~0.3B MACs here
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

class PointNetEncoder_mac(nn.Module):
    def __init__(self, args):
        super(PointNetEncoder_mac, self).__init__()
        self.args = args
        self.k = args.k
        
        self.conv_pos = VNLinearLeakyReLU(3, 64//3, dim=5, negative_slope=0.0)
        self.conv1 = VNLinearLeakyReLU(64//3, 64//3, dim=4, negative_slope=0.0)
        self.conv2 = VNLinearLeakyReLU(64//3*2, 128//3, dim=4, negative_slope=0.0)
        
        self.conv3 = VNLinear(128//3, 1024//3)
        self.bn3 = VNBatchNorm(1024//3, dim=4)
        
        self.std_feature = VNStdFeature_mac(1024//3*2, dim=4, normalize_frame=False, negative_slope=0.0)
        
        if args.pooling == 'max':
            self.pool = VNMaxPool(64//3)
        elif args.pooling == 'mean':
            self.pool = mean_pool
        
        self.fstn = STNkd_mac(args, d=64//3)

    def forward(self, x, macs):
        B, D, N = x.size()
        
        x = x.unsqueeze(1)
        feat = get_graph_feature_cross(x, k=self.k)
        macs = get_mac(macs, 'VNLinearLeakyReLU', feat, (3, 64//3))
        x = self.conv_pos(feat)
        x = self.pool(x)
        
        macs = get_mac(macs, 'VNLinearLeakyReLU', x, (64//3, 64//3))
        x = self.conv1(x)
        
        x_global, macs = self.fstn(x, macs)
        x_global = x_global.unsqueeze(-1).repeat(1,1,1,N)
        x = torch.cat((x, x_global), 1)
        
        pointfeat = x
        macs = get_mac(macs, 'VNLinearLeakyReLU', x, (64//3*2, 128//3))
        x = self.conv2(x)
        macs = get_mac(macs, 'VNLinearBN', x, (128//3, 1024//3))
        x = self.bn3(self.conv3(x))
        
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1)
        x, trans, macs = self.std_feature(x, macs)
        x = x.view(B, -1, N)
        
        x = torch.max(x, -1, keepdim=False)[0]
        
        return x, macs

class VN_PointNet_CLS_mac(nn.Module):
    def __init__(self, args, k=40):
        super(VN_PointNet_CLS_mac, self).__init__()
        self.args = args
        self.binary = False
        self.feat = PointNetEncoder_mac(args)
        self.fc1 = nn.Linear(1024//3*6, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        macs = (0.0, 0.0, 0.0)

        x, macs = self.feat(x, macs)
        macs = get_mac(macs, 'LinearS', x, (1024//3*6, 512))
        x = F.relu(self.bn1(self.fc1(x)))
        macs = get_mac(macs, 'LinearS', x, (512, 256))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        macs = get_mac(macs, 'nn_Linear', x, (256, 40))
        x = self.fc3(x)
        return macs

class VN_PointNet_PSEG_mac(nn.Module):
    def __init__(self, args, num_part=50):
        super(VN_PointNet_PSEG_mac, self).__init__()
        self.args = args
        self.k = args.k
        self.num_part = num_part
        self.binary = False
        
        self.conv_pos = VNLinearLeakyReLU(3, 64//3, dim=5, negative_slope=0.0)
        self.conv1 = VNLinearLeakyReLU(64//3, 64//3, dim=4, negative_slope=0.0)
        self.conv2 = VNLinearLeakyReLU(64//3, 128//3, dim=4, negative_slope=0.0)
        self.conv3 = VNLinearLeakyReLU(128//3, 128//3, dim=4, negative_slope=0.0)
        self.conv4 = VNLinearLeakyReLU(128//3*2, 512//3, dim=4, negative_slope=0.0)
        
        self.conv5 = VNLinear(512//3, 2048//3)
        self.bn5 = VNBatchNorm(2048//3, dim=4)
        
        self.std_feature = VNStdFeature_mac(2048//3*2, dim=4, normalize_frame=False, negative_slope=0.0)
        
        if args.pooling == 'max':
            self.pool = VNMaxPool(64//3)
        elif args.pooling == 'mean':
            self.pool = mean_pool
        
        self.fstn = STNkd_mac(args, d=128//3)
        
        self.convs1 = torch.nn.Conv1d(9025, 256, 1)
        self.convs2 = torch.nn.Conv1d(256, 256, 1)
        self.convs3 = torch.nn.Conv1d(256, 128, 1)
        self.convs4 = torch.nn.Conv1d(128, num_part, 1)
        self.bns1 = nn.BatchNorm1d(256)
        self.bns2 = nn.BatchNorm1d(256)
        self.bns3 = nn.BatchNorm1d(128)

    def forward(self, point_cloud, label):
        macs = (0.0, 0.0, 0.0)

        B, D, N = point_cloud.size()
        
        point_cloud = point_cloud.unsqueeze(1)
        feat = get_graph_feature_cross(point_cloud, k=self.k)
        macs = get_mac(macs, 'VNLinearLeakyReLU', feat, (3, 64//3))
        point_cloud = self.conv_pos(feat)
        point_cloud = self.pool(point_cloud)

        macs = get_mac(macs, 'VNLinearLeakyReLU', point_cloud, (64//3, 64//3))
        out1 = self.conv1(point_cloud)
        macs = get_mac(macs, 'VNLinearLeakyReLU', out1, (64//3, 128//3))
        out2 = self.conv2(out1)
        macs = get_mac(macs, 'VNLinearLeakyReLU', out2, (128//3, 128//3))
        out3 = self.conv3(out2)
        
        net_global, macs = self.fstn(out3, macs)
        net_global = net_global.unsqueeze(-1).repeat(1,1,1,N)
        net_transformed = torch.cat((out3, net_global), 1)

        macs = get_mac(macs, 'VNLinearLeakyReLU', net_transformed, (128//3*2, 512//3))
        out4 = self.conv4(net_transformed)
        macs = get_mac(macs, 'VNLinearBN', out4, (512//3, 2048//3))
        out5 = self.bn5(self.conv5(out4))
        
        out5_mean = out5.mean(dim=-1, keepdim=True).expand(out5.size())
        out5 = torch.cat((out5, out5_mean), 1)
        out5, trans, macs = self.std_feature(out5, macs)
        out5 = out5.view(B, -1, N)
        
        out_max = torch.max(out5, -1, keepdim=False)[0]

        out_max = torch.cat([out_max,label.squeeze(1)],1)
        expand = out_max.view(-1, 2048//3*6+16, 1).repeat(1, 1, N)
        
        out1234 = torch.cat((out1, out2, out3, out4), dim=1)
        macs = get_mac(macs, 'einsum', out1234, trans.size(2))
        out1234 = torch.einsum('bijm,bjkm->bikm', out1234, trans).view(B, -1, N)
        
        concat = torch.cat([expand, out1234, out5], 1)
        
        macs = get_mac(macs, 'Conv1dS', concat, (9025, 256))
        net = F.relu(self.bns1(self.convs1(concat)))
        macs = get_mac(macs, 'Conv1dS', net, (256, 256))
        net = F.relu(self.bns2(self.convs2(net)))
        macs = get_mac(macs, 'Conv1dS', net, (256, 128))
        net = F.relu(self.bns3(self.convs3(net)))
        macs = get_mac(macs, 'nn_Conv1d', net, (128, 50))
        net = self.convs4(net)
        
        return macs

if __name__ == '__main__':
    class Obj(): pass
    args = Obj()
    args.pooling = 'mean'
    args.k = 20

    model = VN_PointNet_CLS_mac(args)
    params = get_param(model)

    x = torch.rand(2, 3, 1024)
    macs, adds, bops = model(x)
    macs /= 1e6 * 2
    adds /= 1e6 * 2
    bops /= 1e6 * 2
    print(f'Params of VN_PointNet on ModelNet40: {params:.6f} M, MACs: {macs:.6f} M, ADDs: {adds:.6f} M, BOPs: {bops:.6f} M')

    args = Obj()
    args.k = 40
    args.dropout = 0
    args.pooling = 'mean'

    model = VN_PointNet_PSEG_mac(args)
    params = get_param(model)

    x = torch.rand(2, 3, 2048)
    l = torch.rand(2, 16)
    macs, adds, bops = model(x, l)
    macs /= 1e6 * 2
    adds /= 1e6 * 2
    bops /= 1e6 * 2
    print(f'Params of VN_PointNet on ShapeNet: {params:.6f} M, MACs: {macs:.6f} M, ADDs: {adds:.6f} M, BOPs: {bops:.6f} M')
