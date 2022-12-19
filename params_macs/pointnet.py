"""
Modified from https://github.com/fxia22/pointnet.pytorch
"""

import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

from macs import get_mac, get_param

class STNkd_mac(nn.Module):
    def __init__(self, k=64):
        super(STNkd_mac, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x, macs):
        batchsize = x.size()[0]
        macs = get_mac(macs, 'nn_Conv1dS', x, (self.k, 64))
        x = F.relu(self.bn1(self.conv1(x)))
        macs = get_mac(macs, 'nn_Conv1dS', x, (64, 128))
        x = F.relu(self.bn2(self.conv2(x)))
        macs = get_mac(macs, 'nn_Conv1dS', x, (128, 1024))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        macs = get_mac(macs, 'LinearS', x, (1024, 512))
        x = F.relu(self.bn4(self.fc1(x)))
        macs = get_mac(macs, 'LinearS', x, (512, 256))
        x = F.relu(self.bn5(self.fc2(x)))
        macs = get_mac(macs, 'nn_Linear', x, (256, self.k * self.k))
        x = self.fc3(x)

        iden = torch.eye(self.k, device=x.device).view(1, -1).repeat(batchsize, 1)
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x, macs


class PointNetEncoder_mac(nn.Module):
    def __init__(self):
        super(PointNetEncoder_mac, self).__init__()
        self.stn = STNkd_mac(3)
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.fstn = STNkd_mac(k=64)

    def forward(self, x, macs):
        B, D, N = x.size()
        trans, macs = self.stn(x, macs)
        x = x.transpose(2, 1)
        if D >3 :
            x, feature = x.split(3,dim=2)

        macs = get_mac(macs, 'einsum', x, trans.size(-1))
        x = torch.bmm(x, trans)
        if D > 3:
            x = torch.cat([x,feature],dim=2)
        x = x.transpose(2, 1)
        macs = get_mac(macs, 'nn_Conv1dS', x, (3, 64))
        x = F.relu(self.bn1(self.conv1(x)))

        trans_feat, macs = self.fstn(x, macs)
        x = x.transpose(2, 1)
        macs = get_mac(macs, 'einsum', x, trans_feat.size(-1))
        x = torch.bmm(x, trans_feat)
        x = x.transpose(2, 1)

        pointfeat = x
        macs = get_mac(macs, 'nn_Conv1dS', x, (64, 128))
        x = F.relu(self.bn2(self.conv2(x)))
        macs = get_mac(macs, 'nn_Conv1dS', x, (128, 1024))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        return x, trans, trans_feat, macs

class PointNet_CLS_mac(nn.Module):
    def __init__(self, args, num_class=40):
        super(PointNet_CLS_mac, self).__init__()
        self.feat = PointNetEncoder_mac()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_class)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.binary = False

    def forward(self, x):
        macs = (0.0, 0.0, 0.0)

        x, trans, trans_feat, macs = self.feat(x, macs)
        macs = get_mac(macs, 'LinearS', x, (1024, 512))
        x = F.relu(self.bn1(self.fc1(x)))
        macs = get_mac(macs, 'LinearS', x, (512, 256))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        macs = get_mac(macs, 'nn_Linear', x, (256, 40))
        x = self.fc3(x)
        return macs

class PointNet_PSEG_mac(nn.Module):
    def __init__(self, args, num_part=50):
        super(PointNet_PSEG_mac, self).__init__()
        self.num_part = num_part
        self.binary = False

        self.stn = STNkd_mac(3)
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, 512, 1)
        self.conv5 = torch.nn.Conv1d(512, 2048, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(2048)
        self.fstn = STNkd_mac(k=128)
        self.convs1 = torch.nn.Conv1d(4944, 256, 1)
        self.convs2 = torch.nn.Conv1d(256, 256, 1)
        self.convs3 = torch.nn.Conv1d(256, 128, 1)
        self.convs4 = torch.nn.Conv1d(128, num_part, 1)
        self.bns1 = nn.BatchNorm1d(256)
        self.bns2 = nn.BatchNorm1d(256)
        self.bns3 = nn.BatchNorm1d(128)

    def forward(self, point_cloud, label):
        macs = (0.0, 0.0, 0.0)

        B, D, N = point_cloud.size()
        trans, macs = self.stn(point_cloud, macs)
        point_cloud = point_cloud.transpose(2, 1)
        if D > 3:
            point_cloud, feature = point_cloud.split(3, dim=2)
        macs = get_mac(macs, 'einsum', point_cloud, trans.size(-1))
        point_cloud = torch.bmm(point_cloud, trans)
        if D > 3:
            point_cloud = torch.cat([point_cloud, feature], dim=2)

        point_cloud = point_cloud.transpose(2, 1)

        macs = get_mac(macs, 'nn_Conv1dS', point_cloud, (3, 64))
        out1 = F.relu(self.bn1(self.conv1(point_cloud)))
        macs = get_mac(macs, 'nn_Conv1dS', out1, (64, 128))
        out2 = F.relu(self.bn2(self.conv2(out1)))
        macs = get_mac(macs, 'nn_Conv1dS', out2, (128, 128))
        out3 = F.relu(self.bn3(self.conv3(out2)))

        trans_feat, macs = self.fstn(out3, macs)
        x = out3.transpose(2, 1)
        macs = get_mac(macs, 'einsum', x, trans_feat.size(-1))
        net_transformed = torch.bmm(x, trans_feat)
        net_transformed = net_transformed.transpose(2, 1)

        macs = get_mac(macs, 'nn_Conv1dS', net_transformed, (128, 512))
        out4 = F.relu(self.bn4(self.conv4(net_transformed)))
        macs = get_mac(macs, 'nn_Conv1dS', out4, (512, 2048))
        out5 = self.bn5(self.conv5(out4))
        out_max = torch.max(out5, 2, keepdim=True)[0]
        out_max = out_max.view(-1, 2048)

        out_max = torch.cat([out_max,label.squeeze(1)],1)
        expand = out_max.view(-1, 2048+16, 1).repeat(1, 1, N)
        concat = torch.cat([expand, out1, out2, out3, out4, out5], 1)
        macs = get_mac(macs, 'nn_Conv1dS', concat, (4944, 256))
        net = F.relu(self.bns1(self.convs1(concat)))
        macs = get_mac(macs, 'nn_Conv1dS', net, (256, 256))
        net = F.relu(self.bns2(self.convs2(net)))
        macs = get_mac(macs, 'nn_Conv1dS', net, (256, 128))
        net = F.relu(self.bns3(self.convs3(net)))
        macs = get_mac(macs, 'nn_Conv1d', net, (128, 50))
        net = self.convs4(net)

        return macs

if __name__ == '__main__':
    class Obj(): pass
    args = Obj()
    args.k = 20
    args.binary = False
    args.dropout = 0
    args.pooling = 'mean'

    model = PointNet_CLS_mac(args)
    params = get_param(model)

    x = torch.rand(2, 3, 1024)
    macs, adds, bops = model(x)
    macs /= 1e6 * 2
    adds /= 1e6 * 2
    bops /= 1e6 * 2
    print(f'Params of PointNet on ModelNet40: {params:.6f} M, MACs: {macs:.6f} M, ADDs: {adds:.6f} M, BOPs: {bops:.6f} M')

    args = Obj()
    args.k = 40
    args.dropout = 0
    args.pooling = 'mean'

    model = PointNet_PSEG_mac(args)
    params = get_param(model)

    x = torch.rand(2, 3, 2048)
    l = torch.rand(2, 16)
    macs, adds, bops = model(x, l)
    macs /= 1e6 * 2
    adds /= 1e6 * 2
    bops /= 1e6 * 2
    print(f'Params of PointNet on ShapeNet: {params:.6f} M, MACs: {macs:.6f} M, ADDs: {adds:.6f} M, BOPs: {bops:.6f} M')
