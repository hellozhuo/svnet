"""
Modified from https://github.com/FlyingGiraffe/vnn-pc
"""

import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from .vn_layers import *
from .utils.vn_util import get_graph_feature_cross

class PointNetEncoder(nn.Module):
    def __init__(self, args):
        super(PointNetEncoder, self).__init__()
        self.args = args
        self.k = args.k
        
        self.conv_pos = VNLinearLeakyReLU(3, 64//3, dim=5, negative_slope=0.0)
        self.conv1 = VNLinearLeakyReLU(64//3, 64//3, dim=4, negative_slope=0.0)
        self.conv2 = VNLinearLeakyReLU(64//3*2, 128//3, dim=4, negative_slope=0.0)
        
        self.conv3 = VNLinear(128//3, 1024//3)
        self.bn3 = VNBatchNorm(1024//3, dim=4)
        
        self.std_feature = VNStdFeature(1024//3*2, dim=4, normalize_frame=False, negative_slope=0.0)
        
        if args.pooling == 'max':
            self.pool = VNMaxPool(64//3)
        elif args.pooling == 'mean':
            self.pool = mean_pool
        
        self.fstn = STNkd(args, d=64//3)

    def forward(self, x):
        B, D, N = x.size()
        
        x = x.unsqueeze(1)
        feat = get_graph_feature_cross(x, k=self.k)
        x = self.conv_pos(feat)
        x = self.pool(x)
        
        x = self.conv1(x)
        
        x_global = self.fstn(x).unsqueeze(-1).repeat(1,1,1,N)
        x = torch.cat((x, x_global), 1)
        
        pointfeat = x
        x = self.conv2(x)
        x = self.bn3(self.conv3(x))
        
        x_mean = x.mean(dim=-1, keepdim=True).expand(x.size())
        x = torch.cat((x, x_mean), 1)
        x, trans = self.std_feature(x)
        x = x.view(B, -1, N)
        
        x = torch.max(x, -1, keepdim=False)[0]
        
        return x

class VN_PointNet_CLS(nn.Module):
    def __init__(self, args, num_class=40):
        super(VN_PointNet_CLS, self).__init__()
        self.args = args
        self.feat = PointNetEncoder(args)
        self.fc1 = nn.Linear(1024//3*6, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_class)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return x
