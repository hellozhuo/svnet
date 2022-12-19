import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

from models.sv_layers import *
from models.utils.sv_util import *
from macs import get_mac, get_param

class SV_STNkd_mac(nn.Module):
    def __init__(self, dim, binary):
        super(SV_STNkd_mac, self).__init__()
        self.binary = binary
        self.dim = dim

        self.conv1 = SVBlock(dim, (64//2, 64//6), binary=binary)
        self.conv2 = SVBlock((64//2, 64//6), (128//2, 128//6), binary=binary)
        self.conv3 = SVBlock((128//2, 128//6), (1024//2, 1024//6), binary=binary)

        self.fc1 = SVBlock((1024//2, 1024//6), (512//2, 512//6), binary=binary)
        self.fc2 = SVBlock((512//2, 512//6), (256//2, 256//6), binary=binary)
        self.fc3 = SVBlock((256//2, 256//6), dim, binary=binary)

    def forward(self, x, macs):
        macs = get_mac(macs, 'SVBlock', x, (self.dim, (64//2, 64//6)), binary=self.binary)
        x = self.conv1(x)
        macs = get_mac(macs, 'SVBlock', x, ((64//2, 64//6), (128//2, 128//6)), binary=self.binary)
        x = self.conv2(x)
        macs = get_mac(macs, 'SVBlock', x, ((128//2, 128//6), (1024//2, 1024//6)), binary=self.binary)
        x = self.conv3(x) # B, N_points, [3,] 1024//(2,6)
        x = svpool(x, dim=1)

        macs = get_mac(macs, 'SVBlock', x, ((1024//2, 1024//6), (512//2, 512//6)), binary=self.binary)
        x = self.fc1(x)
        macs = get_mac(macs, 'SVBlock', x, ((512//2, 512//6), (256//2, 256//6)), binary=self.binary)
        x = self.fc2(x)
        macs = get_mac(macs, 'SVBlock', x, ((256//2, 256//6), self.dim), binary=self.binary)
        x = self.fc3(x) # B, [3,] dim
        
        return x, macs

class SVPointNetEncoder_mac(nn.Module):
    def __init__(self, k, binary):
        super(SVPointNetEncoder_mac, self).__init__()
        self.k = k
        self.binary = binary

        self.init_scalar = Vector2Scalar(3, 3)
        self.conv_pos = SVBlock((9, 3), (64//2, 64//6))
        self.conv1 = SVBlock((64//2, 64//6), (64//2, 64//6), binary=binary)

        self.fstn = SV_STNkd_mac((64//2, 64//6), binary=binary)

        self.conv2 = SVBlock((64//2*2, 64//6*2), (128//2, 128//6), binary=binary)
        self.conv3 = SVBlock((128//2, 128//6), (1024//2, 1024//6), binary=binary)

        self.conv_fuse = SVBlock((1024//2*2, 1024//6*2), (1024//2, 1024//6), binary=binary)

        self.svfuse = SVFuse(1024//6, 3, binary=binary)
        
    def forward(self, x, macs):
        B, D, N = x.size()
        
        x = x.unsqueeze(1)
        v = get_graph_feature_cross(x, k=self.k) # B, N_points, k, 3, 3
        macs = get_mac(macs, 'Vector2Scalar', v, 3)
        s = self.init_scalar(v) # B, N_points, k, 9
        x = (s, v)
        macs = get_mac(macs, 'SVBlock', x, ((9, 3), (64//2, 64//6)))
        x = self.conv_pos(x)
        x = svpool(x)
        
        macs = get_mac(macs, 'SVBlock', x, ((64//2, 64//6), (64//2, 64//6)), binary=self.binary)
        x = self.conv1(x) # B, N_points, [3,] dim
        
        x_global, macs = self.fstn(x, macs)
        x_global = (x_global[0].unsqueeze(1).expand_as(x[0]), x_global[1].unsqueeze(1).expand_as(x[1]))
        x = svcat([x, x_global])
        
        macs = get_mac(macs, 'SVBlock', x, ((64//2*2, 64//6*2), (128//2, 128//6)), binary=self.binary)
        x = self.conv2(x)
        macs = get_mac(macs, 'SVBlock', x, ((128//2, 128//6), (1024//2, 1024//6)), binary=self.binary)
        x = self.conv3(x) # B, N_points, [3,] 1024//(2,6)

        x_mean = svpool(x, dim=1, keepdim=True) # B, 1, [3,] 1024//(2,6)
        x_mean = (x_mean[0].expand_as(x[0]), x_mean[1].expand_as(x[1]))
        x = svcat([x, x_mean])
        macs = get_mac(macs, 'SVBlock', x, ((1024//2*2, 1024//6*2), (1024//2, 1024//6)), binary=self.binary)
        x = self.conv_fuse(x)
        
        x = svpool(x, dim=1) # B, [3,] 1024//2*2 or 1024//6*2
        macs = get_mac(macs, 'SVFuse', x, (1024//6, 3), binary=self.binary)
        x = self.svfuse(x) # B, 1024//2*2+1024//6*2*3
        
        #x = svpool(x, dim=1)
        #macs = get_mac(macs, 'SVFuse', x, (1024//6, 3), binary=self.binary)
        #x = self.svfuse(x) # B, 1024
        
        return x, macs

class SV_Pointnet_CLS_mac(nn.Module):
    def __init__(self, args, num_class=40):
        super(SV_Pointnet_CLS_mac, self).__init__()
        self.binary = args.binary
        self.k = args.k
        p = 0 if self.binary else 0.4

        self.feat = SVPointNetEncoder_mac(k=self.k, binary=self.binary)
        self.fc1 = Linear(1024//2+1024//6*3, 512, bias=False, bw=self.binary, ba=self.binary)
        self.fc2 = Linear(512, 256, bias=False, bw=self.binary, ba=self.binary)
        self.fc3 = nn.Linear(256, num_class)
        self.dropout = nn.Dropout(p=p)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        macs = (0.0, 0.0, 0.0)

        x, macs = self.feat(x, macs)
        macs = get_mac(macs, 'LinearS', x, (1024//2+1024//6*3, 512), binary=self.binary)
        x = F.relu(self.bn1(self.fc1(x)))
        macs = get_mac(macs, 'LinearS', x, (512, 256), binary=self.binary)
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        macs = get_mac(macs, 'nn_Linear', x, (256, 40))
        x = self.fc3(x)
        return macs

class SV_PointNet_PSEG_mac(nn.Module):
    def __init__(self, args, num_part=50):
        super(SV_PointNet_PSEG_mac, self).__init__()
        self.k = args.k
        self.binary = args.binary

        self.init_scalar = Vector2Scalar(3, 3)
        self.conv_pos = SVBlock((9, 3), (64//2, 64//6))
        self.conv1 = SVBlock((64//2, 64//6), (64//2, 64//6), binary=self.binary)
        self.conv2 = SVBlock((64//2, 64//6), (128//2, 128//6), binary=self.binary)
        self.conv3 = SVBlock((128//2, 128//6), (128//2, 128//6), binary=self.binary)
        self.fstn = SV_STNkd_mac((128//2, 128//6), binary=self.binary)
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
        macs = (0.0, 0.0, 0.0)

        B, D, N = x.size()
        
        x = x.unsqueeze(1)
        v = get_graph_feature_cross(x, k=self.k) # B, N_points, k, 3, 3
        macs = get_mac(macs, 'Vector2Scalar', v, 3)
        s = self.init_scalar(v) # B, N_points, k, 9
        x = (s, v)
        macs = get_mac(macs, 'SVBlock', x, ((9, 3), (64//2, 64//6)))
        x = self.conv_pos(x)
        x = svpool(x)

        macs = get_mac(macs, 'SVBlock', x, ((64//2, 64//6), (64//2, 64//6)), binary=self.binary)
        out1 = self.conv1(x)
        macs = get_mac(macs, 'SVBlock', out1, ((64//2, 64//6), (128//2, 128//6)), binary=self.binary)
        out2 = self.conv2(out1)
        macs = get_mac(macs, 'SVBlock', out2, ((128//2, 128//6), (128//2, 128//6)), binary=self.binary)
        out3 = self.conv3(out2)

        x_global, macs = self.fstn(out3, macs)
        x_global = (x_global[0].unsqueeze(1).expand_as(out3[0]), x_global[1].unsqueeze(1).expand_as(out3[1]))
        x_transformed = svcat([out3, x_global])
        macs = get_mac(macs, 'SVBlock', x_transformed, ((128//2*2, 128//6*2), (512//2, 512//6)), binary=self.binary)
        out4 = self.conv4(x_transformed)
        macs = get_mac(macs, 'SVBlock', out4, ((512//2, 512//6), (2048//2, 2048//6)), binary=self.binary)
        out5 = self.conv5(out4)

        x_mean = svpool(out5, dim=1, keepdim=True) # B, 1, [3,] 2048//(2,6)
        x_mean = (x_mean[0].expand_as(out5[0]), x_mean[1].expand_as(out5[1]))
        x = svcat([out5, x_mean]) # B, N, [3,] 4096//(2,6)
        macs = get_mac(macs, 'SVFuse', x, (2048//6*2, 3), binary=self.binary)
        x, trans = self.svfuse(x) # B, N, 2048//2*2+2048//6*2*3 ~ 2048
        x = x.transpose(-1, -2).contiguous()
        macs = get_mac(macs, 'Conv1dS', x, (self.channels, self.channels//8), binary=self.binary)
        x = self.conv_fuse1(x) # B, self.channels//2, N
        macs = get_mac(macs, 'Conv1dS', x, (self.channels//8, self.channels), binary=self.binary)
        x = self.conv_fuse2(x) # B, self.channels, N
        x, _ = x.max(dim=-1)

        x_l = torch.cat([x, l.squeeze(1)], dim=1) # B, ~self.channels+16
        x_l = x_l.view(B, -1, 1).repeat(1, 1, N) # B, ~self.channels, N

        concat = svcat([out1, out2, out3, out4, out5])
        # B, N, d, 3 and B, N, 3, multi
        macs = get_mac(macs, 'einsum', concat[1], trans.size(-1))
        concat_v = torch.einsum('bimj,bijk->bimk', concat[1].transpose(-1, -2), trans).view(B, N, -1)
        concat = torch.cat([concat[0], concat_v], dim=-1).transpose(-1, -2).contiguous() # B, ~64+128+128+512, N
        concat = torch.cat([x_l, concat], dim=1) # B, ~D, N
        macs = get_mac(macs, 'Conv1dS', concat, (self.channels+16+64//2+128//2*2+512//2+2048//2+(64//6+128//6*2+512//6+2048//6)*3, 256), binary=self.binary)
        net = self.convs1(concat)
        macs = get_mac(macs, 'Conv1dS', net, (256, 256), binary=self.binary)
        net = self.convs2(net)
        macs = get_mac(macs, 'Conv1dS', net, (256, 128), binary=self.binary)
        net = self.convs3(net)
        macs = get_mac(macs, 'nn_Conv1d', net, (128, 50))
        net = self.convs4(net)

        return macs

if __name__ == '__main__':
    class Obj(): pass
    args = Obj()
    args.binary = False
    args.k = 20

    model = SV_Pointnet_CLS_mac(args)
    params = get_param(model)

    x = torch.rand(2, 3, 1024)
    macs, adds, bops = model(x)
    macs /= 1e6 * 2
    adds /= 1e6 * 2
    bops /= 1e6 * 2
    print(f'Params of SV_PointNet (FP) on ModelNet40: {params:.6f} M, MACs: {macs:.6f} M, ADDs: {adds:.6f} M, BOPs: {bops:.6f} M')

    args.binary = True

    model = SV_Pointnet_CLS_mac(args)
    params = get_param(model)

    x = torch.rand(2, 3, 1024)
    macs, adds, bops = model(x)
    macs /= 1e6 * 2
    adds /= 1e6 * 2
    bops /= 1e6 * 2
    print(f'Params of SV_PointNet on ModelNet40: {params:.6f} M, MACs: {macs:.6f} M, ADDs: {adds:.6f} M, BOPs: {bops:.6f} M')

    args = Obj()
    args.k = 40
    args.binary = False

    model = SV_PointNet_PSEG_mac(args)
    params = get_param(model)

    x = torch.rand(2, 3, 2048)
    l = torch.rand(2, 16)
    macs, adds, bops = model(x, l)
    macs /= 1e6 * 2
    adds /= 1e6 * 2
    bops /= 1e6 * 2
    print(f'Params of SV_PointNet (FP) on ShapeNet: {params:.6f} M, MACs: {macs:.6f} M, ADDs: {adds:.6f} M, BOPs: {bops:.6f} M')

    args.binary = True

    model = SV_PointNet_PSEG_mac(args)
    params = get_param(model)

    x = torch.rand(2, 3, 2048)
    l = torch.rand(2, 16)
    macs, adds, bops = model(x, l)
    macs /= 1e6 * 2
    adds /= 1e6 * 2
    bops /= 1e6 * 2
    print(f'Params of SV_PointNet on ShapeNet: {params:.6f} M, MACs: {macs:.6f} M, ADDs: {adds:.6f} M, BOPs: {bops:.6f} M')
