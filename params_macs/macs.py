
from functools import reduce
import operator
from models.sv_layers import Linear, Conv1d

def get_param(model):
    n = sum([reduce(operator.mul, i.size(), 1) for i in model.parameters()])
    params = float(n)

    bparams = 0.0
    if model.binary:
        for layer in model.modules():
            if isinstance(layer, (Linear, Conv1d)):
                bparams += reduce(operator.mul, layer.weight.size(), 1)
    params = ((params - bparams) * 32 + bparams) / 1e6

    return params


def get_mac(macs, func_name, x, dims, binary=False):
    mac, add, bop = macs
    if func_name == 'Vector2Scalar':
        v = x
        multi = dims
        op = reduce(operator.mul, v.size(), 1) * multi
        mac += reduce(operator.mul, v.size(), 1) * multi
        if binary:
            add += op
        else:
            mac += op
    elif func_name == 'SVBlock':
        s, v = x
        in_dims, out_dims = dims
        macs = get_mac(macs, 'Vector2Scalar', v, 3, binary)
        mac, add, bop = macs
        mac += s.shape[0] * (in_dims[0] * (out_dims[1]//2) + out_dims[1]//2 + (out_dims[1]//2) * out_dims[1] + out_dims[1])
        op = sum([reduce(operator.mul, i.size(), 1) for i in [s, v]]) * out_dims[0]
        mac += reduce(operator.mul, s.size()[:-1], 1) * out_dims[0] * 2 # bn + relu
        op2 = reduce(operator.mul, v.size(), 1) * out_dims[1]
        mac += reduce(operator.mul, v.size()[:-1], 1) * out_dims[1] * 2 # bn + ele-wise
        if binary:
            bop += op
            add += op2
        else:
            mac += op + op2
    elif func_name == 'SBlock':
        _, out_channels = dims
        op = reduce(operator.mul, x.size(), 1) * out_channels
        mac += reduce(operator.mul, x.size()[:-1], 1) * out_channels * 2
        if binary:
            bop += op
        else:
            mac += op
    elif func_name == 'VBlock':
        v = x
        in_dim, out_dim = dims
        macs = get_mac(macs, 'Vector2Scalar', v, 3, binary)
        mac, add, bop = macs
        mac += v.shape[0] * (in_dim * 3 * (out_dim//2) + out_dim//2 + (out_dim//2) * out_dim + out_dim)
        op = reduce(operator.mul, v.size(), 1) * out_dim
        mac += reduce(operator.mul, v.size()[:-1], 1) * out_dim * 2 # bn + ele-wise
        if binary:
            add += op
        else:
            mac += op
    elif func_name == 'SVFuse':
        s, v = x
        v_dim, multi = dims
        macs = get_mac(macs, 'Vector2Scalar', v, multi, binary)
        mac, add, bop = macs
    elif func_name == 'nn_Conv1dS':
        in_channels, out_channels = dims
        mac += reduce(operator.mul, x.size(), 1) * out_channels
        mac += x.size(0) * out_channels * x.size(2) * 2
    elif func_name == 'nn_Conv1d':
        in_channels, out_channels = dims
        mac += reduce(operator.mul, x.size(), 1) * out_channels
    elif func_name == 'Conv1dS':
        in_channels, out_channels = dims
        op = reduce(operator.mul, x.size(), 1) * out_channels
        mac += x.size(0) * out_channels * x.size(2) * 2
        if binary:
            bop += op
        else:
            mac += op
    elif func_name == 'LinearS':
        _, out_channels = dims
        op = reduce(operator.mul, x.size(), 1) * out_channels
        mac += reduce(operator.mul, x.size()[:-1], 1) * out_channels * 2
        if binary:
            bop += op
        else:
            mac += op
    elif func_name == 'nn_Linear':
        _, out_channels = dims
        mac += reduce(operator.mul, x.size(), 1) * out_channels
    elif func_name == 'VNLinearLeakyReLU':
        in_channels, out_channels = dims
        mac += reduce(operator.mul, x.size(), 1) * out_channels
        mac += reduce(operator.mul, x.size(), 1) / in_channels * out_channels
        mac += reduce(operator.mul, x.size(), 1) * out_channels
        mac += reduce(operator.mul, x.size(), 1) / in_channels * out_channels # p*d
        mac += reduce(operator.mul, x.size(), 1) / in_channels * out_channels # ele-wise
    elif func_name == 'VNLinearLeakyReLU_Share':
        in_channels, out_channels = dims
        mac += reduce(operator.mul, x.size(), 1) * out_channels
        mac += reduce(operator.mul, x.size(), 1) / in_channels * out_channels
        mac += reduce(operator.mul, x.size(), 1) * 1
        mac += reduce(operator.mul, x.size(), 1) / in_channels * out_channels # p*d
        mac += reduce(operator.mul, x.size(), 1) / in_channels * out_channels # ele-wise
    elif func_name == 'VNLinearBN':
        in_channels, out_channels = dims
        mac += reduce(operator.mul, x.size(), 1) * out_channels
        mac += reduce(operator.mul, x.size(), 1) / in_channels * out_channels
    elif func_name == 'VNLinear':
        in_channels, out_channels = dims
        mac += reduce(operator.mul, x.size(), 1) * out_channels
    elif func_name == 'einsum':
        mac += reduce(operator.mul, x.size(), 1) * dims
    else:
        raise ValueError(f'not recognized function name: {func_name}')
    return (mac, add, bop)


        



