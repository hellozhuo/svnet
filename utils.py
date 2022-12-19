from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import os
import shutil
import math
import numpy as np
import time
import logging
import operator
from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F

######################################
#       measurement functions        #
######################################

def get_param_num(model):
    n = sum([reduce(operator.mul, i.size(), 1) for i in model.parameters()])
    return float(n) / 1e6


######################################
#          loss functions            #
######################################

### For dgcnn and vn/sv pointnet
def cal_loss(pred, target, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    target = target.contiguous().view(-1)

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, target, reduction='mean')

    return loss

## for original pointnet
def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    I = torch.eye(d, device=trans.device)[None, :, :]
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1) - I), dim=(1, 2)))
    return loss

def cal_pointnet_loss(outputs, target):
    pred, trans_feat = outputs
    loss = cal_loss(pred, target)
    mat_diff_loss = feature_transform_reguliarzer(trans_feat)

    total_loss = loss + mat_diff_loss * 0.001
    return total_loss


def calculate_shape_IoU(pred_np, seg_np, label, class_choice=None):
    seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
    index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]

    label = label.squeeze()
    shape_ious = []
    for shape_idx in range(seg_np.shape[0]):
        if not class_choice:
            start_index = index_start[label[shape_idx]]
            num = seg_num[label[shape_idx]]
            parts = range(start_index, start_index + num)
        else:
            parts = range(seg_num[label[0]])
        part_ious = []
        for part in parts:
            I = np.sum(np.logical_and(pred_np[shape_idx] == part, seg_np[shape_idx] == part))
            U = np.sum(np.logical_or(pred_np[shape_idx] == part, seg_np[shape_idx] == part))
            if U == 0:
                iou = 1  # If the union of groundtruth and prediction points is empty, then count part IoU as 1
            else:
                iou = I / float(U)
            part_ious.append(iou)
        shape_ious.append(np.mean(part_ious))
    return shape_ious



######################################
#         basic functions            #
######################################

def configure_logging(root, name, extent=None):
    os.makedirs(root, exist_ok=True)

    formatter = logging.Formatter('%(message)s')
    if extent is None:
        extent = time.strftime('%Y-%m-%d-%H-%M-%S')
        formatter = logging.Formatter('%(asctime)s - %(message)s')
    name = f'{name}-{extent}'
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler('%s/%s.txt' % (root, name))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    def log_string(str):
        logger.info(str)
        print(str)
    return log_string

def load_checkpoint(args):

    model_dir = os.path.join(args.save_dir, 'save_models')
    latest_filename = os.path.join(model_dir, 'latest.txt')
    model_filename = ''

    if args.test is not None:
        model_filename = args.test
    elif args.resume_from is not None:
        model_filename = args.resume_from
    elif args.resume and os.path.exists(latest_filename):
        with open(latest_filename, 'r') as fin:
            model_filename = fin.readlines()[0].strip()
    loadinfo = "=> loading checkpoint from '{}'".format(model_filename)
    print(loadinfo)

    state = None
    if os.path.exists(model_filename):
        state = torch.load(model_filename, map_location='cpu')

    return state


def save_checkpoint(state, epoch, root, is_best, saveID, save_freq=20):

    filename = 'checkpoint_%03d.pth' % epoch
    model_dir = os.path.join(root, 'save_models')
    model_filename = os.path.join(model_dir, filename)
    latest_filename = os.path.join(model_dir, 'latest.txt')

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # write new checkpoint 
    torch.save(state, model_filename)
    with open(latest_filename, 'w') as fout:
        fout.write(model_filename)
    print("=> saved checkpoint '{}'".format(model_filename))

    # update best model 
    if is_best:
        best_filename = os.path.join(model_dir, 'model_best.pth')
        shutil.copyfile(model_filename, best_filename)

    # remove old model
    if saveID is not None and (saveID + 1) % save_freq > 0:
        filename = 'checkpoint_%03d.pth' % saveID
        model_filename = os.path.join(model_dir, filename)
        if os.path.exists(model_filename):
            os.remove(model_filename)
            print('=> removed checkpoint %s' % model_filename)

    print('##########Time##########', time.strftime('%Y-%m-%d %H:%M:%S'))
    return epoch


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, accum='mean'):
        self.reset()
        self.accum = accum

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if self.accum == 'mean':
            self.sum += val * n
            self.val = val
        elif self.accum == 'sum':
            self.sum += val
            self.val = val / n
        self.count += n
        self.avg = self.sum / self.count
        self.avg100 = self.sum / self.count * 100
        self.val100 = self.val * 100

def set_binary_modules(model, bound, binary):

    from ops import BConv, LBPConv
    modules = []
    for layer in model.modules():
        if isinstance(layer, BConv):
            layer.reset_state(bound, binary)
            modules.append(layer)
        elif isinstance(layer, LBPConv):
            layer.reset_state(bound)
            modules.append(layer)
    return modules


def adjust_learning_rate(optimizer, epoch, args, method='cosine'):
    if method == 'cosine':
        T_total = float(args.epochs)
        T_cur = float(epoch)
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * T_cur / T_total))
        blr = 0.5 * args.blr * (1 + math.cos(math.pi * T_cur / T_total))
    elif method == 'multistep':
        lr = args.lr
        blr = args.blr
        for epoch_step, lr_gamma in zip(args.lr_steps, args.lr_gammas):
            if epoch >= epoch_step:
                lr = lr * lr_gamma
                blr = blr * lr_gamma
    if epoch < args.warm_epoch:
        lr = args.lr * (epoch + 1) / args.warm_epoch
        blr = args.blr * (epoch + 1) / args.warm_epoch
    str_lr = ''
    for param_group, lr in zip(optimizer.param_groups, [lr, lr, blr]):
        param_group['lr'] = lr
        str_lr = '%s-%.6f' % (str_lr, lr)
    # remove the first '-'
    return str_lr[1:]



