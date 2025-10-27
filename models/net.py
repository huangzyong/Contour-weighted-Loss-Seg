'''
Author       : 
Date         : 2021-10-28 11:53:48
LastEditTime : 2021-10-28 11:53:49
LastEditors  : 
Description  : 

Code is far away from bug with the animal protecting
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import pdb
import numpy as np
from math import cos, pi
#from .sync_batchnorm import SynchronizedBatchNorm3d
#BN = SynchronizedBatchNorm3d
BN = nn.BatchNorm3d
# from .ops.carafe3d import CARAFE3DPack

NUM_PER_GROUP = 8
NUM_GROUP = 8


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def conv3x3(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation_rate=1):

    if kernel_size == (1, 3, 3):
        return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                         padding=(0, 1, 1), bias=False, dilation=dilation_rate)
    else:
        return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                         padding=padding, bias=False, dilation=dilation_rate)


class SELayer(nn.Module):

    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Sequential(
            nn.Conv3d(channel, channel // reduction, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(channel // reduction, channel, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.conv(y)

        return x * y


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, kernel_size=3, stride=1, downsample=None, dilation_rate=1, norm='bn'):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, kernel_size, stride)
        if norm == 'bn':
            self.bn1 = BN(inplanes)
        elif norm == 'gn':
            self.bn1 = nn.GroupNorm(NUM_GROUP, inplanes)
        else:
            raise ValueError('unsupport norm method')
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes, kernel_size=kernel_size,
                             dilation_rate=dilation_rate, padding=dilation_rate)
        if norm == 'bn':
            self.bn2 = BN(planes)
        elif norm == 'gn':
            self.bn2 = nn.GroupNorm(NUM_GROUP, planes)
        else:
            raise ValueError('unsupport norm method')

        self.shortcut = nn.Sequential()

        if stride != 1 or inplanes != planes:
            if norm == 'bn':
                self.shortcut = nn.Sequential(
                    BN(inplanes),
                    self.relu,
                    nn.Conv3d(inplanes, planes, kernel_size=1,
                              stride=stride, bias=False)
                )
            elif norm == 'gn':
                self.shortcut = nn.Sequential(
                    nn.GroupNorm(NUM_GROUP, inplanes),
                    self.relu,
                    nn.Conv3d(inplanes, planes, kernel_size=1,
                              stride=stride, bias=False)
                )
            else:
                raise ValueError('unsupport norm method')
        self.stride = stride

    def forward(self, x):

        # pdb.set_trace()
        residue = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += self.shortcut(residue)

        return out


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, kernel_size=3, stride=1, reduction=4, dilation_rate=1, norm='bn'):
        super(SEBasicBlock, self).__init__()

        self.conv1 = conv3x3(
            inplanes, planes, kernel_size=kernel_size, stride=stride)
        if norm == 'bn':
            self.bn1 = BN(inplanes)
        elif norm == 'gn':
            self.bn1 = nn.GroupNorm(NUM_GROUP, inplanes)
        else:
            raise ValueError('unsupport norm method')
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes, kernel_size=kernel_size,
                             dilation_rate=dilation_rate, padding=dilation_rate)
        if norm == 'bn':
            self.bn2 = BN(planes)
        elif norm == 'gn':
            self.bn2 = nn.GroupNorm(NUM_GROUP, planes)
        else:
            raise ValueError('unsupport norm method')
        self.se = SELayer(planes, reduction)

        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes:
            if norm == 'bn':
                self.shortcut = nn.Sequential(
                    BN(inplanes),
                    self.relu,
                    nn.Conv3d(inplanes, planes, kernel_size=1,
                              stride=stride, bias=False)
                )
            elif norm == 'gn':
                self.shortcut = nn.Sequential(
                    nn.GroupNorm(NUM_GROUP, inplanes),
                    self.relu,
                    nn.Conv3d(inplanes, planes, kernel_size=1,
                              stride=stride, bias=False)
                )
            else:
                raise ValueError('unsupport norm method')

        self.stride = stride

    def forward(self, x):
        # pdb.set_trace()
        residue = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.se(out)

        out += self.shortcut(residue)

        return out


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, se=False, norm='bn'):
        super(inconv, self).__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=(
            1, 3, 3), padding=(0, 1, 1), bias=False)
        self.relu = nn.ReLU(inplace=True)

        if not se:
            self.conv2 = BasicBlock(
                out_ch, out_ch, kernel_size=(1, 3, 3), norm=norm)
        else:
            self.conv2 = SEBasicBlock(
                out_ch, out_ch, kernel_size=(1, 3, 3), norm=norm)

    def forward(self, x):

        out = self.relu(self.conv1(x))
        out = self.conv2(out)

        return out


class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, se=False, reduction=2, dilation_rate=1, norm='bn'):
        super(conv_block, self).__init__()

        if not se:
            self.conv = BasicBlock(
                in_ch, out_ch, stride=stride, dilation_rate=dilation_rate, norm=norm)
        else:
            self.conv = SEBasicBlock(
                in_ch, out_ch, stride=stride, reduction=reduction, dilation_rate=dilation_rate, norm=norm)

    def forward(self, x):

        out = self.conv(x)

        return out


class up_block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=(4, 4, 4), scale=(2, 2, 2), up_mode='deconv', se=False, reduction=2, norm='bn'):
        super(up_block, self).__init__()

        if up_mode == 'bilinear':
            self.up = nn.UpsamplingBilinear3d(scale_factor=scale)
            up_channel = in_ch
        elif up_mode == 'deconv':
            self.up = nn.ConvTranspose3d(
                in_ch, out_ch, kernel, scale, padding=1)
            up_channel = out_ch
        # elif up_mode == 'carafe':
        #     self.up = CARAFE3DPack(in_ch, scale, (3,5,5))
        #     #self.up = CARAFE3DPack(in_ch, scale, (5,5,5))
        #     up_channel = in_ch

        self.conv = nn.Sequential(
            conv_block(up_channel + out_ch, out_ch, se=se,
                       reduction=reduction, norm=norm),
        )

    def forward(self, x1, x2):
        # pdb.set_trace()
        x1 = self.up(x1)

        h1, w1 = x1.shape[-2:]
        h2, w2 = x2.shape[-2:]

        if h1 != h2 or w1 != w2:
            x1 = F.pad(x1, [0, w2-w1, 0, h2-h1])

        out = torch.cat([x2, x1], dim=1)
        out = self.conv(out)

        return out

class up_nocat(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=(4,4,4), scale=(2,2,2), se=False, reduction=2, norm='bn'):
        super(up_nocat, self).__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel, scale, padding=1)
        self.conv = conv_block(out_ch, out_ch, se=se, reduction=reduction, norm=norm)

    def forward(self, x):
        out = self.up(x)
        out = self.conv(out)

        return out

class literal_conv(nn.Module):
    def __init__(self, in_ch, out_ch, se=False, reduction=2, norm='bn'):
        super(literal_conv, self).__init__()

        self.conv = conv_block(in_ch, out_ch, se=se,
                               reduction=reduction, norm=norm)

    def forward(self, x):

        out = self.conv(x)

        return out


class DenseASPPBlock(nn.Sequential):
    """Conv Net block for building DenseASPP"""

    def __init__(self, input_num, num1, num2, dilation_rate, drop_out, bn_start=True, norm='bn'):
        super(DenseASPPBlock, self).__init__()
        if bn_start:
            if norm == 'bn':
                self.add_module('norm_1', BN(input_num))
            elif norm == 'gn':
                self.add_module('norm_1', nn.GroupNorm(NUM_GROUP, input_num))

        self.add_module('relu_1', nn.ReLU(inplace=True))
        self.add_module('conv_1', nn.Conv3d(
            in_channels=input_num, out_channels=num1, kernel_size=1))

        if norm == 'bn':
            self.add_module('norm_2', BN(num1))
        elif norm == 'gn':
            self.add_module('norm_2', nn.GroupNorm(NUM_GROUP, num1))
        self.add_module('relu_2', nn.ReLU(inplace=True))
        self.add_module('conv_2', nn.Conv3d(in_channels=num1, out_channels=num2, kernel_size=3,
                                            dilation=dilation_rate, padding=dilation_rate))

        self.drop_rate = drop_out

    def forward(self, input):
        feature = super(DenseASPPBlock, self).forward(input)

        if self.drop_rate > 0:
            feature = F.dropout3d(
                feature, p=self.drop_rate, training=self.training)

        return feature

# leanrning rate policy


class LRScheduler(object):
    r"""Learning Rate Scheduler
    For mode='step', we multiply lr with `decay_factor` at each epoch in `step`.
    For mode='poly'::
        lr = targetlr + (baselr - targetlr) * (1 - iter / maxiter) ^ power
    For mode='cosine'::
        lr = targetlr + (baselr - targetlr) * (1 + cos(pi * iter / maxiter)) / 2
    If warmup_epochs > 0, a warmup stage will be inserted before the main lr scheduler.
    For warmup_mode='linear'::
        lr = warmup_lr + (baselr - warmup_lr) * iter / max_warmup_iter
    For warmup_mode='constant'::
        lr = warmup_lr
    Parameters
    ----------
    mode : str
        Modes for learning rate scheduler.
        Currently it supports 'step', 'poly' and 'cosine'.
    niters : int
        Number of iterations in each epoch.
    base_lr : float
        Base learning rate, i.e. the starting learning rate.
    epochs : int
        Number of training epochs.
    step : list
        A list of epochs to decay the learning rate.
    decay_factor : float
        Learning rate decay factor.
    targetlr : float
        Target learning rate for poly and cosine, as the ending learning rate.
    power : float
        Power of poly function.
    warmup_epochs : int
        Number of epochs for the warmup stage.
    warmup_lr : float
        The base learning rate for the warmup stage.
    warmup_mode : str
        Modes for the warmup stage.
        Currently it supports 'linear' and 'constant'.
    """

    def __init__(self, optimizer, niters, args):
        super(LRScheduler, self).__init__()

        self.mode = args.lr_mode
        self.warmup_mode = args.warmup_mode if hasattr(
            args, 'warmup_mode') else 'linear'
        assert(self.mode in ['step', 'poly', 'cosine'])
        assert(self.warmup_mode in ['linear', 'constant'])

        self.optimizer = optimizer

        self.base_lr = args.lr if hasattr(args, 'lr') else 0.1
        self.learning_rate = self.base_lr
        self.niters = niters

        self.step = [int(i) for i in args.step.split(
            ',')] if hasattr(args, 'step') else [30, 60, 90]
        self.decay_factor = args.decay_factor if hasattr(
            args, 'decay_factor') else 0.1
        self.targetlr = args.targetlr if hasattr(args, 'targetlr') else 0.0
        self.power = args.power if hasattr(args, 'power') else 2.0
        self.warmup_lr = args.warmup_lr if hasattr(args, 'warmup_lr') else 0.0
        self.max_iter = args.epochs * niters
        self.warmup_iters = (args.warmup_epochs if hasattr(
            args, 'warmup_epochs') else 0) * niters

    def update(self, i, epoch):
        T = epoch * self.niters + i
        #assert (T >= 0 and T <= self.max_iter)

        if self.warmup_iters > T:
            # Warm-up Stage
            if self.warmup_mode == 'linear':
                self.learning_rate = self.warmup_lr + (self.base_lr - self.warmup_lr) * \
                    T / self.warmup_iters
            elif self.warmup_mode == 'constant':
                self.learning_rate = self.warmup_lr
            else:
                raise NotImplementedError
        else:
            if self.mode == 'step':
                count = sum([1 for s in self.step if s <= epoch])
                self.learning_rate = self.base_lr * \
                    pow(self.decay_factor, count)
            elif self.mode == 'poly':
                self.learning_rate = self.targetlr + (self.base_lr - self.targetlr) * \
                    pow(1 - (T - self.warmup_iters) /
                        (self.max_iter - self.warmup_iters), self.power)
            elif self.mode == 'cosine':
                self.learning_rate = self.targetlr + (self.base_lr - self.targetlr) * \
                    (1 + cos(pi * (T - self.warmup_iters) /
                             (self.max_iter - self.warmup_iters))) / 2
            else:
                raise NotImplementedError

        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = self.learning_rate

###############################################################################
# Functions
###############################################################################


def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    #print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError(
            'initialization method [%s] is not implemented' % init_type)
