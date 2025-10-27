"""
UNet3D
@Author: Zhengyong Huang
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
import scipy.stats as st
import scipy.ndimage
# from axial_attention import AxialAttention
# from torch.cuda.amp import autocast

__all__ = ['Proposed', ]


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        nn.init.xavier_uniform_(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        nn.init.constant_(m.weight.data, 1.0)
        nn.init.constant_(m.bias.data, 0.0)
        
def min_max_norm(in_):
    max_ = in_.max(3)[0].max(2)[0].unsqueeze(2).unsqueeze(3).expand_as(in_)
    min_ = in_.min(3)[0].min(2)[0].unsqueeze(2).unsqueeze(3).expand_as(in_)
    in_ = in_ - min_
    return in_.div(max_-min_+1e-8)


def gkern(kernlen=16, nsig=3):
    interval = (2*nsig+1.)/kernlen
    x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel

def gkern3d(kernlen=21, std=3):
    """Returns a 3D Gaussian kernel."""
    # Create 2D Gaussian kernel using scipy's gaussian_filter function
    gkern1d = scipy.ndimage.gaussian_filter(np.zeros((kernlen, kernlen, kernlen)), std)
    
    # Center the kernel to the middle of the array
    gkern1d[kernlen // 2, kernlen // 2, kernlen // 2] = 1
    
    # Create a 3D kernel using the outer product of the 2D kernel
    kernel_raw = scipy.ndimage.gaussian_filter(gkern1d, std)
    
    # Normalize the kernel
    kernel = kernel_raw / kernel_raw.sum()
    return kernel

def conv3x3(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation_rate=1):
    if kernel_size == (1, 3, 3):
        return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                         padding=(0, 1, 1), bias=False, dilation=dilation_rate)
    else:
        return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                         padding=padding, bias=False, dilation=dilation_rate)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, dilation_rate=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, kernel_size, stride, padding=padding)
        self.bn1 = nn.InstanceNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, kernel_size=kernel_size, padding=padding, 
                             dilation_rate=dilation_rate)
        self.bn2 = nn.InstanceNorm3d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.InstanceNorm3d(planes),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        residue = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(residue)
        out = self.relu(out)
        return out

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

class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, kernel_size=3, stride=1, reduction=4):
        super(SEBasicBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, kernel_size=kernel_size, stride=stride)
        self.bn1 = nn.InstanceNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.se = SELayer(planes, reduction)

        self.shortcut = nn.Sequential(nn.Conv3d(inplanes, planes, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        # pdb.set_trace()
        residue = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.se(out)

        out += self.shortcut(residue)

        return out


class _conv_block(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, se=False, reduction=2, dilation_rate=1, norm='bn'):
        super(_conv_block, self).__init__()
        self.conv = BasicBlock(in_ch, out_ch, stride=stride, dilation_rate=dilation_rate)
        
        if not se:
            self.conv = BasicBlock(in_ch, out_ch, stride=stride, dilation_rate=dilation_rate)
        else:
            self.conv = SEBasicBlock(in_ch, out_ch, stride=stride, reduction=reduction)

    def forward(self, x):
        out = self.conv(x)
        return out

class up_block(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=(4, 4, 4), scale=(2, 2, 2), bilinear=False, reduction=2, norm='bn'):
        super(up_block, self).__init__()
        if bilinear:
            self.up = nn.UpsamplingBilinear3d(scale_factor=scale)
        else:
            self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel, scale, padding=1)

        self.conv = _conv_block(2*out_ch, out_ch, reduction=reduction, norm=norm)
        # self.cwca = CWCABlock(2*out_ch)
        # self.conv = _conv_block(4*out_ch, out_ch, reduction=reduction, norm=norm)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        out = torch.cat([x2, x1], dim=1)
        # out = self.cwca(x1, x2)
        # out = torch.cat([out, x2, x1], dim=1)
        
        out = self.conv(out)
        return out

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=4):
        super(ChannelAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y_avg = self.avg_pool(x).view(b, c)  # (1, c)
        y_max = self.max_pool(x).view(b, c)  # (1, c)
        y_avg = self.fc(y_avg).view(b, c, 1, 1, 1)
        y_max = self.fc(y_max).view(b, c, 1, 1, 1)

        return self.sigmoid(y_avg+y_max)  # (1, c, 1, 1, 1)

class CWCABlock(nn.Module):
    def __init__(self, channel):
        super(CWCABlock, self).__init__()

        c = channel
        self.conv = nn.Sequential(
            BasicBlock(c, c, 3, 1, 1),
            BasicBlock(c, c, 3, 1, 1)
        )
        self.weight_conv = nn.Sequential(
            nn.Conv3d(c, 2, 3, 1, 1),
            nn.Softmax(dim=1)
        )
        self.channel_attention = ChannelAttention(c)

    def forward(self, con_fm, decon_fm):
        concat_fm = torch.cat([con_fm, decon_fm], dim=1)
        x = self.conv(concat_fm)
        weight_map = self.weight_conv(x)
        concat = torch.cat([
            con_fm * weight_map[:, 0, ...].unsqueeze(1),
            decon_fm * weight_map[:, 1, ...].unsqueeze(1),
        ], dim=1) # (1, 2*c, h, w, t)
        channel_wise = self.channel_attention(concat)
        return concat*channel_wise


class RFB(nn.Module):
    # RFB-like multi-scale module
    def __init__(self, in_channel, out_channel):
        super(RFB, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicBlock(in_channel, out_channel, 3),
        )
        self.branch1 = nn.Sequential(
            BasicBlock(in_channel, out_channel, kernel_size=(1, 1, 3), padding=(0, 0, 1)),
            BasicBlock(out_channel, out_channel, kernel_size=(1, 3, 1), padding=(0, 1, 0)),
            BasicBlock(out_channel, out_channel, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            BasicBlock(out_channel, out_channel, kernel_size=(1, 1, 1), padding=(0, 0, 0)),
        )
        self.branch2 = nn.Sequential(
            BasicBlock(in_channel, out_channel, kernel_size=(1, 1, 5), padding=(0, 0, 2)),
            BasicBlock(out_channel, out_channel, kernel_size=(1, 5, 1), padding=(0, 2, 0)),
            BasicBlock(out_channel, out_channel, kernel_size=(5, 1, 1), padding=(2, 0, 0)),
            BasicBlock(out_channel, out_channel, kernel_size=(1, 1, 1), padding=(0, 0, 0)),
        )
        self.branch3 = nn.Sequential(
            BasicBlock(in_channel, out_channel, kernel_size=(1, 1, 7), padding=(0, 0, 3)),
            BasicBlock(out_channel, out_channel, kernel_size=(1, 7, 1), padding=(0, 3, 0)),
            BasicBlock(out_channel, out_channel, kernel_size=(7, 1, 1), padding=(3, 0, 0)),
            BasicBlock(out_channel, out_channel, kernel_size=(1, 1, 1), padding=(0, 0, 0)),
        )
        self.conv_cat = BasicBlock(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicBlock(in_channel, out_channel, 3)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class aggregation(nn.Module):
    # dense aggregation, it can be replaced by other aggregation model, such as DSS, amulet, and so on.
    # used after MSF
    def __init__(self, channel):
        super(aggregation, self).__init__()
        self.relu = nn.ReLU(True)

        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv_upsample1 = BasicBlock(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicBlock(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicBlock(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicBlock(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicBlock(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicBlock(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicBlock(3*channel, 3*channel, 3, padding=1)
        self.conv4 = BasicBlock(3*channel, 3*channel, 3, padding=1)
        self.conv5 = nn.Conv3d(3*channel, channel, 3,1,1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = self.conv_upsample2(self.upsample(self.upsample(x1))) \
               * self.conv_upsample3(self.upsample(x2)) * x3

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x
    
class HA(nn.Module):
    # holistic attention module
    def __init__(self):
        super(HA, self).__init__()
        gaussian_kernel = np.float32(gkern3d(31, 4))  # 11 2
        gaussian_kernel = gaussian_kernel[np.newaxis, np.newaxis, ...]
        repeated_array = np.repeat(gaussian_kernel, 96, axis=1)
        self.gaussian_kernel = Parameter(torch.from_numpy(repeated_array))  # 1, 1, 11, 11, 11

    def forward(self, attention, x):
        soft_attention = F.conv3d(attention, self.gaussian_kernel, stride=1, padding=15)
        soft_attention = min_max_norm(soft_attention)
        x = torch.mul(x, soft_attention.max(attention))
        return x

class CPD_Module(nn.Module):
    # resnet based encoder decoder
    def __init__(self, channel=32):
        super(CPD_Module, self).__init__()
        self.conv = nn.Sequential(BasicBlock(channel, channel, 3),
                                  BasicBlock(channel, channel, 3),
                                )
        self.conv3_1 = BasicBlock(channel, 96, 3)
        self.conv4_1 = BasicBlock(96, 128, 3)
        
        self.conv3_2 = BasicBlock(channel, 96, 3)
        self.conv4_2 = BasicBlock(96, 128, 3)
        self.rfb2_1 = RFB(96, channel)
        self.rfb3_1 = RFB(96, channel)
        self.rfb4_1 = RFB(128, channel)
        self.agg1 = aggregation(channel)

        self.rfb2_2 = RFB(96, channel)
        self.rfb3_2 = RFB(96, channel)
        self.rfb4_2 = RFB(128, channel)
        self.agg2 = aggregation(channel)
        self.upsample = nn.Upsample(scale_factor=0.5, mode='trilinear', align_corners=True)

        self.HA = HA()
        
        self.maxpool21 = nn.MaxPool3d((2,2,2))
        self.maxpool31 = nn.MaxPool3d((2,2,2))
        self.maxpool22 = nn.MaxPool3d((2,2,2))
        self.maxpool32 = nn.MaxPool3d((2,2,2))
        
    def forward(self, x):
        x2_1 = self.conv(x)  # b, 64, 16, 60, 60
        x3_1 = self.conv3_1(self.maxpool21(x2_1))  # b, 96, 8, 30, 30
        x4_1 = self.conv4_1(self.maxpool31(x3_1))  # b, 128, 4, 15, 15  
        
        x2_1 = self.rfb2_1(x2_1)  # b, 64, 16, 60, 60
        x3_1 = self.rfb3_1(x3_1)  # b, 64, 8, 30, 30
        x4_1 = self.rfb4_1(x4_1)  # b, 64, 4, 15, 15  
        
        attention_map = self.agg1(x4_1, x3_1, x2_1)  # 1, 1, 16, 60, 60

        x2_2 = self.HA(attention_map.sigmoid(), x2_1)  # b, 64, 16, 60, 60
        x3_2 = self.conv3_2(self.maxpool22(x2_2))  # b, 64, 8, 30, 30
        x4_2 = self.conv4_2(self.maxpool32(x3_2))  # b, 64, 4, 15, 15  

        x2_2 = self.rfb2_2(x2_2)
        x3_2 = self.rfb3_2(x3_2)
        x4_2 = self.rfb4_2(x4_2)
        detection_map = self.agg2(x4_2, x3_2, x2_2)

        # return self.upsample(attention_map), self.upsample(detection_map)
        return self.upsample(detection_map)


class OurNet(nn.Module):
    def __init__(self, in_c, num_cls):
        super(OurNet, self).__init__()

        self.channel = in_c
        self.num_classes = num_cls
        reduction = 4
        init_type = 'normal'
        
        self.conv1x = nn.Sequential(
            conv3x3(self.channel, 32, kernel_size=(3, 3, 3)),
            nn.InstanceNorm3d(32),
            nn.ReLU(inplace=True),
            BasicBlock(32, 32, kernel_size=(3, 3, 3))
            )
        self.maxpool1 = nn.MaxPool3d((2, 2, 2))

        self.conv2x = self._make_layer(
            _conv_block, 32, 64, 2, stride=1, se=False, reduction=reduction)
        self.maxpool2 = nn.MaxPool3d((2, 2, 2))

        self.conv4x = self._make_layer(
            _conv_block, 64, 96, 2, stride=1, se=False, reduction=reduction)
        self.maxpool3 = nn.MaxPool3d((2, 2, 2))
        
        self.cpd = CPD_Module(channel=96)

        # upsample
        self.up1 = up_block(in_ch=96, out_ch=96, reduction=reduction)
        self.literal1 = nn.Conv3d(96, 96, 3, padding=1)

        self.up2 = up_block(in_ch=96, out_ch=64, reduction=reduction)
        self.literal2 = nn.Conv3d(64, 64, 3, padding=1)
        
        self.up3 = up_block(in_ch=64, out_ch=32, reduction=reduction)
        self.literal3 = nn.Conv3d(32, 32, 3, padding=1)
        
        self.out = nn.Conv3d(32, self.num_classes, 1)

        # self.out_conv_new = nn.Conv3d(32, self.num_classes, 1)

        self.init_weights(init_type=init_type)
    
    
    def forward(self, x, **kargs):
        # down  x: 1 1 32 320 320
        o1 = self.conv1x(x)  # 1 32 64 240 240   b, 32, d, h, w
        
        o1_pool = self.maxpool1(o1)

        o2 = self.conv2x(o1_pool)  # 1 48 32 120 120    b, 48, d/2, h/2, w/2
        
        o2_pool = self.maxpool2(o2)
        
        o3 = self.conv4x(o2_pool)  # 1 64 16 60 60    b, 64, d/4, h/4, w/4
        
        o3_pool = self.maxpool3(o3)  # 1 64 8 30 30 
        # o3_pool = self.cpd(o3)  #  b, 64, d/8, h/8, w/8

        # up
        out1 = self.up1(o3_pool, self.literal1(o3))  # 1 64 16 60 60 
        out2 = self.up2(out1, self.literal2(o2))  # 1 48 32 120 120
        out3 = self.up3(out2, self.literal3(o1))  # 1 32 64 240 240
        out_up_new = self.out(out3)
        
        return out_up_new

    def _make_layer(self, block, in_ch, out_ch, num_blocks, stride=1, se=True, reduction=2, dilation_rate=1):
        layers = []
        layers.append(block(in_ch, out_ch, stride=stride, se=True,
                            reduction=reduction, dilation_rate=dilation_rate))
        for i in range(num_blocks-1):
            layers.append(block(out_ch, out_ch, stride=1,
                                reduction=reduction, dilation_rate=dilation_rate))
        return nn.Sequential(*layers)

    def init_weights(self, init_type='normal'):
        if init_type == 'normal':
            self.apply(weights_init_normal)
        elif init_type == 'xavier':
            self.apply(weights_init_xavier)
        else:
            raise NotImplementedError(
                'initialization method [%s] is not implemented' % init_type)


    def _make_layer(self, block, in_ch, out_ch, num_blocks, stride=1, se=True, reduction=2, dilation_rate=1):
        layers = []
        layers.append(block(in_ch, out_ch, stride=stride, se=se,
                            reduction=reduction, dilation_rate=dilation_rate))
        for i in range(num_blocks-1):
            layers.append(block(out_ch, out_ch, stride=1, se=se,
                                reduction=reduction, dilation_rate=dilation_rate))
        return nn.Sequential(*layers)

    def init_weights(self, init_type='normal'):
        if init_type == 'normal':
            self.apply(weights_init_normal)
        elif init_type == 'xavier':
            self.apply(weights_init_xavier)
        else:
            raise NotImplementedError(
                'initialization method [%s] is not implemented' % init_type)



if __name__ == "__main__":
    x_input = torch.rand(1, 1, 64, 240, 240)
    net = OurNet(in_c=1, num_cls=16)
    
    # x_input = torch.rand(1, 64, 16, 60, 60)
    # net = CPD_Module(channel=64)
    out_1 = net(x_input)
    print("out_1 shape: ", out_1.shape)
    print("Good!!!")
    
    total_params = sum(p.numel() for p in net.parameters())
    print(f"Total number of parameters: {total_params}")