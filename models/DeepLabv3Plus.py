'''
Author       : 
Date         : 2022-04-01 18:56:44
LastEditTime : 2022-04-01 21:06:07

Description  : 

Code is far away from bug with the animal protecting
'''


import torch
import torch.nn as nn
import torch.nn.functional as F

from .net import *
from skimage import measure


class _ConvBatchNormReLU(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        relu=True,
    ):
        super(_ConvBatchNormReLU, self).__init__()
        self.add_module(
            "conv",
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
        )
        self.add_module(
            "bn",
            nn.BatchNorm3d(
                num_features=out_channels, eps=1e-5, momentum=0.999, affine=True
            ),
        )

        if relu:
            self.add_module("relu", nn.ReLU(inplace=True))

    def forward(self, x):
        return super(_ConvBatchNormReLU, self).forward(x)

class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling with image pool"""

    def __init__(self, in_channels, out_channels, pyramids):
        super(ASPP, self).__init__()
        self.c0 = _ConvBatchNormReLU(in_channels, out_channels, 1, 1, 0, 1)
        self.c1 = _ConvBatchNormReLU(in_channels, out_channels, 3, 1, padding=(1,pyramids[0],pyramids[0]), dilation=(1,pyramids[0],pyramids[0]))
        self.c2 = _ConvBatchNormReLU(in_channels, out_channels, 3, 1, padding=(1,pyramids[1],pyramids[1]), dilation=(1,pyramids[1],pyramids[1]))
        self.c3 = _ConvBatchNormReLU(in_channels, out_channels, 3, 1, padding=(1,pyramids[2],pyramids[2]), dilation=(1,pyramids[2],pyramids[2]))
        self.imagepool = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(in_channels, out_channels, 1, 1, 0, 1),
            nn.ReLU(True)
        )
    
    def forward(self, x):
        _,_,d,h,w = x.size()
        out_imagepool = self.imagepool(x)
        out_imagepool = F.upsample(out_imagepool, size=(d,h,w), mode='trilinear')

        out_c0 = self.c0(x)
        out_c1 = self.c1(x)
        out_c2 = self.c2(x)
        out_c3 = self.c3(x)
        out = torch.cat([out_c0, out_c1, out_c2, out_c3, out_imagepool], dim=1)

        return out


class DeepLabv3Plus(nn.Module):

    def __init__(self, channel, num_classes=4, se=True, reduction=2, norm='bn'):
        super(DeepLabv3Plus, self).__init__()

        self.conv1x = inconv(channel, 32, norm=norm)
        self.maxpool1 = nn.MaxPool3d((1,2,2)) # [30,120,120]

        self.conv2x = self._make_layer(conv_block, 32, 64, 3, se=se, stride=1, reduction=reduction, norm=norm)
        self.maxpool2 = nn.MaxPool3d((2,2,2)) # [15,60,60]

        self.conv4x = self._make_layer(conv_block, 64, 96, 4, se=se, stride=1, reduction=reduction, norm=norm)
        self.maxpool3 = nn.MaxPool3d((2,2,2)) # [15,60,60]

        self.conv8x_mg1 = self._make_layer(conv_block, 96, 128, 1, se=se, stride=1, reduction=reduction, dilation_rate=(1,2,2))
        self.conv8x_mg2 = self._make_layer(conv_block, 128, 128, 1, se=se, stride=1, reduction=reduction, dilation_rate=(1,4,4))
        self.conv8x_mg3 = self._make_layer(conv_block, 128, 128, 1, se=se, stride=1, reduction=reduction, dilation_rate=(1,2,2))

        pyramids = [6,12,18]
        self.aspp = ASPP(128, 64, pyramids)
        self.f1 = _ConvBatchNormReLU(64*(len(pyramids)+2), 128, 1, 1, 0, 1)

        self.literal = _ConvBatchNormReLU(96, 96, 1, 1, 0, 1)
        self.f2 = nn.Sequential(
            _ConvBatchNormReLU(96+128, 128, 3, 1, 1, 1),
            _ConvBatchNormReLU(128, 128, 3, 1, 1, 1)
        )

        self.out_conv = nn.Conv3d(128, num_classes, 1)

        self.num_classes = num_classes

    def forward(self, x):
        # exp, x: [1, 32, 400, 200]

        o1 = self.conv1x(x) # o1: [32, 32, 400, 200]

        o2 = self.maxpool1(o1)
        o2 = self.conv2x(o2) # o2: [64, 32, 200, 100]

        o3 = self.maxpool2(o2)
        o3 = self.conv4x(o3) # o3: [96, 16, 100, 50]

        o4 = self.maxpool3(o3)

        o4 = self.conv8x_mg1(o4)
        o4 = self.conv8x_mg2(o4)
        o4 = self.conv8x_mg3(o4) # o4: [128, 8, 50, 25]

        out_aspp = self.aspp(o4) # aspp: [320, 8, 50, 25]
        out_f1 = self.f1(out_aspp) # f1: [128, 8, 50, 25]

        out_reduce = self.literal(o3) # reduce: [96, 16, 100, 50]
        out_f1 = F.upsample(out_f1, size=o3.shape[2:], mode='trilinear') # f1: [128, 16, 100, 50]

        out = torch.cat((out_f1, out_reduce), dim=1) # cat: [224, 16, 100, 50]
        out = self.f2(out) # out: [128, 16, 100, 50]
        out = self.out_conv(out) # out: [num_classes, 16, 100, 50]
        out = F.upsample(out, size=x.shape[2:], mode='trilinear') # out: [num_classes, 32, 400, 200]

        return out 


    def _make_layer(self, block, in_ch, out_ch, num_blocks, se=True, stride=1, reduction=2, dilation_rate=1, norm='bn'):
        layers = []
        
        layers.append(block(in_ch, out_ch, se=se, stride=stride, reduction=reduction, dilation_rate=dilation_rate, norm=norm))
        for i in range (num_blocks-1):
            layers.append(block(out_ch, out_ch, se=se, stride=1, reduction=reduction, dilation_rate=dilation_rate, norm=norm))

        return nn.Sequential(*layers)

class DeepLabv3Plus_Encoder(DeepLabv3Plus):
    def __init__(self, channel, num_classes=19, se=True, reduction=2, norm='bn'):
        super(DeepLabv3Plus_Encoder, self).__init__(channel=channel, num_classes=num_classes, se=se, reduction=reduction, norm=norm)

        self.encoder_classif_o4_1 = _ConvBatchNormReLU(128, 128, 1, 1, 0, 1)
        self.encoder_classif_o4_2 = self._make_layer(conv_block, 128, 256, 3, se=se, stride=1, reduction=reduction, norm=norm)

        self.encoder_classif_o3_1 = _ConvBatchNormReLU(96, 96, 1, 1, 0, 1)
        self.encoder_classif_o3_2 = self._make_layer(conv_block, 96, 128, 3, se=se, stride=1, reduction=reduction, norm=norm)

        self.encoder_classif_up_cn = nn.Conv1d(128+256, 256, 3, 1, 1)
        self.encoder_classif_up_bn = nn.BatchNorm1d(num_features=256, eps=1e-5, momentum=0.999, affine=True)

        self.encoder_classif_fc = nn.Linear(256, num_classes-1) # exclude background

    def encoder_classif(self, o3, o4):
        # o3: [96, 16, 100, 50]
        # o4, [128, 8, 50, 25]

        depth_x1 = o4.shape[2]
        depth_x2 = o3.shape[2]

        x1 = self.encoder_classif_o4_1(o4) # [128, 8, 50, 25]
        x1 = self.encoder_classif_o4_2(x1) # [256, 8, 50, 25]
        x1 = F.adaptive_avg_pool3d(x1, (depth_x1,1,1)) # [256, 8, 1, 1]

        x2 = self.encoder_classif_o3_1(o3) # [96, 16, 100, 50]
        x2 = self.encoder_classif_o3_2(x2) # [128, 16, 100, 50]
        x2 = F.adaptive_avg_pool3d(x2, (depth_x2,1,1)) # [128, 16, 1, 1]

        x1 = torch.squeeze(x1, 3)
        x1 = torch.squeeze(x1, 3) # [256, 8]
        x2 = torch.squeeze(x2, 3)
        x2 = torch.squeeze(x2, 3) # [128, 16]
        
        x1_up = F.upsample(x1, size=depth_x1*2, mode='linear') # [256, 16]
    
        out = torch.cat((x1_up, x2), dim=1) # [256+128, 16]

        out = self.encoder_classif_up_cn(out) 
        out = self.encoder_classif_up_bn(out) # [256, 16]
        out = F.upsample(out, size=depth_x2*2, mode='linear') # [256, 32]

        out = out.transpose(1, 2) # [32, 256]
        out = self.encoder_classif_fc(out) # [32, num_classes-1]

        return out


    def forward(self, x):
        # exp, x: [1, 32, 400, 200]

        o1 = self.conv1x(x) # o1: [32, 32, 400, 200]

        o2 = self.maxpool1(o1)
        o2 = self.conv2x(o2) # o2: [64, 32, 200, 100]

        o3 = self.maxpool2(o2)
        o3 = self.conv4x(o3) # o3: [96, 16, 100, 50]

        o4 = self.maxpool3(o3)

        o4 = self.conv8x_mg1(o4)
        o4 = self.conv8x_mg2(o4)
        o4 = self.conv8x_mg3(o4) # o4: [128, 8, 50, 25]

        out_encoder = self.encoder_classif(o3, o4)

        out_aspp = self.aspp(o4) # aspp: [320, 8, 50, 25]
        out_f1 = self.f1(out_aspp) # f1: [128, 8, 50, 25]

        out_reduce = self.literal(o3) # reduce: [96, 16, 100, 50]
        out_f1 = F.upsample(out_f1, size=o3.shape[2:], mode='trilinear') # f1: [128, 16, 100, 50]

        out = torch.cat((out_f1, out_reduce), dim=1)
        out = self.f2(out)
        out = self.out_conv(out)
        out = F.upsample(out, size=x.shape[2:], mode='trilinear') # out: [num_classes, 32, 400, 200]

        return out, out_encoder

class DeepLabv3Plus_Encoder2(DeepLabv3Plus_Encoder):
    def __init__(self, channel, num_classes=19, se=True, reduction=2, norm='bn'):
        super(DeepLabv3Plus_Encoder2, self).__init__(channel=channel, num_classes=num_classes, se=se, reduction=reduction, norm=norm)

    def forward(self, x):
        # exp, x: [1, 32, 400, 200]
        depth = x.shape[2]

        o1 = self.conv1x(x) # o1: [32, 32, 400, 200]

        o2 = self.maxpool1(o1)
        o2 = self.conv2x(o2) # o2: [64, 32, 200, 100]

        o3 = self.maxpool2(o2)
        o3 = self.conv4x(o3) # o3: [96, 16, 100, 50]

        o4 = self.maxpool3(o3)

        o4 = self.conv8x_mg1(o4)
        o4 = self.conv8x_mg2(o4)
        o4 = self.conv8x_mg3(o4) # o4: [128, 8, 50, 25]

        out_encoder = self.encoder_classif(o3, o4) # [32, num_classes-1]

        out_aspp = self.aspp(o4) # aspp: [320, 8, 50, 25]
        out_f1 = self.f1(out_aspp) # f1: [128, 8, 50, 25]

        out_reduce = self.literal(o3) # reduce: [96, 16, 100, 50]
        out_f1 = F.upsample(out_f1, size=o3.shape[2:], mode='trilinear') # f1: [128, 16, 100, 50]

        out = torch.cat((out_f1, out_reduce), dim=1)
        out = self.f2(out)
        out = self.out_conv(out)
        out = F.upsample(out, size=x.shape[2:], mode='trilinear') # out: [num_classes, 32, 400, 200]

        out_encoder_prob = torch.nn.functional.sigmoid(out_encoder)
        out_encoder_prob = out_encoder_prob.transpose(1, 2) # [num_classes-1, 32]
        out_encoder_prob = out_encoder_prob.view((self.num_classes-1, depth, 1, 1))

        out[:, 1:, :, :, :] = out[:, 1:, :, :, :].clone() * out_encoder_prob

        return out, out_encoder


class DeepLabv3Plus_FineTune(nn.Module):

    def __init__(self, channel, num_classes=4, se=True, reduction=2, norm='bn'):
        super(DeepLabv3Plus_FineTune, self).__init__()

        self.conv1x = inconv(channel, 32, norm=norm)
        self.maxpool1 = nn.MaxPool3d((2,2,2)) # [30,120,120]

        self.conv2x = self._make_layer(conv_block, 32, 64, 3, se=se, stride=1, reduction=reduction, norm=norm)
        self.maxpool2 = nn.MaxPool3d((2,2,2)) # [15,60,60]

        self.conv4x = self._make_layer(conv_block, 64, 96, 4, se=se, stride=1, reduction=reduction, norm=norm)
        self.maxpool3 = nn.MaxPool3d((2,2,2)) # [15,60,60]

        self.conv8x_mg1 = self._make_layer(conv_block, 96, 128, 1, se=se, stride=1, reduction=reduction, dilation_rate=(1,2,2))
        self.conv8x_mg2 = self._make_layer(conv_block, 128, 128, 1, se=se, stride=1, reduction=reduction, dilation_rate=(1,4,4))
        self.conv8x_mg3 = self._make_layer(conv_block, 128, 128, 1, se=se, stride=1, reduction=reduction, dilation_rate=(1,2,2))

        pyramids = [6,12,18]
        self.aspp = ASPP(128, 64, pyramids)
        self.f1 = _ConvBatchNormReLU(64*(len(pyramids)+2), 128, 1, 1, 0, 1)

        self.literal = _ConvBatchNormReLU(96, 96, 1, 1, 0, 1)
        self.f2 = nn.Sequential(
            _ConvBatchNormReLU(96+128, 128, 3, 1, 1, 1),
            _ConvBatchNormReLU(128, 128, 3, 1, 1, 1)
        )

        self.out_conv_new = nn.Conv3d(128, num_classes, 1)
        self.se = SEBasicBlock(32, 32)

    def forward(self, x):
        # x : 1 1 32 320 320
        
        o1 = self.conv1x(x)  # 1 32 32 320 320 
        
        o2_mp = self.maxpool1(o1)  # 1 32 32 160 160
        
        o2_conv = self.conv2x(o2_mp)  # 1 64 32 160 160

        o3_mp = self.maxpool2(o2_conv)  # 1 64 16 80 80

        o3_conv = self.conv4x(o3_mp)  # 1 96 16 80 80

        o4_mp = self.maxpool3(o3_conv)  # 1 96 8 40 40

        o4_mg1 = self.conv8x_mg1(o4_mp)  # 1 128 8 40 40
        # o4_mg2 = self.conv8x_mg2(o4_mg1)
        o4_mg3 = self.conv8x_mg3(o4_mg1)  # 1 128 8 40 40 

        out_aspp = self.aspp(o4_mg3)  # 1 320 8 40 40
        out_f1 = self.f1(out_aspp)  # 1 128 8 40 40

        out_reduce = self.literal(o3_conv)  # 1 96 16 80 80
        out_f1_up = F.upsample(out_f1, size=o3_conv.shape[2:], mode='trilinear')  # 1 128 16 80 80

        out = torch.cat((out_f1_up, out_reduce), dim=1)  # 1 224 16 80 80
        out_f2 = self.f2(out)  # 1 128 16 80 80

        out_conv_new = self.out_conv_new(out_f2)  # 1 24 16 80 80
        out = F.upsample(out_conv_new, size=x.shape[2:], mode='trilinear')  # 1 24 32 320 320

        return out
    
    def _make_layer(self, block, in_ch, out_ch, num_blocks, se=True, stride=1, reduction=2, dilation_rate=1, norm='bn'):
        layers = []
        
        layers.append(block(in_ch, out_ch, se=se, stride=stride, reduction=reduction, dilation_rate=dilation_rate, norm=norm))
        for i in range (num_blocks-1):
            layers.append(block(out_ch, out_ch, se=se, stride=1, reduction=reduction, dilation_rate=dilation_rate, norm=norm))

        return nn.Sequential(*layers)

if __name__ == '__main__':
    model = DeepLabv3Plus_FineTune(1, 16)

    input_data = torch.rand((1, 1, 64, 240, 240))

    out = model(input_data)

    print(out.shape)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")