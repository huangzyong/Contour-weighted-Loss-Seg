"""
UNet3D
@Author: Lingyun Wu
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .net import *
# from axial_attention import AxialAttention
# from torch.cuda.amp import autocast

__all__ = ['UNet3D', ]


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

def conv3x3(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation_rate=1):
    if kernel_size == (1, 3, 3):
        return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                         padding=(0, 1, 1), bias=False, dilation=dilation_rate)
    else:
        return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                         padding=padding, bias=False, dilation=dilation_rate)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, downsample=None, dilation_rate=1, norm='bn'):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, kernel_size, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, kernel_size=kernel_size,
                             dilation_rate=dilation_rate, padding=dilation_rate)
        self.bn2 = nn.BatchNorm3d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes),
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


class _conv_block(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, reduction=2, dilation_rate=1, norm='bn'):
        super(_conv_block, self).__init__()
        self.conv = BasicBlock(in_ch, out_ch, stride=stride, dilation_rate=dilation_rate, norm=norm)

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

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        out = torch.cat([x2, x1], dim=1)
        out = self.conv(out)
        return out


class UNet3D(nn.Module):
    def __init__(self, in_c, num_cls):
        super(UNet3D, self).__init__()

        self.channel = in_c
        self.num_classes = num_cls
        reduction = 2
        norm = 'bn'
        init_type = 'normal'
        

        self.conv1x = nn.Sequential(
            conv3x3(self.channel, 32, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            BasicBlock(32, 32, kernel_size=(3, 3, 3), norm=norm)
            )
        self.maxpool1 = nn.MaxPool3d((2, 2, 2))

        self.conv2x = self._make_layer(
            _conv_block, 32, 48, 2, stride=1, reduction=reduction, norm=norm)
        self.maxpool2 = nn.MaxPool3d((2, 2, 2))

        self.conv4x = self._make_layer(
            _conv_block, 48, 64, 2, stride=1, reduction=reduction, norm=norm)
        self.maxpool3 = nn.MaxPool3d((2, 2, 2))

        # upsample
        self.up1 = up_block(in_ch=64, out_ch=64, reduction=reduction, norm=norm)
        self.literal1 = nn.Conv3d(64, 64, 3, padding=1)

        self.up2 = up_block(in_ch=64, out_ch=48, reduction=reduction, norm=norm)
        self.literal2 = nn.Conv3d(48, 48, 3, padding=1)

        self.out_conv_new = nn.Conv3d(48, self.num_classes, 1)

        self.init_weights(init_type=init_type)
    
    
    def forward(self, x, **kargs):
        # down  x: 1 1 32 320 320
        o1 = self.conv1x(x)  # 1 32 32 320 320
        
        o1_pool = self.maxpool1(o1)  # 1 32 32 160 160 
        
        o2 = self.conv2x(o1_pool)  # 1 48 32 160 160 
        
        o2_pool = self.maxpool2(o2)  # 1 48 16 80 80 
        
        o3 = self.conv4x(o2_pool)  # 1 64 16 80 80 
        
        o3_pool = self.maxpool3(o3)  # 1 64 8 40 40

        # up
        out1 = self.up1(o3_pool, self.literal1(o3))  # 1 64 16 80 80
        out2 = self.up2(out1, self.literal2(o2))  # 1 48 32 160 160 
        
        out_conv_new = self.out_conv_new(out2)  # 1 24 32 160 160
        out_up_new = F.upsample(out_conv_new, size=x.shape[2:], mode='trilinear')  # 1 2 32 320 320

        return out_up_new

    def _make_layer(self, block, in_ch, out_ch, num_blocks, stride=1, reduction=2, dilation_rate=1, norm='bn'):
        layers = []
        layers.append(block(in_ch, out_ch, stride=stride,
                            reduction=reduction, dilation_rate=dilation_rate, norm=norm))
        for i in range(num_blocks-1):
            layers.append(block(out_ch, out_ch, stride=1,
                                reduction=reduction, dilation_rate=dilation_rate, norm=norm))
        return nn.Sequential(*layers)

    def init_weights(self, init_type='normal'):
        if init_type == 'normal':
            self.apply(weights_init_normal)
        elif init_type == 'xavier':
            self.apply(weights_init_xavier)
        else:
            raise NotImplementedError(
                'initialization method [%s] is not implemented' % init_type)


class UNet3D_cat(nn.Module):
    def __init__(self,):
        super(UNet3D_cat, self).__init__()

        channel = 1  
        new_num_classes = 24
        brain_classes = 2
        reduction = 2
        norm = 'bn'
        first_conv_kernel_z = 1
        init_type = 'normal'
        

        self.conv1x = nn.Sequential(
            conv3x3(channel, 32, kernel_size=(first_conv_kernel_z, 3, 3)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            BasicBlock(32, 32, kernel_size=(first_conv_kernel_z, 3, 3), norm=norm)
            )
        self.maxpool1 = nn.MaxPool3d((1, 2, 2))

        self.conv2x = self._make_layer(
            _conv_block, 32, 48, 2, stride=1, reduction=reduction, norm=norm)
        self.maxpool2 = nn.MaxPool3d((2, 2, 2))

        self.conv4x = self._make_layer(
            _conv_block, 48, 64, 2, stride=1, reduction=reduction, norm=norm)
        self.maxpool3 = nn.MaxPool3d((2, 2, 2))

        # upsample
        self.up1 = up_block(in_ch=64, out_ch=64, reduction=reduction, norm=norm)
        self.literal1 = nn.Conv3d(64, 64, 3, padding=1)

        self.up2 = up_block(in_ch=64, out_ch=48, reduction=reduction, norm=norm)
        self.literal2 = nn.Conv3d(48, 48, 3, padding=1)

        self.out_conv_new = nn.Conv3d(48, new_num_classes, 1)
        self.out_conv_brain = nn.Conv3d(48, brain_classes, 1)

        # contour
        self.up1_c = up_block(in_ch=64, out_ch=64, reduction=reduction, norm=norm)
        self.literal1_c = nn.Conv3d(64, 64, 3, padding=1)
        self.up2_c = up_block(in_ch=64, out_ch=48, reduction=reduction, norm=norm)
        self.literal2_c = nn.Conv3d(48, 48, 3, padding=1)
        self.out_conv_c = nn.Conv3d(48, new_num_classes, 1)
        self.out = nn.Conv3d(2 * new_num_classes, new_num_classes, 1)

        self.init_weights(init_type=init_type)
    
    
    def forward(self, x, **kargs):
        # down  x: 1 1 32 320 320
        o1 = self.conv1x(x)  # 1 32 32 320 320
        
        o1_pool = self.maxpool1(o1)  # 1 32 32 160 160 
        
        o2 = self.conv2x(o1_pool)  # 1 48 32 160 160 
        
        o2_pool = self.maxpool2(o2)  # 1 48 16 80 80 
        
        o3 = self.conv4x(o2_pool)  # 1 64 16 80 80 
        
        o3_pool = self.maxpool3(o3)  # 1 64 8 40 40

        # up
        out1 = self.up1(o3_pool, self.literal1(o3))  # 1 64 16 80 80
        out2 = self.up2(out1, self.literal2(o2))  # 1 48 32 160 160         

        out_conv_new = self.out_conv_new(out2)  # 1 24 32 160 160
        out_up_new = F.upsample(out_conv_new, size=x.shape[2:], mode='trilinear')  # 1 24 32 320 320 
        
        out_conv_brain = self.out_conv_brain(out2)  # 1 2 32 160 160
        out_up_brain = F.upsample(out_conv_brain, size=x.shape[2:], mode='trilinear')  # 1 2 32 320 320

        # contour
        out1_c = self.up1_c(o3_pool, self.literal1_c(o3))  # 1 64 16 80 80
        out2_c = self.up2_c(out1_c, self.literal2_c(o2))  # 1 48 32 160 160
        out_conv_c = self.out_conv_c(out2_c)  # 1 24 32 160 160
        out_up_c = F.upsample(out_conv_c, size=x.shape[2:], mode='trilinear')  # 1 24 32 320 320 

        out = torch.cat([out_conv_new, out_conv_c], dim=1)
        out = self.out(out)
        out = F.upsample(out_conv_c, size=x.shape[2:], mode='trilinear')  # 1 24 32 320 320 

        return out, out_up_brain, out_up_c

    def _make_layer(self, block, in_ch, out_ch, num_blocks, stride=1, reduction=2, dilation_rate=1, norm='bn'):
        layers = []
        layers.append(block(in_ch, out_ch, stride=stride,
                            reduction=reduction, dilation_rate=dilation_rate, norm=norm))
        for i in range(num_blocks-1):
            layers.append(block(out_ch, out_ch, stride=1,
                                reduction=reduction, dilation_rate=dilation_rate, norm=norm))
        return nn.Sequential(*layers)

    def init_weights(self, init_type='normal'):
        if init_type == 'normal':
            self.apply(weights_init_normal)
        elif init_type == 'xavier':
            self.apply(weights_init_xavier)
        else:
            raise NotImplementedError(
                'initialization method [%s] is not implemented' % init_type)


class UNet3D_2(nn.Module):
    def __init__(self,):
        super(UNet3D_2, self).__init__()

        channel = 1  
        new_num_classes = 24
        brain_classes = 2
        reduction = 2
        norm = 'bn'
        first_conv_kernel_z = 1
        init_type = 'normal'
        

        self.conv1x = nn.Sequential(
            conv3x3(channel, 32, kernel_size=(first_conv_kernel_z, 3, 3)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            BasicBlock(32, 32, kernel_size=(first_conv_kernel_z, 3, 3), norm=norm)
            )
        self.maxpool1 = nn.MaxPool3d((1, 2, 2))

        self.conv2x = self._make_layer(
            _conv_block, 32, 48, 2, stride=1, reduction=reduction, norm=norm)
        self.maxpool2 = nn.MaxPool3d((2, 2, 2))

        self.conv4x = self._make_layer(
            _conv_block, 48, 64, 2, stride=1, reduction=reduction, norm=norm)
        self.maxpool3 = nn.MaxPool3d((2, 2, 2))

        # upsample
        self.up1 = up_block(in_ch=64, out_ch=64, reduction=reduction, norm=norm)
        self.literal1 = nn.Conv3d(64, 64, 3, padding=1)

        self.up2 = up_block(in_ch=64, out_ch=48, reduction=reduction, norm=norm)
        self.literal2 = nn.Conv3d(48, 48, 3, padding=1)

        self.out_conv_new = nn.Conv3d(48, 9, 1)
        self.out_conv_brain = nn.Conv3d(48, brain_classes, 1)

        # small TODO *********************
        self.conv1x_s = nn.Sequential(
            conv3x3(channel, 32, kernel_size=(first_conv_kernel_z, 3, 3)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            BasicBlock(32, 32, kernel_size=(first_conv_kernel_z, 3, 3), norm=norm)
            )
        self.maxpool1_s = nn.MaxPool3d((1, 2, 2))

        self.conv2x_s = self._make_layer(
            _conv_block, 32, 48, 2, stride=1, reduction=reduction, norm=norm)
        self.maxpool2_s = nn.MaxPool3d((2, 2, 2))

        self.conv4x_s = self._make_layer(
            _conv_block, 48, 64, 2, stride=1, reduction=reduction, norm=norm)
        self.maxpool3_s = nn.MaxPool3d((2, 2, 2))

        # upsample
        self.up1_s = up_block(in_ch=64, out_ch=64, reduction=reduction, norm=norm)
        self.literal1_s = nn.Conv3d(64, 64, 3, padding=1)

        self.up2_s = up_block(in_ch=64, out_ch=48, reduction=reduction, norm=norm)
        self.literal2_s = nn.Conv3d(48, 48, 3, padding=1)

        self.out_conv_new_s = nn.Conv3d(48, 15, 1)



        self.init_weights(init_type=init_type)
    
    
    def forward(self, x, **kargs):
        # down  x: 1 1 32 320 320
        o1 = self.conv1x(x)  # 1 32 32 320 320
        
        o1_pool = self.maxpool1(o1)  # 1 32 32 160 160 
        
        o2 = self.conv2x(o1_pool)  # 1 48 32 160 160 
        
        o2_pool = self.maxpool2(o2)  # 1 48 16 80 80 
        
        o3 = self.conv4x(o2_pool)  # 1 64 16 80 80 
        
        o3_pool = self.maxpool3(o3)  # 1 64 8 40 40

        # up
        out1 = self.up1(o3_pool, self.literal1(o3))  # 1 64 16 80 80
        
        out2 = self.up2(out1, self.literal2(o2))  # 1 48 32 160 160 

        out_conv_new = self.out_conv_new(out2)  # 1 24 32 160 160
        out_up_new = F.upsample(out_conv_new, size=x.shape[2:], mode='trilinear')  # 1 24 32 320 320 
        
        out_conv_brain = self.out_conv_brain(out2)  # 1 2 32 160 160
        out_up_brain = F.upsample(out_conv_brain, size=x.shape[2:], mode='trilinear')  # 1 2 32 320 320

        # small TODO *********
        o1_s = self.conv1x_s(x)  # 1 32 32 320 320
        o1_pool_s = self.maxpool1_s(o1_s)  # 1 32 32 160 160 
        o2_s = self.conv2x_s(o1_pool_s)  # 1 48 32 160 160 
        o2_pool_s = self.maxpool2_s(o2_s)  # 1 48 16 80 80 
        o3_s = self.conv4x_s(o2_pool_s)  # 1 64 16 80 80 
        o3_pool_s = self.maxpool3_s(o3_s)  # 1 64 8 40 40

        # up
        out1_s = self.up1_s(o3_pool_s, self.literal1_s(o3_s))  # 1 64 16 80 80
        
        out2_s = self.up2_s(out1_s, self.literal2_s(o2_s))  # 1 48 32 160 160 

        out_conv_new_s = self.out_conv_new_s(out2_s)  # 1 24 32 160 160
        out_up_new_s = F.upsample(out_conv_new_s, size=x.shape[2:], mode='trilinear')  # 1 24 32 320 320 
       
        out_up_new = torch.cat([out_up_new_s, out_up_new], dim=1)
        return out_up_new, out_up_brain

    def _make_layer(self, block, in_ch, out_ch, num_blocks, stride=1, reduction=2, dilation_rate=1, norm='bn'):
        layers = []
        layers.append(block(in_ch, out_ch, stride=stride,
                            reduction=reduction, dilation_rate=dilation_rate, norm=norm))
        for i in range(num_blocks-1):
            layers.append(block(out_ch, out_ch, stride=1,
                                reduction=reduction, dilation_rate=dilation_rate, norm=norm))
        return nn.Sequential(*layers)

    def init_weights(self, init_type='normal'):
        if init_type == 'normal':
            self.apply(weights_init_normal)
        elif init_type == 'xavier':
            self.apply(weights_init_xavier)
        else:
            raise NotImplementedError(
                'initialization method [%s] is not implemented' % init_type)


class UNet3D_SE(nn.Module):
    def __init__(self,):
        super(UNet3D_SE, self).__init__()

        channel = 1  
        new_num_classes = 24
        brain_classes = 2
        reduction = 2
        norm = 'bn'
        first_conv_kernel_z = 1
        init_type = 'normal'
        

        self.conv1x = nn.Sequential(
            conv3x3(channel, 32, kernel_size=(first_conv_kernel_z, 3, 3)),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            BasicBlock(32, 32, kernel_size=(first_conv_kernel_z, 3, 3), norm=norm)
            )
        self.maxpool1 = nn.MaxPool3d((1, 2, 2))

        self.conv2x = self._make_layer(
            conv_block, 32, 48, 3, stride=1, se=True, reduction=reduction, norm=norm)
        self.maxpool2 = nn.MaxPool3d((2, 2, 2))

        self.conv4x = self._make_layer(
            conv_block, 48, 64, 3, stride=1, se=True, reduction=reduction, norm=norm)
        self.maxpool3 = nn.MaxPool3d((2, 2, 2))
        
        # *********TODO***********
        # self.conv8x = self._make_layer(
        #     conv_block, 128, 128, 3, stride=1, se=True, reduction=reduction, norm=norm)
        # self.maxpool4 = nn.MaxPool3d((2, 2, 2))

        # self.up0 = up_block(in_ch=128, out_ch=128, reduction=reduction, norm=norm)
        # self.literal0 = nn.Conv3d(128, 128, 3, padding=1)

        # self.up0 = up_block(in_ch=128, out_ch=128, reduction=reduction, norm=norm)
        # self.literal1 = nn.Conv3d(128, 128, 3, padding=1)
        # ***********TODO************

        # upsample

        self.up1 = up_block(in_ch=64, out_ch=64, reduction=reduction, norm=norm)
        self.literal1 = nn.Conv3d(64, 64, 3, padding=1)

        self.up2 = up_block(in_ch=64, out_ch=48, reduction=reduction, norm=norm)
        self.literal2 = nn.Conv3d(48, 48, 3, padding=1)

        # self.out_conv = nn.Conv3d(48, self.num_classes, 1, 1)

        self.out_conv_new = nn.Conv3d(48, new_num_classes, 1)
        self.out_conv_brain = nn.Conv3d(48, brain_classes, 1)

        self.init_weights(init_type=init_type)

    def forward(self, x, **kargs):
    
        # down  x: 1 1 32 320 320
        o1 = self.conv1x(x)  # 1 32 32 320 320
        
        o1_pool = self.maxpool1(o1)  # 1 32 32 160 160 
        
        o2 = self.conv2x(o1_pool)  # 1 48 32 160 160 
        
        o2_pool = self.maxpool2(o2)  # 1 48 16 80 80 
        
        o3 = self.conv4x(o2_pool)  # 1 64 16 80 80 
        
        o3_pool = self.maxpool3(o3)  # 1 64 8 40 40
        # print("o3_pool: ", o3_pool.shape)
        # xxx = self.literal1(o3)
        # print("xxx: ", xxx.shape)
        # o4 = self.conv8x(o3_pool)
        # o4_pool = self.maxpool4(o3)

        # out0 = self.up0(o4_pool, self.literal0(o4))
        # up
        out1 = self.up1(o3_pool, self.literal1(o3))  # 1 64 16 80 80
        
        out2 = self.up2(out1, self.literal2(o2))  # 1 48 32 160 160 
        
        # out3 = self.out_conv(out2)
        # out = F.interpolate(out3, size=x.shape[2:], mode='trilinear')

        out_conv_new = self.out_conv_new(out2)  # 1 24 32 160 160
        out_up_new = F.upsample(out_conv_new, size=x.shape[2:], mode='trilinear')  # 1 24 32 320 320 
        
        out_conv_brain = self.out_conv_brain(out2)  # 1 2 32 160 160
        out_up_brain = F.upsample(out_conv_brain, size=x.shape[2:], mode='trilinear')  # 1 2 32 320 320
        

        return out_up_new, out_up_brain

    def _make_layer(self, block, in_ch, out_ch, num_blocks, stride=1, se=True, reduction=2, dilation_rate=1, norm='bn'):
        layers = []
        layers.append(block(in_ch, out_ch, stride=stride, se=se,
                            reduction=reduction, dilation_rate=dilation_rate, norm=norm))
        for i in range(num_blocks-1):
            layers.append(block(out_ch, out_ch, stride=1,
                                reduction=reduction, dilation_rate=dilation_rate, norm=norm))
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
    x_input = torch.rand(1, 1, 32, 320, 320)
    net = UNet3D_2()
    out_1, out_2 = net(x_input)
    print("out_1 shape: ", out_1.shape)
    print("out_2 shape: ", out_2.shape)
    print("Good!!!")