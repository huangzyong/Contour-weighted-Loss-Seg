"""
@author: Zhengyong Huang
code: utf-8
data: 2022.6.7
goal: compute edge loss 
"""
from calendar import c
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import SimpleITK as sitk
import scipy.ndimage as scip

def GetContour_2D(target, type_trans='erode', kernel=3, iter=1, num_cls=2):
    '''
    target   : 1 D H W    ---> 1 32 320 320  tensor
    contours : 1 N D H W  ---> 1 24 32 320 320  tensor
    '''
    
    kernel = np.ones((kernel, kernel), dtype=np.uint8)
    N, D, H, W = target.size()
    
    t_one_hot = target.new_zeros(1, num_cls, D, H, W)  # 1 24 32 320 320
    t_one_hot.scatter_(1, target.view(N, 1, D, H, W), 1.)  # 1 24 32 320 320
    contours = np.zeros((1, num_cls, D, H, W))
    
    for i in range(num_cls):
        np_img = t_one_hot[0, i, :, :, :].squeeze(0).cpu().numpy()
        np_img = (np_img).astype(np.uint8)  # D H W
        for j in range(D):
            img = np_img[j, :, :]  # H W
            
            if type_trans == 'erode':
                erosion = cv2.erode(img, kernel, iterations=iter)
                contour = img - erosion
            elif type_trans == 'dilate':
                dilate = cv2.dilate(img, kernel, iterations=iter)
                contour = dilate - img
            elif type_trans == 'ContainBackground':
                dilate = cv2.dilate(img, kernel, iterations=iter)
                erosion = cv2.erode(img, kernel, iterations=iter)
                contour = dilate - erosion
            else:
                raise ValueError("type_trans error")

            contours[:, i, j, :, :] = contour
    contours = torch.tensor(contours).float()
    return contours

def GetContour_3D(target, type_trans='erode', kernel=3, iter=1, num_cls=2):
    '''
    target(tensor)     : 1 32 320 320
    contours   : 1 24 32 320 320
    '''
    kernel_s = np.ones((kernel, kernel, kernel))
    N, D, H, W = target.size()

    # t_one_hot = target.new_zeros(1, num_cls, D, H, W)  # 1 24 32 320 320
    # t_one_hot.scatter_(1, target.view(N, 1, D, H, W), 1.)  # 1 24 32 320 320
    t_one_hot = target.new_zeros(N, num_cls, D, H, W)  # 1 24 32 320 320
    t_one_hot.scatter_(N, target.view(N, 1, D, H, W), 1.)  # 1 24 32 320 320
    contours = np.zeros((N, num_cls, D, H, W))
    for n in range(N):
        for i in range(num_cls):
            img = t_one_hot[n, i, :, :, :].detach().cpu().numpy()
            img_n = img.astype(np.uint8)
            # print(i, type_trans)
            if type_trans == 'erode':
                erosion = scip.binary_erosion(img_n, structure=kernel_s).astype(img.dtype)
                contour = img_n - erosion

            elif type_trans == 'dilate':
                dilate = scip.binary_dilation(img_n, structure=kernel_s).astype(img.dtype)
                contour = dilate - img_n
            
            elif type_trans == 'ContainBackground':
                dilation = scip.binary_dilation(img_n, structure=kernel_s).astype(img.dtype)
                erosion = scip.binary_erosion(img_n, structure=kernel_s).astype(img.dtype)
                contour = dilation - erosion

            else:
                raise ValueError("type error")
            contours[n, i, :, :, :] = contour
    contours = torch.Tensor(contours) 
    return contours


class BoundaryCeLoss(nn.Module):
    def __init__(self, weight=None):
        super(BoundaryCeLoss, self).__init__()

        self.celoss = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=100, reduction='none')

    def forward(self, inputs, targets, contour_ce):
        '''
        inputs  : 1 24 32 320 320
        targets : 1 32 320 320
        contour_ce : 1 32 320 320
        '''
        contour = contour_ce * 2  # 2
        contour = contour.float()
        contour += 1.0
        loss_ce = self.celoss(inputs, targets) # loss_ce shape: [1, 32, 320, 320]
        loss_wce = contour * loss_ce

        return loss_wce.mean() + loss_ce.mean()
        # return loss_wce.mean()


class BoundaryDiceLoss(nn.Module):
    def __init__(self, weight=None):
        super(BoundaryDiceLoss, self).__init__()
        self.smooth = 0.001
        # self.diceloss = DiceLoss(weight=None)

    def forward(self, inputs, targets, contour_dice):
        '''
        inputs     : 1 24 32 320 320
        targets    : 1 32 320 320 
        contour_dice : 1 24 32 320 320
        '''
        # loss_dice = self.diceloss(inputs, targets)
        
        N, C, D, H, W = inputs.size()  
        prob = F.softmax(inputs, dim=1)  #  1 24 32 320 320

        t_one_hot = inputs.new_zeros(inputs.size())  # 1 24 32 320 320
        t_one_hot.scatter_(1, targets.view(N, 1, D, H, W), 1.)  # 1 24 32 320 320

        contour = contour_dice

        # 去掉背景
        contour = contour[:, 1:, :, :, :]
        prob = prob[:, 1:, :, :, :]
        t_one_hot = t_one_hot[:, 1:, :, :, :]
        
        index_c = torch.where(contour == 1)
        index_noc = torch.where(contour != 1)

        prob_c = prob[index_c]
        prob_noc = prob[index_noc]

        t_one_hot_c = t_one_hot[index_c]
        t_one_hot_noc = t_one_hot[index_noc]

        inter_c = (prob_c * t_one_hot_c).sum(dim=0)
        sum_c = prob_c.sum(dim=0) + t_one_hot_c.sum(dim=0)

        loss_c = 1 - ((2. * inter_c + self.smooth) /
                        (sum_c + self.smooth))
        
        inter_noc = (prob_noc * t_one_hot_noc).sum(dim=0)
        sum_noc = prob_noc.sum(dim=0) + t_one_hot_noc.sum(dim=0)
        loss_noc = 1 - ((2. * inter_noc + self.smooth) /
                        (sum_noc + self.smooth))
        # print(loss_noc.mean().shape)
        # print(loss_noc.mean())
        return loss_c.mean() + loss_noc.mean()


class BoundaryDiceLoss_1(nn.Module):
    def __init__(self, weight=None):
        super(BoundaryDiceLoss_1, self).__init__()
        self.smooth = 1.0

    def forward(self, inputs, targets, contour_dice):
        '''
        inputs     : 1 24 32 320 320
        targets    : 1 32 320 320 
        contour_dice : 1 24 32 320 320
        先获取 label 的膨胀结果(膨胀的成都可以大一些), 再获取contour,将contour 反转,以此作为背景,分别计算前景和背景的diceloss
        '''
        N, C, D, H, W = inputs.size()  
        prob = F.softmax(inputs, dim=1)  #  1 24 32 320 320

        t_one_hot = inputs.new_zeros(inputs.size())  # 1 24 32 320 320
        t_one_hot.scatter_(1, targets.view(N, 1, D, H, W), 1.)  # 1 24 32 320 320

        contour = contour_dice  

        # 去掉背景
        contour = contour[:, 1:, :, :, :]
        prob = prob[:, 1:, :, :, :]
        t_one_hot = t_one_hot[:, 1:, :, :, :]

        
        # index_c = torch.where(contour == 1)  # contour 1 24, 32, 320, 320 
        index_gt = torch.where(t_one_hot == 1)
        index_bg = torch.where(contour == 1)
        
        prob_gt = prob[index_gt]
        prob_bg = prob[index_bg]

        temp = 1 - t_one_hot

        t_one_hot_gt = t_one_hot[index_gt]
        t_one_hot_bg = temp[index_bg]
        # print("contour: ", contour.sum())
        # print("t_one_hot_bg: ", t_one_hot_bg.sum())

        inter_gt = (prob_gt * t_one_hot_gt).sum(dim=0)
        sum_gt = prob_gt.sum(dim=0) + t_one_hot_gt.sum(dim=0)

        loss_gt = 1 - ((2. * inter_gt + self.smooth) /
                        (sum_gt + self.smooth))
        
        inter_bg = (prob_bg * t_one_hot_bg).sum(dim=0)
        sum_bg = prob_bg.sum(dim=0) + t_one_hot_bg.sum(dim=0)
        loss_bg = ((2. * inter_bg + self.smooth) /
                        (sum_bg + self.smooth))
        # print("prob_bg: ", prob_bg.sum())
        # print("inter_bg: ", inter_bg.sum())
        # print("sum_bg: ", sum_bg.sum())
        # print("loss_bg: ", loss_bg.mean())
        return loss_gt.mean(), 10 * loss_bg.mean()


class NewDiceLoss(nn.Module):
    def __init__(self, weight=None):
        super(NewDiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, inputs, targets):
        '''
        inputs     : 1 24 32 320 320
        targets    : 1 32 320 320 
        contour_dice : 1 24 32 320 320
        '''
        N, C, D, H, W = inputs.size()  
        prob = F.softmax(inputs, dim=1)  #  1 24 32 320 320

        t_one_hot = inputs.new_zeros(inputs.size())  # 1 24 32 320 320
        t_one_hot.scatter_(1, targets.view(N, 1, D, H, W), 1.)  # 1 24 32 320 320


        # 去掉背景
        prob = prob[:, 1:, :, :, :]
        t_one_hot = t_one_hot[:, 1:, :, :, :]
        
        # contour_temp = torch.zeros_like(contour)
        # contour = contour_temp

        
        index_new = torch.where(t_one_hot == 1)  # contour 1 24, 32, 320, 320 

        prob_new = prob[index_new]

        t_one_hot_new = t_one_hot[index_new]

        inter_new = (prob_new * t_one_hot_new).sum(dim=0)
        sum_new = prob_new.sum(dim=0) + t_one_hot_new.sum(dim=0)

        loss_new = 1 - ((2. * inter_new + self.smooth) /
                        (sum_new + self.smooth))

        return loss_new.mean()


class DiceLoss(nn.Module):

    def __init__(self, weight=None):
        super(DiceLoss, self).__init__()
        self.smooth = 0.001
        self.weight = weight

        if self.weight is not None:
            self.weight = self.weight[1:]
            self.weight = self.weight / self.weight.sum()

    def forward(self, inputs, targets):

        N, C, D, H, W = inputs.size()  
        prob = F.softmax(inputs, dim=1)
        
        t_one_hot = inputs.new_zeros(inputs.size())
        t_one_hot.scatter_(1, targets.view(N, 1, D, H, W), 1.)
        

        # ignore bg
        prob = prob[:, 1:, :, :, :]
        t_one_hot = t_one_hot[:, 1:, :, :, :]
       

        if self.weight is None:
            iflat = prob.view(-1)
            tflat = t_one_hot.view(-1)
            intersection = (iflat * tflat).sum()

            return 1 - ((2. * intersection + self.smooth) /
                        (iflat.sum() + tflat.sum() + self.smooth))
        else:
            prob = prob.permute(0, 2, 3, 4, 1).contiguous().view(-1, C-1)
            t_one_hot = t_one_hot.permute(0, 2, 3, 4, 1).contiguous().view(-1, C-1)
            intersection = (prob * t_one_hot).sum(dim=0)
            summ = prob.sum(dim=0) + t_one_hot.sum(dim=0)

            loss = 1 - ((2. * intersection + self.smooth) /
                        (summ + self.smooth))

            weight = self.weight.type_as(prob)
            loss *= weight
            return loss.mean()


class Dual_DiceLoss(nn.Module):
    def __init__(self, weight=None):
        super(Dual_DiceLoss, self).__init__()
        self.smooth = 0.001

    def forward(self, inputs, targets):
        '''
        inputs     : 1 24 32 320 320
        targets    : 1 32 320 320 
        contour_dice : 1 24 32 320 320
        '''
        N, C, D, H, W = inputs.size()  
        prob = F.softmax(inputs, dim=1)  #  1 24 32 320 320

        t_one_hot = inputs.new_zeros(inputs.size())  # 1 24 32 320 320
        t_one_hot.scatter_(1, targets.view(N, 1, D, H, W), 1.)  # 1 24 32 320 320
        # print("t_one_hot max: ", t_one_hot.max())
        # print("t_one_hot sum: ", t_one_hot.sum())
        # 去掉背景
        prob = prob[:, 1:, :, :, :]
        t_one_hot = t_one_hot[:, 1:, :, :, :]
        
        index_gt = torch.where(t_one_hot == 1)  # contour 1 24, 32, 320, 320 
        index_bg = torch.where(t_one_hot == 0)  # contour 1 24, 32, 320, 320 

        temp = 1 - t_one_hot
        sum_volume = temp.sum()
        # print("bg_volume: ", sum_volume.item())

        prob_gt = prob[index_gt]
        prob_bg = prob[index_bg]

        t_one_hot_gt = t_one_hot[index_gt]
        t_one_hot_bg = t_one_hot[index_bg]
        # print("t_one_hot_gt sum: ", t_one_hot_gt.sum().item())
        # print("t_one_hot_bg sum: ", t_one_hot_bg.sum(dim=0).item())

        inter_gt = (prob_gt * t_one_hot_gt).sum(dim=0)
        inter_bg = (prob_bg * t_one_hot_bg).sum(dim=0)
        # print("inter_gt sum: ", inter_gt.sum().item())
        # print("inter_bg sum: ", inter_bg.sum().item())

        sum_gt = prob_gt.sum(dim=0) + t_one_hot_gt.sum(dim=0)
        sum_bg = prob_bg.sum(dim=0) + t_one_hot_bg.sum(dim=0)
        # print("sum_bg: ", sum_bg.item())

        loss_gt = 1 - ((2. * inter_gt + self.smooth) /
                        (sum_gt + self.smooth))

        # loss_bg = 1 - ((2.* inter_bg + self.smooth) / 
        #                 (sum_bg + self.smooth))
        loss_bg = sum_bg / sum_volume
        
        # print("sum_bg: ", sum_bg.item())
        # print("loss_bg: ", loss_bg.mean())
        # print("loss_gt: ", loss_gt.mean())

        return loss_gt.mean(), loss_bg.mean()


class Dual_DiceLoss_1(nn.Module):
    def __init__(self, weight=None):
        super(Dual_DiceLoss_1, self).__init__()
        self.smooth = 0.001

    def forward(self, inputs, targets):
        '''
        loss分为两个部分, 前景部分loss和背景部分loss,其中前景部分使用diceloss; 背景部分的loss的计算方式为先将背景和前景反转,再计算背景的dice,并以此作为loss
        inputs     : 1 24 32 320 320
        targets    : 1 32 320 320 
        contour_dice : 1 24 32 320 320
        '''
        N, C, D, H, W = inputs.size()  
        prob = F.softmax(inputs, dim=1)  #  1 24 32 320 320

        t_one_hot = inputs.new_zeros(inputs.size())  # 1 24 32 320 320
        t_one_hot.scatter_(1, targets.view(N, 1, D, H, W), 1.)  # 1 24 32 320 320
        # print("t_one_hot max: ", t_one_hot.max())
        # print("t_one_hot sum: ", t_one_hot.sum())
        # 去掉背景
        prob = prob[:, 1:, :, :, :]
        t_one_hot = t_one_hot[:, 1:, :, :, :]
        
        index_gt = torch.where(t_one_hot == 1)  # contour 1 24, 32, 320, 320  

        temp_gt = 1 - t_one_hot  # 反转  背景，前景交换
        index_bg = torch.where(temp_gt == 1)
        # print("bg_volume: ", sum_volume.item())

        prob_gt = prob[index_gt]
        prob_bg = prob[index_bg]

        t_one_hot_gt = t_one_hot[index_gt]
        t_one_hot_bg = temp_gt[index_bg]
        # print("t_one_hot_gt sum: ", t_one_hot_gt.sum().item())
        # print("t_one_hot_bg sum: ", t_one_hot_bg.sum(dim=0).item())

        inter_gt = (prob_gt * t_one_hot_gt).sum(dim=0)
        inter_bg = (prob_bg * t_one_hot_bg).sum(dim=0)
        # print("inter_gt sum: ", inter_gt.sum().item())
        # print("inter_bg sum: ", inter_bg.sum().item())

        sum_gt = prob_gt.sum(dim=0) + t_one_hot_gt.sum(dim=0)
        sum_bg = prob_bg.sum(dim=0) + t_one_hot_bg.sum(dim=0)
        

        loss_gt = 1 - ((2. * inter_gt + self.smooth) /
                        (sum_gt + self.smooth))

        loss_bg = ((2.* inter_bg + self.smooth) / 
                        (sum_bg + self.smooth))
        # loss_bg = sum_bg / sum_volume
        
        # print("sum_bg: ", sum_bg.item())
        # print("loss_bg: ", loss_bg.mean())
        # print("loss_gt: ", loss_gt.mean())

        return loss_gt.mean(), 10*loss_bg.mean()


class Dual_DiceLoss_2(nn.Module):
    def __init__(self, weight=None):
        super(Dual_DiceLoss_2, self).__init__()
        self.smooth = 0.001

    def forward(self, inputs, targets):
        '''
        计算前景和背景的loss,其中前景使用diceloss. 背景计算方式为
        inputs     : 1 24 32 320 320
        targets    : 1 32 320 320 
        contour_dice : 1 24 32 320 320
        '''
        N, C, D, H, W = inputs.size()  
        prob = F.softmax(inputs, dim=1)  #  1 24 32 320 320

        t_one_hot = inputs.new_zeros(inputs.size())  # 1 24 32 320 320
        t_one_hot.scatter_(1, targets.view(N, 1, D, H, W), 1.)  # 1 24 32 320 320
        # print("t_one_hot max: ", t_one_hot.max())
        # print("t_one_hot sum: ", t_one_hot.sum())
        # 去掉背景
        prob = prob[:, 1:, :, :, :]
        t_one_hot = t_one_hot[:, 1:, :, :, :]
        
        index_gt = torch.where(t_one_hot == 1)  # contour 1 24, 32, 320, 320  

        temp_gt = 1 - t_one_hot  # 反转  背景，前景交换
        index_bg = torch.where(temp_gt == 1)
        # print("bg_volume: ", sum_volume.item())

        prob_gt = prob[index_gt]
        prob_bg = prob[index_bg]

        t_one_hot_gt = t_one_hot[index_gt]
        t_one_hot_bg = temp_gt[index_bg]
        print("t_one_hot_gt sum: ", t_one_hot_gt.sum().item())
        print("t_one_hot_bg sum: ", t_one_hot_bg.sum(dim=0).item())

        inter_gt = (prob_gt * t_one_hot_gt).sum(dim=0)
        inter_bg = (prob_bg * t_one_hot_bg).sum(dim=0)
        print("inter_gt sum: ", inter_gt.sum().item())
        print("inter_bg sum: ", inter_bg.sum().item())

        sum_gt = prob_gt.sum(dim=0) + t_one_hot_gt.sum(dim=0)
        sum_bg = prob_bg.sum(dim=0) + t_one_hot_bg.sum(dim=0)
        

        loss_gt = 1 - ((2. * inter_gt + self.smooth) /
                        (sum_gt + self.smooth))

        loss_bg = ((2.* inter_bg + self.smooth) / 
                        (sum_bg + self.smooth))
        # loss_bg = sum_bg / sum_volume
        
        print("sum_bg: ", sum_bg.item())
        print("loss_bg: ", loss_bg.mean())
        print("loss_gt: ", loss_gt.mean())

        return loss_gt.mean(), 10*loss_bg.mean()


if __name__ == "__main__":
    inputs = torch.rand(1, 24, 32, 32, 32)
    target = torch.rand(1, 32, 32, 32)
    contour_dice = torch.rand(1, 24, 32, 32, 32)
    contour_ce = torch.rand(1, 32, 32, 32)
    target = target > 0.5
    target = target.long()

    organ_weight = np.array([0.5, 1, 2, 2, 1, 8, 8, 8, 8, 2, 2, 8, 8, 1, 2, 2, 1, 2, 2, 2, 8, 8, 8, 8])
    organ_weight = torch.from_numpy(organ_weight).float().unsqueeze(1)
    
    BDL = BoundaryDiceLoss()
    loss_c, loss_noc, dice_c, dice_noc = BDL(inputs, target, contour_dice)
    print("EDL: ", loss_c, loss_noc)
    print(np.mean(dice_c))
    print(np.mean(dice_noc))
    print(len(dice_c))
    DL = DiceLoss()
    loss_dice = DL(inputs, target)
    print("DL: ", loss_dice)

    EL = BoundaryCeLoss(weight=organ_weight)
    loss = EL(inputs, target, contour_ce)
    print("EL: ", loss)
    



