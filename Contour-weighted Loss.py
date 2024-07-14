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

    t_one_hot = target.new_zeros(N, num_cls, D, H, W)  # 1 24 32 320 320
    t_one_hot.scatter_(N, target.view(N, 1, D, H, W), 1.)  # 1 24 32 320 320
    contours = np.zeros((N, num_cls, D, H, W))
    for n in range(N):
        for i in range(num_cls):
            img = t_one_hot[n, i, :, :, :].detach().cpu().numpy()
            img_n = img.astype(np.uint8)
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



if __name__ == "__main__":
    inputs = torch.rand(1, 24, 32, 32, 32)
    target = torch.rand(1, 32, 32, 32)
    contour_dice = torch.rand(1, 24, 32, 32, 32)
    contour_ce = torch.rand(1, 32, 32, 32)
    target = target > 0.5
    target = target.long()

    BDL = BoundaryDiceLoss()
    loss_c, loss_noc = BDL(inputs, target, contour_dice)
    print("BDL: ", loss_c, loss_noc)





