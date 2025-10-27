import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pdb
import numpy as np


class GDL(nn.Module):
    """
    Computes Generalized Dice Loss (GDL) as described in https://arxiv.org/pdf/1707.03237.pdf
    """

    def __init__(self, weight=None):
        super(GDL, self).__init__()
        self.epsilon = 1e-5
        self.weight = weight


    def forward(self, inputs, target):
        # get probabilities from logits
        '''
        inouts : 1 24 32 320 320
        target : 1 32 320 320
        '''
        N, C, D, H, W = inputs.size()
        inputs = F.softmax(inputs, dim=1)
        t_one_hot = inputs.new_zeros(inputs.size())
        t_one_hot.scatter_(1, target.view(N, 1, D, H, W), 1.)
        
        target = t_one_hot
        
        inputs = inputs[:, 1:, :, :, :]  # 1 23 32 320 320
        target = target[:, 1:, :, :, :]  # 1 23 32 320 320

        inputs = inputs.view(C-1, -1)  # C-1 N*H*D*W
        target = target.view(C-1, -1)
        target = target.float()
        target_sum = target.sum(-1)  # C-1 1
        # print('target_sum: ', target_sum.shape)

        class_weights = 1. / ((target_sum * target_sum) + 0.001)
        # print("class weights: ", class_weights)
        # class_weights = class_weights / class_weights.sum()
        
        intersect = (inputs * target).sum(-1) * class_weights
        # print("intersect: ", intersect)
        if self.weight is not None:
            intersect = self.weight * intersect
        intersect = intersect.sum()
        
        denominator = ((inputs + target).sum(-1) * class_weights).sum()
        # print("denominator: ", denominator)
        
        GDL_loss = 1. - 2. * (intersect + self.epsilon) / (denominator + self.epsilon)
        
        return GDL_loss.mean()
    
def generalized_dice_loss(pred, target):
    """compute the weighted dice_loss
    Args:
        pred (tensor): prediction after softmax, shape(bath_size, channels, height, width)
        target (tensor): gt, shape(bath_size, channels, height, width)
    Returns:
        gldice_loss: loss value
    """    
    N, C, D, H, W = pred.size()
    inputs = F.softmax(pred, dim=1)
    t_one_hot = inputs.new_zeros(inputs.size())
    t_one_hot.scatter_(1, target.view(N, 1, D, H, W), 1.)
    
    target = t_one_hot
    
    pred = inputs[:, 1:, :, :, :]  # 1 23 32 320 320
    target = target[:, 1:, :, :, :]  # 1 23 32 320 320
    print("target: ", target.shape)
    
    wei = torch.sum(target, axis=[0,2,3,4]) # (n_class,)
    print('wei: ', wei)
    wei = 1/(wei**2+1e-5)
    intersection = torch.sum(wei*torch.sum(pred * target, axis=[0,2,3, 4]))
    union = torch.sum(wei*torch.sum(pred + target, axis=[0,2,3,4]))
    gldice_loss = 1 - (2. * intersection) / (union + 1e-5)
    return gldice_loss.mean()

class GDiceLoss(nn.Module):

    def __init__(self, weight=None):
        super(GDiceLoss, self).__init__()
        self.smooth = 0.001
        self.weight = weight

        if self.weight is not None:
            self.weight = self.weight[1:]
            self.weight = self.weight / self.weight.sum()

    def forward(self, inputs, targets):
        '''
        inouts : 1 24 32 320 320
        target : 1 32 320 320
        '''
        N, C, D, H, W = inputs.size()
        prob = F.softmax(inputs, dim=1)
        t_one_hot = inputs.new_zeros(inputs.size())
        t_one_hot.scatter_(1, targets.view(N, 1, D, H, W), 1.)

        # ignore bg
        prob = prob[:, 1:, :, :, :]  # 1 23 32 320 320
        t_one_hot = t_one_hot[:, 1:, :, :, :]  # 1 23 32 320 320
        print('t_one_hot: ', t_one_hot.shape)
        # prob = prob.permute(0, 2, 3, 4, 1).contiguous().view(-1, C-1)
        # t_one_hot = t_one_hot.permute(
        #     0, 2, 3, 4, 1).contiguous().view(-1, C-1)
        
        target_sum = t_one_hot.sum(dim=(0,2,3,4))  # C-1 1
        print('target_sum: ', target_sum)

        class_weights = 1. / ((target_sum * target_sum) + 0.001)
        print("class weights: ", class_weights)
        
        intersection = (prob * t_one_hot).sum(dim=0)
        summ = prob.sum(dim=0) + t_one_hot.sum(dim=0)

        loss = 1 - ((2. * intersection + self.smooth) /
                    (summ + self.smooth))

        # weight = self.weight.type_as(prob)  # 23
        # loss *= weight
        return loss.mean() 

if __name__ == '__main__':
    score = torch.zeros(1, 4, 32, 120, 120)
    target = torch.rand(1, 32, 120, 120)
    target = target > 0.5
    target = target.long()

    DL = GDiceLoss()
    loss = DL(score, target)
    # loss_gdl = generalized_dice_loss(score, target)
    print('dice_loss  %.8f' % loss)
    # print('gdl  %.8f' % loss_gdl)
