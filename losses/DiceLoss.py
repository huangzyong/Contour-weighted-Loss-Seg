import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pdb
import numpy as np
  

class DiceLoss(nn.Module):

    def __init__(self, weight=None):
        super(DiceLoss, self).__init__()
        self.smooth = 1e-3
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

        if self.weight is None:
            iflat = prob.view(-1)
            tflat = t_one_hot.view(-1)
            intersection = (iflat * tflat).sum()

            return 1 - ((2. * intersection + self.smooth) /
                        (iflat.sum() + tflat.sum() + self.smooth))
        else:
            prob = prob.permute(0, 2, 3, 4, 1).contiguous().view(-1, C-1)
            t_one_hot = t_one_hot.permute(
                0, 2, 3, 4, 1).contiguous().view(-1, C-1)
            intersection = (prob * t_one_hot).sum(dim=0)
            summ = prob.sum(dim=0) + t_one_hot.sum(dim=0)

            loss = 1 - ((2. * intersection + self.smooth) /
                        (summ + self.smooth))

            weight = self.weight.type_as(prob)  # 23
            loss *= weight
            return loss.mean() 

# 支持多分类和二分类
class FocalLoss(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)^gamma*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """
 
    def __init__(self, num_class, alpha=None, gamma=2, balance_index=-1, smooth=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.size_average = size_average
 
        if self.alpha is None:
            self.alpha = torch.ones(self.num_class, 1)
        elif isinstance(self.alpha, (list, np.ndarray)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.FloatTensor(alpha).view(self.num_class, 1)
            self.alpha = self.alpha / self.alpha.sum()
        elif isinstance(self.alpha, float):
            alpha = torch.ones(self.num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[balance_index] = self.alpha
            self.alpha = alpha
        else:
            raise TypeError('Not support alpha type')
 
        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')
 
    def forward(self, input, target):
        logit = F.softmax(input, dim=1)
 
        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = target.view(-1, 1)
 
        # N = input.size(0)
        # alpha = torch.ones(N, self.num_class)
        # alpha = alpha * (1 - self.alpha)
        # alpha = alpha.scatter_(1, target.long(), self.alpha)
        epsilon = 1e-10
        alpha = self.alpha
        if alpha.device != input.device:
            alpha = alpha.to(input.device)
 
        idx = target.cpu().long()
        one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)
 
        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth, 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + epsilon
        logpt = pt.log()
 
        gamma = self.gamma
 
        alpha = alpha[idx]
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt
 
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss
    
    
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

        class_weights = 1. / ((target_sum * target_sum) + 1)
        
        intersect = (inputs * target).sum(-1) * class_weights
        if self.weight is not None:
            intersect = self.weight * intersect
        intersect = intersect.sum()
        
        denominator = ((inputs + target).sum(-1) * class_weights).sum()
        
        GDL_loss = 1. - 2. * (intersect + self.epsilon) / (denominator + self.epsilon)
        
        return GDL_loss

if __name__ == '__main__':
    score = torch.rand(1, 16, 32, 320, 320)
    target = torch.rand(1, 32, 320, 320)
    # target = target > 0.5
    target = target.long()

    weights_19 = torch.FloatTensor(
        [0.5, 1, 1, 1, 8, 8, 5, 5, 5, 1, 8, 1, 1, 8, 5, 5, 5, 1, 1])
    organ_weight = np.array([0.5, 1, 2, 2, 1, 8, 8, 8, 8, 2, 2, 8, 8, 1, 2, 2, 1, 2, 2, 2, 8, 8, 8, 8])
    organ_weight = torch.from_numpy(organ_weight).float().unsqueeze(1)
    DL = DiceLoss()
    FL = FocalLoss(num_class=16, alpha=0.5)
    loss = DL(score, target)
    loss_focal = FL(score, target)
    print('dice_loss  %.8f' % loss)
    print('focal_loss  %.8f' % loss_focal)