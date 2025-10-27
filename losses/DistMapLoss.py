import torch
from torch import nn
from scipy.ndimage import distance_transform_edt
import numpy as np


def softmax_helper(x):
    # copy from: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/utilities/nd_softmax.py
    rpt = [1 for _ in range(len(x.size()))]
    rpt[1] = x.size(1)
    x_max = x.max(1, keepdim=True)[0].repeat(*rpt)
    e_x = torch.exp(x - x_max)
    return e_x / e_x.sum(1, keepdim=True).repeat(*rpt)

def compute_edts_forPenalizedLoss(GT):
    """
    GT.shape = (batch_size, x,y,z)
    only for binary segmentation
    """
    res = np.zeros(GT.shape)
    for i in range(GT.shape[0]):
        posmask = GT[i]
        negmask = ~posmask
        pos_edt = distance_transform_edt(posmask)
        pos_edt = (np.max(pos_edt)-pos_edt)*posmask 
        neg_edt =  distance_transform_edt(negmask)
        neg_edt = (np.max(neg_edt)-neg_edt)*negmask
        
        res[i] = pos_edt/(np.max(pos_edt) + neg_edt/np.max(neg_edt) + 1e-4)
    return res

class DistBinaryDiceLoss(nn.Module):
    """
    Distance map penalized Dice loss
    Motivated by: https://openreview.net/forum?id=B1eIcvS45V
    Distance Map Loss Penalty Term for Semantic Segmentation        
    """
    def __init__(self, smooth=1e-5):
        super(DistBinaryDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, net_output, gt):
        """
        net_output: (batch_size, 2, x,y,z)
        target: ground truth, shape: (batch_size, 1, x,y,z)
        """
        net_output = softmax_helper(net_output)
        # one hot code for gt
        with torch.no_grad():
            if len(net_output.shape) != len(gt.shape):
                gt = gt.view((gt.shape[0], 1, *gt.shape[1:]))

            if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = gt
            else:
                gt = gt.long()
                y_onehot = torch.zeros(net_output.shape)
                if net_output.device.type == "cuda":
                    y_onehot = y_onehot.cuda(net_output.device.index)
                y_onehot.scatter_(1, gt, 1)
        
        gt_temp = gt[:,0, ...].type(torch.float32)
        with torch.no_grad():
            dist = compute_edts_forPenalizedLoss(gt_temp.cpu().numpy()>0.5) + 1.0
        # print('dist.shape: ', dist.shape)
        dist = torch.from_numpy(dist)

        if dist.device != net_output.device:
            dist = dist.to(net_output.device).type(torch.float32)
 
        tp = net_output * y_onehot
        tp = torch.sum(tp[:,1,...] * dist, (1,2,3))
        dc = (2 * tp + self.smooth) / (torch.sum(net_output[:,1,...], (1,2,3)) + torch.sum(y_onehot[:,1,...], (1,2,3)) + self.smooth)
        dc = dc.mean()

        return 1 - dc

class DistBinaryCeLoss(nn.Module):
    """
    Distance map penalized Dice loss
    Motivated by: https://openreview.net/forum?id=B1eIcvS45V
    Distance Map Loss Penalty Term for Semantic Segmentation        
    """
    def __init__(self, smooth=1e-5):
        super(DistBinaryCeLoss, self).__init__()
        self.smooth = smooth
        self.ce = torch.nn.CrossEntropyLoss(weight=None, ignore_index=100)

    def forward(self, net_output, gt):
        """
        net_output: (batch_size, 2, x,y,z)
        target: ground truth, shape: (batch_size, 1, x,y,z)
        """
        net_output = softmax_helper(net_output)
        # one hot code for gt
        with torch.no_grad():
            if len(net_output.shape) != len(gt.shape):
                gt = gt.view((gt.shape[0], 1, *gt.shape[1:]))

            if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = gt
            else:
                gt = gt.long()
                y_onehot = torch.zeros(net_output.shape)
                if net_output.device.type == "cuda":
                    y_onehot = y_onehot.cuda(net_output.device.index)
                y_onehot.scatter_(1, gt, 1)
        
        gt_temp = gt[:,0, ...].type(torch.float32)
        with torch.no_grad():
            dist = compute_edts_forPenalizedLoss(gt_temp.cpu().numpy()>0.5) + 1.0
        # print('dist.shape: ', dist.shape)
        dist = torch.from_numpy(dist)

        if dist.device != net_output.device:
            dist = dist.to(net_output.device).type(torch.float32)
 
        tp = net_output * y_onehot
        ce = -(y_onehot*torch.log(net_output) + (1-y_onehot)*torch.log(1-net_output))
        return ce.mean()
    
class DWCE(nn.Module):
    def __init__(self, ):
        super(DWCE, self).__init__()
        self.dist = DistBinaryCeLoss()
        
    def forward(self, inputs, target):
        '''
        inputs: b, n_cls, d, h, w
        target: b, d, h, w
        '''
        N, C, D, H, W = inputs.size()
        t_one_hot = inputs.new_zeros(inputs.size())
        t_one_hot.scatter_(1, target.view(N, 1, D, H, W), 1.)
        loss = []
        for i in range(1, C):
            in_x = torch.cat([inputs[:, 0, :, :, :].unsqueeze(1), inputs[:, i, :, :, :].unsqueeze(1)], dim=1)
            tar = t_one_hot[:, i, :, :, :].unsqueeze(1)
            # print('in_x: ', in_x.shape)
            # print('tar: ', tar.shape)
            dist_loss = self.dist(in_x, tar)
            loss.append(dist_loss)
            
        return sum(loss) / len(loss)

class DistMapLoss(nn.Module):
    def __init__(self, ):
        super(DistMapLoss, self).__init__()
        self.dist = DistBinaryDiceLoss()
        
    def forward(self, inputs, target):
        '''
        inputs: b, n_cls, d, h, w
        target: b, d, h, w
        '''
        N, C, D, H, W = inputs.size()
        t_one_hot = inputs.new_zeros(inputs.size())
        t_one_hot.scatter_(1, target.view(N, 1, D, H, W), 1.)
        loss = []
        for i in range(1, C):
            in_x = torch.cat([inputs[:, 0, :, :, :].unsqueeze(1), inputs[:, i, :, :, :].unsqueeze(1)], dim=1)
            tar = t_one_hot[:, i, :, :, :].unsqueeze(1)
            # print('in_x: ', in_x.shape)
            # print('tar: ', tar.shape)
            dist_loss = self.dist(in_x, tar)
            loss.append(dist_loss)
            
        return sum(loss) / len(loss)
        
        
        
    
if __name__ == "__main__":
    x_input = torch.zeros(2, 4, 64, 240, 240)
    target = torch.rand(2, 64, 240, 240)
    target = target > 0.5
    target = target.long()
    
    net = DistMapLoss()
    loss = net(x_input, target)
    print("loss: ", loss)

    print("Good!!!")