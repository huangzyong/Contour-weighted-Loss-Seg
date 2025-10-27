import torch
from medpy import metric

def calculate_hd95_asd(pred, gt):

    dice = metric.binary.dc(pred, gt)
    # jc = metric.binary.jc(pred, gt)
    # hd95 = metric.binary.hd95(pred, gt)
    # asd = metric.binary.asd(pred, gt)
    # return dice, hd95, asd
    return dice

pt = torch.rand(1, 1, 144, 144, 144)
pt = pt > 0.5
gt = torch.ones(1, 1, 144, 144, 144)
gt = gt[0, 0, ...].cpu().detach().numpy()
pt = pt[0, 0, ...].cpu().detach().numpy()

# hd95, asd = calculate_hd95_asd(pt, gt)
# print("hd95: ", hd95)
# print("asd: ", asd)



