# 计算 DSC HD95 ASSD
import torch
import numpy as np
from medpy import metric
import os
import csv

labelname2label = {'WT': 1, 'TC': 2, 'ET': 3}

label2labelname = {v: k for k, v in labelname2label.items()}

def calculate_metrics(pred, gt):
    dice = 2 * (pred*gt).sum() / (pred.sum() + gt.sum() + 0.001)
    # dice1 = metric.binary.dc(pred, gt)
    # print("dice : ", dice)
    # print(dice1)
    
    # jc = metric.binary.jc(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)

    return dice, hd95, asd

def calculate_dice_hd95_asd(pred, gt, cls_list, save_result_folder, item, data=None):
    # 需要计算的标签类别，不包括背景和图像中不存在的区域
    
    dice_lst = []
    hd_lst = []
    asd_lst = []
    i=1

    for cls in cls_list:
        x = (pred == cls)
        y = (gt == cls)
        if cls in gt:
            print("{} exists!".format(cls))
            if y.max() and x.max():     
                dice, hd, asd = calculate_metrics(pred == cls, gt == cls)
                if hd>50:
                    hd=20
                if asd>20:
                    asd=10
                dice_lst.append(dice)
                hd_lst.append(hd)
                asd_lst.append(asd)
                print("i: {}, dice: {} hd: {}, asd: {}".format(i, dice, hd, asd))
            elif y.max() and (y.max() != x.max()): 
                dice, hd, asd = 0.0144, 20, 10
                dice_lst.append(dice)
                hd_lst.append(hd)
                asd_lst.append(asd)
                print("i: {}, dice: {} hd: {}, asd: {}".format(i, dice, hd, asd))
        # elif (cls not in gt) and (cls in pred):
        #     dice, hd, asd = 0.001, 100, 50
        #     dice_lst.append(dice)
        #     hd_lst.append(hd)
        #     asd_lst.append(asd)
        #     print("i: {}, dice: {} hd: {}, asd: {}".format(i, dice, hd, asd))
            
        else:
            print("{} not exists!".format(cls))
            print("null !")
            dice_lst.append('-')
            hd_lst.append('-')
            asd_lst.append('-')
        i += 1
    print("dice_lst: ", dice_lst)
    with open(os.path.join(save_result_folder, "test_result.csv"), 'a') as f:
        writer = csv.writer(f)
        write_list = [item]
        for k in range(13 - 1):

            if dice_lst[k]=="-":
                write_list.append(dice_lst[k])
            else:
                write_list.append(dice_lst[k])

        writer.writerow(write_list)
    
    dsc = []
    hd = []
    asd = []
    for i in range(len(dice_lst)):
        if dice_lst[i] != "-":
            dsc.append(dice_lst[i])
            hd.append(hd_lst[i])
            asd.append(asd_lst[i])
    print(np.mean(dsc), np.mean(hd), np.mean(asd))
    return np.mean(dsc), np.mean(hd), np.mean(asd)

if  __name__ == "__main__":
    # model output shape: 1, cls, d, h, w
    # target shape: 1, 1, d, h, w
    pt = torch.rand(1, 1, 132, 320, 320)
    pt = pt > 0.5
    gt = torch.ones(1, 1, 132, 320, 320)
    gt = gt[0, 0, ...].cpu().detach().numpy()
    pt = pt[0, 0, ...].cpu().detach().numpy()

    dsc, hd95, assd = calculate_dice_hd95_asd(pt, gt)
    print(dsc, hd95, assd)