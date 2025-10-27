import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from utils.util import *
import pdb

from tqdm import tqdm

def cal_dice_multi_class(pred, target, C):
    """
        Args:
            pred: (N, C)
            target: (N, C)
    """
    intersection = pred.cpu() * target.cpu()
    summ = pred.cpu() + target.cpu()
    intersection = intersection.sum(0).type(torch.float64)
    summ = summ.sum(0).type(torch.float64)

    eps = torch.rand(C, dtype=torch.float64).fill_(0.0000001)
    summ += eps
    dice = 2 * intersection / summ

    return dice, intersection, summ


def cal_dice(pred, target, C, data=None):
    """
        Args:
            pred: (N, )
            target: (N, 1)
    """
    N = pred.size(0)
    target_mask = target.data.new(N, C).fill_(0)
    target_mask.scatter_(1, target, 1.)
    pred_mask = pred.data.new(N, C).fill_(0)
    pred_mask.scatter_(1, pred.unsqueeze(1), 1.)
    
    intersection = pred_mask.cpu() * target_mask.cpu()
    summ = pred_mask.cpu() + target_mask.cpu()
    intersection = intersection.sum(0).type(torch.float64)
    summ = summ.sum(0).type(torch.float64)

    eps = torch.rand(C, dtype=torch.float64).fill_(0.0000001)
    summ += eps
    dice = 2 * intersection / summ

    return dice, intersection, summ

def validate_with_contour(net, dataloader, num_classes, max_len=None):
    total_intersection = torch.zeros((num_classes,)).double()
    total_summ = torch.zeros((num_classes,)).double()

    with torch.no_grad():
        for i, (data, label, contour, _) in enumerate(dataloader):
            inputs = data.cuda()
            pred = net(inputs)
            if isinstance(pred, list):
                pred = torch.cat(pred, dim=0)
            label_pred = pred.permute(0, 2, 3, 4, 1).contiguous().view(-1, num_classes)
            _, label_pred = torch.max(F.softmax(label_pred, dim=1), 1)
            label_true = label.view(-1, 1)

            dice, intersection, summ = cal_dice(label_pred, label_true, num_classes)
            #pdb.set_trace()
            total_intersection += intersection
            total_summ += summ
            # val on part of trainset
            if max_len is not None:
                if i > max_len:
                    break

    total_dice = 2 * total_intersection / total_summ
    avg_dice = total_dice[1:num_classes].mean()

    return total_dice, avg_dice


def validate(net, dataloader, num_classes, max_len=None):
    total_intersection = torch.zeros((num_classes,)).double()
    total_summ = torch.zeros((num_classes,)).double()

    with torch.no_grad():
        for i, (data, label, _) in enumerate(dataloader):
            inputs = data.cuda()
            pred = net(inputs)
            if isinstance(pred, list):
                pred = torch.cat(pred, dim=0)
            label_pred = pred.permute(0, 2, 3, 4, 1).contiguous().view(-1, num_classes)
            _, label_pred = torch.max(F.softmax(label_pred, dim=1), 1)
            label_true = label.view(-1, 1)

            dice, intersection, summ = cal_dice(label_pred, label_true, num_classes)
            #pdb.set_trace()
            total_intersection += intersection
            total_summ += summ
            # val on part of trainset
            if max_len is not None:
                if i > max_len:
                    break

    total_dice = 2 * total_intersection / total_summ
    avg_dice = total_dice[1:num_classes].mean()

    return total_dice, avg_dice

def validate_new(net, dataloader, num_classes, device, max_len=None):
    total_intersection = torch.zeros((num_classes,)).double()
    total_summ = torch.zeros((num_classes,)).double()

    with torch.no_grad():
        dataiter = iter(dataloader)
        niters = len(dataloader)
        if max_len is not None:
            pbar = tqdm(total=max_len)
        else:
            pbar = tqdm(total=niters)
        # for i, (data, label, _) in enumerate(dataloader):
        for i in range(niters):
            data, label, _ = next(dataiter)
            inputs = data.to(device)
            pred = net(inputs)
            if isinstance(pred, list):
                pred = torch.cat(pred, dim=0)
            label_pred = pred.permute(0, 2, 3, 4, 1).contiguous().view(-1, num_classes)
            _, label_pred = torch.max(F.softmax(label_pred, dim=1), 1)
            label_true = label.view(-1, 1)

            dice, intersection, summ = cal_dice(label_pred, label_true, num_classes)
            #pdb.set_trace()
            total_intersection += intersection
            total_summ += summ
            # val on part of trainset
            if max_len is not None:
                if i > max_len:
                    break
            pbar.update(1)
        
        pbar.close()

    total_dice = 2 * total_intersection / total_summ
    avg_dice = total_dice[1:num_classes].mean()

    return total_dice, avg_dice

def metrictor(pred_y, truth_y):
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    N, C = pred_y.shape
    num = N*C
    for i in range(N):
        for j in range(C):
            if pred_y[i][j] == truth_y[i][j]:
                if truth_y[i][j] == 1:
                    tp += 1
                else:
                    tn += 1
            elif truth_y[i][j] == 1:
                fn += 1
            elif truth_y[i][j] == 0:
                fp += 1
    
    accuracy = (tp + tn + 1e-10) / (num + 1e-10)
    precision = (tp + 1e-10) / (tp + fp + 1e-10)
    recall = (tp + 1e-10) / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    return accuracy, precision, recall, f1


def validate_encoder(net, dataloader, num_classes, device, max_len=None):
    total_intersection = torch.zeros((num_classes,)).double()
    total_summ = torch.zeros((num_classes,)).double()

    sum_precision = 0.0
    sum_recall = 0.0
    sum_f1 = 0.0
    sum_num = 0

    with torch.no_grad():
        for i, (data, label, depth_label, _) in enumerate(dataloader):
            inputs = data.to(device)
            pred, encoder_pred = net(inputs)

            label_pred = pred.permute(0, 2, 3, 4, 1).contiguous().view(-1, num_classes)
            _, label_pred = torch.max(F.softmax(label_pred, dim=1), 1)
            label_true = label.view(-1, 1)

            m = nn.Sigmoid()
            encoder_pred_result = (m(encoder_pred) > 0.5).type(torch.FloatTensor)

            encoder_pred_result = encoder_pred_result.view(-1, num_classes-1)
            depth_label = depth_label.view(-1, num_classes-1)

            accuracy, precision, recall, f1 = metrictor(encoder_pred_result, depth_label)
            sum_precision += precision
            sum_recall += recall
            sum_f1 += f1
            sum_num += 1

            dice, intersection, summ = cal_dice(label_pred, label_true, num_classes)
            #pdb.set_trace()
            total_intersection += intersection
            total_summ += summ
            # val on part of trainset
            if max_len is not None:
                if i > max_len:
                    break

    total_dice = 2 * total_intersection / total_summ
    avg_dice = total_dice[1:num_classes].mean()

    avg_precision = sum_precision / sum_num
    avg_recall = sum_recall / sum_num
    avg_f1 = sum_f1 / sum_num

    return total_dice, avg_dice, avg_precision, avg_recall, avg_f1

def validate_mutli_class(net, dataloader, num_classes, max_len=None):
    total_intersection = torch.zeros((num_classes,)).double()
    total_summ = torch.zeros((num_classes,)).double()

    with torch.no_grad():
        for i, (data, label, _) in enumerate(dataloader):
            inputs = data.cuda()
            pred = net(inputs)
            if isinstance(pred, list):
                pred = torch.cat(pred, dim=0)
            label_pred = pred.permute(0, 2, 3, 4, 1).contiguous().view(-1, num_classes)
            label_true = label.permute(0, 2, 3, 4, 1).contiguous().view(-1, num_classes)
            m = nn.Sigmoid()
            label_pred = (m(label_pred) > 0.5).type(torch.FloatTensor)
            # _, label_pred = torch.max(F.softmax(label_pred, dim=1), 1)
            # label_true = label.view(-1, 1)

            dice, intersection, summ = cal_dice_multi_class(label_pred, label_true, num_classes)
            #pdb.set_trace()
            total_intersection += intersection
            total_summ += summ
            # val on part of trainset
            if max_len is not None:
                if i > max_len:
                    break

    total_dice = 2 * total_intersection / total_summ
    avg_dice = total_dice.mean()

    return total_dice, avg_dice

def validate_mutli_class_with_contour(net, dataloader, num_classes, max_len=None):
    total_intersection = torch.zeros((num_classes,)).double()
    total_summ = torch.zeros((num_classes,)).double()

    with torch.no_grad():
        for i, (data, label, contour, _) in enumerate(dataloader):
            inputs = data.cuda()
            pred = net(inputs)
            if isinstance(pred, list):
                pred = torch.cat(pred, dim=0)
            label_pred = pred.permute(0, 2, 3, 4, 1).contiguous().view(-1, num_classes)
            label_true = label.permute(0, 2, 3, 4, 1).contiguous().view(-1, num_classes)
            m = nn.Sigmoid()
            label_pred = (m(label_pred) > 0.5).type(torch.FloatTensor)
            # _, label_pred = torch.max(F.softmax(label_pred, dim=1), 1)
            # label_true = label.view(-1, 1)

            dice, intersection, summ = cal_dice_multi_class(label_pred, label_true, num_classes)
            #pdb.set_trace()
            total_intersection += intersection
            total_summ += summ
            # val on part of trainset
            if max_len is not None:
                if i > max_len:
                    break

    total_dice = 2 * total_intersection / total_summ
    avg_dice = total_dice.mean()

    return total_dice, avg_dice