'''
Author       : huangzhengyong
Date         : 2022-12-21
Description  :
Code is far away from bug with the animal protecting
'''
import os
import sys
import pathlib
import shutil
import logging
import time
import copy
import pandas as pd
from dataset.datasetTask1 import datasetTask1
import re
import csv
import os.path as osp
from tkinter.messagebox import NO
from sklearn.model_selection import KFold
import yaml
import numpy as np
import random
import argparse
import SimpleITK as sitk
# from apex import amp
import torch
import torch.nn as nn
from torch import mode, optim
from torch.utils.data import DataLoader, ConcatDataset, dataset
import torch.nn.functional as F

from models.net import LRScheduler

from scripts.metrics import calculate_dice_hd95_asd
from utils.util import *
from utils.timer import Timer
from validate import validate, cal_dice
from tensorboardX import SummaryWriter

from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import torch.multiprocessing as mp
import torch.distributed as dist

from tqdm import tqdm
from collections import OrderedDict
from torch.cuda.amp import GradScaler
import dataset.transformer as transforms
from monai.inferers import sliding_window_inference

TORCH_USE_CUDA_DSA=1

labelname2label = {'WT': 1, 'TC': 2, 'ET': 3}

label2labelname = {v: k for k, v in labelname2label.items()}

def save_nii_gz(save_result_folder, item, pred_pabel, name):
    data = sitk.ReadImage("/home/hzy/Projects/PENGWIN/data1/dataset/task1/testset/images/100_data.nii.gz")
    origin = data.GetOrigin()
    space = data.GetSpacing()
    direction = data.GetDirection()
    np_pred = np.array(pred_pabel)
    if name != 'image':
        np_pred = np_pred.astype(np.int16)
    np_image = sitk.GetImageFromArray(np_pred)
    np_image.SetOrigin(origin)
    np_image.SetSpacing(space)
    np_image.SetDirection(direction)
    out_path = os.path.join(save_result_folder, 'pred_nii/')
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    file_path = out_path + item[0] + '_' + name + '.nii.gz'

    sitk.WriteImage(np_image, file_path)
    

def validate_my(net, dataloader, num_classes, device, max_len=None):
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
            sample = next(dataiter)
            img, label = sample['input'], sample['target']
            inputs = img.to(device)  # 1 1 32 320 320
            
            pred = net(inputs)

            label_pred = pred.permute(0, 2, 3, 4, 1).contiguous().view(-1, num_classes)
            _, label_pred = torch.max(F.softmax(label_pred, dim=1), 1)
            label_true = label.view(-1, 1)

            dice, intersection, summ = cal_dice(label_pred, label_true, num_classes, data='brats')

            # pdb.set_trace()
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


def eval_dice(net, dataloader, cfg, save_result_folder, device):
    # depth_input = cfg.BASE.depth_input  # for newwork input
    n_items = len(dataloader)
    total_dice = torch.zeros(n_items, cfg.num_classes - 1)

    with open(os.path.join(save_result_folder, "Best_model_result.csv"), 'w') as f:
        writer = csv.writer(f)
        write_list = ["pid"]
        for i in range(len(labelname2label)):
            write_list.append("{}".format(label2labelname[i + 1]))
        # write_list.append("organ_brain")
        writer.writerow(write_list)
    with open(os.path.join(save_result_folder, "test_result.csv"), 'w') as f:
        writer = csv.writer(f)
        write_list = ["pid"]
        for i in range(len(labelname2label)):
            write_list.append("{}".format(label2labelname[i + 1]))
    net.eval()
    cls_list = np.arange(1,13,1)
    with torch.no_grad():
        DSC = []
        HD = []
        ASD = []
        for i, sample in enumerate(dataloader):
            data, label, item = sample['input'], sample['target'], sample['id']
            print("data size: ", data.size())
            print("label size: ", label.size())

            # save_nii_gz(save_result_folder, item, data.squeeze(0).squeeze(0), name='image')
            save_nii_gz(save_result_folder, item, torch.squeeze(label), name='label')
            
            
            data = data.to(device)
            item = sample['id']
            start_time = time.time()
            
            depth = data.size(2)  # Z 轴切片数
            depth_input = depth
            scores = torch.zeros(1, cfg.num_classes, data.size(
                2), data.size(3), data.size(4))

            # for j in range(depth // depth_input):
            #     inputs = data[:, :, j * depth_input:(j + 1) * depth_input, :, :]
            #     preds = net(inputs)

            #     scores[:, :, j * depth_input:(j + 1) * depth_input, :, :] = preds.cpu()

            # if depth % depth_input != 0:

            #     inputs = data[:, :, (depth - depth_input):depth, :, :]
            #     preds = net(inputs)

            #     scores[:, :, (j + 1) * depth_input:depth, :, :] = \
            #         preds[:, :, ((j + 2) * depth_input - depth):depth_input, :, :].cpu()

            '''
            scores : 1 2 D H W
            label  : 1 D H W

            '''
            out = sliding_window_inference(data, (64,args.rand_crop,args.rand_crop), 1, net, overlap=0.1)
            scores = out
           
            end_time = time.time()
            print("each case running time: ", end_time - start_time)

            scores = torch.squeeze(scores)  # 去掉维数为1的维度
            label = torch.squeeze(label)

            label_pred = scores.permute(1, 2, 3, 0).contiguous()
            pre, label_pred = torch.max(F.softmax(label_pred, dim=3), 3)
            
            # save
            print("pred_label shape: ", label_pred.shape)
            save_nii_gz(save_result_folder, item, label_pred.cpu(), name='pred')
            
            # 计算 DSC HD95 ASSD
            pred = label_pred.cpu().numpy()  # d, h, w
            gt = label.cpu().numpy()  # d, h, w
            print(item)
            dice, hd, asd = calculate_dice_hd95_asd(pred, gt, cls_list, save_result_folder, item, data == "pengwin")
            DSC.append(dice)
            HD.append(hd)
            ASD.append(asd)
            logging.info("name: {:} dice: {:.8f} hd: {:.8f} asd: {:.8f}".format(item, dice, hd, asd)) 
                
            # save
            print("pred_label shape: ", label_pred.shape)  # d, h, w
            # save_nii_gz(save_result_folder, item, label_pred)

            label_pred = label_pred.view(-1)  # 转为一维张量
            label_true = label.view(-1, 1)  # 二维，第2个维度为1，

            dice, intersection, summ = cal_dice(
                label_pred, label_true, cfg.num_classes, data='pengwin')

            avg_dice = dice[1:cfg.num_classes].mean()
            total_dice[i] = dice[1:cfg.num_classes]

            print('%s %d, avg_dice: %f' % (item[0], i + 1, avg_dice.item()))
            print(dice[1:cfg.num_classes])

            with open(os.path.join(save_result_folder, "Best_model_result.csv"), 'a') as f:
                writer = csv.writer(f)
                write_list = [item]
                for k in range(cfg.num_classes - 1):
                    write_list.append(dice[k + 1].item())

                writer.writerow(write_list)

        print('total, avg_dice: {}'.format(total_dice.mean()))
        print("std: ", total_dice.std())

        avg_dice_organ = total_dice.mean(dim=0)
        print(avg_dice_organ)

        with open(osp.join(save_result_folder, 'Best_model_result.txt'), 'a') as f:
            f.write('total, avg_dice: %f' % (total_dice.mean()))
            for k in range(cfg.num_classes-1):
                f.write(' %.4f' % avg_dice_organ[k].item())
        print('write result.txt')
        
        print("mean(DSC): ", np.mean(DSC), "   std(DSC): ", np.std(DSC)) 
        print("mean(HD): ", np.mean(HD), "   std(HD): ", np.std(HD))    
        print("mean(ASD): ", np.mean(ASD), "   std(ASD): ", np.std(ASD)) 
        logging.info("Average score:")
        logging.info("mean_dice: {:.8f} mean_std: {:.8f}".format(np.mean(DSC), np.std(DSC)))  
        logging.info("mean_hd: {:.8f} mean_std: {:.8f}".format(np.mean(HD), np.std(HD))) 
        logging.info("mean_asd: {:.8f} mean_std: {:.8f}".format(np.mean(ASD), np.std(ASD))) 
            

        # with open(osp.join(cfg.output_dir, '{}_result.txt'.format(model_class)), 'a') as f:
        #     writer = csv.writer(f)
        #     write_list = [item]
        #     for k in range(cfg.BASE.num_classes - 1):
        #         write_list.append(dice[k + 1].item())
        #     for k in range(cfg.BASE.num_classes - 1):
        #         f.write(' %.4f' % avg_dice_organ[k].item())
        #     writer.writerow(write_list)
        # print('write result.txt')


def test(cfg, net, save_result_folder, device, load_best_model=True):
    # assert isinstance(model_class, str)
    test_transforms = transforms.Compose([
        # transforms.RandomRotation(p=0.2, angle_range=[0, 15]),
        # transforms.Mirroring(p=0.2),
        # transforms.NormalizeIntensity(),
        transforms.ToTensor(mode='test')
    ])
    
    print('===> Loading test datasets')
    all_paths = []
    test_path = os.path.join(args.testFolder, 'images')
    for item in os.listdir(test_path):
        f_path = os.path.join(test_path, item)
        all_paths.append(f_path)

    print("All data number: ", len(all_paths))

    test_data = datasetTask1(all_paths, transform=test_transforms, mode='test')
    
    test_loader = DataLoader(dataset=test_data, num_workers=0, batch_size=1, shuffle=False, drop_last=False)

    eval_dice(net, test_loader, cfg, save_result_folder, device)


def run(args):
    
    run_name = get_run_name()
    date = run_name.split('_')[0]
    if args.lossType:
        save_result_folder = '../Results_test_new/' + args.model + '/' + args.model + '_' + date[:11] + '_' + args.lossType
    else:
        save_result_folder = '../Results_test_new/' + args.model + '_' + date[:11]
    
    if not os.path.exists(save_result_folder):
        os.makedirs(save_result_folder)

     # make logger file
    if os.path.exists(save_result_folder + '/code'):
        shutil.rmtree(save_result_folder + '/code')
    shutil.copytree('.', save_result_folder + '/code', shutil.ignore_patterns(['.git', '__pycache__']))
    logging.basicConfig(filename=save_result_folder + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')


    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    torch.backends.cudnn.benchmark = True
    
    if args.model == "U_Net":
        from models.UNet3D import UNet3D as mymodel
        model = mymodel(in_c=1, num_cls=args.num_classes)
    elif args.model == "DeepLabv3Plus":
        from models.DeepLabv3Plus import DeepLabv3Plus_FineTune as mymodel
        model = mymodel(channel=1, num_classes=args.num_classes)
    elif args.model == 'ProposedNet':
        from models.Proposed import ProposedNet as mymodel
        model = mymodel(in_c=1, num_cls=args.num_classes)
    elif args.model == "UNETR":
        from monai.networks.nets import UNETR as mymodel
        model = mymodel(
                    in_channels=1,
                    out_channels=args.num_classes,
                    img_size=(64, args.rand_crop, args.rand_crop),
                    feature_size=16,
                    hidden_size=768,
                    mlp_dim=3072,
                    num_heads=12,
                    pos_embed="perceptron",
                    norm_name="instance",
                    conv_block=True,
                    res_block=True,
                    dropout_rate=0.0,
                    )
        # from models.UNETR import UNETR as mymodel
        # model = mymodel(img_shape=(64, args.rand_crop, args.rand_crop), input_dim=1,
        #                 output_dim=args.num_classes, embed_dim=768, patch_size=16, num_heads=12, dropout=0.2)
    elif args.model == "Slim_UNETR":
        from models.Slim_UNETR import SlimUNETR as mymodel
        model = mymodel(
                        in_channels=1,
                        out_channels=args.num_classes,
                        embed_dim=64,
                        embedding_dim=32,
                        channels=(24, 48, 60),
                        blocks=(1, 2, 3, 2),
                        heads=(1, 2, 4, 4),
                        r=(4, 2, 2, 1),
                        dropout=0.2,
                        )
    elif args.model == "SwinUNETR":
        from monai.networks.nets import SwinUNETR as mymodel
        model = mymodel(
                        img_size=(64, args.rand_crop, args.rand_crop),
                        in_channels=1,
                        out_channels=args.num_classes,
                        feature_size=48,
                        use_checkpoint=True,
                        )
    elif args.model == 'UXNET3D':
        from models.UXNET3D import UXNET as mymodel
        model = mymodel(
                in_chans=1,
                out_chans=args.num_classes,
                depths=[2, 2, 2, 2],
                feat_size=[48, 96, 192, 384],
                drop_path_rate=0,
                layer_scale_init_value=1e-6,
                spatial_dims=3,
                )
    elif args.model == "RepUXNET":
        from models.REPUXNET import REPUXNET as mymodel
        model = mymodel( 
                        in_chans=1,
                        out_chans=args.num_classes,                       
                        depths=[2, 2, 2, 2],                               
                        feat_size=[48, 96, 192, 384],                                       
                        ks=11,
                        a=1,
                        drop_path_rate=0,
                        layer_scale_init_value=1e-6,
                        spatial_dims=3,
                        deploy=False
                        )
    elif args.model == "OurNet":
        from models.OurNet import OurNet as mymodel
        model = mymodel(in_c=1, num_cls=args.num_classes)
    else:
        raise ValueError('not support model:', args.model)
    
    load_path = '/home/hzy/Projects/PENGWIN/Results_PENGWIN/OurNet/OurNet_Dec18-23-14_edgeLoss/save_model/model_best.pth'
    model.load_state_dict(torch.load(load_path)['net'])
    print('Model loaded from {}'.format(load_path))
    
    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    test(args, model, save_result_folder, device=device, load_best_model=True)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Training Script')
    parser.add_argument('--testFolder', default='/home/hzy/Projects/PENGWIN/data/dataset/task1/testset', type=str, help='testset path')
    parser.add_argument('--gpu', default=1, type=int, help='select gpu')
    parser.add_argument('--model', default='OurNet', type=str, help='select model OurNet Slim_UNETR | U_Net | VNet | UNETR | DeepLabv3Plus | AttU_Net | TransUnet')
    parser.add_argument('--num_classes', default=13, type=int, help='num_classes')
    parser.add_argument('--rand_crop', default=192, type=int, help='rand crop')
    parser.add_argument('--lossType', default="edgeLoss", type=str, help='edgeLoss | cedl | GDL | DistMap')
    args = parser.parse_args()

    if True:
        run(args)
