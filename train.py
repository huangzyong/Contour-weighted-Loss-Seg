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
import pandas as pd
from dataset.datasetTask1 import datasetTask1
from losses import GDLoss, CEDRLoss, DistMapLoss
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
from losses.DiceLoss import DiceLoss, GDL
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

from losses.BoundaryWeightedLoss import GetContour_2D, GetContour_3D, BoundaryCeLoss, BoundaryDiceLoss

TORCH_USE_CUDA_DSA=1

labelname2label = {
    'spleen': 1,
    'kidney R': 2,
    'kidney L': 3,
    'gall ': 4,
    'esophagus': 5,
    'liver': 6,
    'stomach': 7,
    'arota': 8,
    'postcava': 9,
    'pancreas': 10,
    'adrenal': 11,
    'adrenal L': 12,
}

label2labelname = {v: k for k, v in labelname2label.items()}


def validate_my(net, dataloader, num_classes, device, max_len=None):
    total_intersection = torch.zeros((num_classes,)).double()
    total_summ = torch.zeros((num_classes,)).double()
    
    total_dice = torch.zeros((num_classes,)).double()

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
            # print("val img shape: ", img.shape)
            # print("val label shape: ", label.shape)
            
            pred = net(inputs)

            label_pred = pred.permute(0, 2, 3, 4, 1).contiguous().view(-1, num_classes)
            _, label_pred = torch.max(F.softmax(label_pred, dim=1), 1)
            label_true = label.view(-1, 1)

            dice, intersection, summ = cal_dice(label_pred, label_true, num_classes, data="pengwin")

            # pdb.set_trace()
            total_intersection += intersection
            total_summ += summ
            
            dice = 2 * intersection / summ
            total_dice += dice
            

            # val on part of trainset
            if max_len is not None:
                if i > max_len:
                    break
            pbar.update(1)

        pbar.close()

    # total_dice = 2 * total_intersection / total_summ
    total_dice = total_dice / (i+1)
    avg_dice = total_dice[1:num_classes].mean()
    
    return total_dice, avg_dice


def eval_dice(net, dataloader, cfg, save_result_folder, device, load_best_model=True):
    # depth_input = cfg.BASE.depth_input  # for newwork input
    n_items = len(dataloader)
    total_dice = torch.zeros(n_items, cfg.num_classes - 1)
    
    name = "Best_model_result.csv"
    if not load_best_model:
        name = "Last_model_result.csv"
    with open(os.path.join(save_result_folder, name), 'w') as f:
        writer = csv.writer(f)
        write_list = ["pid"]
        for i in range(len(labelname2label)):
            write_list.append("{}".format(label2labelname[i + 1]))
        writer.writerow(write_list)
    
    net.eval()
    with torch.no_grad():
        for i, sample in enumerate(dataloader):
            data, label = sample['input'], sample['target']
            print("data size: ", data.size())
            print("label size: ", label.size())
            data = data.to(device)
            item = sample['id']
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
            out = sliding_window_inference(data, (64, args.rand_crop, args.rand_crop), 1, net, overlap=0.1)
            scores = out

            scores = torch.squeeze(scores)  # 去掉维数为1的维度
            label = torch.squeeze(label)

            label_pred = scores.permute(1, 2, 3, 0).contiguous()
            pre, label_pred = torch.max(F.softmax(label_pred, dim=3), 3)

            label_pred = label_pred.view(-1)  # 转为一维张量
            label_true = label.view(-1, 1)  # 二维，第2个维度为1，

            dice, intersection, summ = cal_dice(
                label_pred, label_true, cfg.num_classes, data="pengwin")

            avg_dice = dice[1:cfg.num_classes].mean()
            
            # avg_dice = 0
            # num = 0
            # for i in range(1, 13):
            #     if i in label_true:
            #         avg_dice += dice[i]
            #         num += 1
            #     else:
            #         dice[i] = 99
            # avg_dice = avg_dice / num
            
            total_dice[i] = dice[1:cfg.num_classes]

            print('item %d, avg_dice: %f' % (i + 1, avg_dice.item()))
            print(dice[1:cfg.num_classes])

            with open(os.path.join(save_result_folder, name), 'a') as f:
                writer = csv.writer(f)
                write_list = [item]
                for k in range(cfg.num_classes - 1):
                    write_list.append(dice[k + 1].item())

                writer.writerow(write_list)
            # break
        print('total, avg_dice: {}'.format(total_dice.mean()))

        avg_dice_organ = total_dice.mean(dim=0)
        print(avg_dice_organ)

        name_txt = 'Best_model_result.txt'
        if not load_best_model:
            name_txt = 'Last_model_result.txt'
        with open(osp.join(save_result_folder, name_txt), 'a') as f:
            f.write('total, avg_dice: %f' % (total_dice.mean()))
            for k in range(cfg.num_classes-1):
                f.write(' %.4f' % avg_dice_organ[k].item())
        print('write result.txt')


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
    
    test_set_loader = DataLoader(dataset=test_data, num_workers=4, batch_size=1, shuffle=False, drop_last=False)
    print("load model...")
    if load_best_model:
        load_path = os.path.join(save_result_folder, 'save_model', 'model_best.pth')
        net.load_state_dict(torch.load(load_path)['net'])
        print('Model loaded from {}'.format(load_path))
    print("load model complete...")

    eval_dice(net, test_set_loader, cfg, save_result_folder, device, load_best_model=load_best_model)


def run(args):
    
    run_name = get_run_name()
    date = run_name.split('_')[0]
    if args.lossType:
        save_result_folder = '../Results_PENGWIN/' + args.model + '/' + args.model + '_' + date[:11] + '_' + args.lossType
    else:
        save_result_folder = '../Results_PENGWIN/' + args.model + '_' + date[:11]
    
    if not os.path.exists(save_result_folder):
        os.makedirs(save_result_folder)

     # make logger file
    if os.path.exists(save_result_folder + '/code'):
        shutil.rmtree(save_result_folder + '/code')
    shutil.copytree('.', save_result_folder + '/code', shutil.ignore_patterns(['.git', '__pycache__']))
    logging.basicConfig(filename=save_result_folder + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    writer = SummaryWriter(save_result_folder+'/log')

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    torch.backends.cudnn.benchmark = True
    
    
    train_transforms = transforms.Compose([
        # transforms.RandomRotation(p=0.2, angle_range=[0, 10]),
        # transforms.Mirroring(p=0.2),
        # transforms.NormalizeIntensity(),
        transforms.ToTensor()
    ])
    val_transforms = transforms.Compose([
        # transforms.RandomRotation(p=0.2, angle_range=[0, 15]),
        # transforms.Mirroring(p=0.2),
        # transforms.NormalizeIntensity(),
        transforms.ToTensor()
    ])
    
    print('===> Loading datasets')
    all_paths = []
    train_path= os.path.join(args.trainFolder, 'images')
    for item in os.listdir(train_path):
        f_path = os.path.join(train_path, item)
        all_paths.append(f_path)

    print("All data number: ", len(all_paths))
    # ***************** 五折交叉验证 *******************
    folder = KFold(n_splits=5, random_state=42, shuffle=True)
    train_paths = []  # 存放5折的训练集划分
    val_paths = []  # 存放5折的验证集划分
    for k, (Trindex, Tsindex) in enumerate(folder.split(all_paths)):
        train_paths.append(np.array(all_paths)[Trindex].tolist())
        val_paths.append(np.array(all_paths)[Tsindex].tolist())
    df = pd.DataFrame(data=train_paths, index=['0', '1', '2', '3', '4'])
    df.to_csv(os.path.join(save_result_folder, 'train.csv'))
    df1 = pd.DataFrame(data=val_paths, index=['0', '1', '2', '3', '4'])
    df1.to_csv(os.path.join(save_result_folder, 'val.csv'))
    
    fold = 1
    train_data = datasetTask1(all_paths, transform=train_transforms, random_crop=args.rand_crop, mode='train')
    val_data = datasetTask1(val_paths[fold], transform=val_transforms, random_crop=args.rand_crop, mode='validation')
    
    train_set_loader = DataLoader(dataset=train_data, num_workers=4, batch_size=args.train_batch_size,
                                  shuffle=True, drop_last=False)
    val_set_loader = DataLoader(dataset=val_data, num_workers=4, batch_size=args.val_batch_size,
                                shuffle=False, drop_last=False)
    
    print("train data: {}, val data: {}".format(len(train_set_loader)*args.train_batch_size, len(val_set_loader)*args.val_batch_size))
    test_transforms = transforms.Compose([
        # transforms.RandomRotation(p=0.2, angle_range=[0, 15]),
        # transforms.Mirroring(p=0.2),
        # transforms.NormalizeIntensity(),
        transforms.ToTensor(mode='test')
    ])
    
    # print('===> Loading test datasets')
    # all_paths = []
    # test_path = os.path.join(args.testFolder, 'images')
    # for item in os.listdir(test_path):
    #     f_path = os.path.join(test_path, item)
    #     all_paths.append(f_path)

    # print("All data number: ", len(all_paths))

    # test_data = datasetTask1(all_paths, transform=test_transforms, mode='test')
    
    # test_set_loader = DataLoader(dataset=test_data, num_workers=4, batch_size=1,
    #                               shuffle=False, drop_last=False)

    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")
    print("------------ device: {} -----------".format(device))


    print('===> Selecting model')
    if args.model == "DeepLabv3Plus":
        from models.DeepLabv3Plus import DeepLabv3Plus_FineTune as mymodel
        model = mymodel(channel=1, num_classes=args.num_classes)
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
    elif args.model == "nnFormer":
        from models.nnFormer import nnFormer as mymodel
        model = mymodel(crop_size=[64,args.rand_crop,args.rand_crop],
                        embedding_dim=144,
                        input_channels=1, 
                        num_classes=args.num_classes, 
                        conv_op=nn.Conv3d, 
                        depths=[2,2,2,2],
                        num_heads=[6, 12, 24, 48],
                        patch_size=[2,4,4],
                        window_size=[4,4,8,4],
                        deep_supervision=False)
    elif args.model == "U_Net":
        from models.UNet3D import UNet3D as mymodel
        model = mymodel(in_c=1, num_cls=args.num_classes)
    elif args.model == "Proposed_Att":
        from model.Proposed import Proposed_Att as mymodel
        model = mymodel(in_c=1, num_cls=args.num_classes)
    elif args.model=='ProposedNet':
        from models.Proposed import ProposedNet as mymodel
        model = mymodel(in_c=1, num_cls=args.num_classes)
    elif args.model == "VNet":
        from model.VNet import VNet as mymodel
        model = mymodel(in_c=4, out_c=4)
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
                    # pos_embed="perceptron",
                    norm_name="instance",
                    conv_block=True,
                    res_block=True,
                    dropout_rate=0.0,
                    )
        # from models.UNETR import UNETR as mymodel
        # model = mymodel(img_shape=(64, args.rand_crop, args.rand_crop), input_dim=1,
        #                 output_dim=args.num_classes, embed_dim=768, patch_size=16, num_heads=12, dropout=0.2)
    elif args.model == 'AttU_Net':
        from model.AttUnet import AttU_Net as mymodel
        model = mymodel(img_ch=4, output_ch=4)
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
    elif args.model == "SwinUNETR":
        from monai.networks.nets import SwinUNETR as mymodel
        model = mymodel(
                        img_size=(64, args.rand_crop, args.rand_crop),
                        in_channels=1,
                        out_channels=args.num_classes,
                        feature_size=48,
                        use_checkpoint=True,
                        )
    elif args.model == "OurNet":
        from models.OurNet import OurNet as mymodel
        model = mymodel(in_c=1, num_cls=args.num_classes)
    else:
        raise ValueError('not support model:', args.BASE.model)

    

    # ****************** TODO 加载预训练模型 ****************
    if args.load_model:
        # model.load_state_dict(torch.load(args.load_model_path)['net'])
        model.load_state_dict(torch.load("/home/hzy/Projects/ICIP/Results_BraTS_Train/U_Net_May07-23-26_edgeLoss/save_model/model_best.pth")['net'])
        model.to(device)
        print('==> resume from trained model, model path: {}'.format(args.load_model))

    model = model.to(device)
 
    # optim
    if args.optim == 'SGD':
        optimizer_seg = optim.SGD(
            model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)
    elif args.optim == 'Adam':
        optimizer_seg = optim.Adam(
            model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0005)
    elif args.optim == 'AdamW':
        optimizer_seg = optim.Adam(
            model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0005)

    all_niters = len(train_set_loader)
    lr_scheduler = LRScheduler(optimizer_seg, all_niters, args)
    # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_seg, T_0=25, eta_min=1e-5)

    if args.organ_weight:
        organ_weight = np.array(args.organ_weight)
        organ_weight = torch.from_numpy(organ_weight).float().unsqueeze(1)

    # loss
    if args.lossType=='edgeLoss':
        criterion_ce = BoundaryCeLoss(weight=None).to(device)
        criterion_dl = BoundaryDiceLoss(weight=None).to(device)
    elif args.lossType=='GDL':
        # criterion_ce = torch.nn.CrossEntropyLoss(weight=None, ignore_index=100).to(device)
        criterion_dl = GDLoss.GDL().to(device)
    elif args.lossType=='ce':
        criterion_ce = torch.nn.CrossEntropyLoss(weight=None, ignore_index=100).to(device)
        # criterion_dl = GDLoss.GDL().to(device)
    elif args.lossType=='edgeCE':
        criterion_ce = BoundaryCeLoss(weight=None).to(device)
        # criterion_dl = GDLoss.GDL().to(device)
    elif args.lossType=='DL':
        # criterion_ce = torch.nn.CrossEntropyLoss(weight=None, ignore_index=100).to(device)
        criterion_dl = DiceLoss().to(device)
    elif args.lossType=='edgeDL':
        # criterion_ce = torch.nn.CrossEntropyLoss(weight=None, ignore_index=100).to(device)
        criterion_dl = BoundaryDiceLoss(weight=None).to(device)
    else:
        criterion_ce = torch.nn.CrossEntropyLoss(weight=None, ignore_index=100).to(device)
        criterion_dl = DiceLoss().to(device)
        

    best_dice = 0.0
    t = Timer()
    start_epoch = 0
    for epoch in range(start_epoch, args.epochs):
        epoch_loss = 0
        trainloader_iter = iter(train_set_loader)
        model.train()
        for i in range(all_niters):
        # for i in range(50):
            sample = next(trainloader_iter)
            t.tic()
            lr_scheduler.update(i, epoch)
      
            img, label = sample['input'], sample['target']
            inputs = img.to(device)  # 1 4 128 160 160
            labels = label.to(device)  # 1 128 160 160
            # print("img shape: ", img.shape)
            # print("label shape: ", labels.shape)

            # ***** TODO *****
            if (args.lossType=='edgeLoss') or (args.lossType=='edgeDL') or (args.lossType=='edgeCE'):
                # contour = GetContour_3D(labels, type_trans=args.lossType, kernel=3, iter=1, num_cls=4)  # 1 4 128 160 160
                contour = sample['contour'].squeeze(1).float()
                contour = contour.to(device)
                contour_dice = contour  # 1 64 320 320
                contour_ce = contour.sum(dim=1).squeeze(0)
                contour_ce[contour_ce > 1] = 1
                contour_ce = contour_ce.to(device)
                contour_dice = contour_dice.to(device)
                print("compute contour")
            
            optimizer_seg.zero_grad() 
            result = model(inputs)


            if args.lossType=='edgeLoss':
                loss_ce = criterion_ce(result, labels, contour_ce)
                loss_dl = criterion_dl(result, labels, contour_dice)
                print("edgeLoss !!!")
            elif args.lossType=='GDL':
                loss_dl = criterion_dl(result, labels)
                loss_ce = torch.tensor(0)
            elif args.lossType=='ce':
                loss_ce = criterion_ce(result, labels)
                loss_dl = torch.tensor(0)
                print("CE !!!")
            elif args.lossType=='edgeCE':
                loss_ce = criterion_ce(result, labels, contour_ce)
                loss_dl = torch.tensor(0)
                print("edgeCE !!!")
            elif args.lossType=='DL':
                loss_dl = criterion_dl(result, labels)
                loss_ce = torch.tensor(0)
                print("DL !!!")
            elif args.lossType=='edgeDL':
                loss_dl = criterion_dl(result, labels, contour_dice)
                loss_ce = torch.tensor(0)
                print("edgeDL !!!")
            else:
                loss_ce = criterion_ce(result, labels)
                loss_dl = criterion_dl(result, labels)

            loss_seg = loss_ce + loss_dl

            loss_seg.backward()
            optimizer_seg.step()
    

            epoch_loss += loss_seg.item()
            batch_time = t.toc()

            print('Epoch: {}/{}, step: {}/{}, batch_loss: {}, ce_loss: {}, dl_loss: {}, '
                  'batch_time: {}'.format(epoch, args.epochs, i, all_niters, loss_seg.item(), loss_ce.item(), loss_dl.item(), batch_time))
            if i % 100 ==0:
                logging.info(
                    'Epoch: {}/{}, step: {}/{}, batch_loss: {}, ce_loss: {}, dl_loss: {}, '
                    'batch_time: {}'.format(epoch, args.epochs, i, all_niters, loss_seg.item(), loss_ce.item(), loss_dl.item(), batch_time))
            
            writer.add_scalar(
                'loss_seg', loss_seg.item(), epoch * all_niters + i)
            writer.add_scalar(
                'loss_ce', loss_ce.item(), epoch * all_niters + i)
            writer.add_scalar(
                'loss_dl', loss_dl.item(), epoch * all_niters + i)
            # break
        # ************* TODO val *************
        model.eval()
        print('val model on validation set...')
        # val on  valset
        val_dice, val_avg_dice = validate_my(model, val_set_loader, args.num_classes, device, max_len=50)
        # val on part of trainset
        print('val model on part of training set...')
        train_dice, train_avg_dice = validate_my(model, train_set_loader, args.num_classes, device, max_len=2)

        dice_each_class_dice = {}

        writer.add_scalars(
            'dice_avg_group', {'train_avg_dice': train_avg_dice}, epoch + 1
        )
        writer.add_scalars(
            'dice_avg_group', {'val_avg_dice': val_avg_dice}, epoch + 1
        )

        for k in range(1, args.num_classes):
            dice_each_class_dice["train_class_{}".format(k)] = train_dice[k]
            dice_each_class_dice["val_class_{}".format(k)] = val_dice[k]

        writer.add_scalars(
            'dice_all_group', dice_each_class_dice, epoch + 1)

        if val_avg_dice >= best_dice:
            best_dice = val_avg_dice
            save_model_folder = os.path.join(save_result_folder, "save_model")
            if not os.path.exists(save_model_folder):
                os.mkdir(save_model_folder)
            torch.save({'net': model.state_dict(), 'epoch': epoch + 1}, '{}/model_best.pth'
                       .format(save_model_folder))
            print('best average dice: ', val_avg_dice)

        print(
            "this epoch average dice: {}, the best average dice: {}".format(val_avg_dice, best_dice))
        
        log_path = osp.join(save_result_folder, 'train_log.txt')
        with open('%s' % log_path, 'a') as f:
            f.write(
                '[epoch {}] epoch_loss: {:.6f}, lr: {}, val_dataset: val_avg_dice: {:.6f}, best_dice: {:.6f}'.format(
                    epoch + 1, epoch_loss, optimizer_seg.param_groups[0]['lr'], val_avg_dice, best_dice))

            f.write('\n')

            f.write("each classes dice: ")
            for k in range(1, args.num_classes):
                f.write(' {:.4f}'.format(val_dice[k]))
            f.write('\n')

        if (epoch + 1) % args.save_freq == 0:
            torch.save({'net': model.state_dict(), 'epoch': epoch + 1}, '%s/%s/model_epoch_%d.pth' %
                       (save_result_folder, 'save_model', epoch + 1))

        # write summary
        writer.add_scalar('epoch_loss', epoch_loss, epoch + 1)
        writer.add_scalar(
            'lr', optimizer_seg.param_groups[0]['lr'], epoch + 1)

    test(args, model, save_result_folder, device=device, load_best_model=False)
    test(args, model, save_result_folder, device=device, load_best_model=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Model Training Script')
    parser.add_argument('--trainFolder', default='/home/hzy/Projects/PENGWIN/data/dataset/task1/trainset', type=str, help='trainset path')
    parser.add_argument('--testFolder', default='/home/hzy/Projects/PENGWIN/data/dataset/task1/testset', type=str, help='testset path')
    parser.add_argument('--load_model', default=False, type=str, help='pretrained model path')
    parser.add_argument('--gpu', default=0, type=int, help='select gpu')
    parser.add_argument('--model', default='DeepLabv3Plus', type=str, help='select model DeepLabv3Plus Slim_UNETR ProposedNet | U_Net | OurNet | UNETR |  | OurNet | TransUnet')
    parser.add_argument('--optim', default='Adam', type=str, help='optimizer  SGD | Adam')
    parser.add_argument('--lr', default=0.0003, type=float, help='lr')
    parser.add_argument('--num_classes', default=13, type=int, help='num_classes')
    parser.add_argument('--lr_mode', default='step', type=str, help='lr scheluder  step | poly | cos | linear')
    parser.add_argument('--step', default='50, 150, 300, 400', type=str, help='lr scheluder step')
    parser.add_argument('--decay_factor', default=0.5, type=str, help='lr scheluder step decay_factor')
    parser.add_argument('--warmup_mode', default='linear', type=str, help='warmup_mode')
    parser.add_argument('--warmup_lr', default=0.00001, type=float, help='warmup_lr')
    parser.add_argument('--warmup_epochs', default=1, type=int, help='warmup_epochs')
    parser.add_argument('--epochs', default=500, type=int, help='epochs')
    parser.add_argument('--save_freq', default=100, type=int, help='save_freq')
    parser.add_argument('--rand_crop', default=192, type=int, help='rand crop')
    
    parser.add_argument('--train_batch_size', default=1, type=int, help='train batch_size')
    parser.add_argument('--val_batch_size', default=1, type=int, help='val batch_size')
    parser.add_argument('--start_epochs', default=-1, type=int, help='start_epochs')
    parser.add_argument('--organ_weight', default=None, type=str, help='organ_weight')
    parser.add_argument('--lossType', default='edgeLoss', type=str, help='edgeLoss | None | GDL | cedl')
    args = parser.parse_args()

    if True:
        run(args)
