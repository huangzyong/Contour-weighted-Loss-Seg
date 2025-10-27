'''
Author       : Zhengyong Huang
Date         : 2022-04-01 17:32:03
Description  :
FilePath     :
Code is far away from bug with the animal protecting
'''
import SimpleITK as sitk
import os
import csv
import numpy as np
import argparse
import torch
import scipy.ndimage as scip
from multiprocessing import Pool


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def batch_data_b(in_dir, out_dir, item, thickness, overlap):
    data_path = os.path.join(in_dir, item, "data.nii.gz")
    label_path = os.path.join(in_dir, item, "label.nii.gz")

    data = sitk.ReadImage(data_path)
    label = sitk.ReadImage(label_path)
    origin = data.GetOrigin()
    spacing = data.GetSpacing()
    direction = data.GetDirection()

    npdata = sitk.GetArrayFromImage(data)
    nplabel = sitk.GetArrayFromImage(label)
    # 1-12 总共12个标签
    nplabel[nplabel==11] = 5
    nplabel[nplabel==12] = 6
    nplabel[nplabel==13] = 7 
    nplabel[nplabel==14] = 8
    nplabel[nplabel==21] = 9
    nplabel[nplabel==22] = 10
    nplabel[nplabel==23] = 11
    nplabel[nplabel==24] = 12

    
    num = int(thickness - overlap)

    d, h, w = nplabel.shape
    for j in range(d // num):
        # print(j * num + thickness)
        assert (j * num + thickness) < d, "(j * num + thickness) must be smaller than h" 
        batch_data = npdata[j * num:j * num + thickness, :, :]
        batch_label = nplabel[j * num:j * num + thickness, :, :]

        data2save = sitk.GetImageFromArray(batch_data)
        data2save.SetOrigin(origin)
        data2save.SetSpacing(spacing)
        data2save.SetDirection(direction)

        label2save = sitk.GetImageFromArray(batch_label)
        label2save.SetOrigin(origin)
        label2save.SetSpacing(spacing)
        label2save.SetDirection(direction)

        name = [item + '_data_%d.nii.gz' % j, item + '_label_%d.nii.gz' % j]
        # print(name)
        # print(os.path.join(out_dir, 'data', name[0]))

        sitk.WriteImage(data2save, os.path.join(out_dir, 'images', name[0]))
        sitk.WriteImage(label2save, os.path.join(out_dir, 'labels', name[1]))

    batch_data = npdata[-thickness:, :, :]
    batch_label = nplabel[-thickness:, :, :]

    data2save = sitk.GetImageFromArray(batch_data)
    data2save.SetOrigin(origin)
    data2save.SetSpacing(spacing)
    data2save.SetDirection(direction)
    label2save = sitk.GetImageFromArray(batch_label)
    label2save.SetOrigin(origin)
    label2save.SetSpacing(spacing)
    label2save.SetDirection(direction)

    name = [item + '_data_%d.nii.gz' % (j + 1), item + '_label_%d.nii.gz' % (j + 1)]
    # print(name)

    sitk.WriteImage(data2save, os.path.join(out_dir, 'images', name[0]))
    sitk.WriteImage(label2save, os.path.join(out_dir, 'labels', name[1]))

    print("pid: {}, Done".format(item))


def batch_data_whole(in_dir, out_dir, item):
    data_path = os.path.join(in_dir, item, "data.nii.gz")
    label_path = os.path.join(in_dir, item, "label.nii.gz")

    data = sitk.ReadImage(data_path)
    label = sitk.ReadImage(label_path)
    origin = data.GetOrigin()
    spacing = data.GetSpacing()
    direction = data.GetDirection()

    npdata = sitk.GetArrayFromImage(data)
    nplabel = sitk.GetArrayFromImage(label)
    
    # 1-12 总共12个标签
    nplabel[nplabel==11] = 5
    nplabel[nplabel==12] = 6
    nplabel[nplabel==13] = 7 
    nplabel[nplabel==14] = 8
    nplabel[nplabel==21] = 9
    nplabel[nplabel==22] = 10
    nplabel[nplabel==23] = 11
    nplabel[nplabel==24] = 12

    data2save = sitk.GetImageFromArray(npdata)
    data2save.SetOrigin(origin)
    data2save.SetSpacing(spacing)
    data2save.SetDirection(direction)
    label2save = sitk.GetImageFromArray(nplabel)
    label2save.SetOrigin(origin)
    label2save.SetSpacing(spacing)
    label2save.SetDirection(direction)

    name = [item + '_data.nii.gz', item + '_label.nii.gz']
    print(name)

    sitk.WriteImage(data2save, os.path.join(out_dir, 'images', name[0]))
    sitk.WriteImage(label2save, os.path.join(out_dir, 'labels', name[1]))

    print("pid: {}, Done".format(item))


if __name__ == "__main__":
    # 1-80 for train
    # 81-100 for test
    in_dir = r"/home/hzy/Projects/PENGWIN/data1/data-resample/task1"
    print(in_dir)
    trainset_dir = r"/home/hzy/Projects/PENGWIN/data1/dataset/task1/trainset"
    testset_dir = r"/home/hzy/Projects/PENGWIN/data1/dataset/task1/testset"
    print("beginning...")
    if not os.path.exists(trainset_dir):
        os.makedirs(trainset_dir)
        os.makedirs(testset_dir)
    if not os.path.exists(os.path.join(trainset_dir, 'images')):
        os.makedirs(os.path.join(trainset_dir, 'images'))
        os.makedirs(os.path.join(trainset_dir, 'labels'))
        os.makedirs(os.path.join(trainset_dir, 'contours'))
        
        os.makedirs(os.path.join(testset_dir, 'images'))
        os.makedirs(os.path.join(testset_dir, 'labels'))
        
    thickness = 64
    overlap = 16
    cnt = 0
    item_list = sorted(os.listdir(in_dir))
    print(item_list)
    print("the number of item: ", len(item_list))
    
    # item_train = item_list[:80]
    # item_test = item_list[80:]
    item_train = []
    item_test = []
    idx_test = ['014', '082', '024', '084', '085', '086', '087', '088', '089', '090', '091', '041', '055', '094', '095', '096', '097', '098', '099', '100']
    for item in item_list:
        if item in idx_test:
            item_test.append(item)
        else:
            item_train.append(item)

    print("the number of train item: ", len(item_train))
    print("the number of test item: ", len(item_test))
    print(item_test)
    
    # for item in item_test:
    #     print(item)
    #     batch_data_whole(in_dir, testset_dir, item)
    
    print("start run multi process code")
    p = Pool(30)
    for item in item_train:
        print(item)
        p.apply_async(batch_data_b, args=(in_dir, trainset_dir, item, thickness, overlap))
    p.close()
    p.join()
    print("end run multi process code")
    
    # print("start run multi process code")
    # p = Pool(30)
    # for item in item_test:
    #     print(item)
    #     p.apply_async(batch_data_whole, args=(in_dir, testset_dir, item))
    # p.close()
    # p.join()
    # print("end run multi process code")
