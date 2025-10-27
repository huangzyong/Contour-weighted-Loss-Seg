'''
Author       : huangzhengyong
Date         : 2022-06-26
Description  : resample, center crop, and transfer .mha to .nii
FilePath     :
Code is far away from bug with the animal protecting
'''

import SimpleITK as sitk
import numpy as np
from PIL import Image
import os.path as osp
import os
import torch
import torch.nn.functional as f
import pdb
import argparse
import glob

from multiprocessing import Pool
from functools import partial


def resample_one(image_dir, label_dir, out_dir, out_spacing, item, crop_size):
    cnt = 0
    item_name = item.split(".")[0]
    print('item name:', item_name)
    data = os.path.join(image_dir, item)
    label = os.path.join(label_dir, item)

    image_ct = sitk.ReadImage(data)
    image_gt = sitk.ReadImage(label)

    # data label
    ct_origin = image_ct.GetOrigin()
    ct_spacing = image_ct.GetSpacing()
    ct_direction = image_ct.GetDirection()
    ct_size = image_ct.GetSize()
    print('ct spacing:', ct_spacing, 'ct size:', ct_size)
    # print("np size: ", sitk.GetArrayFromImage(image_ct).shape)


    out_size_c = [int(np.round(ct_size[0] * (ct_spacing[0] / out_spacing[0]))),
                  int(np.round(ct_size[1] * (ct_spacing[1] / out_spacing[1]))),
                  int(np.round(ct_size[2] * (ct_spacing[2] / out_spacing[2])))]
    print('ct out_size', out_size_c)

    # ct gt插值
    reference_image = sitk.Image(out_size_c, sitk.sitkFloat32)
    reference_image.SetOrigin(ct_origin)
    reference_image.SetSpacing(out_spacing)
    reference_image.SetDirection(ct_direction)
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(reference_image)


    resample.SetInterpolator(sitk.sitkLinear)  # 线性插值
    newimage_ct = resample.Execute(image_ct)
    resample.SetInterpolator(sitk.sitkNearestNeighbor)  # 线性插值
    newimage_gt = resample.Execute(image_gt)

    
    # center crop
    origin = newimage_ct.GetOrigin()
    spacing = newimage_ct.GetSpacing()
    direction = newimage_ct.GetDirection()

    npdata = sitk.GetArrayFromImage(newimage_ct)
    nplabel = sitk.GetArrayFromImage(newimage_gt)
    print("resample size: ", npdata.shape)

    npdata = center_crop_one(npdata, crop_size[0], crop_size[1])
    nplabel = center_crop_one(nplabel, crop_size[0], crop_size[1])

    d, h, w = nplabel.shape

    data2save = sitk.GetImageFromArray(npdata)
    data2save.SetOrigin(origin)
    data2save.SetSpacing(spacing)
    data2save.SetDirection(direction)
    label2save = sitk.GetImageFromArray(nplabel)
    label2save.SetOrigin(origin)
    label2save.SetSpacing(spacing)
    label2save.SetDirection(direction)
    print("crop size: ", d,h,w)

    # save ct gt
    out_item_dir = osp.join(out_dir, item_name)
    if not osp.exists(out_item_dir):
        os.mkdir(out_item_dir)
    sitk.WriteImage(data2save, osp.join(out_item_dir, 'data.nii.gz'))
    sitk.WriteImage(label2save, osp.join(out_item_dir, 'label.nii.gz'))
    cnt += 1
    print("{} is completed!".format(item_name))
    print("**"*20)


def center_crop_one(npdata, size_x, size_y):
    """center crop at x, y axis"""
    D, H, W = npdata.shape
    #assert size <= H
    if size_x <= H and size_y <= W:
        center_x = H // 2
        center_y = W // 2
        half_size_x = size_x // 2
        half_size_y = size_y // 2

        npdata_crop = npdata[:, (center_x-half_size_x):(center_x + half_size_x), (center_y-half_size_y):(center_y+half_size_y)]
    elif size_x > H and size_y < W:
        npdata_crop = np.ones((npdata.shape[0], size_x, W), dtype=npdata.dtype) * npdata.min()
        start_x = (size_x - H) // 2
        npdata_crop[:, start_x:start_x+H, :] = npdata
        
        center_y = W // 2
        half_size_y = size_y // 2
        
        npdata_crop = npdata_crop[:, :, (center_y-half_size_y):(center_y+half_size_y)]
        
    elif size_x < H and size_y > W:
        npdata_crop = np.ones((npdata.shape[0], H, size_y), dtype=npdata.dtype) * npdata.min()
        start_y = (size_y - W) // 2
        npdata_crop[:, :, start_y:start_y+W] = npdata
        
        center_x = H // 2
        half_size_x = size_x // 2
        
        npdata_crop = npdata_crop[:, (center_x-half_size_x):(center_x+half_size_x), :]
        

    else: 
        print('image size small than crop size, do padding')
        # return False
        npdata_crop = np.ones((npdata.shape[0], size_x, size_y), dtype=npdata.dtype) * npdata.min()
        start_x = (size_x - H) // 2
        start_y = (size_y - W) // 2
        npdata_crop[:, start_x:start_x+H, start_y:start_y+W] = npdata
    return npdata_crop


if __name__ == "__main__":
    image_dir = r"/home/hzy/Projects/PENGWIN/data/data-ori/task1/PENGWIN_CT_train_images"
    label_dir = r"/home/hzy/Projects/PENGWIN/data/data-ori/task1/PENGWIN_CT_train_labels"
    out_dir = r"/home/hzy/Projects/PENGWIN/data1/data-resample/task1/"
    
    crop_size = [144, 192]

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    item_list = sorted(os.listdir(image_dir))
    
    # for item in item_list:
    #     resample_one(image_dir, label_dir, out_dir, [1.0, 1.0, 1.0], item, crop_size)
    
    print("start run multi process code")
    pool = Pool(30)
    for item in item_list:
        pool.apply_async(resample_one, args=(image_dir, label_dir, out_dir, [2.0, 2.0, 2.0], item, crop_size))
    pool.close()
    pool.join()
    print("end run multi process code")


