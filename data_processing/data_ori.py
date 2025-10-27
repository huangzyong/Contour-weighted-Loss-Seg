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


def save_ori(image_dir, label_dir, out_dir, item):
    cnt = 0
    item_name = item.split(".")[0]
    print('item name:', item_name)
    data = os.path.join(image_dir, item)
    label = os.path.join(label_dir, item)

    image_ct = sitk.ReadImage(data)
    image_gt = sitk.ReadImage(label)


    # save ct gt
    out_item_dir = osp.join(out_dir, item_name)
    if not osp.exists(out_item_dir):
        os.mkdir(out_item_dir)
    sitk.WriteImage(image_ct, osp.join(out_item_dir, 'data.nii.gz'))
    sitk.WriteImage(image_gt, osp.join(out_item_dir, 'label.nii.gz'))
    cnt += 1
    print("{} is completed!".format(item_name))
    print("**"*20)


if __name__ == "__main__":
    image_dir = r"/home/hzy/projects/PENGWIN/data/data-ori/task1/PENGWIN_CT_train_images"
    label_dir = r"/home/hzy/projects/PENGWIN/data/data-ori/task1/PENGWIN_CT_train_labels"
    out_dir = "/home/hzy/projects/PENGWIN/data/data-ori/task1/ori"
    
    crop_size = [320, 400]

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    item_list = sorted(os.listdir(image_dir))
    
    # for item in item_list:
    #     resample_one(image_dir, label_dir, out_dir, [1.0, 1.0, 1.0], item, crop_size)
    
    print("start run multi process code")
    pool = Pool(30)
    for item in item_list:
        pool.apply_async(save_ori, args=(image_dir, label_dir, out_dir, item))
    pool.close()
    pool.join()
    print("end run multi process code")


