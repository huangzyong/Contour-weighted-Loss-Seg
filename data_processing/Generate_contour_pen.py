import SimpleITK as sitk
import os
import csv
import numpy as np
import argparse
import torch
import scipy.ndimage as scip
from multiprocessing import Pool

def GetContour_3D(target, type_trans='erode', kernel=3, iter=1, num_cls=2):
    '''
    target(tensor)     : 1 32 320 320
    contours   : 1 24 32 320 320
    '''
    kernel_s = np.ones((kernel, kernel, kernel))
    N, D, H, W = target.size()
    # print("target max num:{}".format(target.max()))
    print(target.max())
    b = target.view(N, 1, D, H, W)
    # print(b.shape)
    t_one_hot = target.new_zeros(1, num_cls, D, H, W)  # 1 24 32 320 320
    # print("t_one_hot shape: {}".format(t_one_hot.shape))
    t_one_hot.scatter_(1, target.view(N, 1, D, H, W), 1.)  # 1 24 32 320 320
    
    contours = np.zeros((1, num_cls, D, H, W))
    for i in range(num_cls):
        img = t_one_hot[0, i, :, :, :].squeeze(0).cpu().numpy()
        img_n = (img).astype(np.uint8)
        
        # target = img_n
        # if target.sum()>30000:
        #     iteration = 2
        # else:
        #     iteration = 1
        # print("iteration: ", iteration)
        
        
        if type_trans == 'erode':
            erosion = scip.binary_erosion(img_n, structure=kernel_s, iterations=iter).astype(img.dtype)
            contour = img_n - erosion

        elif type_trans == 'dilate':
            dilate = scip.binary_dilation(img_n, structure=kernel_s, iterations=iter).astype(img.dtype)
            contour = dilate - img_n
        
        elif type_trans == 'ContainBackground':
            dilation = scip.binary_dilation(img_n, structure=kernel_s, iterations=iter).astype(img.dtype)
            erosion = scip.binary_erosion(img_n, structure=kernel_s).astype(img.dtype)
            contour = dilation - erosion

        else:
            raise ValueError("type error")
        contours[:, i, :, :, :] = contour
    contours = torch.Tensor(contours) 
    return contours

def generate_contour(in_dir, out_dir, item):

    # data_path = os.path.join(in_dir, item)
    data_path = os.path.join(in_dir, item)

    data = sitk.ReadImage(data_path)
    origin = data.GetOrigin()
    spacing = data.GetSpacing()

    npdata = sitk.GetArrayFromImage(data).astype(np.int64)

    # **** TODO ***** 获取 contour 
    target = torch.tensor(npdata).unsqueeze(0)

    contours = GetContour_3D(target, type_trans='erode', kernel=2, iter=1, num_cls=13)  # 1 24 D H W
    npcontour = contours.cpu().numpy()
    # print(npcontour.shape)
    # print(npcontour[0, 0, :, :, :].max())

    d, h, w = npdata.shape
    save_path = os.path.join(out_dir, 'contour21')
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    np.savez(os.path.join(save_path, item[:-7] + '.npz'), contour=npcontour)  # BraTS  31
    # sitk.WriteImage(contour2save, os.path.join(out_dir, item))
    
    print("pid: {}, Done".format(item))


if __name__ == "__main__":
    in_dir = '/home/hzy/Projects/PENGWIN/data1/dataset/task1/trainset/labels'
    out_dir = '/home/hzy/Projects/PENGWIN/data1/dataset/task1/trainset/contours'

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    item_list = os.listdir(in_dir)
    print(len(item_list))
    
    # for i, item in enumerate(item_list):
    #     print(i+1, item)
    #     generate_contour(in_dir, out_dir, item)
    
    print("start run multi process code")
    p = Pool(30)
    for i, item in enumerate(item_list):
        p.apply_async(generate_contour, args=(in_dir, out_dir, item))
    print("end run multi process code")
    p.close()
    p.join()
    

    