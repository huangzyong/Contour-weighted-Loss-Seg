import os
import random
import numpy as np
import SimpleITK as sitk
import torch
from scipy import ndimage
from torch.utils.data import Dataset
from natsort import natsorted
from scipy.ndimage.interpolation import zoom


def windowing(img, win):
    # scale intensity from win[0]~win[1] to float numbers in 0~1
    # img1 = img.astype(float)
    # img1 -= win[0]
    # img1 /= win[1] - win[0]
    # img1[img1 > 1] = 1
    # img1[img1 < 0] = 0
    
    img = np.clip(img, win[0], win[1])
    img1 = (img - np.mean(img)) / (np.std(img) + 1e-3)

    return img1

class datasetTask1(Dataset):
    def __init__(self, data_root, transform=None, random_crop=400, mode='train'):
        self.sample_list_A = [s for s in data_root]
        self.mode = mode
        self.transform = transform  
        self.random_crop = random_crop
        

    def __len__(self):
        return len(self.sample_list_A)

    def __getitem__(self, idx):
        sample = dict()
        data_path = self.sample_list_A[idx]
        name = data_path.split('/')[-1].replace("data", "label")
        if self.mode=='train' or self.mode=='validation':
            label_path = '/home/hzy/Projects/PENGWIN/data/dataset/task1/trainset/labels/' + name
            contour_path = '/home/hzy/Projects/PENGWIN/data/dataset/task1/trainset/contours/contour21/' + name[:-7] + '.npz'
            contour = np.load(contour_path)['contour']
        if self.mode=='test':
            label_path = '/home/hzy/Projects/PENGWIN/data/dataset/task1/testset/labels/' + name

        # print("data: ", data_path)
        # print("label: ", label_path)
        sample['id'] = data_path.split('/')[-1][:3]
        # print(sample['id'])
        
        np_data = sitk.GetArrayFromImage(sitk.ReadImage(data_path))
        np_label = sitk.GetArrayFromImage(sitk.ReadImage(label_path))
        
        # scale to 0~1
        np_data = windowing(np_data, win=[-500, 1000])
     
        if self.mode == 'train':
            # random crop
            d, h, w = np_data.shape 
            # print(d,h,w)
            if self.random_crop<320:
                num = self.random_crop
                random_x_index = np.random.randint(0, h-num) if h>num else 0
                random_y_index = np.random.randint(0, w-num) if w>num else 0
                random_z_index = np.random.randint(0, d-num) if d>num else 0
                np_data = np_data[random_z_index:random_z_index+num, random_x_index:random_x_index+num, random_y_index:random_y_index+num]
                np_label = np_label[random_z_index:random_z_index+num, random_x_index:random_x_index+num, random_y_index:random_y_index+num]
                contour = contour[:, :, random_z_index:random_z_index+num, random_x_index:random_x_index+num, random_y_index:random_y_index+num]
            
            sample['input'] = np.expand_dims(np_data, axis=0)
            sample['target'] = np_label
            sample['contour'] = contour
            
            if self.transform:
                sample = self.transform(sample)
            return sample
        
        if self.mode=='validation' or self.mode=='test':
            d, h, w = np_data.shape 
            if self.random_crop<320:
                num = self.random_crop // 2
                center_x = h // 2
                center_y = w // 2
                center_z = d // 2
                np_data = np_data[:, center_x-num:center_x+num, center_y-num:center_y+num]
                np_label = np_label[:, center_x-num:center_x+num, center_y-num:center_y+num]
                # contour = contour[:, :, center_z-num:center_z+num, center_x-num:center_x+num, center_y-num:center_y+num]
            
            sample['input'] = np.expand_dims(np_data, axis=0)
            sample['target'] = np_label
            # sample['contour'] = contour.astype(np.int8)
         
            if self.transform:
                sample = self.transform(sample)
            return sample
