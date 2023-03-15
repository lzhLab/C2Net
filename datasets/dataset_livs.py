import numpy as np
import os, cv2
import PIL.Image as Image
from torch.utils.data import DataLoader, Dataset
import glob
from sklearn.model_selection import KFold
import sys
from utils.transforms import *
import random


class labeledDataset(Dataset):
    def __init__(self, paths, transforms=None):
        
        super(labeledDataset, self).__init__()
        self.paths = paths
        self.transforms = transforms

    def __getitem__(self, item):
        liver_paths = self.paths[item]
        train_dir = r'/image'
        vessel_dir = r'/label'

        if isinstance(liver_paths, list):
            liver_imgs = []
            for liver_path in liver_paths:
                liver_imgs.append(Image.open(liver_path).convert('L'))
            vessel_maskpath = liver_paths[len(liver_paths) // 2].replace(train_dir, vessel_dir)
            vessel_mask = Image.open(vessel_maskpath).convert('L')
        else:
            liver_paths = liver_paths.replace(vessel_dir, train_dir)
            liver_imgs = Image.open(liver_paths).convert('L')
            vessel_maskpath = liver_paths.replace(train_dir, vessel_dir)
            vessel_mask = Image.open(vessel_maskpath).convert('L')

        if self.transforms is not None:
            liver_imgs, vessel_mask = self.transforms(liver_imgs, vessel_mask)
        
        return liver_imgs, vessel_mask

    def __len__(self):
        return len(self.paths)


def split_data(folderpath, nfolds=5, expand_size=0, random_state=1):
    '''
    :param folderpath:
    :param nfolds:
    :param expand_size:
    :param random_state:
    :return: train_paths, val_paths
    '''
    group = []
    fold_names = os.listdir(folderpath)
    kf = KFold(n_splits=nfolds, random_state=random_state, shuffle=True)
    for train_idx, val_idx in kf.split(fold_names):
        trainpart = [fold_names[i] for i in train_idx]
        valpart = [fold_names[i] for i in val_idx]
        group.append([trainpart, valpart])

    result = []
    for fold in range(nfolds):
        train_paths = []
        val_paths = []
        train_folders = group[fold][0]
        val_folders = group[fold][1]

        for foldername in train_folders:
            train_folder = os.path.join(folderpath, foldername)
            expand_list = expand_data(train_folder, expand_size=expand_size)
            for data in expand_list:
                train_paths.append(data)
        
        for foldername in val_folders:
            val_folder = os.path.join(folderpath, foldername)
            expand_list = expand_data(val_folder, expand_size=expand_size)
            for data in expand_list:
                val_paths.append(data)
        result.append((train_paths, val_paths))
    return result


def expand_data(foldername, expand_size=0):
    '''
    获取文件夹中的图片文件名，返回一个list
    foldername: 文件夹名称
    expand_size：要扩展的图片张数
    '''
    group = []
    imggroup = []
    for name in os.listdir(foldername):
        group.append(os.path.join(foldername, name))
    if expand_size == 0:   #若连续张数为0，直接返回group
        return group
    else:                   #否则获取连续的图片list
        dict = {}
        imgdict = {}
        result = []
        for filename in group:
            num = filename.split('/')[-1].split('.')[0].split('_')[1]
            dict[int(num)] = filename
        
        # 找mask对应的图片
        imgfoldername = foldername.replace(r'label', r'image')
        for imgname in os.listdir(imgfoldername):
            imggroup.append(os.path.join(imgfoldername, imgname))
        for imgfilename in imggroup:
            imgnum = imgfilename.split('/')[-1].split('.')[0].split('_')[1]
            imgdict[int(imgnum)] = imgfilename

        for filename in group:
            sub_group = []
            find_filename = filename.replace(r'label', r'image')
            for i in range(0 - expand_size, 1 + expand_size): #-1,2
                imgnum = find_filename.split('/')[-1].split('.')[0].split('_')[1]
                if imgdict.get(int(imgnum) + i):
                    sub_group.append(imgdict[int(imgnum) + i])
                else:
                    break
                if i == expand_size:
                    result.append(sub_group)
    return result
