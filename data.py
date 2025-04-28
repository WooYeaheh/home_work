import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim as optim
import random


def get_display_path(config):
    print('get_display_path')
    resize = config.resize
    root_path = config.root_path
    batch_size = config.batch_size
    inside_path = root_path# + '/Inside_{}'.format(resize)
    In_and_Out = ['Inside_{}'.format(resize)]
    objects = os.listdir(inside_path + '/train')
    objects = ['shoes', 'cosmetics', 'music_album', 'beverage', 'dvd', 'household_goods']
    train_displays = config.train_display
    train_displays.append('real')
    valid_displays = config.valid_display
    test_displays = config.test_display

    trainDataset = []
    testDataset = []
    valDataset = []

    for r_f in train_displays:  #display1, display2, display3, real
        for obj in objects:
            print("object : {}".format(obj))
            folder_path = inside_path + '/train/{}/{}/'.format(obj, r_f)
            folders = os.listdir(folder_path)
            if len(folders) > 0:
                trainDataset.append(datasets.ImageFolder(root=folder_path))
#     print("valid_displays size", len(valid_displays))
    for r_f in valid_displays:
        datas = []
        for obj in objects:
            folder_path = inside_path + '/validation/{}/{}/'.format(obj, r_f)
            folders = os.listdir(folder_path)
            if len(folders) > 0:
                datas.append(datasets.ImageFolder(root=folder_path))
        for obj in objects:
            folder_path = inside_path + '/validation/{}/{}/'.format(obj, 'real')
            folders = os.listdir(folder_path)
            if len(folders) > 0:
                datas.append(datasets.ImageFolder(root=folder_path))

        valDataset.append(datas)

    for r_f in test_displays:  #[display1, real], [display2, real], [display3, real]
        datas = []
        for obj in objects:
            folder_path = inside_path + '/test/{}/{}/'.format(obj, r_f)
            folders = os.listdir(folder_path)
            if len(folders) > 0:
                datas.append(datasets.ImageFolder(root=folder_path))
        for obj in objects:
            folder_path = inside_path + '/test/{}/{}/'.format(obj, 'real')
            folders = os.listdir(folder_path)
            if len(folders) > 0:
                datas.append(datasets.ImageFolder(root=folder_path))
        testDataset.append(datas) # [shoes/display, ..., shoes/real, ...]

    return trainDataset, valDataset, testDataset


class CustomDataset():
    def __init__(self, Dataset, resize):
        resultList = []
        transform = transforms.Compose([
            transforms.RandomCrop(int(resize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        befor_folder_name = ''
        for obj in range(len(Dataset)):
            data = Dataset[obj]
            is_concat = False
            index_count = 0
            for i in range(len(data)):
                img = data.__getitem__(i)
                data_name = data.imgs[i]
                if len(data_name[0].split('/')[-1].split('_')) == 1:
                    center = transform(img[0])
                else:
                    focus_num = int(data_name[0].split('_')[-1].split('.jpg')[0]) # number of Focus
                    if focus_num == 0:
                        background = transform(img[0])
                    elif focus_num == 1:
                        continue

                index_count += 1

                if befor_folder_name == data_name[0].split('/')[-2]:
                    is_concat = True

                if is_concat:
                    if not befor_folder_name == data_name[0].split('/')[-2]:
                        print(befor_folder_name, data_name[0].split('/')[-2])
                    result = torch.cat([center, background])
                    is_concat = False
                    index_count = 0
                    r_f = data_name[0].split('/')[-2]
                    if r_f == 'real':
                        result = (result, 0)
                    else:
                        result = (result, 1)
                    # print(result[0].shape, result[1])
                    resultList.append(result)

                befor_folder_name = data_name[0].split('/')[-2]
        self.len = resultList.__len__()
        self.resultData = resultList

    def __getitem__(self, index):
        return self.resultData[index]

    def __len__(self):
        return self.len