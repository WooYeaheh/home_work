from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
from torchvision.transforms import transforms,Compose
import json
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class AFS_dataset(Dataset):
    def __init__(self, train_rgb_path, train_ir_path, train_depth_path, label, text = None, transform = None):
        self.train_rgb_path = train_rgb_path
        self.train_ir_path = train_ir_path
        self.train_depth_path = train_depth_path
        self.label = label
        self.text = text
        self.transform = transform

    def __getitem__(self, idx):
        rgb = Image.open(self.train_rgb_path[idx]).convert('RGB')
        ir = Image.open(self.train_ir_path[idx]).convert('RGB')
        depth = Image.open(self.train_depth_path[idx]).convert('RGB')
        label = self.label[idx]
        try:rgb, ir, depth = self.transform(rgb, ir, depth)
        except:
            [rgb1, ir1, depth1],[rgb2, ir2, depth2] = self.transform(rgb, ir, depth)
            return [rgb1, ir1, depth1], [rgb2, ir2, depth2], label
        if self.text is not None:
            text = self.text[idx]
            return rgb, ir, depth, label, text
        return rgb, ir, depth, label

    def __len__(self):
        return len(self.label)

class RemoveBlackBorders(object):
    def __call__(self, im):
        if type(im) == list:
            return [self.__call__(ims) for ims in im]
        V = np.array(im)
        V = np.mean(V, axis=2)
        X = np.sum(V, axis=0)
        Y = np.sum(V, axis=1)
        y1 = np.nonzero(Y)[0][0]
        y2 = np.nonzero(Y)[0][-1]

        x1 = np.nonzero(X)[0][0]
        x2 = np.nonzero(X)[0][-1]
        return im.crop([x1, y1, x2, y2])

def apply_same_transform(img_a, img_b, img_c):
    remove = RemoveBlackBorders()
    seed = 42
    transform = Compose([
        remove,
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
    ])

    torch.manual_seed(seed)
    img_a = transform(img_a)

    torch.manual_seed(seed)
    img_b = transform(img_b)

    torch.manual_seed(seed)
    img_c = transform(img_c)

    return img_a, img_b, img_c

def apply_same_transform_two(img_a, img_b, img_c):
    remove = RemoveBlackBorders()
    seed = 42
    transform = Compose([
        remove,
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.ToTensor(),
    ])

    torch.manual_seed(seed)
    img_a_1 = transform(img_a)
    torch.manual_seed(seed+1000)
    img_a_2 = transform(img_a)

    torch.manual_seed(seed)
    img_b_1 = transform(img_b)
    torch.manual_seed(seed+1000)
    img_b_2 = transform(img_b)

    torch.manual_seed(seed)
    img_c_1 = transform(img_c)
    torch.manual_seed(seed+1000)
    img_c_2 = transform(img_c)

    return [img_a_1, img_b_1, img_c_1], [img_a_2, img_b_2, img_c_2]


# with open('D:/antispoof/WACV/data_idx/train_idx.json','r') as f:
#     train_dataset = json.load(f)
# with open('D:/antispoof/WACV/data_idx/val_idx.json','r') as f:
#     val_dataset = json.load(f)
# with open('D:/antispoof/WACV/data_idx/test_idx.json','r') as f:
#     test_dataset = json.load(f)
#
# #train_dataset = train_Race.main(train_Race.parse_arguments([]))
# #val_dataset, test_dataset = devtest_Race.main(devtest_Race.parse_arguments([]))
# text1 = []
# text2 = []
# for file in train_dataset['real_rgb'][0]:
#     file_name = file.split('\\')[-3]
#     name = file_name.split('_')[-1]
#     text1.append("real face" if name == '1' else "paper attack" if name == '2' else "display attack" if name == '4' else " ")
#
# for file in train_dataset['fake_rgb'][0]:
#     text2.append("mask attack")
#
# train_rgb = train_dataset['real_rgb'][0] + train_dataset['fake_rgb'][0]  #pa+mask attack
# train_rgb_label = train_dataset['real_rgb'][1] + train_dataset['fake_rgb'][1]
# train_text = text1 + text2
# train_ir = train_dataset['real_ir'][0] + train_dataset['fake_ir'][0]
# train_ir_label = train_dataset['real_ir'][1] + train_dataset['fake_ir'][1]
# train_depth = train_dataset['real_depth'][0] + train_dataset['fake_depth'][0]
# train_depth_label = train_dataset['real_depth'][1] + train_dataset['fake_depth'][1]
#
# val_rgb = val_dataset['real_rgb'][0] + val_dataset['fake_rgb'][0]
# val_rgb_label = val_dataset['real_rgb'][1] + val_dataset['fake_rgb'][1]
# val_ir = val_dataset['real_ir'][0] + val_dataset['fake_ir'][0]
# val_ir_label = val_dataset['real_ir'][1] + val_dataset['fake_ir'][1]
# val_depth = val_dataset['real_depth'][0] + val_dataset['fake_depth'][0]
# val_depth_label = val_dataset['real_depth'][1] + val_dataset['fake_depth'][1]
#
# test_rgb = test_dataset['real_rgb'][0] + test_dataset['fake_rgb'][0]
# test_rgb_label = test_dataset['real_rgb'][1] + test_dataset['fake_rgb'][1]
# test_ir = test_dataset['real_ir'][0] + test_dataset['fake_ir'][0]
# test_ir_label = test_dataset['real_ir'][1] + test_dataset['fake_ir'][1]
# test_depth = test_dataset['real_depth'][0] + test_dataset['fake_depth'][0]
# test_depth_label = test_dataset['real_depth'][1] + test_dataset['fake_depth'][1]

import os
import json
import os

base_root = r'C:/Users/USER/PycharmProjects/PythonProject/Antispoofing/dataset'

all_data = {
    f"{split}_{modality}": [
        os.path.join(f"{base_root}/{split}/{modality}", f)
        for f in os.listdir(f"{base_root}/{split}/{modality}")
        if os.path.isfile(os.path.join(f"{base_root}/{split}/{modality}", f))
    ]
    for split in ['train', 'val', 'test']
    for modality in ['rgb', 'ir', 'depth']
}

# 변수처럼 꺼내 쓰고 싶다면 예시:
train_rgb = all_data['train_rgb']; train_ir = all_data['train_ir']; train_depth = all_data['train_depth']
val_rgb = all_data['val_rgb']; val_ir = all_data['val_ir']; val_depth = all_data['val_depth']
test_rgb = all_data['test_rgb']; test_ir = all_data['test_ir']; test_depth = all_data['test_depth']

with open(r'C:/Users/USER/PycharmProjects/PythonProject/Antispoofing/dataset/train/train_label.json', 'r') as f:
    train_rgb_label = json.load(f)
with open(r'C:/Users/USER/PycharmProjects/PythonProject/Antispoofing/dataset/val/val_label.json', 'r') as f:
    val_rgb_label = json.load(f)
with open(r'C:/Users/USER/PycharmProjects/PythonProject/Antispoofing/dataset/test/test_label.json', 'r') as f:
    test_rgb_label = json.load(f)

AFS_dataset_train = AFS_dataset(train_rgb_path=train_rgb, train_ir_path=train_ir, train_depth_path=train_depth, label=train_rgb_label, transform=apply_same_transform)
AFS_dataset_train_two = AFS_dataset(train_rgb_path=train_rgb, train_ir_path=train_ir, train_depth_path=train_depth, label=train_rgb_label, transform=apply_same_transform_two)


AFS_dataset_val = AFS_dataset(train_rgb_path=val_rgb, train_ir_path=val_ir, train_depth_path=val_depth, label=val_rgb_label, transform=apply_same_transform)
AFS_dataset_test = AFS_dataset(train_rgb_path=test_rgb, train_ir_path=test_ir, train_depth_path=test_depth, label=test_rgb_label, transform=apply_same_transform)

