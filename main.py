from __future__ import print_function

import argparse
from torch.backends import cudnn
import train
import data
import torch
import numpy as np
from AFS_dataset import *


def main(config):
    trainDataset, valDataset, testDataset = data.get_display_path(config)
    train.train(config, trainDataset, valDataset, testDataset)
def main_new(config):
    train.train_new(config, AFS_dataset_train_two, AFS_dataset_val, AFS_dataset_test)
    #train.train_new(config, AFS_dataset_train, AFS_dataset_val, AFS_dataset_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='Diff_attention', type=str, help='name of training model')
    parser.add_argument('--weight_name', default='Diff_attention', type=str, help='weight name')
    parser.add_argument('--ckpt_folder_root', default='./weight/c_model/', type=str, help='folder name for ckpt folder')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--test_batch_size', type=int, default=260)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--last_lr', default=0.00005, type=float, help='last learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum value for momentum SGD')
    parser.add_argument('--lr_change_epoch', default=40, type=int, help='epoch when the learning rate changes')
    parser.add_argument('--resize', type=int, default=224, help='image size for resize')
    parser.add_argument('--crop_rate', type=float, default=1)
    parser.add_argument('--is_random_crop', type=bool, default=True)
    parser.add_argument('--init_lr', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('--weight_decay', default=5e-5, type=float, help='weight decay factor for loss')
    parser.add_argument('--num_workers', default=4, type=int, help='weight decay factor for loss')

    parser.add_argument('--train_display', type=str, nargs='+', default=['display'], choices=['display', 'display_s', 'projector']) # apple, samsung, projector
    parser.add_argument('--valid_display', type=str, nargs='+', default=['display', 'display_s', 'projector'], choices=['display', 'display_s', 'projector'])
    parser.add_argument('--test_display', type=str, nargs='+', default=['display', 'display_s', 'projector'], choices=['display', 'display_s', 'projector'])
    parser.add_argument('--root_path', type=str, default=r'C:\Users\MJ\Desktop\dofnet_data'.replace('\\','/'))
    parser.add_argument('--rand_seed', default=15, type=int, help='random seed value')
    parser.add_argument('--lmd1', type=float, default=0.4, help='dual lambda 1 value')
    parser.add_argument('--lmd2', type=float, default=0.3, help='center lambda 2 value')
    parser.add_argument('--lmd3', type=float, default=0.3, help='background lambda 3 value')
    parser.add_argument('--debug', type=bool, default=True)

    config = parser.parse_args()
    #main(config)
    main_new(config)
