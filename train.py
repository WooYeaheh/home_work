import os
import numpy as np
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data as data
from torch.utils.data import RandomSampler
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.optim as optim
import random
from sklearn.metrics import average_precision_score, accuracy_score, roc_curve, auc
import data
import torch.backends.cudnn as cudnn
import pandas as pd
# import seaborn as sns
# from dataset_load import dataset_loader
from model_load import model_loader
import seaborn as sns
import matplotlib.pyplot as plt

from loss import *
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
from sklearn.manifold import TSNE
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class train():
    def __init__(self, config, trainDataset, valDataset, testDataset):
        self.config = config
        self.lmd1 = config.lmd1
        self.lmd2 = config.lmd2
        self.lmd3 = config.lmd3


        self.debug = config.debug
        self.epochs = config.epochs
        self.lr_change_epoch = config.lr_change_epoch
        print("config.rand_seed: ", config.rand_seed)
        self.rand_seed = config.rand_seed
        self.batch_size = config.batch_size

        self.test_display = config.test_display
        self.valid_display = config.valid_display

        # random seed arrange
        np.random.seed(config.rand_seed)
        torch.manual_seed(config.rand_seed)
        train_dataset = data.CustomDataset(trainDataset, config.resize)
        dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                                                 num_workers=8, pin_memory=True)
        testDataloader = []
        for test_data in testDataset:
            test_dataset = data.CustomDataset(test_data, config.resize)
            testDataloader.append(
                torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=8,
                                            pin_memory=True))

        valDataloader = []
        for valid_data in valDataset:
            val_dataset = data.CustomDataset(valid_data, config.resize)
            valDataloader.append(
                torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=8,
                                            pin_memory=True))
        print('dataload end')
        print("train set : {}, test set : {}".format(train_dataset.__len__(), test_dataset.__len__()))
        print(
            "random_seed : {}, batch_size : {}, epochs : {}".format(config.rand_seed, config.batch_size, config.epochs))

        classes = ('real', 'fake')
        (self.trainloader, self.testloader, self.classes) = (dataloader, testDataloader, classes)
        # model loading...
        net = model_loader(config.model_name)

        # CUDA & cudnn checking...
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.net = net.to(self.device)
        if self.device == 'cuda':
            self.net = torch.nn.DataParallel(self.net)
            cudnn.deterministic = True
            cudnn.benchmark = False

        # loss loading...
        self.criterion = nn.CrossEntropyLoss()

        # optimizer loading...
        self.init_optimizer = optim.SGD(net.parameters(), lr=config.init_lr, momentum=config.momentum,
                                        weight_decay=config.weight_decay)
        self.last_optimizer = optim.SGD(net.parameters(), lr=config.last_lr, momentum=config.momentum,
                                        weight_decay=config.weight_decay)

        self.train()
        # self.test('39')

    def train(self):

        # Network initialization...
        self.net.train()
        optimizer = self.init_optimizer
        best_epoch = 0
        min_loss = float('inf')
        train_log_path = 'D:/antispoof/result/weight/train_log.txt'

        for i_epoch in range(self.epochs):

            # epoch initialization
            train_loss = 0
            background_loss = 0.
            adv_loss = 0.
            l2_loss = 0.
            correct = 0
            correct_single = 0
            correct_single2 = 0
            total = 0

            # lr change processing...
            if i_epoch == self.lr_change_epoch:
                optimizer = self.last_optimizer

            # batch iterations...
            for batch_idx, (inputs, targets) in enumerate(self.trainloader):
                inputs1 = inputs[:, :3]  # B, 3, h, w 이미지
                inputs2 = inputs[:, 3:]  # B, 3, h, w 이미지, targets : torch.Size([B]) (0 or 1)

                inputs1, inputs2, targets = inputs1.to(self.device), inputs2.to(self.device), targets.to(self.device)

                ## dual image conventional process
                # forward
                optimizer.zero_grad()
                outputs = self.net(inputs1, inputs2)
                out_all, feat1, feat2 = outputs['out_all'], outputs['feat_1'], outputs['feat_2']
                loss_c1 = self.criterion(feat1, targets)
                loss_c2 = self.criterion(feat2, targets)
                loss_c3 = self.criterion(out_all, targets)

                # c1, c2, m1, m2, c3 = outputs['1c'], outputs['2c'], outputs['1m'], outputs['2m'], outputs['3c']
                ##loss_c1 = self.criterion(c1, targets)
                # loss_c2 = self.criterion(c2, targets)
                # loss_c3 = self.criterion(c3, targets)
                # loss_m1 = self.criterion(m1, torch.tensor([0]).repeat(c1.shape[0]))
                # loss_m2 = self.criterion(m2, torch.tensor([1]).repeat(c1.shape[0]))

                loss = loss_c1 + loss_c2 + loss_c3  # + loss_m1 + loss_m2

                loss.backward()
                optimizer.step()

                # logging...
                train_loss += loss.item()
                total += targets.size(0)
                # print('{} iteration finished >> Training Loss: c2:{} | c3:{} | m1:{} | m2:{}'.format(batch_idx, loss_c2, loss_c3, loss_m1, loss_m2))
                print(
                    '{} iteration finished >> Training Loss: c1:{} | c2:{} | c3:{}'.format(batch_idx, loss_c1, loss_c2,
                                                                                           loss_c3))
            print('{} epoch finished >> Training Loss: {}'.format(i_epoch, train_loss / (batch_idx + 1)))
            self.save(f'./weight/{self.config.model_name}/', i_epoch)
            with open(train_log_path, 'a') as f:
                f.write('{} epoch >> Training Loss: {}\n\n'.format(i_epoch, train_loss / (batch_idx + 1)))
            if min_loss > (train_loss / (batch_idx + 1)):
                best_epoch = i_epoch
                min_loss = train_loss / (batch_idx + 1)
        print('Best epoch: {}'.format(best_epoch))
        with open(train_log_path, 'a') as f:
            f.write('Best epoch: {}'.format(best_epoch))
        self.test(best_epoch)

    def test(self, best_epoch):

        state = torch.load('D:/antispoof/result/weight/model_ckpt_{}.pth'.format(best_epoch))
        # state = torch.load('D:/antispoof/DoFNet/weight/model_ckpt_49.pth')
        self.net.load_state_dict(state['net'], strict=True)
        self.net.eval()
        test_log_path = 'D:/antispoof/result/weight/test_log.txt'
        for data_name_index, test_data in enumerate(self.testloader):

            test_loss = 0
            correct = 0
            total = 0
            y_true, y_pred = [], []
            with torch.no_grad():
                # batch iterations...
                for batch_idx, (inputs, targets) in enumerate(test_data):
                    inputs1 = inputs[:, :3]
                    inputs2 = inputs[:, 3:]
                    inputs1, inputs2, targets = inputs1.to(self.device), inputs2.to(self.device), targets.to(
                        self.device)

                    outputs = self.net(inputs1, inputs2)
                    out_all, feat1, feat2 = outputs['out_all'], outputs['feat_1'], outputs['feat_2']

                    # c1, c2, m1, m2, c3 = outputs['1c'], outputs['2c'], outputs['1m'], outputs['2m'], outputs['3c']
                    out_all = out_all

                    loss = self.criterion(out_all, targets)

                    # logging...
                    test_loss += loss.item()
                    _, predicted = out_all.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                    prediction = torch.max(out_all.data, 1)[1]

                    y_pred.extend(prediction)
                    y_true.extend(targets.flatten().tolist())
                    # print(prediction)
                    # correct += prediction.eq(labels.data.view_as(prediction)).cpu().sum()
                y_true, y_pred = np.array(y_true), np.array(y_pred)
                metrics = self.calculate_metrics(y_true, y_pred)
                auc_score, min_hter, acer, tpr_at_target_fpr = metrics['AUC'], metrics['Min HTER'], metrics['ACER'], \
                metrics['TPR@FPR=0.1%']
                acc = accuracy_score(y_true, y_pred > 0.5)
                ap = average_precision_score(y_true, y_pred)
                print('type : ', self.test_display[data_name_index], ', accuracy : ', acc, ', average precision : ', ap)
                with open(test_log_path, 'a') as f:
                    f.write(
                        'type : {} >> accuracy : {}, average precision : {}, AUC : {}, min HTER : {}, ACER : {}, TPR@FPR=0.1% : {}\n\n'.format(
                            self.test_display[data_name_index], acc, ap, auc_score, min_hter, acer, tpr_at_target_fpr))
            # Test Result
            print('Test Loss: %.3f | Acc: %.3f%% (%d/%d)\n' % (
                test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    def save(self, ckpt_folder, epoch):
        # state building...
        state = {'net': self.net.state_dict()}

        # Save checkpoint...
        if not os.path.isdir(ckpt_folder):
            os.mkdir(ckpt_folder)
        torch.save(state, ckpt_folder + '/model_ckpt_{}.pth'.format(epoch))
        return ckpt_folder

    def calculate_metrics(self, y_true, y_pred):
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)

        # AUC
        auc_score = auc(fpr, tpr)

        # HTER (Half Total Error Rate)
        far = fpr
        frr = 1 - tpr
        hter = (far + frr) / 2
        min_hter = np.min(hter)

        # ACER (Average Classification Error Rate)
        apcer = np.mean(y_pred[y_true == 0] > 0.5)  # Attack Presentation Classification Error Rate
        bpcer = np.mean(y_pred[y_true == 1] <= 0.5)  # Bona Fide Presentation Classification Error Rate
        acer = (apcer + bpcer) / 2

        # TPR at FPR = 0.1%
        target_fpr = 0.001
        tpr_at_target_fpr = 0.0
        if np.any(fpr <= target_fpr):
            tpr_at_target_fpr = np.interp(target_fpr, fpr, tpr)

        return {
            'AUC': auc_score,
            'Min HTER': min_hter,
            'ACER': acer,
            'TPR@FPR=0.1%': tpr_at_target_fpr
        }


class train_new():
    def __init__(self, config, trainDataset, valDataset, testDataset):
        self.writer = SummaryWriter(f'runs/{config.model_name}')

        self.config = config
        self.lmd1 = config.lmd1
        self.lmd2 = config.lmd2
        self.lmd3 = config.lmd3
        self.vis = {'patch_mean': 0, 'cls_tsne': 0}
        self.debug = config.debug
        self.epochs = config.epochs
        self.lr_change_epoch = config.lr_change_epoch
        print("config.rand_seed: ", config.rand_seed)
        self.rand_seed = config.rand_seed
        self.batch_size = config.batch_size

        self.test_display = config.test_display
        self.valid_display = config.valid_display

        # random seed arrange
        np.random.seed(config.rand_seed)
        torch.manual_seed(config.rand_seed)
        # sampler = RandomSampler(trainDataset, num_samples=1000, replacement=False)
        # dataloader = torch.utils.data.DataLoader(trainDataset, batch_size=config.batch_size,
        # num_workers=8, pin_memory=True,sampler=sampler, persistent_workers=True)

        dataloader = torch.utils.data.DataLoader(trainDataset, batch_size=config.batch_size, shuffle=True,
                                                 num_workers=config.num_workers, pin_memory=True, persistent_workers=True)

        testDataloader = torch.utils.data.DataLoader(testDataset, batch_size=config.test_batch_size, shuffle=False,
                                                     num_workers=config.num_workers,
                                                     pin_memory=True, persistent_workers=True)

        valDataloader = torch.utils.data.DataLoader(valDataset, batch_size=config.batch_size, shuffle=False,
                                                    num_workers=config.num_workers,
                                                    pin_memory=True, persistent_workers=True)
        print('dataload end')
        print("train set : {}, val set : {}, test set : {}".format(trainDataset.__len__(), valDataset.__len__(),
                                                                   testDataset.__len__()))
        print(
            "random_seed : {}, batch_size : {}, epochs : {}".format(config.rand_seed, config.batch_size, config.epochs))

        classes = ('real', 'fake')
        (self.trainloader, self.valloader, self.testloader, self.classes) = (
        dataloader, valDataloader, testDataloader, classes)
        # model loading...
        net = model_loader(config.model_name)

        # CUDA & cudnn checking...
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.net = net.to(self.device)
        # if self.device == 'cuda':
        # self.net = torch.nn.DataParallel(self.net)
        # cudnn.deterministic = True
        # cudnn.benchmark = False

        # self.criterion = nn.CrossEntropyLoss()
        self.criterion = FocalLoss(gamma=2.0,alpha=0.25)
        self.supcon = SupConLoss(temperature=0.1)
        self.init_optimizer = optim.SGD(net.parameters(), lr=config.init_lr, momentum=config.momentum,
                                        weight_decay=config.weight_decay)
        self.last_optimizer = optim.SGD(net.parameters(), lr=config.last_lr, momentum=config.momentum,
                                        weight_decay=config.weight_decay)

        self.train()
        # self.train_two()
        self.test('46')

    def train(self):

        self.net.train()
        optimizer = self.init_optimizer
        best_epoch = 0
        min_loss = float('inf')
        os.makedirs(f'./weight/{self.config.model_name}/',exist_ok=True)
        train_log_path = f'./weight/{self.config.model_name}/train_log.txt'
        for i_epoch in range(self.epochs):

            # epoch initialization
            train_loss = 0
            background_loss = 0.
            adv_loss = 0.
            l2_loss = 0.
            correct = 0
            correct_single = 0
            correct_single2 = 0
            total = 0

            # lr change processing...
            if i_epoch == self.lr_change_epoch:
                optimizer = self.last_optimizer

            # batch iterations...
            for batch_idx, (rgb, ir, depth, labels) in enumerate(self.trainloader):
                # to_pil = transforms.ToPILImage()
                # img = to_pil(rgb[0])
                # img.show(title="Image")

                inputs1, inputs2, inputs3, targets = rgb.to(self.device), ir.to(self.device), depth.to(
                    self.device), labels.to(self.device)

                ## dual image conventional process
                # forward
                optimizer.zero_grad()
                outputs = self.net(inputs1, inputs2)
                # outputs = self.net(inputs1, inputs2, inputs3)
                # out_all, feat1, feat2 = outputs['out_all'], outputs['feat_1'], outputs['feat_2']
                feat1, feat2 = outputs['feat_1'], outputs['feat_2']

                # out_all, feat1, feat2, feat3 = outputs['out_all'], outputs['feat_1'], outputs['feat_2'], outputs['feat_3']
                loss_c1 = self.criterion(feat1, targets)
                loss_c2 = self.criterion(feat2, targets)
                # loss_c3 = self.criterion(feat3, targets)
                # loss_c3 = self.criterion(out_all, targets)

                # c1, c2, m1, m2, c3 = outputs['1c'], outputs['2c'], outputs['1m'], outputs['2m'], outputs['3c']
                ##loss_c1 = self.criterion(c1, targets)
                # loss_c2 = self.criterion(c2, targets)
                # loss_c3 = self.criterion(c3, targets)
                # loss_m1 = self.criterion(m1, torch.tensor([0]).repeat(c1.shape[0]))
                # loss_m2 = self.criterion(m2, torch.tensor([1]).repeat(c1.shape[0]))

                loss = loss_c1 + loss_c2 # + loss_c3

                loss.backward()
                optimizer.step()

                # logging...
                train_loss += loss.item()
                total += targets.size(0)
                # print('{} iteration finished >> Training Loss: c2:{} | c3:{} | m1:{} | m2:{}'.format(batch_idx, loss_c2, loss_c3, loss_m1, loss_m2))
                # print('{} iteration finished >> Training Loss: c1:{} | c2:{} | c3:{}'.format(batch_idx, loss_c1, loss_c2))#,loss_c3))
                print('{} iteration finished >> Training Loss: c1:{} | c2:{}'.format(batch_idx, loss_c1, loss_c2))
            val_loss = 0
            total_val = 0
            with torch.no_grad():
                for val_idx, (rgb_val, ir_val, depth_val, labels_val) in enumerate(self.valloader):
                    inputs1, inputs2, inputs3, targets = rgb_val.to(self.device), ir_val.to(self.device), depth_val.to(
                        self.device), labels_val.to(self.device)

                    outputs = self.net(inputs1, inputs2)
                    # outputs = self.net(inputs1, inputs2, inputs3)
                    # out_all = outputs['out_all']
                    out_all = outputs['feat_2']
                    # out_all, feat1, feat2, feat3 = outputs['out_all'], outputs['feat_1'], outputs['feat_2'], outputs['feat_3']
                    # c1, c2, m1, m2, c3 = outputs['1c'], outputs['2c'], outputs['1m'], outputs['2m'], outputs['3c']
                    # out_all = feat2

                    loss = self.criterion(out_all, targets)
                    val_loss += loss.item()
                    total_val += targets.size(0)

            print('{} epoch >> Training Loss: {} | Validation Loss: {}'.format(i_epoch, train_loss / (batch_idx + 1),
                                                                               val_loss / (val_idx + 1)))
            self.writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, i_epoch)
            with open(train_log_path, 'a') as f:
                f.write('{} epoch >> Training Loss: {} | Validation Loss: {}\n\n'.format(i_epoch,
                                                                                         train_loss / (batch_idx + 1),
                                                                                         val_loss / (val_idx + 1)))
            if min_loss > (val_loss / (val_idx + 1)):
                best_epoch = i_epoch
                min_loss = val_loss / (val_idx + 1)
                self.save(f'./weight/{self.config.model_name}/')

        self.writer.close()
        print('Best epoch: {}'.format(best_epoch))
        with open(train_log_path, 'a') as f:
            f.write('Best epoch: {}'.format(best_epoch))
        self.test(best_epoch)

    def train_two(self):

        self.net.train()
        optimizer = self.init_optimizer
        best_epoch = 0
        min_loss = float('inf')
        train_log_path = f'./weight/{self.config.model_name}/train_log.txt'
        os.makedirs(os.path.dirname(train_log_path),exist_ok=True)
        for i_epoch in range(self.epochs):

            # epoch initialization
            train_loss = 0
            background_loss = 0.
            adv_loss = 0.
            l2_loss = 0.
            correct = 0
            correct_single = 0
            correct_single2 = 0
            total = 0

            # lr change processing...
            if i_epoch == self.lr_change_epoch:
                optimizer = self.last_optimizer

            # batch iterations...
            for batch_idx, ([rgb1, ir1, depth1], [rgb2, ir2, depth2], labels) in enumerate(self.trainloader):
                # to_pil = transforms.ToPILImage()
                # img = to_pil(rgb[0])
                # img.show(title="Image")
                rgb1, ir1, depth1 = rgb1, ir1, depth1
                rgb2, ir2, depth2 = rgb2, ir2, depth2
                inputs1 = torch.cat((rgb1,rgb2),dim=0)
                inputs2 = torch.cat((ir1,ir2),dim=0)
                inputs3 = torch.cat((depth1,depth2),dim=0)
                inputs1, inputs2, inputs3, targets = inputs1.to(self.device), inputs2.to(self.device), inputs3.to(
                    self.device), labels.to(self.device)
                targets2 = torch.cat((targets,targets),dim=0)

                bsz = rgb1.shape[0]

                ## dual image conventional process
                # forward
                optimizer.zero_grad()
                outputs = self.net(inputs1, inputs2)
                # outputs = self.net(inputs1, inputs2, inputs3)
                out_all, feat1, feat2 = outputs['out_all'], outputs['feat_1'], outputs['feat_2']
                # out_all, feat1, feat2, feat3 = outputs['out_all'], outputs['feat_1'], outputs['feat_2'], outputs['feat_3']

                last_feats = outputs['mlp_feats'] # 'rgb', 'ir' # b*2,d
                f1, f2 = torch.split(last_feats['rgb'][0],[bsz,bsz], dim=0)
                features_rgb = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

                f1, f2 = torch.split(last_feats['ir'][0], [bsz, bsz], dim=0)
                features_ir = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

                loss_supcon = self.supcon(features_rgb, targets) + self.supcon(features_ir, targets)
                loss_c1 = self.criterion(feat1, targets2)
                loss_c2 = self.criterion(feat2, targets2)
                # loss_c3 = self.criterion(feat3, targets)
                # loss_c3 = self.criterion(out_all, targets)

                # c1, c2, m1, m2, c3 = outputs['1c'], outputs['2c'], outputs['1m'], outputs['2m'], outputs['3c']
                ##loss_c1 = self.criterion(c1, targets)
                # loss_c2 = self.criterion(c2, targets)
                # loss_c3 = self.criterion(c3, targets)
                # loss_m1 = self.criterion(m1, torch.tensor([0]).repeat(c1.shape[0]))
                # loss_m2 = self.criterion(m2, torch.tensor([1]).repeat(c1.shape[0]))

                loss = loss_c1 + loss_c2 + 0.1 * loss_supcon.float()# + loss_c3

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                optimizer.step()

                # logging...
                train_loss += loss.item()
                total += targets.size(0)
                # print('{} iteration finished >> Training Loss: c2:{} | c3:{} | m1:{} | m2:{}'.format(batch_idx, loss_c2, loss_c3, loss_m1, loss_m2))
                # print('{} iteration finished >> Training Loss: c1:{} | c2:{} | c3:{}'.format(batch_idx, loss_c1, loss_c2))#,loss_c3))
                print('{} iteration finished >> Training Loss: c1:{} | c2:{} | suploss:{}'.format(batch_idx, loss_c1, loss_c2, loss_supcon))
            val_loss = 0
            total_val = 0
            with torch.no_grad():
                for val_idx, (rgb_val, ir_val, depth_val, labels_val) in enumerate(self.valloader):
                    inputs1, inputs2, inputs3, targets = rgb_val.to(self.device), ir_val.to(self.device), depth_val.to(
                        self.device), labels_val.to(self.device)

                    outputs = self.net(inputs1, inputs2)
                    # outputs = self.net(inputs1, inputs2, inputs3)
                    # out_all = outputs['out_all']
                    out_all = outputs['feat_2']
                    # out_all, feat1, feat2, feat3 = outputs['out_all'], outputs['feat_1'], outputs['feat_2'], outputs['feat_3']
                    # c1, c2, m1, m2, c3 = outputs['1c'], outputs['2c'], outputs['1m'], outputs['2m'], outputs['3c']
                    # out_all = feat2

                    loss = self.criterion(out_all, targets)
                    val_loss += loss.item()
                    total_val += targets.size(0)

            print('{} epoch >> Training Loss: {} | Validation Loss: {}'.format(i_epoch, train_loss / (batch_idx + 1),
                                                                               val_loss / (val_idx + 1)))
            with open(train_log_path, 'a') as f:
                f.write('{} epoch >> Training Loss: {} | Validation Loss: {}\n\n'.format(i_epoch,
                                                                                         train_loss / (batch_idx + 1),
                                                                                         val_loss / (val_idx + 1)))
            if min_loss > (val_loss / (val_idx + 1)):
                best_epoch = i_epoch
                min_loss = val_loss / (val_idx + 1)
                self.save(f'./weight/{self.config.model_name}/')

        print('Best epoch: {}'.format(best_epoch))
        with open(train_log_path, 'a') as f:
            f.write('Best epoch: {}'.format(best_epoch))
        self.test(best_epoch)

    def train_contrast(self):
        self.net.train()
        optimizer = self.init_optimizer
        best_epoch = 0
        min_loss = float('inf')
        train_log_path = 'D:/antispoof/result/weight/train_log.txt'

        for i_epoch in range(self.epochs):

            train_loss = 0
            total = 0

            if i_epoch == self.lr_change_epoch:
                optimizer = self.last_optimizer

            # batch iterations...
            for batch_idx, (rgb, ir, depth, labels, text) in enumerate(self.trainloader):
                # to_pil = transforms.ToPILImage()
                # img = to_pil(rgb[0])
                # img.show(title="Image")

                # if batch_idx % 10 == 0:
                # vis.close()
                # vis.images(rgb)
                # vis.images(ir)

                inputs1, inputs2, inputs3, targets = rgb.to(self.device), ir.to(self.device), depth.to(
                    self.device), labels.to(self.device)

                optimizer.zero_grad()
                loss = self.net(inputs1, inputs2, text)

                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                total += targets.size(0)
                # print('{} iteration finished >> Training Loss: c2:{} | c3:{} | m1:{} | m2:{}'.format(batch_idx, loss_c2, loss_c3, loss_m1, loss_m2))
                # print('{} iteration finished >> Training Loss: c1:{} | c2:{} | c3:{}'.format(batch_idx, loss_c1, loss_c2, loss_c3))
                print('{} iteration finished >> Training Loss: {}'.format(batch_idx, loss))

            print('{} epoch >> Training Loss: {} '.format(i_epoch, train_loss / (batch_idx + 1)))
            self.save(f'./weight/{self.config.model_name}/', i_epoch)
            with open(train_log_path, 'a') as f:
                f.write('{} epoch >> Training Loss: {} \n\n'.format(i_epoch, train_loss / (batch_idx + 1)))
        print('Best epoch: {}'.format(best_epoch))
        with open(train_log_path, 'a') as f:
            f.write('Best epoch: {}'.format(best_epoch))

    def test(self, best_epoch):
        state = torch.load(f'./weight/{self.config.model_name}/best.pth', weights_only=True)
        self.net.load_state_dict(state['net'], strict=True)
        self.net.eval()
        test_log_path = f'./weight/{self.config.model_name}/test_log.txt'

        test_loss = 0
        correct = 0
        total = 0
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch_idx, (rgb, ir, depth, labels) in enumerate(self.testloader):
                inputs1, inputs2, inputs3, targets = rgb.to(self.device), ir.to(self.device), depth.to(
                    self.device), labels.to(self.device)

                outputs = self.net(inputs1, inputs2)
                # outputs = self.net(inputs1, inputs2, inputs3)
                # feat1, feat2, out_all = outputs['feat_1'], outputs['feat_2'], outputs['out_all']
                out_all = outputs['feat_2']
                # out_all, feat1, feat2, feat3 = outputs['out_all'], outputs['feat_1'], outputs['feat_2'], outputs['feat_3']
                # c1, c2, m1, m2, c3 = outputs['1c'], outputs['2c'], outputs['1m'], outputs['2m'], outputs['3c']
                # out_all = out_all
                # self.analyze(out_all, feat1, feat2, targets)
                if batch_idx == 0 and outputs.get('rgb_feats',None):
                    self.visualize_feat2TSNE(outputs['rgb_feats'],labels,'rgb.jpg')
                    self.visualize_feat2TSNE(outputs['ir_feats'],labels,'ir.jpg')

                loss = self.criterion(out_all, targets)

                # logging...
                test_loss += loss.item()
                _, predicted = out_all.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                prediction = torch.max(out_all.data, 1)[1]

                y_pred.extend(prediction)
                y_true.extend(targets.flatten().tolist())
                # print(prediction)
                # correct += prediction.eq(labels.data.view_as(prediction)).cpu().sum()

            # self.writer.add_embedding(outputs['attn_feats']['rgb'][-1][:,0], metadata = labels.tolist(), label_img=F.interpolate(inputs2,size=(32,32)), global_step=batch_idx)
            # self.writer.add_images()
            # self.writer.close()
            y_pred = [np.atleast_1d(t.detach().cpu().numpy()) if isinstance(t, torch.Tensor) else np.atleast_1d(t) for t
                      in y_pred]
            y_pred = np.concatenate(y_pred)
            # y_true, y_pred = np.array(y_true), np.array(y_pred)
            metrics = self.calculate_metrics(y_true, y_pred)
            auc_score, min_hter, acer, tpr_at_target_fpr = metrics['AUC'], metrics['Min HTER'], metrics['ACER'], \
                metrics['TPR@FPR=0.1%']
            acc = accuracy_score(y_true, y_pred > 0.5)
            ap = average_precision_score(y_true, y_pred)
            print('accuracy : ', acc, ', average precision : ', ap)
            with open(test_log_path, 'a') as f:
                f.write(
                    'accuracy : {}, average precision : {}, AUC : {}, min HTER : {}, ACER : {}, TPR@FPR=0.1% : {}\n\n'.format(
                        acc, ap, auc_score, min_hter, acer, tpr_at_target_fpr))
        # Test Result
        print('Test Loss: %.3f | Acc: %.3f%% (%d/%d)\n' % (
            test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        with open(test_log_path, 'a') as f:
            f.write('Test Loss: %.3f | Acc: %.3f%% (%d/%d)\n' % (
                test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    def analyze(self, out_all, feat1, feat2, targets):
        probs1 = F.softmax(feat1, dim=1)
        pred1 = probs1.argmax(dim=1)
        probs2 = F.softmax(feat2, dim=1)
        pred2 = probs2.argmax(dim=1)
        # probs3 = F.softmax(feat3, dim=1)
        # pred3 = probs3.argmax(dim=1)
        probs_all = F.softmax(out_all, dim=1)
        pred_all = probs_all.argmax(dim=1)

        df = pd.DataFrame({
            'feat1_0': probs1[:, 0].cpu().detach().numpy(),
            'feat1_1': probs1[:, 1].cpu().detach().numpy(),

            'feat2_0': probs2[:, 0].cpu().detach().numpy(),
            'feat2_1': probs2[:, 1].cpu().detach().numpy(),

            # 'feat3_0': probs3[:, 0].cpu().detach().numpy(),
            # 'feat3_1': probs3[:, 1].cpu().detach().numpy(),

            'out_all_0': probs_all[:, 0].cpu().detach().numpy(),
            'out_all_1': probs_all[:, 1].cpu().detach().numpy(),

            'feat1_pred': pred1.cpu().detach().numpy(),
            'feat2_pred': pred2.cpu().detach().numpy(),
            # 'feat3_pred': pred3.cpu().detach().numpy(),
            'out_all_pred': pred_all.cpu().detach().numpy(),

            'target': targets.cpu().detach().numpy(),
        })
        df.to_csv('./result/diff2.csv', mode='a', index=False)

    def save(self, ckpt_folder):
        # state building...
        state = {'net': self.net.state_dict()}

        # Save checkpoint...
        if not os.path.isdir(ckpt_folder):
            os.mkdir(ckpt_folder)
        torch.save(state, ckpt_folder + '/best.pth')
        return ckpt_folder

    def calculate_metrics(self, y_true, y_pred):
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)

        # AUC
        auc_score = auc(fpr, tpr)

        # HTER (Half Total Error Rate)
        far = fpr
        frr = 1 - tpr
        hter = (far + frr) / 2
        min_hter = np.min(hter)

        # ACER (Average Classification Error Rate)
        apcer = np.mean(y_pred[y_true == 0] > 0.5)  # Attack Presentation Classification Error Rate
        bpcer = np.mean(y_pred[y_true == 1] <= 0.5)  # Bona Fide Presentation Classification Error Rate
        acer = (apcer + bpcer) / 2

        # TPR at FPR = 0.1%
        target_fpr = 0.001
        tpr_at_target_fpr = 0.0
        if np.any(fpr <= target_fpr):
            tpr_at_target_fpr = np.interp(target_fpr, fpr, tpr)

        return {
            'AUC': auc_score,
            'Min HTER': min_hter,
            'ACER': acer,
            'TPR@FPR=0.1%': tpr_at_target_fpr
        }

    def visualize(self, feats: list, name: str, img_num, img):  # [B N D] * num_feats
        ''' batchsize = 1 각 이미지 별로 패치 피처의 평균값을 시각화 '''
        if not os.path.exists(f'D:/antispoof/feat_map/{self.config.weight_name}/{img_num}/'): os.makedirs(
            f'D:/antispoof/feat_map/{self.config.weight_name}/{img_num}/')
        num_feats = len(feats)
        fig, axes = plt.subplots(1, num_feats, figsize=(30, 5))
        img = (img.squeeze().permute(1, 2, 0).numpy() + 1) / 2
        for idx in range(num_feats):
            axes[idx].imshow(img, zorder=1)
            f = feats[idx][:, 1:, :].mean(dim=-1).squeeze()  # N-1
            f = (f - f.min()) / (f.max() - f.min() + 1e-6)
            f = (F.interpolate(f.reshape(14, 14)[None, None, ...], size=(224, 224), mode='bilinear').squeeze()).numpy()
            hm = sns.heatmap(f, cmap='jet', ax=axes[idx], vmin=0.0, vmax=1, zorder=2)
            hm.collections[0].set_alpha(0.45)
        plt.tight_layout()
        plt.savefig(f'D:/antispoof/feat_map/{self.config.weight_name}/{img_num}/{name}.jpg')
        plt.close()

    def visualize_tsne(self, fig, axes, feats: list, name: str, img_num, labels):  # [B N D] * num_feats
        ''' 모든 데이터에 대해 t-sne
        batch > 1해서 batch내에 feature들이 언제 분기 되는지 설명함 '''
        if not os.path.exists(f'D:/antispoof/TSNE/{self.config.weight_name}/'): os.makedirs(
            f'D:/antispoof/TSNE/{self.config.weight_name}/')
        num_feats = len(feats)
        for idx in range(num_feats):  # 각 레이어마다
            f = feats[idx][:, 0, :]  # B D : cls token의 분기점을 설명하려 함 # 대신에 patch feature들의 합이어도 됨
            tsne_results = self.tsne.fit_transform(f.detach().cpu().numpy())
            sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=labels.numpy(), ax=axes[idx])

    def visualize_grad_cam(self, feats: list, name: str, img_num, img):  # [B N D] * num_feats
        ''' batchsize = 1 각 이미지 별로 패치 피처의 평균값을 시각화 '''
        if not os.path.exists(f'D:/antispoof/grad_cam/{self.config.weight_name}/{img_num}/'): os.makedirs(
            f'D:/antispoof/grad_cam/{self.config.weight_name}/{img_num}/')
        num_feats = len(feats)
        fig, axes = plt.subplots(1, num_feats, figsize=(30, 5))
        img = (img.squeeze().permute(1, 2, 0).numpy() + 1) / 2
        for idx in range(num_feats):
            axes[idx].imshow(img, zorder=1)
            f = feats[idx][:, 1:, :].mean(dim=-1).squeeze()  # N-1
            f = (f - f.min()) / (f.max() - f.min() + 1e-6)
            f = (F.interpolate(f.reshape(14, 14)[None, None, ...], size=(224, 224), mode='bilinear').squeeze()).numpy()
            hm = sns.heatmap(f, cmap='jet', ax=axes[idx], vmin=0.0, vmax=1, zorder=2)
            hm.collections[0].set_alpha(0.45)
        plt.tight_layout()
        plt.savefig(f'D:/antispoof/feat_map/{self.config.weight_name}/{img_num}/{name}.jpg')
        plt.close()

    def visualize_feat2TSNE(self, feats:torch.Tensor, labels, path):
        '''
        feats.shape = N,D
        labels.shape = N
        '''
        if not path : path = f'weight/{self.config.model_name}/tsne.jpg'
        else : path = f'weight/{self.config.model_name}/{path}'
        tsne = TSNE(n_components=2,perplexity=30,init='pca',random_state=42)
        tsne_results = tsne.fit_transform(feats.detach().cpu().numpy())
        fig,ax = plt.subplots(1,1,figsize=(10,10))
        sns.scatterplot(x=tsne_results[:, 0], y=tsne_results[:, 1], hue=labels.numpy(),ax=ax)
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

class FigureWrapper:
    def __init__(self, num_feats=5):
        self.fig, self.axes = plt.subplots(1, num_feats, figsize=(30, 5))

    def __call__(self):
        return self.fig, self.axes

    def save(self, path):
        self.fig.tight_layout()
        self.fig.savefig(path)


class train_debug():
    def __init__(self, config, trainDataset, valDataset, testDataset):
        self.config = config
        self.lmd1 = config.lmd1
        self.lmd2 = config.lmd2
        self.lmd3 = config.lmd3

        self.debug = config.debug
        self.epochs = config.epochs
        self.lr_change_epoch = config.lr_change_epoch
        print("config.rand_seed: ", config.rand_seed)
        self.rand_seed = config.rand_seed
        self.batch_size = config.batch_size

        self.test_display = config.test_display
        self.valid_display = config.valid_display

        # random seed arrange
        np.random.seed(config.rand_seed)
        torch.manual_seed(config.rand_seed)
        # sampler = RandomSampler(trainDataset, num_samples=1000, replacement=False)
        # dataloader = torch.utils.data.DataLoader(trainDataset, batch_size=config.batch_size,
        # num_workers=8, pin_memory=True,sampler=sampler, persistent_workers=True)

        dataloader = torch.utils.data.DataLoader(trainDataset, batch_size=config.batch_size, shuffle=True,
                                                 num_workers=8, pin_memory=True, persistent_workers=True)

        testDataloader = torch.utils.data.DataLoader(testDataset, batch_size=config.batch_size, shuffle=False,
                                                     num_workers=8,
                                                     pin_memory=True, persistent_workers=True)

        valDataloader = torch.utils.data.DataLoader(valDataset, batch_size=config.batch_size, shuffle=False,
                                                    num_workers=8,
                                                    pin_memory=True, persistent_workers=True)
        print('dataload end')
        print("train set : {}, val set : {}, test set : {}".format(trainDataset.__len__(), valDataset.__len__(),
                                                                   testDataset.__len__()))
        print(
            "random_seed : {}, batch_size : {}, epochs : {}".format(config.rand_seed, config.batch_size, config.epochs))

        classes = ('real', 'fake')
        (self.trainloader, self.valloader, self.testloader, self.classes) = (
        dataloader, valDataloader, testDataloader, classes)
        # model loading...
        net = model_loader(config.model_name)

        # CUDA & cudnn checking...
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.net = net.to(self.device)
        if self.device == 'cuda':
            self.net = torch.nn.DataParallel(self.net)
            cudnn.deterministic = True
            cudnn.benchmark = False

        self.criterion = nn.CrossEntropyLoss()

        self.init_optimizer = optim.SGD(net.parameters(), lr=config.init_lr, momentum=config.momentum,
                                        weight_decay=config.weight_decay)
        self.last_optimizer = optim.SGD(net.parameters(), lr=config.last_lr, momentum=config.momentum,
                                        weight_decay=config.weight_decay)

        # self.train_contrast()
        self.test('40')

    def train(self):

        self.net.train()
        optimizer = self.init_optimizer
        best_epoch = 0
        min_loss = float('inf')
        train_log_path = 'D:/antispoof/result/weight/train_log.txt'

        for i_epoch in range(self.epochs):

            # epoch initialization
            train_loss = 0
            background_loss = 0.
            adv_loss = 0.
            l2_loss = 0.
            correct = 0
            correct_single = 0
            correct_single2 = 0
            total = 0

            # lr change processing...
            if i_epoch == self.lr_change_epoch:
                optimizer = self.last_optimizer

            # batch iterations...
            for batch_idx, (rgb, ir, depth, labels, text) in enumerate(self.trainloader):
                # to_pil = transforms.ToPILImage()
                # img = to_pil(rgb[0])
                # img.show(title="Image")

                # if batch_idx % 10 == 0:
                # vis.close()
                # vis.images(rgb)
                # vis.images(ir)

                inputs1, inputs2, inputs3, targets = rgb.to(self.device), ir.to(self.device), depth.to(
                    self.device), labels.to(self.device)

                ## dual image conventional process
                # forward
                optimizer.zero_grad()
                outputs = self.net(inputs1, inputs2)
                # outputs = self.net(inputs1, inputs2, inputs3)
                out_all, feat1, feat2 = outputs['out_all'], outputs['feat_1'], outputs['feat_2']
                # out_all = outputs['out_all']
                # out_all, feat1, feat2, feat3 = outputs['out_all'], outputs['feat_1'], outputs['feat_2'], outputs['feat_3']
                loss_c1 = self.criterion(feat1, targets)
                loss_c2 = self.criterion(feat2, targets)
                # loss_c3 = self.criterion(feat3, targets)
                # loss_c3 = self.criterion(out_all, targets)

                # c1, c2, m1, m2, c3 = outputs['1c'], outputs['2c'], outputs['1m'], outputs['2m'], outputs['3c']
                ##loss_c1 = self.criterion(c1, targets)
                # loss_c2 = self.criterion(c2, targets)
                # loss_c3 = self.criterion(c3, targets)
                # loss_m1 = self.criterion(m1, torch.tensor([0]).repeat(c1.shape[0]))
                # loss_m2 = self.criterion(m2, torch.tensor([1]).repeat(c1.shape[0]))

                loss = loss_c1 + loss_c2

                loss.backward()
                optimizer.step()

                # logging...
                train_loss += loss.item()
                total += targets.size(0)
                # print('{} iteration finished >> Training Loss: c2:{} | c3:{} | m1:{} | m2:{}'.format(batch_idx, loss_c2, loss_c3, loss_m1, loss_m2))
                # print('{} iteration finished >> Training Loss: c1:{} | c2:{} | c3:{}'.format(batch_idx, loss_c1, loss_c2, loss_c3))
                print('{} iteration finished >> Training Loss: c1:{} | c2:{}'.format(batch_idx, loss_c1, loss_c2))
            val_loss = 0
            total_val = 0
            with torch.no_grad():
                for val_idx, (rgb_val, ir_val, depth_val, labels_val) in enumerate(self.valloader):
                    inputs1, inputs2, inputs3, targets = rgb_val.to(self.device), ir_val.to(self.device), depth_val.to(
                        self.device), labels_val.to(self.device)

                    outputs = self.net(inputs1, inputs2)
                    # outputs = self.net(inputs1, inputs2, inputs3)
                    out_all, feat1, feat2 = outputs['out_all'], outputs['feat_1'], outputs['feat_2']
                    # out_all = outputs['out_all']
                    # out_all, feat1, feat2, feat3 = outputs['out_all'], outputs['feat_1'], outputs['feat_2'], outputs['feat_3']
                    # c1, c2, m1, m2, c3 = outputs['1c'], outputs['2c'], outputs['1m'], outputs['2m'], outputs['3c']
                    out_all = feat2

                    loss = self.criterion(out_all, targets)
                    val_loss += loss.item()
                    total_val += targets.size(0)

            print('{} epoch >> Training Loss: {} | Validation Loss: {}'.format(i_epoch, train_loss / (batch_idx + 1),
                                                                               val_loss / (val_idx + 1)))
            self.save(f'./weight/{self.config.model_name}/', i_epoch)
            with open(train_log_path, 'a') as f:
                f.write('{} epoch >> Training Loss: {} | Validation Loss: {}\n\n'.format(i_epoch,
                                                                                         train_loss / (batch_idx + 1),
                                                                                         val_loss / (val_idx + 1)))
            if min_loss > (val_loss / (val_idx + 1)):
                best_epoch = i_epoch
                min_loss = val_loss / (val_idx + 1)
        print('Best epoch: {}'.format(best_epoch))
        with open(train_log_path, 'a') as f:
            f.write('Best epoch: {}'.format(best_epoch))
        self.test(best_epoch)

    def train_contrast(self):
        self.net.train()
        optimizer = self.init_optimizer
        best_epoch = 0
        min_loss = float('inf')
        train_log_path = 'D:/antispoof/result/weight/train_log.txt'

        for i_epoch in range(self.epochs):

            train_loss = 0
            total = 0

            if i_epoch == self.lr_change_epoch:
                optimizer = self.last_optimizer

            # batch iterations...
            for batch_idx, (rgb, ir, depth, labels, text) in enumerate(self.trainloader):
                # to_pil = transforms.ToPILImage()
                # img = to_pil(rgb[0])
                # img.show(title="Image")

                # if batch_idx % 10 == 0:
                # vis.close()
                # vis.images(rgb)
                # vis.images(ir)

                inputs1, inputs2, inputs3, targets = rgb.to(self.device), ir.to(self.device), depth.to(
                    self.device), labels.to(self.device)

                optimizer.zero_grad()
                loss = self.net(inputs1, inputs2, text)

                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                total += targets.size(0)
                # print('{} iteration finished >> Training Loss: c2:{} | c3:{} | m1:{} | m2:{}'.format(batch_idx, loss_c2, loss_c3, loss_m1, loss_m2))
                # print('{} iteration finished >> Training Loss: c1:{} | c2:{} | c3:{}'.format(batch_idx, loss_c1, loss_c2, loss_c3))
                print('{} iteration finished >> Training Loss: {}'.format(batch_idx, loss))

            print('{} epoch >> Training Loss: {} '.format(i_epoch, train_loss / (batch_idx + 1)))
            self.save(f'./weight/{self.config.model_name}/', i_epoch)
            with open(train_log_path, 'a') as f:
                f.write('{} epoch >> Training Loss: {} \n\n'.format(i_epoch, train_loss / (batch_idx + 1)))
        print('Best epoch: {}'.format(best_epoch))
        # with open(train_log_path, 'a') as f:
        # f.write('Best epoch: {}'.format(best_epoch))

    def test(self, best_epoch):

        state = torch.load('D:/antispoof/result/ca2 all/model_ckpt_{}.pth'.format(best_epoch))
        # state = torch.load('D:/antispoof/result/weight/model_ckpt_49.pth')
        self.net.load_state_dict(state['net'], strict=True)
        self.net.eval()
        test_log_path = 'D:/antispoof/result/weight/test_log.txt'
        test_loss = 0
        correct = 0
        total = 0
        y_true, y_pred = [], []
        with torch.no_grad():
            # batch iterations...
            for batch_idx, (rgb, ir, depth, labels) in enumerate(self.testloader):
                inputs1, inputs2, inputs3, targets = rgb.to(self.device), ir.to(self.device), depth.to(
                    self.device), labels.to(self.device)

                outputs = self.net(inputs1, inputs2)
                # outputs = self.net(inputs1, inputs2, inputs3)
                out_all, feat1, feat2 = outputs['out_all'], outputs['feat_1'], outputs['feat_2']
                # out_all = outputs['out_all']
                # out_all, feat1, feat2, feat3 = outputs['out_all'], outputs['feat_1'], outputs['feat_2'], outputs['feat_3']

                # c1, c2, m1, m2, c3 = outputs['1c'], outputs['2c'], outputs['1m'], outputs['2m'], outputs['3c']
                out_all = feat2
                # self.analyze(out_all, feat1, feat2, targets)

                loss = self.criterion(out_all, targets)

                # logging...
                test_loss += loss.item()
                _, predicted = out_all.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                prediction = torch.max(out_all.data, 1)[1]

                y_pred.extend(prediction)
                y_true.extend(targets.flatten().tolist())
                # print(prediction)
                # correct += prediction.eq(labels.data.view_as(prediction)).cpu().sum()
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            metrics = self.calculate_metrics(y_true, y_pred)
            auc_score, min_hter, acer, tpr_at_target_fpr = metrics['AUC'], metrics['Min HTER'], metrics['ACER'], \
                metrics['TPR@FPR=0.1%']
            acc = accuracy_score(y_true, y_pred > 0.5)
            ap = average_precision_score(y_true, y_pred)
            print('accuracy : ', acc, ', average precision : ', ap)
            # with open(test_log_path, 'a') as f:
            # f.write('accuracy : {}, average precision : {}, AUC : {}, min HTER : {}, ACER : {}, TPR@FPR=0.1% : {}\n\n'.format(acc, ap, auc_score, min_hter, acer, tpr_at_target_fpr))
        # Test Result
        print('Test Loss: %.3f | Acc: %.3f%% (%d/%d)\n' % (
            test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        # with open(test_log_path, 'a') as f:
        # f.write('Test Loss: %.3f | Acc: %.3f%% (%d/%d)\n' % (
        # test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    def analyze(self, out_all, feat1, feat2, targets):
        probs1 = F.softmax(feat1, dim=1)
        pred1 = probs1.argmax(dim=1)
        probs2 = F.softmax(feat2, dim=1)
        pred2 = probs2.argmax(dim=1)
        # probs3 = F.softmax(feat3, dim=1)
        # pred3 = probs3.argmax(dim=1)
        probs_all = F.softmax(out_all, dim=1)
        pred_all = probs_all.argmax(dim=1)

        df = pd.DataFrame({
            'feat1_0': probs1[:, 0].cpu().detach().numpy(),
            'feat1_1': probs1[:, 1].cpu().detach().numpy(),

            'feat2_0': probs2[:, 0].cpu().detach().numpy(),
            'feat2_1': probs2[:, 1].cpu().detach().numpy(),

            # 'feat3_0': probs3[:, 0].cpu().detach().numpy(),
            # 'feat3_1': probs3[:, 1].cpu().detach().numpy(),

            'out_all_0': probs_all[:, 0].cpu().detach().numpy(),
            'out_all_1': probs_all[:, 1].cpu().detach().numpy(),

            'feat1_pred': pred1.cpu().detach().numpy(),
            'feat2_pred': pred2.cpu().detach().numpy(),
            # 'feat3_pred': pred3.cpu().detach().numpy(),
            'out_all_pred': pred_all.cpu().detach().numpy(),

            'target': targets.cpu().detach().numpy(),
        })
        df.to_csv('D:/antispoof/result/ca2.csv', mode='a', index=False)

    def save(self, ckpt_folder, epoch):
        # state building...
        state = {'net': self.net.state_dict()}

        # Save checkpoint...
        if not os.path.isdir(ckpt_folder):
            os.mkdir(ckpt_folder)
        torch.save(state, ckpt_folder + '/model_ckpt_{}.pth'.format(epoch))
        return ckpt_folder

    def calculate_metrics(self, y_true, y_pred):
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)

        # AUC
        auc_score = auc(fpr, tpr)

        # HTER (Half Total Error Rate)
        far = fpr
        frr = 1 - tpr
        hter = (far + frr) / 2
        min_hter = np.min(hter)

        # ACER (Average Classification Error Rate)
        apcer = np.mean(y_pred[y_true == 0] > 0.5)  # Attack Presentation Classification Error Rate
        bpcer = np.mean(y_pred[y_true == 1] <= 0.5)  # Bona Fide Presentation Classification Error Rate
        acer = (apcer + bpcer) / 2

        # TPR at FPR = 0.1%
        target_fpr = 0.001
        tpr_at_target_fpr = 0.0
        if np.any(fpr <= target_fpr):
            tpr_at_target_fpr = np.interp(target_fpr, fpr, tpr)

        return {
            'AUC': auc_score,
            'Min HTER': min_hter,
            'ACER': acer,
            'TPR@FPR=0.1%': tpr_at_target_fpr
        }