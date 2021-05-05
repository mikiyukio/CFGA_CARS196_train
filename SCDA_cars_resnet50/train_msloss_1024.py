#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 15:45:46 2021

@author: ggh
"""

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
import argparse
from os.path import join
import uuid
import time
import json
from about_msloss.net_resnet50_1024 import FGIAnet100_metric5_ms
from about_msloss import dataloader_msloss
from about_msloss.ms_loss import MultiSimilarityLoss
from about_msloss.lr_schedule import WarmupMultiStepLR
import os
#from circle_loss import CircleLoss,convert_label_to_similarity

os.environ['CUDA_VISIBLE_DEVICES']='1'



parser = argparse.ArgumentParser()

parser.add_argument("--lr_step", type=int, dest="lr_step", help="Learning rate decrease step", default=100)
parser.add_argument("--lr_decay", type=float, dest="lr_decay", help="ADAM Learning rate decay (on step)", default=0.1)
parser.add_argument("--num_epochs", type=int, dest="num_epochs", help="Total number of epochs to train", default=1)
parser.add_argument('--img_size', type=int, default = 256,help='训练阶段你想将图片resize为多少')
parser.add_argument("--batch_size", type=int, dest="batch_size", help=" ", default=80)#应当为80
parser.add_argument('--test_freq', default=1, type=int, required=False, help="Frequency to run over test_dataset")
parser.add_argument('-start_epoch', default=0, type=int, help="")
parser.add_argument('--savedir',default='./models/', help="Path to save weigths and logs")
parser.add_argument('-global_lr', default=0.0001, type= float, required=False)#最初0.0001
parser.add_argument('--grad_update_freq', default=1, type=int, required=False, help="经过多少次反向传播更新一次梯度，每次反向传播对应batch_size张图片")#最初1


args = parser.parse_args()


global_lr=args.global_lr
lr_decay = args.lr_decay
lr_step = args.lr_step
num_epochs = args.num_epochs




train_paths_name=join('./datafile/train_paths.json')
test_paths_name=join('./datafile/test_paths.json')
train_labels_name=join('./datafile/train_labels.json')
test_labels_name=join('./datafile/test_labels.json')
# print(train_paths_name)
with open(train_paths_name) as miki:
    train_paths = json.load(miki)
with open(test_paths_name) as miki:
    test_paths = json.load(miki)
with open(train_labels_name) as miki:
    train_labels = json.load(miki)
with open(test_labels_name) as miki:
    test_labels = json.load(miki)
print(len(train_paths))
print(len(train_labels))
# print(len(test_paths))
# print(len(test_labels))
##############################################################################
loaders = dataloader_msloss.get_dataloaders(train_paths, train_paths,train_labels,train_labels,args.img_size,train_batch_size=args.batch_size,test_batch_size=1, flag=1, SCDA_flag=0)#返回值为由可迭代DataLoader对象所组成的字典
#################################################################################
print(len(loaders['train']))   #3000
################################################################################33
#其实这两行本质上没有区别，只是说FGIA100_metric5_1是对于除backzone之外对于DGCRL的完全复现
model = FGIAnet100_metric5_ms().cuda()#max+avg    #弱监督检索

print(model)
####################################################################################33

# print(model)
print("Total train images: {}".format(len(loaders['train'].dataset)))
print("Total test images: {}".format(len(loaders['test'].dataset)))
total_train_images = len(loaders['train'].dataset)
total_test_images = len(loaders['test'].dataset)


model_dict = model.state_dict()
# for param_tensor in model_dict:  #相当于隐式的执行了for param_tensor in model_dict.keys():
#         打印 key value字典
        # print(param_tensor,'\t',model.state_dict()[param_tensor].size())
for key,value in model_dict.items():
        #打印 key value字典
        print(key,'\t',model.state_dict()[key].size())

save_model = torch.load(join(args.savedir,'resnet50-19c8e357.pth'))

print(save_model.keys())
model_dict =  model.state_dict()
state_dict = {k:v for k,v in save_model.items() if k in model_dict.keys()}
print(state_dict.keys())  # dict_keys(['w', 'conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias'])
model_dict.update(state_dict)#这样就对我们自定义网络的cnn部分的参数进行了更新，更新为vgg16网络中cnn部分的参数值
model.load_state_dict(model_dict)
print(model_dict.keys())


criteria=MultiSimilarityLoss()


def build_optimizer(model):
    params = []
    params_look=[]
    for key, value in model.named_parameters():
        print(key)
        if not value.requires_grad:
            continue
        lr_mul = 0.1
        if  key in ["class_fc.weight",'class_fc.bias','class_fc_2.weight','class_fc_2.bias']:
            lr_mul = 1.0
        params += [{"params": [value], "lr_mul": lr_mul}]
        params_look += [{"params": key, "lr_mul": lr_mul}]

    optimizer = getattr(torch.optim, 'Adam')(params,lr=0.00003,weight_decay=0.0005)
    print(params_look)
    return optimizer

def build_lr_scheduler(optimizer):
    return WarmupMultiStepLR(
        optimizer,
        [1200, 2400],
        0.1,
        warmup_factor=0.01,
        warmup_iters=0,
        warmup_method='linear',
    )


optimizer=build_optimizer(model)
scheduler = build_lr_scheduler(optimizer)

#
# #0.001 0.4 80 40 不行
# optimizer = optim.SGD(model.parameters(), lr=global_lr,
#                               momentum=0.9, weight_decay=0.00005)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_decay)


# center_criterion = nn.MSELoss()

best_acc=0.0

# print(model.parameters)

# for param in model.parameters():
#     print(param.size())#第一个param就是100*512的类别代理

def set_bn_evel(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def train(best_acc):
    print("Starting training....")
    losses = 0.0
    loss_list=[]
    for epoch in range(args.start_epoch, num_epochs):
        total_loss = 0.
        # scheduler.step()
        start = time.time()
        for idx, (images, labels) in enumerate(loaders['train']):
            #print(images.size())
            #print(labels.size())
            #print(labels)
            model.train()
            model.apply(set_bn_evel)
            images = images.cuda()
            labels=labels.cuda()
            if (idx+1)%args.grad_update_freq == 0:
                optimizer.zero_grad()
            _,img_preds = model(images)  # (N, 200)
            #print(img_preds.size())
            # print(img_preds.size())
            # print(img_classifier_weight.size())
            # print(labels.size())
            # print(torch.mm(weight, torch.t(weight)))
            # print(weight)
            # print(weight.requires_grad)
            score=img_preds
            # loss = criteria(score, labels)
            # loss = criteria(*convert_label_to_similarity(score, labels))
            loss = criteria(score , labels)
            # print(weight[0][0:10])
            # print(weight.size())
            # img_preds = nn.functional.linear(img_preds, nn.functional.normalize(weight, p=2, dim=1, eps=1e-12))
            # print(img_preds)
            # print(self.id_agent.data.size())
            # print(torch.mm(weight, torch.t(weight)))

            # print(losses)
            losses = loss
            losses = losses / args.grad_update_freq
            losses.backward()
            if (idx+1)%args.grad_update_freq == 0:
                optimizer.step()
            total_loss += losses.item()  #item()将tensor转化为单纯的数
            loss_list.append(losses.item())
            scheduler.step()
            if (idx % 50) == 0:
                print("Iter {} error on train dataset till: {:.3f} ".format(idx, losses))   #每经过200个batch，计算一次平局的损失函数，并且呢，每一次新的epoch也是重新开始
                # print("Number of corrects: {}".format(corrects.item()))
                print("Iter {} Accuracy time taken {:.3f}".format(idx, (time.time() - start)))#输出本次epoch的平均分类正确率以及训练的总时间
        # scheduler.step()
            if idx==2800 or idx == 2999:
                torch.save({'model': model.state_dict(),
                            #'optimizer': optimizer.state_dict(),
                            }, args.savedir + '/checkpoint_' + str(idx) + "_" + str(losses.item())[0:6] + '.pth',_use_new_zipfile_serialization=False)

train(best_acc)





