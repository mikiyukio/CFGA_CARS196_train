#对于部分预训练的VGG16风格的网络进行微调
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
import argparse
from os.path import join
import uuid
import time
import json
from net_res50 import FGIAnet100_metric5,FGIAnet100_metric5_2
import dataloader


parser = argparse.ArgumentParser()

parser.add_argument("--lr", type=float, dest="lr", help="ADAM Learning rate", default=0.0004)
parser.add_argument("--lr_step", type=int, dest="lr_step", help="Learning rate decrease step", default=100)
parser.add_argument("--lr_decay", type=float, dest="lr_decay", help="ADAM Learning rate decay (on step)", default=0.1)
parser.add_argument("--eps", type=float, dest="eps", help="ADAM epsilon parameter", default=1e-3)
parser.add_argument("--num_epochs", type=int, dest="num_epochs", help="Total number of epochs to train", default=111)
parser.add_argument("--num_classes", type=int, dest="num_classes", help="Total number of epochs to train", default=100)
parser.add_argument('--img_size', type=int, default = 280,help='训练阶段你想将图片resize为多少')
parser.add_argument("--patch_size", type=int, dest="patch_size", help="Total number of epochs to train", default=128)
parser.add_argument("--batch_size", type=int, dest="batch_size", help=" ", default=60)#最初14
parser.add_argument('--test_freq', default=1, type=int, required=False, help="Frequency to run over test_dataset")
parser.add_argument('-start_epoch', default=0, type=int, help="")
parser.add_argument('--savedir',default='./models/', help="Path to save weigths and logs")
parser.add_argument('-global_lr', default=0.001, type= float, required=False)#最初0.0001
parser.add_argument('--grad_update_freq', default=1, type=int, required=False, help="经过多少次反向传播更新一次梯度，每次反向传播对应batch_size张图片")#最初1
parser.add_argument('--max_size', type=int, default = 448,help='进行特征提取的图片的尺寸的上界所对应的数量级')

args = parser.parse_args()

lr = args.lr#学习率
global_lr=args.global_lr
eps = args.eps#与梯度更新有关的一个参数
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
# loaders = dataloader.get_dataloaders(train_paths, test_paths,train_labels,test_labels,args.img_size,train_batch_size=args.batch_size,test_batch_size=1, flag=1, SCDA_flag=0)#返回值为由可迭代DataLoader对象所组成的字典
loaders = dataloader.get_dataloaders(train_paths, train_paths,train_labels,train_labels,args.img_size,train_batch_size=args.batch_size,test_batch_size=1, flag=1, SCDA_flag=0)#返回值为由可迭代DataLoader对象所组成的字典
#################################################################################

################################################################################33
#其实这两行本质上没有区别，只是说FGIA100_metric5_1是对于除backzone之外对于DGCRL的完全复现
# model = FGIAnet100_metric5().cuda()#avg
model = FGIAnet100_metric5_2(scale=100).cuda()#max+avg    #弱监督检索

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
#for key,value in model_dict.items():
        #打印 key value字典
 #       print(key,'\t',model.state_dict()[key].size())
# print(model)
# print(len(model.img_features))
# print(model.img_features[27])
# print(model.img_features[30])


# save_model = torch.load(join(args.savedir,'vgg19.pth'))
# save_model = torch.load(join(args.savedir,'vgg16-397923af.pth'))
save_model = torch.load(join(args.savedir,'resnet50-19c8e357.pth'))

#print(save_model.keys())
model_dict =  model.state_dict()
state_dict = {k:v for k,v in save_model.items() if k in model_dict.keys()}
#print(state_dict.keys())  # dict_keys(['w', 'conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias'])
model_dict.update(state_dict)#这样就对我们自定义网络的cnn部分的参数进行了更新，更新为vgg16网络中cnn部分的参数值
model.load_state_dict(model_dict)
#print(model_dict.keys())





optimizer = optim.SGD(model.parameters(), lr=global_lr,
                              momentum=0.9, weight_decay=0.000005)
# optimizer = optim.Adam(model.parameters(), lr=global_lr, eps=eps,weight_decay=0.000005)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step, gamma=lr_decay)
#torch.optim.lr_scheduler provides several methods to adjust the learning rate based on the number of epochs.
#StepLR：Sets the learning rate of each parameter group to the initial lr decayed by gamma every step_size epochs.
#也就是说每step_size个epochs，将学习率调整为原来的grama倍
#{train:.......，test:.........}
criteria = nn.CrossEntropyLoss()
center_criterion = nn.MSELoss()
criteria_softmax = nn.Softmax(dim=1)
criteria_NLLLoss = nn.NLLLoss()




best_acc=0.0



def train(best_acc):
    print("Starting training....")
    curr_acc = 0.0
    losses = 0.0
    loss_list=[]
    train_acc, test_acc = [], []
    for epoch in range(args.start_epoch, num_epochs):
        total_loss = 0.
        # scheduler.step()
        corrects = 0##corrects对所有batch的预测正确的数量进行一个累加,并且在每一次新的epoch清零，记录本次训练epoch中正确预测的数量
        start = time.time()
        for idx, (images, labels) in enumerate(loaders['train']):
            model.train()
            # getitem()每一次迭代的输出都是一个三维的元组，原始图像【3，448，448】,图像的标签，该图像的一个patch【1,3,224,224】
            # 但是train_loader的每一次迭代相当于执行了batch_size次的__getitem__()    [batch_size,3,488,488],[batch_size,1],[batch_size,1,3,224,224]
            # [batch_size,1,3,224,224] 的意思是一次处理batch_size张图片，对于每一张图片，我们选出一个【3,224,224】的patch
            # save_preprocessed_img2('./testing_loader/path_{}_{}.png'.format(index, labels[epoch].item()), patches_batch, index)
            images = images.cuda()
            labels=labels.cuda()
            if (idx + 1) % args.grad_update_freq == 0:
                optimizer.zero_grad()
            _,img_preds,img_classifier_weight = model(images)  # (N, 200)
            #print(img_classifier_weight.size())
            #print(img_classifier_weight.requires_grad)


            index_sample = torch.arange(0, images.size()[0]).long()  # 真值标签的下标 [batch]
            index_class = labels.view(-1).long()  # 真值标签，即所属类别 [batch]
            # score1 = torch.zeros_like(img_preds).cuda()
            # score1[index_sample, index_class] = 4
            # score = img_preds - score1
            score=img_preds
            #1
            #score_exp=torch.exp(score)
            #score_exp_sum = torch.sum(score_exp, dim=1)
            #score_exp_sum=score_exp_sum.unsqueeze(0).T
            #prob=score_exp/score_exp_sum
            #2
            prob=criteria_softmax(score)

            # print(prob.size())
            # print(torch.sum(prob,dim=1))

            prob_bool=prob[index_sample, index_class]>=0.7
            prob_bool=prob_bool.float()
            prob_bool_2 = prob[index_sample, index_class] < 0.7
            prob_bool_2 = -1 * prob_bool_2.float()
            bool_prob=prob_bool + prob_bool_2
            # print(bool_prob)
            score2 = torch.zeros_like(img_preds).cuda()
            score2[index_sample, index_class] = bool_prob
            # print(index_class)
            # print(index_sample)
            # score2=score2.requires_grad_(False)
            # print(score2.requires_grad)
            
            #1
            #loss=torch.sum(score2 * torch.log(prob))/score2.size()[0]
            #2
            loss=criteria_NLLLoss(-1*score2 * torch.log(prob),labels)

            # print(loss)
            #
            # loss11 = criteria(score, labels)
            # print(loss11)
            # print(score2.requires_grad)
            # print(prob.requires_grad)
            #
            # print(score2.size()[0])



            # loss = criteria(score, labels)
            # print(self.id_agent.data.size())
            # print(torch.mm(self.id_agent.data, torch.t(self.id_agent.data)))
            # center_loss = criteria(
            #     torch.mm(img_classifier_weight, torch.t(img_classifier_weight)).cuda(),
            #     torch.arange(100).cuda().long())
            center_loss = center_criterion(torch.mm(img_classifier_weight,torch.t(img_classifier_weight)),torch.eye(98).cuda().float())

            # print(losses)
            # print(img_classifier_weight.size())
            losses = loss + 0.1 * center_loss
            losses = losses / args.grad_update_freq
##########################3
            # batch_size = labels.size()[0]
            # labels1 = torch.zeros((batch_size, 100)).scatter_(1, labels.cpu().view((batch_size, -1)), 1)
            # img_preds = img_preds - 4 * labels1.cuda()
            ###################################################
            losses.backward()
            if (idx + 1) % args.grad_update_freq == 0:
                optimizer.step()
            total_loss += losses.item()  #item()将tensor转化为单纯的数
            weighted_logits = score
            predicts = torch.argmax(weighted_logits, 1)
            corrects += torch.sum(predicts == labels).item()
            #torch.sum(predicts == labels).item()计算出了这个batch的图片中的预测正确的数量
            #corrects对所有batch的预测正确的数量进行一个累加,并且在每一次新的epoch清零，记录本次训练epoch中正确预测的数量
            loss_list.append(losses.item())

            if (idx % 50) == 0:
                print("Epoch {} Iter {} Avg error on train dataset till now: {:.3f} ".format(epoch, idx, total_loss / (idx + 1)))
                print(prob[index_sample, index_class])
                print(bool_prob)

                #每经过200个batch，计算一次平局的损失函数，并且呢，每一次新的epoch也是重新开始
        # print("Number of corrects: {}".format(corrects.item()))
        curr_train_acc = corrects / total_train_images    #每当遍历一次训练集，便重新计算它一次，并将本次epoch的平均正确率存储到列表中
        train_acc.append(curr_train_acc)
        print("Epoch {} Accuracy {:.4f} time taken {:.3f}".format(epoch, curr_train_acc, (time.time() - start)))#输出本次epoch的平均分类正确率以及训练的总时间
        scheduler.step()
        if epoch in [1,90,95,100,105,110]:
            torch.save({'start_epoch': epoch + 1,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_acc': curr_train_acc
                        }, args.savedir + '/checkpoint_' + str(epoch) + "_" + str(curr_train_acc)[0:6] + '.pth',_use_new_zipfile_serialization=False)

train(best_acc)


# 假设网络为model = Net(), optimizer = optim.Adam(model.parameters(), lr=args.lr), 假设在某个epoch，我们要保存模型参数，优化器参数以及epoch
# 一、
# 1. 先建立一个字典，保存三个参数：
# state = {‘net':model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch':epoch}
# 2.调用torch.save():
# torch.save(state, dir)
# 其中dir表示保存文件的绝对路径+保存文件名，如'/home/qinying/Desktop/modelpara.pth'
# 二、
# 当你想恢复某一阶段的训练（或者进行测试）时，那么就可以读取之前保存的网络模型参数等。
# checkpoint = torch.load(dir)
# model.load_state_dict(checkpoint['net'])
# optimizer.load_state_dict(checkpoint['optimizer'])
# start_epoch = checkpoint['epoch'] + 1







