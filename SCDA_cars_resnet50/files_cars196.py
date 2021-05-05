import scipy.io
import os
import re
import shutil
import argparse
import json
#please download the cars_annos.mat from https://ai.stanford.edu/~jkrause/cars/car_dataset.html,
# which contain the labels of testdataset,and please it at Stanford car dataset\car_devkit\devkit
parser = argparse.ArgumentParser()
parser.add_argument('--datasetdir', default=r'/home/ggh/lzy/Stanford car dataset/car_ims',  help="cars196 train_and_test images")
# C:\Users\于涵\Desktop\Stanford car dataset\car_devkit\devkit
parser.add_argument('--annotationsdir', default=r'/home/ggh/lzy/Stanford car dataset/car_devkit/devkit',  help="cars196 train_and_test annotations")
#需要新建一个cars_196的文件夹，这个文件夹是类似于cub数据集那样子，分类别存储着数据的
parser.add_argument('--targetdatasetdir', default=r'/home/ggh/lzy/Stanford car dataset/car_ims/cars_196',  help="cars196 train_and_test images")

args = parser.parse_args()
source = args.datasetdir
# target = './cars196/'
target=args.targetdatasetdir
data = scipy.io.loadmat(os.path.join(args.annotationsdir,'cars_annos.mat'))
class_names = data['class_names']
# print(class_names)
annotations = data['annotations']
print(data)
#print(annotations)
train_paths=[]
test_paths=[]
train_labels=[]
test_labels=[]
ii=0
j=0
for i in range(annotations.shape[1]):
    name = str(annotations[0, i][0])[2:-2]#提取出图片名字
    image_path = os.path.join(source, name)
    print(image_path)
    clas = int(annotations[0, i][5])#提取出图片的类别属性

    class_name = str(class_names[0, clas-1][0]).replace(' ', '_')
    class_name = class_name.replace('/', '')#避免路径出错
    target_path = os.path.join(target, str(clas)+class_name)
    if not os.path.isdir(target_path):
        os.mkdir(target_path)
    print(target_path)
    if clas <= 98:
        ii=ii+1
        train_labels.append(clas-1)
        # print(name)
        train_paths.append(os.path.join(target_path,name[8:]))
    else:
        j=j+1
        test_labels.append(clas-1)
        test_paths.append(os.path.join(target_path, name[8:]))
    shutil.copy(image_path, target_path)

filename1='./datafile/train_labels.json'
filename2='./datafile/test_labels.json'
with open(filename1,'w') as f_obj:
    json.dump(train_labels,f_obj)
with open(filename2,'w') as f_obj:
    json.dump(test_labels,f_obj)

filename3='./datafile/train_paths.json'
filename4='./datafile/test_paths.json'
with open(filename3,'w') as f_obj:
    json.dump(train_paths,f_obj)
with open(filename4,'w') as f_obj:
    json.dump(test_paths,f_obj)
print('训练集数量 ： ',ii)
print('测试集数量 ： ',j)