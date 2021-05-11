# CFGA_CARS196_train
This is the set of  trained procedures of 'Convolution Filters Gradient Aggregation for Fine-grained Image Retrieval', and the corresponding dataset is CARS196


0-0:
---------------------------------
**requirements:**

python 3.6.13

pytorch 1.9.0 (i.e., nightly version) + torchvision 0.10.0 (i.e., nightly version) + cuda 11.1

numpy 1.19.2

*********************************************************************************************************************************************************************************
0-1:
---------
all trained models of this work can be got in https://drive.google.com/drive/folders/17qfulk7GsDxmIb21Df_M-_pvWP1lwLTO

As a result, you can simply download the above provided trained model and go to https://github.com/mikiyukio/CFGA_CARS196_test , which is the core of our released code.
********************************************************************************************************************************************
1:
------------------------------------------------
https://download.pytorch.org/models/resnet50-19c8e357.pth

please download the resnet50 pretrained model from the above link，and save it in .\SCDA_cars_resnet50\model

2:
------------------------------------------
download CARS196 dataset from https://ai.stanford.edu/~jkrause/cars/car_dataset.html

after you've downloaded it, go into **.\Stanford car dataset\car_ims** of the downloaded dataset, and creat a new folder named as 'cars_196' 

3:
--------------------------------------------------------------------------------
.\SCDA_cars_resnet50\files_cars196.py is the first procrdure you need to run. But before you run it, please change the following code exists in .\SCDA_cars_resnet50\files_cars196.py first. 

(1)

`parser.add_argument('--datasetdir', default=r'/home/ggh/lzy/Stanford car dataset/car_ims',  help="cars196 train_and_test images")`


 replace `r'/home/ggh/lzy/Stanford car dataset/car_ims'` with the path you save CARS196 dataset , e.g.,`r'C:\Users\于涵\Desktop\Stanford car dataset\car_ims'`
 
 (2)
 
` parser.add_argument('--annotationsdir', default=r'/home/ggh/lzy/Stanford car dataset/car_devkit/devkit',  help="cars196 train_and_test annotations")`

replace `r'/home/ggh/lzy/Stanford car dataset/car_devkit/devkit'` with the path you save CARS196 dataset , e.g.,`r'C:\Users\于涵\Desktop\Stanford car dataset\car_devkit\devkit'`
 
 (3)
 
`parser.add_argument('--targetdatasetdir', default=r'/home/ggh/lzy/Stanford car dataset/car_ims/cars_196',  help="cars196 train_and_test images")`

replace `r'/home/ggh/lzy/Stanford car dataset/car_ims/cars_196'` with the path you save CARS196 dataset , e.g.,`r'C:\Users\于涵\Desktop\Stanford car dataset\car_ims\cars_196'`
 
 
 
 
 

===========
after you have run this procedure, four files with json format will be generated in .\SCDA_cub_resnet50\datafile, which will be named as 

train_labels.json, train_paths.json, test_labels.json, test_paths.json

respectively.

4:
----------------------------------------------------------------------------------
Let's go to .\SCDA_cub_resnet50\ dataset

***The following procedures all learn 512-dimensional embedding***

train_DGPCRL.py ====> corresponding to DGPCRL baseline 

train_cosface.py ====> corresponding to cosface baseline 

train_arcface.py ====> corresponding to arcface baseline 

train_msloss.py ====> corresponding to msloss  baseline 

***The following procedures all learn 1024-dimensional embedding***

train_cosface_1024.py ====> corresponding to cosface baseline 

train_arcface_1024.py ====> corresponding to arcface baseline 

train_msloss_1024.py ====> corresponding to msloss baseline


if you have run any above train procedures and want to evaluate the performance of **these baselines/ the corresponding CFGA features**, please take your trained models and go to https://github.com/mikiyukio/CFGA_CUB-200-2011_test for more information.

5:
