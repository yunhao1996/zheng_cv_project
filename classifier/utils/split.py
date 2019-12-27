import random 
import argparse
import os
import numpy as np
import shutil
random.seed(999)

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, help='path to the dataset', default = '/home/ouc/dataset/dtd-r1.0.1/dtd/images/')
parser.add_argument('--output', type=str, help='path to the file list' , default = '/home/ouc/dataset/dtd-r1.0.1/dtd/test/')
args = parser.parse_args()

file_all = os.listdir(args.path)
for file in file_all: 
    img_all = os.listdir(os.path.join(args.path,file))  # 列出路径下的图片名
    img_number=len(img_all)
    rate=0.2           
    picknumber=int(img_number*rate)    # 按照rate比例从文件夹中取一定数量图片
    sample = random.sample(img_all, picknumber)  # 随机选取picknumber数量的样本图片
    print (sample)
    
    if (not os.path.isdir(args.output+file)):
        os.mkdir(args.output+file)

    for name in sample:

        shutil.move(args.path+file+'/'+name, args.output+file+'/'+name)
    

