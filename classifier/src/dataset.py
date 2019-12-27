import os 
import glob
import scipy
import torch
import random
import numpy as np
import cv2
import torchvision.transforms.functional as F

from torch.utils.data import DataLoader
from PIL import Image
from imageio import imread
from .utils import random_crop, center_crop, side_crop, random_crop_pen, oneside_crop
from skimage.feature import canny
from skimage.color import rgb2gray, gray2rgb


class Dataset(torch.utils.data.Dataset):
    def __init__(self, config, input_flist, fmap_flist, fmask_flist, augment=True, training=True):
        super(Dataset, self).__init__()
        self.input_size = config.INPUT_SIZE
        self.center = config.CENTER
        self.model = config.MODEL
        self.augment = augment
        self.training = training
        self.data = self.load_flist(input_flist)
        self.side = config.SIDE
        
        self.count = 0
        self.pos = None
        self.batchsize = config.BATCH_SIZE
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):

        item = self.load_one_side_expand(index)
        return item

    def resize(self, img, height, width):
        img = cv2.resize(img, dsize=(height, width), interpolation=cv2.INTER_CUBIC)

        return img

    def load_name(self, index):
        name = self.data[index]
        return os.path.basename(name)
    
    def load_item(self, index):
        size = self.input_size
        data = imread(self.data[index])
        
        if len(data.shape) == 2:
            data = data[:, :, np.newaxis]
            data = data.repeat(3, axis=2)
        if size != 0:
            data = self.img_resize(data, size, size)
            half_data = self.img_resize(data, size//2, size//2)
            
        pdata, pos, mask = self.cpimage(data)
        fmask_data = mask
        
        self.count += 1
        if self.count == self.batchsize:
            self.count = 0
        
        if self.augment and np.random.binomial(1, 0.5) > 0:
            half_data = half_data[:, ::-1, ...]
            data = data[:, ::-1, ...]
            pdata = pdata[:, ::-1, ...]
            fmask_data = fmask_data[:, ::-1, ...]
            # temp_mask = temp_mask[:, ::-1, ...]
        
        return self.to_tensor(half_data if self.model == 2 else data), self.to_tensor(pdata), torch.IntTensor(pos),\
                self.to_tensor(fmask_data), self.to_tensor(data) * (1 - self.to_tensor(fmask_data))
    
    def load_item_side_expand(self, index):
        size = self.input_size
        data = imread(self.data[index])
        
        if len(data.shape) == 2:
            data = data[:, :, np.newaxis]
            data = data.repeat(3, axis=2)
            
        pos, mask = side_crop(data, 256)
        
        pdata = self.to_tensor(data) * self.to_tensor(mask)
        
        return self.to_tensor(data), pdata, torch.IntTensor(pos),\
                self.to_tensor(mask), pdata
    
        
    def load_one_side_expand(self, index):
        size = self.input_size
        data = imread(self.data[index])
        if len(data.shape) == 2:
            data = data[:, :, np.newaxis]
            data = data.repeat(3, axis=2)
            
        pdata, mask, w = oneside_crop(data, 0)  
        pos = self.to_tensor(data) * self.to_tensor(mask)
        data = self.resize(data ,w, 256)
        #print(pdata.shape,data.shape)
        return self.to_tensor(data), self.to_tensor(pdata),pos,\
                self.to_tensor(data), self.to_tensor(pdata)

    
        
    def img_resize(self, img, height, width, centerCrop=True):
        imgh, imgw = img.shape[0:2]

        if centerCrop and imgh != imgw:
            # center crop
            side = np.minimum(imgh, imgw)
            j = (imgh - side) // 2
            i = (imgw - side) // 2
            img = img[j:j + side, i:i + side, ...]

        img = cv2.resize(img, dsize=(height, width))

        return img

    def cpimage(self, data):
        if self.center == 0:
            if self.model == 4:
                rc, pos, mask = random_crop_pen(data, int(data.shape[0]/2), self.count, self.pos)
                self.pos = pos
            else:
                rc, pos, mask = random_crop(data, int(data.shape[0]/2))
        else:
            rc, pos, mask = center_crop(data, int(data.shape[0]/2))
        return rc, pos, mask
    
    def gray_fmap(self, fmap_data):
        fmap_data = cv2.cvtColor(fmap_data, cv2.COLOR_BGR2GRAY)
        fmap_data[fmap_data < fmap_data.mean()+15] = 0
        fmap_data = cv2.equalizeHist(fmap_data)
        
        return fmap_data


    def load_flist(self, flist):
        if isinstance(flist, list):
            return flist

        # flist: image file path, image directory path, text file flist path
        if isinstance(flist, str):
            if os.path.isdir(flist):
                flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
                flist.sort()
                return flist

            if os.path.isfile(flist):
                try:
                    return np.genfromtxt(flist, dtype=np.str, encoding='utf-8')
                except:
                    return [flist]
        
        return []
    
    def to_tensor(self, img):
        img = Image.fromarray(img)
        img_t = F.to_tensor(img).float()
        return img_t

    def create_iterator(self, batch_size):
        while True:
            sample_loader = DataLoader(
                dataset=self,
                batch_size=batch_size,
                drop_last=True
            )

            for item in sample_loader:
                yield item