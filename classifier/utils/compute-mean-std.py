from torchvision.datasets import ImageFolder
import torch
from torchvision import transforms 
import numpy as np


def compute_mean_std(dataset):

    data_r = np.dstack([dataset[i][0][0, :, :] for i in range(len(dataset))])
    data_g = np.dstack([dataset[i][0][1, :, :] for i in range(len(dataset))])
    data_b = np.dstack([dataset[i][0][2, :, :] for i in range(len(dataset))])
    print(3)
    mean = np.mean(data_r), np.mean(data_g), np.mean(data_b)
    std = np.std(data_r), np.std(data_g), np.std(data_b)

    return mean, std
print(1)
train_set = ImageFolder('/home/ouc/Documents/dataset/train/',transform=transforms.Compose([
                                                           transforms.Resize((224,224)),
                                                   transforms.ToTensor()]))
train_data=torch.utils.data.DataLoader(train_set,batch_size=20,shuffle=True)
print(2)
# print(len(train_data.dataset))
# # print(train_set[11000][0])
mean, std = compute_mean_std(train_data.dataset)
print(mean)
print(std)

