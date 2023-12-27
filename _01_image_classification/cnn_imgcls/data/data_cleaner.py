"""
2022.9.26
从文件夹读取数据
提供dataloader
数据增强
将数据加载到GPU

dataset 为数据集
dataloader 为数据集加载器
"""

import os
import cv2
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torch.nn.functional as f
from torchvision.transforms import InterpolationMode

input_size = (512, 612)
class_num = 2


trans_train = torchvision.transforms.Compose([
    torchvision.transforms.Resize(input_size),
    torchvision.transforms.Pad(100, padding_mode='symmetric'),
    #torchvision.transforms.GaussianBlur(kernel_size=(3, 15), sigma=(0.1, 15.0)),  # 随机高斯模糊
    torchvision.transforms.ColorJitter(brightness=(0.3, 3), contrast=(0.5, 1.5), saturation=0.9),  # 亮度、对比度、饱和度
    torchvision.transforms.RandomRotation(10, expand=False, interpolation=InterpolationMode.BILINEAR),
    torchvision.transforms.CenterCrop(input_size),
    torchvision.transforms.ToTensor(),
])

trans_val = torchvision.transforms.Compose([
    torchvision.transforms.Resize(input_size),
    torchvision.transforms.ToTensor(),
])

datasets_train = ImageFolder('D:/work/files/deeplearn_datasets/清洗机料筐', transform=trans_train)
datasets_val = ImageFolder('D:/work/files/deeplearn_datasets/清洗机料筐', transform=trans_val)

if __name__ == '__main__':
    dataloader_train = DataLoader(datasets_train, 1, shuffle=True)
    dataloader_val = DataLoader(datasets_val, 1, shuffle=True)

    max_width = 0
    max_height = 0
    for imgs, labels in dataloader_val:
        img1 = imgs[0, :, :, :]
        img1 = torchvision.transforms.ToPILImage()(img1)
        img1.show()
        if img1.size[0] > max_height:
            max_height = img1.size[0]
        if img1.size[1] > max_width:
            max_width = img1.size[1]
        pass
    print(max_width, max_height)
