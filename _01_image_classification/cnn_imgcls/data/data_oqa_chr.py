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

import sys
sys.path.append("../../../")
from our1314 import work

input_size = (200, 200)
class_num = 36


class SquarePad3:
    def __call__(self, image):
        # image = torchvision.transforms.Pad()
        w, h = image.size
        max_wh = 195  # max(w, h)

        left = int((max_wh - w) / 2)
        right = max_wh - w - left
        top = int((max_wh - h) / 2)
        bottom = max_wh - h - top

        padding = [left, top, right, bottom]
        image = torchvision.transforms.Pad(padding=padding, fill=0)(image)  # left, top, right and bottom
        return image


class SquarePad2:
    def __call__(self, image):
        max_wh = 200  # max(w, h)
        img = work.pil2mat(image)
        h, w, c = img.shape
        f = max_wh / max(h, w)
        resize_img = cv2.resize(img, (0, 0), fx=f, fy=f)
        image = work.mat2pil(resize_img)

        # image = torchvision.transforms.Pad()
        w, h = image.size

        left = int((max_wh - w) / 2)
        right = max_wh - w - left
        top = int((max_wh - h) / 2)
        bottom = max_wh - h - top

        padding = [left, top, right, bottom]
        image = torchvision.transforms.Pad(padding=padding, fill=0)(image)  # left, top, right and bottom

        return image


class SquarePad:
    def __call__(self, image):
        max_wh = 200  # max(w, h)
        img = work.pil2mat(image)
        resize_img = cv2.resize(img, (200, 200))
        image = work.mat2pil(resize_img)
        return image


trans_train = torchvision.transforms.Compose([
    torchvision.transforms.Resize(input_size),
    torchvision.transforms.Pad(100, padding_mode='symmetric'),
    torchvision.transforms.GaussianBlur(kernel_size=(3, 15), sigma=(0.1, 15.0)),  # 随机高斯模糊
    torchvision.transforms.ColorJitter(brightness=(0.5, 1.5), contrast=(0.5, 1.5), saturation=0.9),  # 亮度、对比度、饱和度
    torchvision.transforms.RandomRotation(10, expand=False, interpolation=InterpolationMode.BILINEAR),
    torchvision.transforms.CenterCrop(input_size),
    torchvision.transforms.ToTensor(),
])

trans_val = torchvision.transforms.Compose([
    torchvision.transforms.Resize(input_size),
    torchvision.transforms.ToTensor(),
    # torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

datasets_train = ImageFolder('d:/work/files/deeplearn_datasets/OQA/cls', transform=trans_train)
datasets_val = ImageFolder('d:/work/files/deeplearn_datasets/OQA/cls', transform=trans_val)

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
