from typing import Any
from torch.utils.data import Dataset, DataLoader
import torch
import os
import torchvision
from PIL import Image
import cv2
from torchvision.transforms import InterpolationMode
import numpy as np
import torchvision.transforms.functional as F
from our1314.work import *

class addnoise():
    def __call__(self, x):
        noise = torch.randn_like(x[0])*(random.random()*0.02+0.01)
        x[0] = x[0] + noise
        return x
    
# 数据增强的种类：1.平移、翻转、旋转、尺寸、仿射变换 2.亮度、颜色、噪声，其中1部分需要同时对图像和标签进行操作，2部分只对图像有效部分进行操作
#input_size = (448-32, 448-32)#图像尺寸应该为16的倍数
input_size = (304,304)
transform1 = torchvision.transforms.Compose([
    ToTensors(),
    Resize1(input_size[0]),#等比例缩放
    PadSquare(),
    randomaffine_imgs(0.5, [-5,5], [-0.1,0.1], [-0.1,0.1], [0.7,0.9]),
    #randomaffine_imgs(1, [-0,0], [-0.0,0.0], [-0.0,0.0], [0.7,1/0.8]),
    randomvflip_imgs(0.5),
    randomhflip_imgs(0.5),
    #addnoise()
])

transform2 = torchvision.transforms.RandomApply([
    torchvision.transforms.GaussianBlur(kernel_size=(3, 13)),  # 随机高斯模糊
    torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.3)], p=0.6)

transform_val = torchvision.transforms.Compose([
    ToTensors(),
    Resize1(input_size[0]),  # 按比例缩放
    PadSquare()  # 四周补零
])



class data_seg(Dataset):
    def __init__(self, data_path, transform1=None, transform2=None):
        self.transform1 = transform1
        self.transform2 = transform2

        self.Images = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.jpg')]
        self.Labels = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.png')]

    def __len__(self):
        return len(self.Images)

    def __getitem__(self, item):
        
        file_name1, extension = os.path.splitext(os.path.basename(self.Images[item]))
        file_name2, extension = os.path.splitext(os.path.basename(self.Labels[item]))

        assert file_name1 == file_name2,"文件不相同！"

        image = Image.open(self.Images[item]).convert("RGB")
        label = Image.open(self.Labels[item]).convert("L")

        if self.transform1 != None:
            image,label = self.transform1([image,label])

        if self.transform2 != None:
            image = self.transform2(image)

        #将label转换为one hot
        eye_ = torch.eye(2,2,dtype=torch.float32)
        idx = label.reshape(-1).int()
        eye_ = eye_[idx]
        label = eye_.reshape(label.shape[1], label.shape[2], 2)
        label = label.permute(2,0,1)
        return image, label


if __name__ == '__main__':
    #a = data_seg.__dict__
    e = np.ones((5,5))
    print(e)
    e1 = e[np.array([1,2,3,3,2,1,2,1,3,4,1,2,1]).reshape(-1)]
    print(e1)

    e=torch.eye(5)
    print(e)
    e1 = e[torch.tensor([1,2,3,3,2,1,2,1,3,4,1,2,1])]
    print(e1)

    data = data_seg('D:/work/files/deeplearn_datasets/xray空洞检测/空洞检测生成数据集/train', transform1=transform1, transform2=transform2)
    data_loader = DataLoader(data, batch_size=1, shuffle=True)
    for image, label in data_loader:
        img = image[0]
        mask = label[0]

        img = np.transpose(img.numpy(), (1, 2, 0))
        mask = np.transpose(mask.numpy(), (1, 2, 0))
        
        tmp = img.copy()
        tmp[:,:,2:3] = tmp[:,:,2:3]*0.5 + mask*0.5

        dis = cv2.hconcat([img, cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR), tmp])
        cv2.imshow("dis", dis)        
        cv2.waitKey()
        
