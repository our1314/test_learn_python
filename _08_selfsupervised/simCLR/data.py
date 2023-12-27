import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision.transforms import transforms
from torch.utils.data import Dataset
import sys
sys.path.append('../../')
import myutils.myutils
from myutils.myutils import yolostr2data


train_transform = transforms.Compose([
        transforms.Resize((100,100)),
        # transforms.CenterCrop((200,200)),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        transforms.RandomApply([transforms.ColorJitter(0.4,0.4,0.4,0.2)], p=0.8),
        #transforms.RandomGrayscale(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    
test_transform = transforms.Compose([
        transforms.Resize((100,100)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])


class ChouJianJi(Dataset):
    def __init__(self, data_path, transform=None):
        self.files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith(".png") or f.endswith(".jpg") or f.endswith(".bmp")]  # 列表解析
        self.transform = transform

    def __getitem__(self, item):
        image_path = self.files[item]
        src = Image.open(image_path).convert('RGB')
        if self.transform!=None:
            img1 = self.transform(src)
            img2 = self.transform(src)
        return img1, img2
    
    def __len__(self):
        return len(self.files)
    

if __name__ == '__main__':
    mydata = ChouJianJi('D:/work/proj/抽检机/program/ChouJianJi/data/ic', train_transform)

    for img1, img2 in mydata:
        img = torch.cat((img1,img2),dim=2)
        mean = torch.tensor([0.485, 0.456, 0.406]).reshape((3,1,1))
        std = torch.tensor([0.229, 0.224, 0.225]).reshape((3,1,1))
        img = img*std+mean

        dis = myutils.myutils.tensor2mat(img)
        cv2.imshow("dis", dis)
        cv2.waitKey(500)
