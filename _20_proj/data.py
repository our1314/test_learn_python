import os
import random
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision
from torch.utils.data import Dataset
import halcon

class data_ic(Dataset):
    def __init__(self, data_path):
        files = [os.path.join(data_path,p) for p in os.listdir(data_path)]

        self.back = [os.path.basename(p).startswith('back') for p in files]
        self.fore = [os.path.basename(p).startswith('fore') for p in files]

    def __len__(self):
        return 100

    def __getitem__(self, item):
        #直接合成图像
        back_image_path = random.choice(self.back, 1)
        img = cv2.imdecode(np.fromfile(back_image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        
        return img, pos

class match1():
    def __init__(temp_path):

        pass

    def findmodel():
        pass

if __name__ == '__main__':
    mydata = data_ic('D:/work/files/data/test_yolo/ic')
    # im = Image.open(data[0])
    # img.show(img)
    img, pos = mydata[0]
    img = img.numpy()
    img *= 255

    cv2.namedWindow("dis")
    cv2.imshow('dis', img)
    cv2.waitKey()
