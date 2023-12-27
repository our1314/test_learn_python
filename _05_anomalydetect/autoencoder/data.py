import os
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision
from torch.utils.data import Dataset
from our1314.myutils.myutils import mat2tensor

class data1(Dataset):
    def __init__(self, data_path):
        self.Images = [os.path.join(data_path, f) for f in os.listdir(data_path) if
                       f.endswith(".png") or f.endswith(".jpg") or f.endswith(".bmp")]  # 列表解析
        # self.transform = transform

    def __len__(self):
        return len(self.Images)

    def __getitem__(self, item):
        image_path = self.Images[item]

        # 读取图像
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)  #type:cv2.Mat
        img = cv2.resize(img, (256, 256))  #resize为256,256
        img_tensor = mat2tensor(img)
        return img_tensor, img_tensor


if __name__ == '__main__':
    pass