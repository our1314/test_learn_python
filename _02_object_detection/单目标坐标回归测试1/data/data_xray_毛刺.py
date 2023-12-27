import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
import torchvision
from torch.utils.data import Dataset

import sys
sys.path.append("../../")
from our1314 import work

class data_xray_毛刺(Dataset):
    def __init__(self, data_path, transform):
        self.Images = [os.path.join(data_path, f) for f in os.listdir(data_path) if
                       f.endswith(".png") or f.endswith(".jpg") or f.endswith(".bmp")]  # 列表解析
        self.Labels = [os.path.join(data_path, f) for f in os.listdir(data_path) if
                       f.endswith('.txt') and (f != 'classes.txt')]  # 列表解析
        self.transform = transform
        
    def __len__(self):
        return len(self.Images)

    def __getitem__(self, item):
        image_path = self.Images[item]
        label_path = self.Labels[item]

        # 读取图像
        img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)  # type:cv2.Mat

        # 读取标签
        f = open(label_path)
        str = f.read()
        f.close()
        label = work.yolostr2data(str)

        #
        orgH, orgW, ch = img.shape
        newH, newW = 140, 410
        fx = newW / orgW
        fy = newH / orgH
        img = cv2.resize(img, None, fx=fx, fy=fy)

        x0=0
        y0=0
        w=0
        h=0
        for i, l in enumerate(label):
            _, x0, y0, w, h = l
            # _x0 = x0 * newW
            # _y0 = y0 * newH
            # _w = w * newW
            # _h = h * newH
            # pt1 = (int(_x0 - _w / 2), int(_y0 - _h / 2))
            # pt2 = (int(_x0 + _w / 2), int(_y0 + _h / 2))
            # cv2.rectangle(img, pt1, pt2, (0, 0, 255), 1)
            # cv2.imshow("dis", img)
            # cv2.waitKey()

        img_tensor = work.mat2tensor(img)
        pos = torch.tensor([x0, y0, w, h])
        return img_tensor, pos


if __name__ == '__main__':
    transf = torchvision.transforms.ToTensor()
    mydata = data_xray_毛刺('D:\desktop\XRay毛刺检测\TO252样品图片\TO252编带好品\ROI\out1/train', transf)
    # im = Image.open(data[0])
    # img.show(img)

    for img, pos in mydata:
        dis = work.tensor2mat(img)
        H, W, CH = dis.shape
        x0, y0, w, h = pos
        x0 *= W
        y0 *= H
        w *= W
        h *= H

        pt1 = (int(x0 - w / 2), int(y0 - h / 2))
        pt2 = (int(x0 + w / 2), int(y0 + h / 2))
        cv2.rectangle(dis, pt1, pt2, (0, 0, 255), 1)
        cv2.imshow("dis", dis)
        cv2.waitKey(500)
