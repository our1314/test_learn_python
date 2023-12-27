from math import *
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class TestData(Dataset):
    def __init__(self, path):
        self.Images = [os.path.join(path, f) for f in os.listdir(path) if
                       f.endswith(".png") or f.endswith(".jpg") or f.endswith(".bmp")]  # 列表解析

    def __getitem__(self, item):
        image_path = self.Images[item]

        # 1、读取图像
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)  # type:cv2.Mat
        image = self.pad(image)
        # cv2.imshow('dis', image)
        # cv2.waitKey()

        # 2、生成放射变换矩阵
        t = 0 / 180 * pi
        x = np.random.randint(0, 20)
        y = np.random.randint(0, 20)
        M = np.array([[cos(t), -sin(t), x],
                      [sin(t), cos(t), y]], dtype=float)

        # 3、对图像进行变换
        target = cv2.warpAffine(image, M, image.shape)
        # cv2.imshow('dis', target)
        # cv2.waitKey()

        # 4、对点进行变换
        edge = 36
        pts1 = np.array([[36, 36 + 168, 36],
                         [36, 36, 36 + 168],
                         [1, 1, 1]], dtype=float)
        pts2 = M.dot(pts1)

        pts1 = pts1[0:2, :].T
        pts2 = pts2.T

        train_image = np.dstack((image, target))
        train_pts = pts2 - pts1

        train_image = torch.Tensor(train_image) / 255.0  # type:torch.Tensor
        train_pts = torch.Tensor(train_pts) / 20.0  # type:torch.Tensor

        train_image = torch.permute(train_image, (2, 1, 0))
        #train_pts = train_pts.view(-1, 6)
        # m = torch.max(train_image)
        # m1 = torch.max(train_pts)
        return train_image, train_pts

    def __len__(self):
        return len(self.Images)

    def pad(self, img):
        h, w = img.shape
        m = max(h, w)
        l = int(sqrt(m ** 2 + m ** 2))
        top = (l - h) // 2
        bottom = l - h - top
        left = (l - w) // 2
        right = l - w - left
        img = cv2.copyMakeBorder(img, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT)

        img = cv2.resize(img, (128, 128))
        return img


if __name__ == '__main__':
    a = TestData('D:/desktop/ccc')
    c = a[0]
