import argparse
import os

import torch
import torchvision
from PIL.Image import Image
from torch import nn
from torch.utils.data import DataLoader

from image_super.SRCNN.data.data_test_srcnn import data_srcnn
from image_super.SRCNN.test_train import SRCNN


def detect():
    datasets_test = data_srcnn("D:/work/files/deeplearn_datasets/test_datasets/xray_super")
    dataloader_test = DataLoader(datasets_test, 1, shuffle=True)

    net = SRCNN(3)

    checkpoint = torch.load('epoch=999.pth')

    net.load_state_dict(checkpoint['net'], strict=False)  # 加载checkpoint的网络权重

    for img, label in dataloader_test:
        out = net(img)

        ones = torch.ones_like(label)
        zeros = torch.zeros_like(label)

        mask = torch.where(label == 0, label, ones)  # 只有线条区域为0
        xx1 = ones - mask
        xx2 = torch.where(mask == 0, out, zeros)

        img2 = torchvision.transforms.ToPILImage()(xx2[0])
        img2.show()

        loss = nn.L1Loss()(xx1, xx2)

        img1 = torchvision.transforms.ToPILImage()(out[0])
        img1.show()

        img1.save('D:/desktop/3.png')


if __name__ == '__main__':
    detect()
