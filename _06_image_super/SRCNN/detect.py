import argparse
import os

import torch
import torchvision
from PIL.Image import Image
from torch.utils.data import DataLoader

from image_super.SRCNN.data.data_test_srcnn import data_srcnn
from image_super.SRCNN.models.net_srcnn import SRCNN


def detect(opt):
    datasets_test = data_srcnn("D:/work/files/deeplearn_datasets/test_datasets/xray_super")
    dataloader_test = DataLoader(datasets_test, 1, shuffle=True)

    net = SRCNN(3)

    checkpoint = torch.load(opt.weights)
    net.load_state_dict(checkpoint['net'], strict=False)  # 加载checkpoint的网络权重

    for img, label in dataloader_test:
        out = net(img)
        img1 = torchvision.transforms.ToPILImage()(out[0])
        img1.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='epoch=9999.pth')
    parser.add_argument('--img_path',
                        default='D:/work/files/deeplearn_datasets/test_datasets/gen_xray/out/super_test/images',
                        type=str)
    parser.add_argument('--out_path', default='run/detect/exp', type=str)

    opt = parser.parse_args()
    detect(opt)
