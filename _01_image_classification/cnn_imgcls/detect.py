import argparse
import os

import torch
from PIL import Image
from data.data_xray_sot23 import data_xray_sot23
from models.net_xray import net_xray


def detect(opt):
    path = 'D:/work/files/deeplearn_datasets/x-ray/cls-dataset/sot23/train/ok/2023-03-15_09.42.39-161.png'
    img = Image.open(path)
    net = net_xray()
    checkpoint = torch.load()
    img.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--weights', default='run/train/exp/weights/best.pth')
    # parser.add_argument('--input_size', default=data_xray_sot23.input_size, type=dict)  # 修改
    # parser.add_argument('--img_path', default='', type=str)
    # parser.add_argument('--out_path', default='run/detect/exp_xray_sot23', type=str)

    opt = parser.parse_args()
    detect(opt)
