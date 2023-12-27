import argparse
import cv2
import torch
from torch.utils.data import DataLoader

# from image_segmentation.unet.data.data_segment import data_oqa
# from image_segmentation.unet.models.unet import UNet
# from utils123 import utils


def detect(opt):
    # 0、加载参数
    conf = opt.conf
    cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # 1、加载网络
    checkpoint = torch.load(opt.weights)
    net = UNet(3, 2)
    net.load_state_dict(checkpoint['net'])

    # 2、加载数据
    datasets = data_oqa("D:/work/files/deeplearn_datasets/test_datasets/自动生成数据集/out")
    data_loader = DataLoader(datasets, 1)
    for images, labels in data_loader:
        pred = net(images)
        pred = torch.permute(pred, (0, 2, 3, 1))
        pred = torch.softmax(pred, dim=-1)
        mask = torch.argmax(pred, -1)

        mat = utils.tensor2mat(mask)
        a, tt = cv2.threshold(mat, 0, 255, cv2.THRESH_BINARY)

        cv2.imshow("123", tt)
        cv2.waitKey()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='./run/train/exp_oqa/weights/epoch=2640.pth')
    parser.add_argument('--conf', type=float, default=0.3)
    opt = parser.parse_args()
    detect(opt)
