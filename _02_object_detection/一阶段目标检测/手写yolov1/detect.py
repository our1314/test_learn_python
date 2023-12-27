import argparse
import cv2
import torch
from torch.utils.data import DataLoader
import numpy as np
from object_detection.手写yolov1.model.yolov1 import yolov1
from object_detection.手写yolov1.datasets.data_test_yolov1 import data_test_yolov1
from utils123 import utils


def detect(opt):
    # 0、加载参数
    conf = opt.conf

    # 1、加载网络
    checkpoint = torch.load(opt.weights)
    net = yolov1()
    net.load_state_dict(checkpoint['net'])

    # 2、加载数据
    datasets = data_test_yolov1("D:/work/files/deeplearn_datasets/test_datasets/test_yolo_xray/train")
    data_loader = DataLoader(datasets, 1)
    for images, labels in data_loader:
        pred = net(images)
        for index in range(images.size(0)):

            img = images[index]
            img = utils.tensor2mat(img)

            for i in range(7):
                for j in range(7):
                    box_pred = pred[index, i, j]
                    x0 = 0.0
                    y0 = 0.0
                    w = 0.0
                    h = 0.0
                    if box_pred[4] > conf or box_pred[9] > conf:
                        box = []
                        if box_pred[4] > box_pred[9]:
                            box = box_pred[0:4]
                        else:
                            box = box_pred[4:9]
                        x0 = j + box[0]
                        y0 = i + box[1]
                        w = box[2]
                        h = box[3]
                        x0 *= 416 / 7
                        y0 *= 416 / 7
                        w *= 416
                        h *= 416

                        x0 = x0.detach().numpy()
                        y0 = y0.detach().numpy()
                        w = w.detach().numpy()
                        h = h.detach().numpy()

                        img = utils.rectangle(img, np.array([x0, y0]), np.array([w, h]), (0, 0, 255), 1)
            cv2.imshow("123", img)
            cv2.waitKey()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='run/train/exp/weights/best.pth')
    parser.add_argument('--conf', type=float, default=0.3)
    opt = parser.parse_args()
    detect(opt)
