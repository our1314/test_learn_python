import os
import cv2
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from our1314.work import Utils
from ..model import basic
import numpy as np


class data_test_yolov1(Dataset):
    def __init__(self, data_path, image_size=416, grid_size=7, num_bboxes=2, num_classes=2):
        self.image_size = image_size
        self.grid_size = grid_size
        self.num_bboxes = num_bboxes
        self.num_classes = num_classes

        self.Images = []
        self.Labels = []

        images_path = os.path.join(data_path, "images")
        labels_path = os.path.join(data_path, "labels")

        self.Images = [images_path + '/' + f for f in os.listdir(images_path) if
                       f.endswith('.jpg') or f.endswith('.png') or f.endswith('.bmp')]
        self.Labels = [labels_path + '/' + f for f in os.listdir(labels_path) if f.endswith('.txt')]

    def __len__(self):
        return len(self.Images)

    def __getitem__(self, item):
        image_path = self.Images[item]
        label_path = self.Labels[item]

        # region 1、读取图像,并按比例缩放到指定尺寸,不足的地方补零
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        max_len = max(img.shape)
        fx = fy = self.image_size / max_len
        img = cv2.resize(img, dsize=(0, 0), fx=fx, fy=fy)
        shape_resize = img.shape

        padding_up = int((self.image_size - img.shape[0]) / 2)
        padding_lr = int((self.image_size - img.shape[1]) / 2)
        img = cv2.copyMakeBorder(img, padding_up, padding_up, padding_lr, padding_lr, borderType=cv2.BORDER_CONSTANT,
                                 value=0)
        img = torchvision.transforms.ToTensor()(img)
        # endregion

        # region 2、读取标签,并根据图像的缩放方式调整bbox坐标值
        labels = []
        with open(label_path, 'r') as f:
            lines = f.readlines()
            lines = [l.strip().split(' ') for l in lines]

            d1 = len(lines)
            d2 = len(lines[0])
            labels = torch.empty((d1, d2))

            for i, data_list in enumerate(lines):
                data_list = [float(f.strip()) for f in data_list]
                cls, x, y, w, h = data_list

                # 根据图像的处理方式，重新计算label坐标
                y = (y * shape_resize[0] + padding_up) / self.image_size
                x = (x * shape_resize[1] + padding_lr) / self.image_size

                h = h * shape_resize[0] / self.image_size
                w = w * shape_resize[1] / self.image_size
                labels[i, :] = torch.tensor([cls, x, y, w, h])
        # endregion 

        labels = basic.encode(labels, self.grid_size, self.num_bboxes, self.num_classes)
        return img, labels


if __name__ == '__main__':
    image_size = 416
    datasets_train = data_test_yolov1('D:/work/files/deeplearn_datasets/test_datasets/test_yolo_xray/train',
                                      image_size=image_size)
    dataloader_train = DataLoader(datasets_train, 1, shuffle=True)
    for imgs, labels in dataloader_train:
        img1 = imgs[0, ...]
        dis = Utils.tensor2mat(img1)
        dis = Utils.drawgrid(dis, (7, 7))

        label = labels[0]
        # obj_mask = label[:, :, 4] > 0
        # objs = label[obj_mask]indexing with dtype torch.uint8 is now deprecated
        for j in range(label.shape[0]):
            for i in range(label.shape[1]):
                c = label[j, i][4]
                if c > 0:
                    data = label[j, i]
                    d = image_size / 7
                    ij = torch.tensor([i, j]) * d
                    xy = ij + data[:2] * d
                    wh = data[2:4] * image_size
                    Utils.rectangle(dis, xy.numpy(), wh.numpy(), (255, 0, 0), 2)
                    cv2.circle(dis, tuple(xy.numpy().astype(np.int)), 3, (255, 0, 0), -1)
        cv2.imshow('dis', dis)
        cv2.waitKey()
