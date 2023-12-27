import os
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision
from torch.utils.data import Dataset


class data_ic(Dataset):
    def __init__(self, data_path):
        self.Images = []
        self.Labels = []

        images_path = os.path.join(data_path, "Images")
        labels_path = os.path.join(data_path, "Labels")

        for filename in os.listdir(images_path):
            self.Images.append(os.path.join(images_path, filename))

        for filename in os.listdir(labels_path):
            self.Labels.append(os.path.join(labels_path, filename))

    def __len__(self):
        return len(self.Images)

    def __getitem__(self, item):
        image_path = self.Images[item]
        label_path = self.Labels[item]

        img = Image.open(image_path)
        img = torchvision.transforms.ToTensor()(img)

        # img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
        # img = torchvision.transforms.Resize(300)(img)
        # img = torch.float(img)
        # img /= 255.
        # img = torch.permute(img, (2, 0, 1))  # 交换维度

        pos = []
        with open(label_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            data_list = lines[0].split(' ')
            x = float(data_list[1])
            y = float(data_list[2])
            w = float(data_list[3])
            h = float(data_list[4])
            pos = torch.tensor([x, y, w, h])

        return img, pos


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
