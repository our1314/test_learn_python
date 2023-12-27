import os
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision
from torch.utils.data import Dataset
from our1314 import work

class data_keypoints(Dataset):
    def __init__(self, data_path):
        self.Images = []
        self.Labels = []

        images_path = os.path.join(data_path, "images")
        labels_path = os.path.join(data_path, "labels")

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
        img = torchvision.transforms.Resize((300,300))(img)
        # img = work.PadSquare(img)

        # img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
        # img = torchvision.transforms.Resize(300)(img)
        # img = torch.float(img)
        # img /= 255.
        # img = torch.permute(img, (2, 0, 1))  # 交换维度

        pos = []
        with open(label_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            data_list = lines[0].split(' ')
            x1 = float(data_list[5])
            y1 = float(data_list[6])
            x2 = float(data_list[8])
            y2 = float(data_list[9])
            x3 = float(data_list[11])
            y3 = float(data_list[12])
            x4 = float(data_list[14])
            y4 = float(data_list[15])
            pos = torch.tensor([x1, y1, x2, y2, x3, y3, x4, y4])

        return img, pos


if __name__ == '__main__':
    mydata = data_keypoints('D:/work/files/deeplearn_datasets/choujianji/src-keypoint')
    # im = Image.open(data[0])
    # img.show(img)
    img, pos = mydata[0]
    img = img.numpy()
    img *= 255

    cv2.namedWindow("dis")
    cv2.imshow('dis', img)
    cv2.waitKey()
