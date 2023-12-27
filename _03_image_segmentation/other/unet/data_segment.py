import os
import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torch.functional as F


class data_oqa(Dataset):
    def __init__(self, data_path, transform=None):
        super(data_oqa, self).__init__()
        self.transform = transform

        path_images = data_path + "/images"
        path_labels = data_path + "/masks"

        image_files = os.listdir(path_images)
        label_files = os.listdir(path_labels)

        self.image_files = [path_images + '/' + f for f in os.listdir(path_images) if
                            f.endswith(".jpg") or f.endswith(".png") or f.endswith(".bmp")]  # 列表解析
        self.label_files = [path_labels + '/' + f for f in os.listdir(path_labels) if f.endswith(".png")]  # 列表解析

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image = Image.open(self.image_files[index])
        # label = Image.open(self.label_files[index])
        label = cv2.imdecode(np.fromfile(self.label_files[index], dtype=np.uint8), -1)
        B, G, R = cv2.split(label)

        # max = torch.max(label)
        # min = torch.min(label)

        image = torchvision.transforms.ToTensor()(image)

        a, R = cv2.threshold(R, 0, 1, cv2.THRESH_BINARY)
        label = torch.from_numpy(R).to(torch.int64)
        # label = torchvision.transforms.ToTensor()(R)
        # torch.functional.Tensor()
        max = torch.max(label)
        min = torch.min(label)

        return image, label


datasets_train = data_oqa("D:/work/files/deeplearn_datasets/test_datasets/gen_oqa/out/val")
datasets_val = data_oqa("D:/work/files/deeplearn_datasets/test_datasets/gen_oqa/out/val")

if __name__ == '__main__':
    data = data_oqa("D:/work/files/deeplearn_datasets/test_datasets/gen_oqa/out/val")
    loader = DataLoader(data, 1, True)
    for i, (image, label) in enumerate(loader):
        img1 = image[0, ...]
        label1 = label#label[0, ...]

        # m1 = torch.max(label1)
        # m2 = torch.min(label1)

        # label1 = torch.unsqueeze(label1, dim=0)
        label1 = label1.expand(3, 512, 512)

        m1 = torch.max(label1)
        m2 = torch.min(label1)

        aa = img1 + label1
        aa = (aa-torch.min(aa)) / (torch.max(aa)-torch.min(aa))

        dd = torchvision.transforms.ToPILImage()(aa)
        dd.show()

        m1 = torch.max(aa)
        m2 = torch.min(aa)
        # if img1.size[0] > max_height:
        #     max_height = img1.size[0]
        # if img1.size[1] > max_width:
        #     max_width = img1.size[1]
        pass
