import torch.utils.data
from torchvision.datasets import ImageFolder

import torchvision
from torch.utils.data import dataloader

dataset = torchvision.datasets.ImageFolder('D:\\桌面\\有无识别数据集2022.6.20', transform=torchvision.transforms.ToTensor())
a = torch.utils.data.Subset(dataset, range(int(0.8 * len(dataset))))
b = torch.utils.data.Subset(dataset, range(int(0.2 * len(dataset))))
print(len(a))
print(len(b))
for (img, index) in dataloader.DataLoader(dataset, 5, True):
    pass
