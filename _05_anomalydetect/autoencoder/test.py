import cv2
import numpy as np
import torch
import torchvision
from torch import nn

from anomalydetect.data import data1

#
from anomalydetect.mvtecCAE import mvtecCAE
from myutils import myutils

checkpoint = torch.load('best.pth')
net = mvtecCAE()
net.load_state_dict(checkpoint['net'])
loss_fn = nn.MSELoss()
loss_fn = nn.BCELoss()

list1 = []
mydata_train = data1('D:/work/files/deeplearn_datasets/anomalydetection/bottle/train/good')
for img, label in mydata_train:  # type:torch.Tensor
    src = myutils.tensor2mat(img)
    input = torch.unsqueeze(img, 0)
    out = net(input)
    t = (out - input).view(-1)
    s = t.sigmoid().sum() / len(t)

    b = myutils.tensor2mat(out[0])
    s1 = loss_fn(out, input)
    cv2.putText(b, f'{str(s1)}', (0, 30), cv2.FONT_ITALIC, 1, (0, 0, 255), 2)
    list1.append(s1)

    b = cv2.hconcat([src, b])
    cv2.imshow('b', b)
    cv2.waitKey()
max = max(list1)
min = min(list1)
print(f'max:{max},min:{min}')
