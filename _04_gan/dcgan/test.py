import os
import numpy as np
import cv2
import torch
import torchvision.transforms

from gan.dcgan import train
from gan.dcgan.g_net import G_Net

g_weight_file = 'dcgan_params/g_net.pth'

g_net = G_Net()
if os.path.exists(g_weight_file) and os.path.getsize(g_weight_file) != 0:
    g_net.load_state_dict(torch.load(g_weight_file))

device = torch.device("cuda")

input_size = (288, 288)
hh = 184
while True:
    z = torch.randn(1, 128, 1, 1)

    out = g_net(z)  # type:torch.Tensor

    temp = out[0]
    temp = temp * 0.5 + 0.5
    # img = torchvision.transforms.ToPILImage()(temp)
    # img.show()

    a = int((input_size[0] - hh) / 2)
    img = temp[:, a:a + hh, :]
    img1 = torchvision.transforms.ToPILImage()(img)
    img1.show()
