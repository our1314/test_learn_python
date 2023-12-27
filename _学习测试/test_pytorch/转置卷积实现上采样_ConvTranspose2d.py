import torch
from torch import nn

data = torch.rand((2, 5, 19, 19))
con_trans = nn.ConvTranspose2d(in_channels=5, out_channels=3, kernel_size=9)
d = con_trans(data)
print(d.shape)  # torch.Size([2, 3, 27, 27])

from torch.functional import F

data = torch.rand(1, 3, 100, 100)
F.conv2d(data)
