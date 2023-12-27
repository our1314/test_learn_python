import torch
from torch import nn

#滑窗提取数据
batches_img = torch.rand(1,1,6,6)
print("batches_img:\n",batches_img,batches_img.shape)

nn_Unfold = nn.Unfold(kernel_size=(2,2), dilation=1, padding=0, stride=2)
patche_img = nn_Unfold(batches_img)

print("patche_img.shape:",patche_img.shape)
print("patch_img:\n",patche_img)
