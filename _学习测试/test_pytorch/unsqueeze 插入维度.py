# 创建一个3*4的全1二维tensor
import torch

a = torch.ones(3, 4)
print(a)
print(a.dim())


b = a.unsqueeze(a.dim())
print(b.shape)

b = a.unsqueeze(-1)
print(b.shape)

b = a.unsqueeze(0)
print(b.shape)
