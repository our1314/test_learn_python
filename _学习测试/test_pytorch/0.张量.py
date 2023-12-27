import torch
from torch import *

t2 = torch.Tensor([[1, 2, 3, 4, 5, 6]])  # type:torch.Tensor
print(t2)

print(t2.size(-1))

print(t2.view(2, 3))
print(t2.reshape(2, 3))

# t2.transpos(0, 1)

print(torch.Tensor(2))
print(torch.tensor(2).size())

print(t2.permute(0, 1))  # permute可以重新排列维度
print(t2.permute(1, 0))
pass


