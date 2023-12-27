import torch
from torch import *

t2 = torch.rand((42, 6))  # type:torch.Tensor
t3 = t2.repeat(3, 1, 1)
pass
