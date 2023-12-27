from types import MethodType, FunctionType

import torch
from torch.autograd import grad

print(isinstance(grad, FunctionType))
print(isinstance(grad, MethodType))

a = torch.tensor(1)
print(isinstance(a.storage, MethodType))

