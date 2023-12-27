from types import FunctionType, MethodType

import torch

x = torch.tensor(1., requires_grad=True)
print(x[0])

loss = x ** 2 * torch.sin(x)
grads = torch.autograd.grad(loss, [x])
print(grads)

x1 = torch.range(1, 10, 0.01)
loss
pass



