import torch


a = torch.randn(1, 3)
print(a)

q = torch.tensor([0, 0.5, 1])
b = torch.quantile(a,q)
print(b)
