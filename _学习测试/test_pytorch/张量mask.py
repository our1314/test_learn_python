import torch

a = torch.rand(3, 4)
print(a)
mask = torch.randint(2, (3, 4))
print(mask)
mask = mask == 1
print(mask)

print(a[mask])
