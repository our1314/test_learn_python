import torch

a = torch.rand(4, 6, dtype=float)
b = a.split((2, 2, 1, 1), 1)
pass
